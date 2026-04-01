#!/usr/bin/env python3
"""LCM Engine - Main orchestration layer for Lossless Context Management.

Provides the high-level API for:
- Conversation management
- Message ingestion
- Context assembly with auto-compaction
- Retrieval tools (describe, grep, expand, hybrid_search)

Port of TypeScript engine.ts from lossless-claw-enhanced.
"""

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Awaitable

from estimate_tokens import estimate_tokens
from lcm_types import (
    ConversationRecord,
    MessageRecord,
    MessagePartRecord,
    MessageRole,
    MessagePartType,
    SummaryRecord,
    ContextItemRecord,
    ContextItemType,
    CompletionResult,
    LcmDependencies,
)
from db.config import LcmConfig, resolve_lcm_config
from db.migration import run_lcm_migrations
from db.connection import open_lcm_database
from store import ConversationStore, SummaryStore
from assembler import (
    ContextAssembler,
    AssembledContext,
    AssemblerConfig,
    create_assembler_from_config,
)
from compaction import (
    CompactionEngine,
    CompactionConfig,
    CompactionDecision,
    CompactionResult,
    create_compaction_engine_from_config,
)
from retrieval import (
    RetrievalEngine,
    RetrievalConfig,
    DescribeResult,
    GrepResult,
    ExpandResult,
    HybridSearchResult,
    create_retrieval_engine,
)
from summarize import create_lcm_summarize_from_legacy_params, LegacyParams


@dataclass
class EngineConfig:
    """Configuration for LCM engine.
    
    Controls behavior of context assembly and auto-compaction.
    """
    token_budget: int = 128000
    context_threshold: float = 0.75
    fresh_tail_count: int = 32
    auto_compact: bool = True
    auto_compact_threshold: float = 0.90
    max_tool_calls_per_response: int = 10


@dataclass
class IngestResult:
    """Result from ingesting a message."""
    message_id: int
    conversation_id: int
    seq: int
    token_count: int
    compacted: bool = False
    compaction_result: Optional[CompactionResult] = None


@dataclass
class ContextAssemblyResult:
    """Result from assembling context."""
    messages: List[Dict[str, Any]]
    total_tokens: int
    token_budget: int
    context_items_used: int
    has_summaries: bool
    warnings: List[str] = field(default_factory=list)
    compaction_needed: bool = False
    compaction_decision: Optional[CompactionDecision] = None


class LcmEngine:
    """Main LCM engine for orchestrating conversation management.
    
    Provides:
    - Conversation lifecycle management
    - Message ingestion with automatic compaction
    - Context assembly under token budget
    - Retrieval tools for agents
    """
    
    @staticmethod
    async def create(
        db_path: str,
        config: Optional[EngineConfig] = None,
        dependencies: Optional[LcmDependencies] = None
    ) -> "LcmEngine":
        """Create a new LCM engine instance.
        
        Args:
            db_path: Path to SQLite database
            config: Engine configuration (uses defaults if not provided)
            dependencies: LCM dependencies for external services
            
        Returns:
            Initialized LcmEngine instance
        """
        # Resolve configuration
        if config is None:
            config = EngineConfig()
        
        lcm_config = resolve_lcm_config()
        
        # Open database with migrations
        db = open_lcm_database(db_path, fts5_available=True)
        run_lcm_migrations(db)
        
        # Create stores
        conversation_store = ConversationStore(db)
        summary_store = SummaryStore(db)
        
        # Create assembler
        assembler = create_assembler_from_config(
            lcm_config,
            get_message_parts=conversation_store.get_message_parts,
            log=None
        )
        
        # Create compaction engine (async)
        async def summarizer(text: str, aggressive: bool = False, options: Optional[Dict] = None) -> str:
            if dependencies and dependencies.complete:
                # Use actual LLM completion
                result = await dependencies.complete(text, aggressive, options)
                if result.error:
                    raise Exception(f"Summarization failed: {result.error}")
                return result.content[0].get("text", text) if result.content else text
            # Fallback: simple truncation
            max_tokens = options.get("max_tokens", 1200) if options else 1200
            tokens = estimate_tokens(text)
            if tokens <= max_tokens:
                return text
            # Simple proportional truncation
            ratio = max_tokens / tokens
            lines = text.split('\n')
            target_lines = max(1, int(len(lines) * ratio))
            return '\n'.join(lines[:target_lines])
        
        compaction_engine = create_compaction_engine_from_config(
            lcm_config,
            summarizer=summarizer,
            log=None
        )
        
        # Create retrieval engine
        retrieval_config = RetrievalConfig(max_results=50)
        retrieval_engine = RetrievalEngine(retrieval_config, db, log=None)
        
        return LcmEngine(
            db=db,
            conversation_store=conversation_store,
            summary_store=summary_store,
            assembler=assembler,
            compaction_engine=compaction_engine,
            retrieval_engine=retrieval_engine,
            config=config,
            dependencies=dependencies
        )
    
    def __init__(
        self,
        db: sqlite3.Connection,
        conversation_store: ConversationStore,
        summary_store: SummaryStore,
        assembler: ContextAssembler,
        compaction_engine: CompactionEngine,
        retrieval_engine: RetrievalEngine,
        config: EngineConfig,
        dependencies: Optional[LcmDependencies] = None
    ):
        """Initialize the engine (use create() factory method).
        
        Args:
            db: SQLite database connection
            conversation_store: Conversation data store
            summary_store: Summary data store
            assembler: Context assembler
            compaction_engine: Compaction engine
            retrieval_engine: Retrieval engine
            config: Engine configuration
            dependencies: LCM dependencies
        """
        self._db = db
        self._store = conversation_store
        self._summary_store = summary_store
        self._assembler = assembler
        self._compaction = compaction_engine
        self._retrieval = retrieval_engine
        self._config = config
        self._deps = dependencies
    
    # =========================================================================
    # Conversation Management
    # =========================================================================
    
    async def get_or_create_conversation(
        self,
        session_id: str,
        session_key: Optional[str] = None,
        title: Optional[str] = None
    ) -> Tuple[ConversationRecord, bool]:
        """Get existing conversation or create a new one.
        
        Args:
            session_id: Unique session identifier
            session_key: Optional session key for deduplication
            title: Optional conversation title
            
        Returns:
            Tuple of (ConversationRecord, created: bool)
        """
        # Try to find existing conversation
        if session_key:
            existing = self._store.get_conversation_by_session_key(session_key)
            if existing:
                return (existing, False)
        
        existing = self._store.get_conversation_by_session_id(session_id)
        if existing:
            return (existing, False)
        
        # Create new conversation
        conversation_id = self._store.create_conversation(
            session_id=session_id,
            session_key=session_key,
            title=title
        )
        conversation = self._store.get_conversation_by_id(conversation_id)
        return (conversation, True)
    
    # =========================================================================
    # Message Ingestion
    # =========================================================================
    
    async def ingest_message(
        self,
        conversation_id: int,
        role: MessageRole,
        content: str,
        parts: Optional[List[Dict[str, Any]]] = None
    ) -> IngestResult:
        """Ingest a message into a conversation.
        
        Args:
            conversation_id: Target conversation ID
            role: Message role (user/assistant/system/tool)
            content: Message content
            parts: Optional message parts (tool calls, etc.)
            
        Returns:
            IngestResult with message info and potential compaction result
        """
        # Calculate token count
        token_count = estimate_tokens(content)
        
        # Get next sequence number
        latest_msg = self._store.get_latest_message(conversation_id)
        seq = (latest_msg.seq + 1) if latest_msg else 1
        
        # Create message
        message_id = self._store.create_message(
            conversation_id=conversation_id,
            seq=seq,
            role=role,
            content=content,
            token_count=token_count
        )
        
        # Add message parts if provided
        if parts:
            for ordinal, part_data in enumerate(parts):
                self._store.create_message_part(
                    message_id=message_id,
                    session_id=f"session_{conversation_id}",
                    part_type=MessagePartType(part_data.get("type", "text")),
                    ordinal=ordinal,
                    text_content=part_data.get("text_content"),
                    tool_call_id=part_data.get("tool_call_id"),
                    tool_name=part_data.get("tool_name"),
                    tool_status=part_data.get("tool_status"),
                    tool_input=part_data.get("tool_input"),
                    tool_output=part_data.get("tool_output"),
                    tool_error=part_data.get("tool_error")
                )
        
        # Add to context items
        self._store.add_context_message(conversation_id, message_id, seq)
        
        result = IngestResult(
            message_id=message_id,
            conversation_id=conversation_id,
            seq=seq,
            token_count=token_count,
            compacted=False
        )
        
        # Check if auto-compaction is needed
        if self._config.auto_compact:
            compaction_needed = await self._check_auto_compaction(conversation_id)
            if compaction_needed:
                compaction_result = await self._perform_auto_compaction(conversation_id)
                if compaction_result:
                    result.compacted = True
                    result.compaction_result = compaction_result
        
        return result
    
    async def _check_auto_compaction(self, conversation_id: int) -> bool:
        """Check if auto-compaction is needed.
        
        Args:
            conversation_id: The conversation ID
            
        Returns:
            True if compaction threshold exceeded
        """
        messages = self._store.get_messages_by_conversation(conversation_id)
        if not messages:
            return False
        
        # Get all summaries
        summaries = self._summary_store.get_summaries_by_conversation(conversation_id)
        
        # Get context items
        context_items_raw = self._store.get_context_items(conversation_id)
        context_items = [
            ContextItemRecord(
                conversation_id=item["conversation_id"],
                ordinal=item["ordinal"],
                item_type=ContextItemType(item["item_type"]),
                message_id=item["message_id"],
                summary_id=item["summary_id"],
                created_at=item["created_at"]
            )
            for item in context_items_raw
        ]
        
        # Evaluate compaction need
        decision = self._compaction.evaluate(messages, summaries, context_items)
        
        if not decision.should_compact:
            return False
        
        # Check against auto_compact_threshold
        ratio = decision.current_tokens / decision.target_tokens if decision.target_tokens > 0 else 0
        return ratio >= self._config.auto_compact_threshold
    
    async def _perform_auto_compaction(self, conversation_id: int) -> Optional[CompactionResult]:
        """Perform automatic compaction.
        
        Args:
            conversation_id: The conversation ID
            
        Returns:
            CompactionResult or None if failed
        """
        messages = self._store.get_messages_by_conversation(conversation_id)
        summaries = self._summary_store.get_summaries_by_conversation(conversation_id)
        
        context_items_raw = self._store.get_context_items(conversation_id)
        context_items = [
            ContextItemRecord(
                conversation_id=item["conversation_id"],
                ordinal=item["ordinal"],
                item_type=ContextItemType(item["item_type"]),
                message_id=item["message_id"],
                summary_id=item["summary_id"],
                created_at=item["created_at"]
            )
            for item in context_items_raw
        ]
        
        try:
            result = await self._compaction.compact(
                conversation_id=conversation_id,
                messages=messages,
                summaries=summaries,
                context_items=context_items
            )
            
            # Persist new summaries
            for summary in result.new_summaries:
                self._summary_store.create_summary(
                    summary_id=summary.summary_id,
                    conversation_id=summary.conversation_id,
                    kind=summary.kind,
                    depth=summary.depth,
                    content=summary.content,
                    token_count=summary.token_count,
                    file_ids=summary.file_ids,
                    earliest_at=summary.earliest_at,
                    latest_at=summary.latest_at,
                    descendant_count=summary.descendant_count,
                    descendant_token_count=summary.descendant_token_count,
                    source_message_token_count=summary.source_message_token_count,
                    model=summary.model,
                    category=summary.category,
                    tier=summary.tier,
                    importance=summary.importance,
                    scope=summary.scope
                )
                
                # Link to messages if available
                msg_links = self._summary_store.get_messages_for_summary(summary.summary_id)
                if not msg_links:
                    # Try to infer from context
                    pass
            
            # Persist new context items
            for item in result.new_context_items:
                self._summary_store.add_context_summary(
                    conversation_id=item.conversation_id,
                    summary_id=item.summary_id,
                    ordinal=item.ordinal
                )
            
            # Update summary DAG relationships
            for summary in result.new_summaries:
                parents = self._summary_store.get_parent_summaries(summary.summary_id)
                for parent in parents:
                    self._summary_store.add_summary_parent(
                        summary_id=summary.summary_id,
                        parent_summary_id=parent.summary_id
                    )
            
            return result
            
        except Exception as e:
            # Log error but don't fail ingestion
            return None
    
    # =========================================================================
    # Context Assembly
    # =========================================================================
    
    async def assemble_context(
        self,
        conversation_id: int,
        system_prompt: Optional[str] = None,
        force_compact: bool = False
    ) -> ContextAssemblyResult:
        """Assemble context for a conversation.
        
        Args:
            conversation_id: The conversation ID
            system_prompt: Optional system prompt to prepend
            force_compact: Force compaction before assembly
            
        Returns:
            ContextAssemblyResult with assembled messages
        """
        # Get messages
        messages = self._store.get_messages_by_conversation(conversation_id)
        
        # Get summaries
        summaries = self._summary_store.get_summaries_by_conversation(conversation_id)
        
        # Get context items
        context_items_raw = self._store.get_context_items(conversation_id)
        context_items = [
            ContextItemRecord(
                conversation_id=item["conversation_id"],
                ordinal=item["ordinal"],
                item_type=ContextItemType(item["item_type"]),
                message_id=item["message_id"],
                summary_id=item["summary_id"],
                created_at=item["created_at"]
            )
            for item in context_items_raw
        ]
        
        # Check if compaction is needed
        decision = self._compaction.evaluate(messages, summaries, context_items)
        
        if force_compact or (self._config.auto_compact and decision.should_compact):
            await self._perform_auto_compaction(conversation_id)
            # Refresh after compaction
            summaries = self._summary_store.get_summaries_by_conversation(conversation_id)
            context_items_raw = self._store.get_context_items(conversation_id)
            context_items = [
                ContextItemRecord(
                    conversation_id=item["conversation_id"],
                    ordinal=item["ordinal"],
                    item_type=ContextItemType(item["item_type"]),
                    message_id=item["message_id"],
                    summary_id=item["summary_id"],
                    created_at=item["created_at"]
                )
                for item in context_items_raw
            ]
        
        # Assemble context
        assembler_config = AssemblerConfig(
            token_budget=self._config.token_budget,
            fresh_tail_count=self._config.fresh_tail_count,
            context_threshold=self._config.context_threshold
        )
        self._assembler.config = assembler_config
        
        assembled = self._assembler.assemble(
            messages=messages,
            summaries=summaries,
            context_items=context_items,
            system_prompt=system_prompt
        )
        
        return ContextAssemblyResult(
            messages=assembled.messages,
            total_tokens=assembled.total_tokens,
            token_budget=assembled.token_budget,
            context_items_used=assembled.context_items_used,
            has_summaries=assembled.has_summaries,
            warnings=assembled.warnings,
            compaction_needed=decision.should_compact,
            compaction_decision=decision
        )
    
    # =========================================================================
    # Retrieval Tools (for agents)
    # =========================================================================
    
    def describe(
        self,
        item_id: str,
        include_children: bool = True,
        max_depth: int = 3
    ) -> DescribeResult:
        """Look up a summary, file, or message by ID.
        
        Args:
            item_id: ID of the item to look up
            include_children: Include child summary IDs
            max_depth: Maximum depth for child traversal
            
        Returns:
            DescribeResult with item details
        """
        # Increment access count if it's a summary
        summary = self._summary_store.get_summary_by_id(item_id)
        if summary:
            self._summary_store.increment_access_count(item_id)
        
        return self._retrieval.describe(
            item_id=item_id,
            include_children=include_children,
            max_depth=max_depth
        )
    
    def grep(
        self,
        query: str,
        mode: str = "full_text",
        conversation_id: Optional[int] = None,
        since: Optional[datetime] = None,
        before: Optional[datetime] = None,
        limit: int = 50
    ) -> GrepResult:
        """Search for content across messages and summaries.
        
        Args:
            query: Search query
            mode: "regex" or "full_text"
            conversation_id: Optional conversation filter
            since: Optional start datetime filter
            before: Optional end datetime filter
            limit: Maximum results
            
        Returns:
            GrepResult with matching items
        """
        return self._retrieval.grep(
            query=query,
            mode=mode,
            conversation_id=conversation_id,
            since=since,
            before=before,
            limit=limit
        )
    
    def expand(
        self,
        summary_id: str,
        max_tokens: int = 4000,
        include_files: bool = True
    ) -> ExpandResult:
        """Expand a summary to get detailed context.
        
        Traverses the summary hierarchy to collect source messages
        and child summaries.
        
        Args:
            summary_id: ID of the summary to expand
            max_tokens: Maximum tokens for expanded content
            include_files: Whether to include file references
            
        Returns:
            ExpandResult with expanded content
        """
        # Increment access count
        self._summary_store.increment_access_count(summary_id)
        
        return self._retrieval.expand(
            summary_id=summary_id,
            max_tokens=max_tokens,
            include_files=include_files
        )
    
    def hybrid_search(
        self,
        query: str,
        scope: Optional[str] = None,
        min_tier: Optional[str] = None,
        max_tier: Optional[str] = None,
        min_decay_score: Optional[float] = None,
        limit: int = 50
    ) -> HybridSearchResult:
        """Hybrid search with BM25 + decay + recency scoring.
        
        Args:
            query: Search query string
            scope: Optional scope filter
            min_tier: Minimum memory tier
            max_tier: Maximum memory tier
            min_decay_score: Minimum decay score threshold
            limit: Maximum results
            
        Returns:
            HybridSearchResult with scored summaries
        """
        from lcm_types import MemoryTier
        
        min_tier_enum = MemoryTier(min_tier) if min_tier else None
        max_tier_enum = MemoryTier(max_tier) if max_tier else None
        
        return self._retrieval.hybrid_search(
            query=query,
            scope=scope,
            min_tier=min_tier_enum,
            max_tier=max_tier_enum,
            min_decay_score=min_decay_score,
            limit=limit
        )
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_conversation_stats(self, conversation_id: int) -> Dict[str, Any]:
        """Get statistics for a conversation.
        
        Args:
            conversation_id: The conversation ID
            
        Returns:
            Dict with conversation statistics
        """
        message_count = self._store.get_message_count(conversation_id)
        token_sum = self._store.get_message_token_sum(conversation_id)
        summary_stats = self._summary_store.get_summary_stats(conversation_id)
        
        return {
            "conversation_id": conversation_id,
            "message_count": message_count,
            "total_tokens": token_sum,
            "summary_stats": summary_stats
        }
    
    def close(self) -> None:
        """Close the engine and release resources."""
        self._db.close()


if __name__ == "__main__":
    import asyncio
    import tempfile
    from pathlib import Path
    
    async def demo():
        """Demo usage of the LCM engine."""
        print("LCM Engine Demo")
        print("=" * 50)
        
        # Create temporary database
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "lcm_demo.db")
            
            # Create engine
            engine = await LcmEngine.create(db_path)
            
            # Create conversation
            conversation, created = await engine.get_or_create_conversation(
                session_id="demo_session_001",
                session_key="demo_key_001",
                title="Demo Conversation"
            )
            print(f"Conversation created: {created}")
            print(f"Conversation ID: {conversation.conversation_id}")
            
            # Ingest messages
            print("\n--- Ingesting messages ---")
            
            await engine.ingest_message(
                conversation_id=conversation.conversation_id,
                role=MessageRole.USER,
                content="Hello, I need help with Python programming."
            )
            
            response = await engine.ingest_message(
                conversation_id=conversation.conversation_id,
                role=MessageRole.ASSISTANT,
                content="I'd be happy to help with Python! What specific topic would you like to explore?"
            )
            print(f"Response: message_id={response.message_id}, seq={response.seq}")
            
            await engine.ingest_message(
                conversation_id=conversation.conversation_id,
                role=MessageRole.USER,
                content="Can you explain how to use decorators?"
            )
            
            await engine.ingest_message(
                conversation_id=conversation.conversation_id,
                role=MessageRole.ASSISTANT,
                content="""Decorators in Python are functions that modify the behavior of other functions or methods.
                They use the @decorator_name syntax above the function definition.
                Here's an example:
                
                ```python
                def my_decorator(func):
                    def wrapper(*args, **kwargs):
                        print('Before function')
                        result = func(*args, **kwargs)
                        print('After function')
                        return result
                    return wrapper
                
                @my_decorator
                def say_hello():
                    print('Hello!')
                ```
                
                The decorator wraps the original function, adding behavior before and after its execution."""
            )
            
            # Assemble context
            print("\n--- Assembling context ---")
            context = await engine.assemble_context(
                conversation_id=conversation.conversation_id,
                system_prompt="You are a helpful Python programming assistant."
            )
            
            print(f"Total tokens: {context.total_tokens}")
            print(f"Token budget: {context.token_budget}")
            print(f"Context items used: {context.context_items_used}")
            print(f"Has summaries: {context.has_summaries}")
            print(f"Messages count: {len(context.messages)}")
            
            # Test retrieval tools
            print("\n--- Testing retrieval tools ---")
            
            # Grep for decorators
            grep_result = engine.grep("decorator")
            print(f"Grep 'decorator': {grep_result.total_count} matches")
            
            # Get conversation stats
            stats = engine.get_conversation_stats(conversation.conversation_id)
            print(f"\nConversation stats:")
            print(f"  Messages: {stats['message_count']}")
            print(f"  Total tokens: {stats['total_tokens']}")
            print(f"  Summaries: {stats['summary_stats']['total_summaries']}")
            
            # Cleanup
            engine.close()
            
            print("\n" + "=" * 50)
            print("Demo completed successfully!")
    
    # Run demo
    asyncio.run(demo())
