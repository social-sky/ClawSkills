#!/usr/bin/env python3
"""LCM Hook - Decorator/Hook wrapper for nanobot integration.

Provides non-invasive integration with nanobot via decorators and hooks:
- @lcm_hook: Decorator to wrap agent message handlers
- @lcm_tool: Decorator to expose LCM tools to agents
- Hook functions for manual integration

Usage:
    from lcm_wrapper.hook import lcm_hook, lcm_tool, LcmHook

    # Option 1: Decorator - wrap your message handler
    @lcm_hook
    async def handle_message(ctx):
        # ctx.messages is auto-populated with LCM-assembled context
        return ctx

    # Option 2: Hook class - for more control
    hook = LcmHook()
    await hook.initialize()
    
    # On session start
    await hook.on_session_start(session_id="sess_001")
    
    # On each message
    await hook.on_message(role="user", content="Hello")
    
    # Get assembled context
    context = await hook.assemble_context()

    # Option 3: Tools for agents
    @lcm_tool
    def describe(item_id: str) -> str:
        '''Look up a summary or file by ID.'''
        return hook.describe(item_id)
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic
from functools import wraps

# Import from nanobot-lossless-claw-enhanced
from ..nanobot_lossless_claw_enhanced.db.config import resolve_lcm_config
from ..nanobot_lossless_claw_enhanced.db.connection import open_lcm_database
from ..nanobot_lossless_claw_enhanced.db.migration import run_lcm_migrations
from ..nanobot_lossless_claw_enhanced.store.conversation_store import ConversationStore
from ..nanobot_lossless_claw_enhanced.store.summary_store import SummaryStore
from ..nanobot_lossless_claw_enhanced.assembler import ContextAssembler, AssemblerConfig, create_assembler_from_config
from ..nanobot_lossless_claw_enhanced.compaction import CompactionEngine, CompactionConfig, create_compaction_engine_from_config
from ..nanobot_lossless_claw_enhanced.retrieval import RetrievalEngine, RetrievalConfig
from ..nanobot_lossless_claw_enhanced.lcm_types import MessageRole, MessagePartType

F = TypeVar('F', bound=Callable[..., Any])


@dataclass
class HookContext:
    """Context passed through lcm_hook decorated functions."""
    conversation_id: int
    session_id: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    system_prompt: Optional[str] = None
    token_count: int = 0
    compacted: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LcmHookConfig:
    """Configuration for LcmHook."""
    db_path: Optional[str] = None
    token_budget: int = 128000
    context_threshold: float = 0.75
    fresh_tail_count: int = 32
    auto_compact: bool = True
    auto_compact_threshold: float = 0.90
    system_prompt: Optional[str] = None


class LcmHook:
    """LCM Hook for nanobot integration.
    
    Provides methods for:
    - Session lifecycle (on_session_start, on_session_end)
    - Message ingestion (on_message)
    - Context assembly (assemble_context)
    - Retrieval tools (describe, grep, expand, hybrid_search)
    
    Example:
        hook = LcmHook()
        await hook.initialize()
        
        # On session start
        await hook.on_session_start(session_id="sess_001", title="My Chat")
        
        # On user message
        await hook.on_message(role="user", content="Hello!")
        
        # On assistant response
        await hook.on_message(role="assistant", content="Hi there!")
        
        # Get context for LLM
        context = await hook.assemble_context()
        
        # Use retrieval tools
        result = hook.describe("sum_abc123")
    """
    
    _instance: Optional["LcmHook"] = None
    
    def __init__(self, config: Optional[LcmHookConfig] = None):
        """Initialize LCM Hook.
        
        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or LcmHookConfig()
        self._db = None
        self._conversation_store = None
        self._summary_store = None
        self._assembler = None
        self._compaction = None
        self._retrieval = None
        self._current_conversation_id: Optional[int] = None
        self._current_session_id: Optional[str] = None
        self._initialized = False
    
    @classmethod
    def get_instance(cls) -> Optional["LcmHook"]:
        """Get the singleton instance."""
        return cls._instance
    
    async def initialize(self) -> None:
        """Initialize the hook (call once before use)."""
        if self._initialized:
            return
        
        # Resolve database path
        lcm_config = resolve_lcm_config()
        db_path = self.config.db_path or lcm_config.database_path
        
        # Open database with migrations
        self._db = open_lcm_database(db_path, fts5_available=True)
        run_lcm_migrations(self._db)
        
        # Create stores
        self._conversation_store = ConversationStore(self._db)
        self._summary_store = SummaryStore(self._db)
        
        # Create assembler
        self._assembler = create_assembler_from_config(
            lcm_config,
            get_message_parts=self._conversation_store.get_message_parts,
            log=None
        )
        
        # Create compaction engine
        self._compaction = create_compaction_engine_from_config(
            lcm_config,
            summarizer=self._create_summarizer(),
            log=None
        )
        
        # Create retrieval engine
        self._retrieval = RetrievalEngine(
            RetrievalConfig(max_results=50),
            self._db,
            log=None
        )
        
        self._initialized = True
        LcmHook._instance = self
    
    def _create_summarizer(self) -> Callable:
        """Create async summarizer function."""
        async def summarize(
            text: str,
            aggressive: bool = False,
            options: Optional[Dict] = None
        ) -> str:
            # Fallback: simple truncation
            # In production, you would call an actual LLM here
            max_tokens = options.get("max_tokens", 1200) if options else 1200
            from ..nanobot_lossless_claw_enhanced.estimate_tokens import estimate_tokens
            tokens = estimate_tokens(text)
            if tokens <= max_tokens:
                return text
            # Simple proportional truncation
            ratio = max_tokens / tokens
            lines = text.split('\n')
            target_lines = max(1, int(len(lines) * ratio))
            return '\n'.join(lines[:target_lines])
        
        return summarize
    
    async def on_session_start(
        self,
        session_id: str,
        title: Optional[str] = None,
        session_key: Optional[str] = None
    ) -> int:
        """Called when a new session starts.
        
        Args:
            session_id: Unique session identifier
            title: Optional conversation title
            session_key: Optional key for deduplication
            
        Returns:
            conversation_id
        """
        if not self._initialized:
            await self.initialize()
        
        # Check for existing conversation
        if session_key:
            existing = self._conversation_store.get_conversation_by_session_key(session_key)
            if existing:
                self._current_conversation_id = existing.conversation_id
                self._current_session_id = session_id
                return existing.conversation_id
        
        existing = self._conversation_store.get_conversation_by_session_id(session_id)
        if existing:
            self._current_conversation_id = existing.conversation_id
            self._current_session_id = session_id
            return existing.conversation_id
        
        # Create new conversation
        conv_id = self._conversation_store.create_conversation(
            session_id=session_id,
            session_key=session_key,
            title=title
        )
        
        self._current_conversation_id = conv_id
        self._current_session_id = session_id
        
        return conv_id
    
    async def on_message(
        self,
        role: str,
        content: str,
        parts: Optional[List[Dict[str, Any]]] = None,
        conversation_id: Optional[int] = None
    ) -> int:
        """Called when a message is received.
        
        Args:
            role: Message role (user/assistant/system/tool)
            content: Message content
            parts: Optional message parts (tool calls, etc.)
            conversation_id: Override conversation ID (uses current if not provided)
            
        Returns:
            message_id
        """
        if not self._initialized:
            await self.initialize()
        
        conv_id = conversation_id or self._current_conversation_id
        if not conv_id:
            raise ValueError("No conversation ID. Call on_session_start first.")
        
        # Estimate tokens
        from ..nanobot_lossless_claw_enhanced.estimate_tokens import estimate_tokens
        token_count = estimate_tokens(content)
        
        # Get next sequence number
        latest = self._conversation_store.get_latest_message(conv_id)
        seq = (latest.seq + 1) if latest else 1
        
        # Create message
        message_id = self._conversation_store.create_message(
            conversation_id=conv_id,
            seq=seq,
            role=MessageRole(role),
            content=content,
            token_count=token_count
        )
        
        # Create message parts if provided
        if parts:
            for ordinal, part_data in enumerate(parts):
                self._conversation_store.create_message_part(
                    message_id=message_id,
                    session_id=self._current_session_id or f"session_{conv_id}",
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
        self._conversation_store.add_context_message(conv_id, message_id, seq)
        
        # Auto-compact if enabled
        if self.config.auto_compact:
            await self._maybe_compact(conv_id)
        
        return message_id
    
    async def _maybe_compact(self, conversation_id: int) -> bool:
        """Check and perform auto-compaction if needed."""
        messages = self._conversation_store.get_messages_by_conversation(conversation_id)
        if not messages:
            return False
        
        summaries = self._summary_store.get_summaries_by_conversation(conversation_id)
        context_items_raw = self._conversation_store.get_context_items(conversation_id)
        
        from ..nanobot_lossless_claw_enhanced.lcm_types import ContextItemRecord, ContextItemType
        context_items = [
            ContextItemRecord(
                conversation_id=item["conversation_id"],
                ordinal=item["ordinal"],
                item_type=ContextItemType(item["item_type"]),
                message_id=item.get("message_id"),
                summary_id=item.get("summary_id"),
                created_at=item.get("created_at")
            )
            for item in context_items_raw
        ]
        
        decision = self._compaction.evaluate(messages, summaries, context_items)
        
        if not decision.should_compact:
            return False
        
        ratio = decision.current_tokens / decision.target_tokens if decision.target_tokens > 0 else 0
        if ratio < self.config.auto_compact_threshold:
            return False
        
        # Perform compaction
        result = await self._compaction.compact(
            conversation_id=conversation_id,
            messages=messages,
            summaries=summaries,
            context_items=context_items
        )
        
        if result and result.success:
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
            
            for item in result.new_context_items:
                self._summary_store.add_context_summary(
                    conversation_id=item.conversation_id,
                    summary_id=item.summary_id,
                    ordinal=item.ordinal
                )
            
            return True
        
        return False
    
    async def assemble_context(
        self,
        conversation_id: Optional[int] = None,
        system_prompt: Optional[str] = None,
        force_compact: bool = False
    ) -> HookContext:
        """Assemble context for LLM.
        
        Args:
            conversation_id: Override conversation ID
            system_prompt: Override system prompt
            force_compact: Force compaction before assembly
            
        Returns:
            HookContext with assembled messages
        """
        if not self._initialized:
            await self.initialize()
        
        conv_id = conversation_id or self._current_conversation_id
        if not conv_id:
            raise ValueError("No conversation ID. Call on_session_start first.")
        
        # Get messages
        messages = self._conversation_store.get_messages_by_conversation(conv_id)
        
        # Get summaries
        summaries = self._summary_store.get_summaries_by_conversation(conv_id)
        
        # Get context items
        context_items_raw = self._conversation_store.get_context_items(conv_id)
        from ..nanobot_lossless_claw_enhanced.lcm_types import ContextItemRecord, ContextItemType
        context_items = [
            ContextItemRecord(
                conversation_id=item["conversation_id"],
                ordinal=item["ordinal"],
                item_type=ContextItemType(item["item_type"]),
                message_id=item.get("message_id"),
                summary_id=item.get("summary_id"),
                created_at=item.get("created_at")
            )
            for item in context_items_raw
        ]
        
        # Check if compaction needed
        decision = self._compaction.evaluate(messages, summaries, context_items)
        
        if force_compact or (self.config.auto_compact and decision.should_compact):
            await self._maybe_compact(conv_id)
            # Refresh after compaction
            summaries = self._summary_store.get_summaries_by_conversation(conv_id)
            context_items_raw = self._conversation_store.get_context_items(conv_id)
            context_items = [
                ContextItemRecord(
                    conversation_id=item["conversation_id"],
                    ordinal=item["ordinal"],
                    item_type=ContextItemType(item["item_type"]),
                    message_id=item.get("message_id"),
                    summary_id=item.get("summary_id"),
                    created_at=item.get("created_at")
                )
                for item in context_items_raw
            ]
        
        # Assemble context
        assembler_config = AssemblerConfig(
            token_budget=self.config.token_budget,
            fresh_tail_count=self.config.fresh_tail_count,
            context_threshold=self.config.context_threshold
        )
        self._assembler.config = assembler_config
        
        assembled = self._assembler.assemble(
            messages=messages,
            summaries=summaries,
            context_items=context_items,
            system_prompt=system_prompt or self.config.system_prompt
        )
        
        # Increment access counts for used summaries
        for summary in summaries:
            self._summary_store.increment_access_count(summary.summary_id)
        
        return HookContext(
            conversation_id=conv_id,
            session_id=self._current_session_id or "",
            messages=assembled.messages,
            system_prompt=system_prompt or self.config.system_prompt,
            token_count=assembled.total_tokens,
            compacted=decision.should_compact,
            metadata={
                "token_budget": assembled.token_budget,
                "context_items_used": assembled.context_items_used,
                "has_summaries": assembled.has_summaries
            }
        )
    
    # =========================================================================
    # Retrieval Tools
    # =========================================================================
    
    def describe(
        self,
        item_id: str,
        include_children: bool = True,
        max_depth: int = 3
    ) -> str:
        """Look up a summary, file, or message by ID.
        
        Args:
            item_id: ID of the item to look up
            include_children: Include child summary IDs
            max_depth: Maximum depth for child traversal
            
        Returns:
            Formatted description string
        """
        result = self._retrieval.describe(
            item_id=item_id,
            include_children=include_children,
            max_depth=max_depth
        )
        
        if not result.found:
            return f"Item not found: {item_id}"
        
        lines = [f"[{result.item_type.upper()}] {item_id}"]
        lines.append(f"Tokens: {result.token_count}")
        if result.content:
            lines.append(f"\n{result.content[:500]}...")
        
        if result.children:
            lines.append(f"\nChild summaries: {', '.join(result.children[:5])}")
        
        return "\n".join(lines)
    
    def grep(
        self,
        query: str,
        mode: str = "full_text",
        conversation_id: Optional[int] = None,
        limit: int = 10
    ) -> str:
        """Search for content across messages and summaries.
        
        Args:
            query: Search query
            mode: "regex" or "full_text"
            conversation_id: Optional filter by conversation
            limit: Maximum results
            
        Returns:
            Formatted search results
        """
        result = self._retrieval.grep(
            query=query,
            mode=mode,
            conversation_id=conversation_id,
            limit=limit
        )
        
        if not result.matches:
            return f"No matches found for: {query}"
        
        lines = [f"Found {result.total_count} matches for '{query}':\n"]
        for match in result.matches[:limit]:
            lines.append(f"  [{match['type']}] {match.get('snippet', '')[:100]}")
        
        return "\n".join(lines)
    
    def expand(
        self,
        summary_id: str,
        max_tokens: int = 4000
    ) -> str:
        """Expand a summary to get detailed context.
        
        Args:
            summary_id: ID of the summary to expand
            max_tokens: Maximum tokens for expanded content
            
        Returns:
            Expanded content
        """
        # Increment access count
        self._summary_store.increment_access_count(summary_id)
        
        result = self._retrieval.expand(
            summary_id=summary_id,
            max_tokens=max_tokens
        )
        
        if not result.expanded_content:
            return f"Summary not found or empty: {summary_id}"
        
        lines = [f"[Summary {summary_id}] (depth: {result.depth})"]
        lines.append(f"Source messages: {len(result.source_messages)}")
        lines.append(f"\n{result.expanded_content[:2000]}...")
        
        return "\n".join(lines)
    
    def hybrid_search(
        self,
        query: str,
        scope: Optional[str] = None,
        min_tier: Optional[str] = None,
        limit: int = 10
    ) -> str:
        """Hybrid search with BM25 + decay + recency scoring.
        
        Args:
            query: Search query
            scope: Optional scope filter
            min_tier: Minimum memory tier (peripheral/working/core)
            limit: Maximum results
            
        Returns:
            Formatted search results with scores
        """
        from ..nanobot_lossless_claw_enhanced.lcm_types import MemoryTier
        
        min_tier_enum = MemoryTier(min_tier) if min_tier else None
        
        result = self._retrieval.hybrid_search(
            query=query,
            scope=scope,
            min_tier=min_tier_enum,
            limit=limit
        )
        
        if not result.results:
            return f"No results for: {query}"
        
        lines = [
            f"Found {result.total_count} results (BM25={result.bm25_weight}, "
            f"decay={result.decay_weight}, recency={result.recency_weight}):\n"
        ]
        
        for scored in result.results[:limit]:
            lines.append(
                f"  [{scored.tier.value}] score={scored.final_score:.3f} | "
                f"{scored.snippet[:80]}..."
            )
        
        return "\n".join(lines)
    
    def close(self) -> None:
        """Close the hook and release resources."""
        if self._db:
            self._db.close()
            self._db = None
            self._initialized = False
            LcmHook._instance = None


# =============================================================================
# Decorators
# =============================================================================

def lcm_hook(
    func: Optional[F] = None,
    *,
    config: Optional[LcmHookConfig] = None
) -> F:
    """Decorator to wrap an agent message handler with LCM.
    
    Automatically:
    - Initializes LCM hook on first call
    - Injects assembled context into ctx.messages
    - Handles auto-compaction
    
    Args:
        func: The function to decorate
        config: Optional LCM hook configuration
        
    Example:
        @lcm_hook
        async def handle_message(ctx):
            # ctx.messages is now LCM-assembled context
            response = await llm.chat(ctx.messages)
            await lcm_hook.get_hook().on_message("assistant", response)
            return ctx
    """
    def decorator(f: F) -> F:
        hook = LcmHook(config or LcmHookConfig())
        hook_ref = [hook]  # Mutable container for late binding
        
        @wraps(f)
        async def wrapper(*args, **kwargs):
            h = hook_ref[0]
            if not h._initialized:
                await h.initialize()
            
            # Inject hook into kwargs if not present
            if 'lcm_hook' not in kwargs:
                kwargs['lcm_hook'] = h
            
            # Execute the wrapped function
            return await f(*args, **kwargs)
        
        # Attach hook getter
        wrapper.get_hook = lambda: hook_ref[0]
        wrapper._lcm_hook = hook_ref
        
        return wrapper  # type: ignore
    
    if func is None:
        # Called with arguments: @lcm_hook(config=...)
        return decorator
    else:
        # Called without arguments: @lcm_hook
        hook = LcmHook(config or LcmHookConfig())
        return decorator(func)


def lcm_tool(
    func: Optional[F] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None
) -> F:
    """Decorator to expose a function as an LCM tool for agents.
    
    The decorated function becomes callable as an LCM retrieval/management tool.
    
    Args:
        func: The function to decorate
        name: Tool name (defaults to function name)
        description: Tool description (defaults to docstring)
        
    Example:
        hook = LcmHook()
        await hook.initialize()
        
        @lcm_tool(name="lcm_describe")
        def describe(item_id: str) -> str:
            '''Look up a summary by ID.'''
            return hook.describe(item_id)
        
        # Agent can now call: describe("sum_abc123")
    """
    def decorator(f: F) -> F:
        tool_name = name or f.__name__
        tool_desc = description or f.__doc__ or ""
        
        # Attach tool metadata
        f._lcm_tool_name = tool_name
        f._lcm_tool_description = tool_desc
        f._is_lcm_tool = True
        
        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        
        wrapper._lcm_tool_name = tool_name
        wrapper._lcm_tool_description = tool_desc
        wrapper._is_lcm_tool = True
        
        return wrapper  # type: ignore
    
    if func is None:
        return decorator
    else:
        return decorator(func)


# =============================================================================
# Convenience Functions
# =============================================================================

async def quick_start(session_id: str, **kwargs) -> LcmHook:
    """Quick start: create hook, initialize, and start session.
    
    Args:
        session_id: Unique session identifier
        **kwargs: Additional LcmHookConfig options
        
    Returns:
        Initialized LcmHook with session started
    """
    hook = LcmHook(LcmHookConfig(**kwargs))
    await hook.initialize()
    await hook.on_session_start(session_id=session_id)
    return hook


async def quick_ingest(
    conversation_id: int,
    role: str,
    content: str,
    parts: Optional[List[Dict[str, Any]]] = None
) -> int:
    """Quick message ingestion using global hook instance.
    
    Args:
        conversation_id: Target conversation
        role: Message role
        content: Message content
        parts: Optional message parts
        
    Returns:
        message_id
    """
    hook = LcmHook.get_instance()
    if not hook:
        raise RuntimeError("LCM Hook not initialized. Call quick_start() first.")
    
    return await hook.on_message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        parts=parts
    )


async def quick_assemble(conversation_id: int, **kwargs) -> HookContext:
    """Quick context assembly using global hook instance.
    
    Args:
        conversation_id: Target conversation
        **kwargs: Additional assemble_context options
        
    Returns:
        HookContext with assembled messages
    """
    hook = LcmHook.get_instance()
    if not hook:
        raise RuntimeError("LCM Hook not initialized. Call quick_start() first.")
    
    return await hook.assemble_context(conversation_id=conversation_id, **kwargs)
