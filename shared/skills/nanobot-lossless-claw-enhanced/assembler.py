#!/usr/bin/env python3
"""Context assembly module for LCM.

Builds model context under token budget with:
- Fresh tail protection (keep last N messages)
- Message and summary resolution
- System prompt guidance for LCM recall
- Tool call/result block reconstruction

Port of TypeScript assembler.ts from lossless-claw-enhanced.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Awaitable

from estimate_tokens import estimate_tokens
from transcript_repair import sanitize_tool_use_result_pairing
from lcm_types import (
    MessageRole,
    ContextItemType,
    MessageRecord,
    MessagePartRecord,
    SummaryRecord,
    ContextItemRecord,
    LargeFileRecord,
)
from db.config import LcmConfig



# System prompt for LCM recall guidance
LCM_RECALL_SYSTEM_PROMPT = """You have access to a lossless conversation history through the LCM (Lossless Context Management) system.

When you need information from earlier in the conversation:
1. Use the `describe` tool to look up summaries or files by ID
2. Use the `grep` tool to search for specific content
3. Use the `expand` tool to traverse the summary hierarchy

The summaries provided in your context are high-level overviews. For detailed information, always use these retrieval tools to access the complete conversation history."""


@dataclass
class AssemblerConfig:
    """Configuration for context assembler."""
    token_budget: int = 128000
    fresh_tail_count: int = 32
    context_threshold: float = 0.75
    max_expand_tokens: int = 4000
    include_system_prompt: bool = True


@dataclass
class AssembledContext:
    """Result of context assembly."""
    messages: List[Dict[str, Any]]
    total_tokens: int
    token_budget: int
    fresh_tail_tokens: int
    summary_tokens: int
    has_summaries: bool
    context_items_used: int
    warnings: List[str] = field(default_factory=list)


@dataclass
class ResolvedMessage:
    """A resolved message with its parts."""
    record: MessageRecord
    parts: List[MessagePartRecord] = field(default_factory=list)
    token_count: int = 0


@dataclass 
class ResolvedSummary:
    """A resolved summary with its details."""
    record: SummaryRecord
    token_count: int = 0


class ContextAssembler:
    """Assembles context under token budget.
    
    Features:
    - Fresh tail protection (keeps last N messages)
    - Summary inclusion for compressed context
    - Tool call/result block reconstruction
    - System prompt guidance for LCM recall
    """
    
    def __init__(
        self,
        config: AssemblerConfig,
        get_message_parts: Optional[Callable[[int], List[MessagePartRecord]]] = None,
        get_large_file: Optional[Callable[[str], Optional[LargeFileRecord]]] = None,
        log: Optional[Callable[[str], None]] = None
    ):
        """Initialize context assembler.
        
        Args:
            config: Assembler configuration
            get_message_parts: Function to get message parts by message_id
            get_large_file: Function to get large file by file_id
            log: Optional logging function
        """
        self.config = config
        self._get_message_parts = get_message_parts
        self._get_large_file = get_large_file
        self._log = log or print
    
    def assemble(
        self,
        messages: List[MessageRecord],
        summaries: List[SummaryRecord],
        context_items: List[ContextItemRecord],
        system_prompt: Optional[str] = None
    ) -> AssembledContext:
        """Assemble context under token budget.
        
        Args:
            messages: All message records
            summaries: Available summary records
            context_items: Context item ordering records
            system_prompt: Optional system prompt to include
            
        Returns:
            AssembledContext with assembled messages and metadata
        """
        warnings = []
        
        # Calculate effective token budget
        effective_budget = int(self.config.token_budget * self.config.context_threshold)
        
        # Build message lookup
        message_lookup = {m.message_id: m for m in messages}
        summary_lookup = {s.summary_id: s for s in summaries}
        
        # Identify fresh tail (last N messages)
        sorted_messages = sorted(messages, key=lambda m: m.seq)
        fresh_tail_messages = sorted_messages[-self.config.fresh_tail_count:] if len(sorted_messages) >= self.config.fresh_tail_count else sorted_messages
        
        # Calculate fresh tail tokens
        fresh_tail_tokens = sum(m.token_count for m in fresh_tail_messages)
        
        # Reserve budget for fresh tail
        remaining_budget = effective_budget - fresh_tail_tokens
        
        # Reserve tokens for system prompt
        system_tokens = 0
        if system_prompt:
            system_tokens = estimate_tokens(system_prompt)
            remaining_budget -= system_tokens
        
        # Reserve tokens for LCM recall guidance
        lcm_guidance_tokens = 0
        if self.config.include_system_prompt:
            lcm_guidance_tokens = estimate_tokens(LCM_RECALL_SYSTEM_PROMPT)
            remaining_budget -= lcm_guidance_tokens
        
        # Build result messages
        result_messages: List[Dict[str, Any]] = []
        total_tokens = 0
        context_items_used = 0
        summary_tokens = 0
        has_summaries = False
        
        # Add system prompt if provided
        if system_prompt:
            result_messages.append({
                "role": "system",
                "content": system_prompt
            })
            total_tokens += system_tokens
        
        # Add LCM recall guidance if enabled
        if self.config.include_system_prompt:
            result_messages.append({
                "role": "system",
                "content": LCM_RECALL_SYSTEM_PROMPT
            })
            total_tokens += lcm_guidance_tokens
        
        # Get fresh tail message IDs
        fresh_tail_ids = {m.message_id for m in fresh_tail_messages}
        
        # Process context items in order, stopping when budget exhausted
        sorted_items = sorted(context_items, key=lambda c: c.ordinal)
        
        for item in sorted_items:
            if remaining_budget <= 0:
                break
            
            if item.item_type == ContextItemType.MESSAGE:
                # Check if in fresh tail (already handled)
                if item.message_id in fresh_tail_ids:
                    continue
                
                msg = message_lookup.get(item.message_id)
                if not msg:
                    continue
                
                if msg.token_count > remaining_budget:
                    # Can't fit this message
                    continue
                
                # Build message dict
                msg_dict = self._build_message_dict(msg)
                if msg_dict:
                    result_messages.append(msg_dict)
                    total_tokens += msg.token_count
                    remaining_budget -= msg.token_count
                    context_items_used += 1
            
            elif item.item_type == ContextItemType.SUMMARY:
                summary = summary_lookup.get(item.summary_id)
                if not summary:
                    continue
                
                if summary.token_count > remaining_budget:
                    continue
                
                # Build summary as system message
                summary_dict = self._build_summary_dict(summary)
                if summary_dict:
                    result_messages.append(summary_dict)
                    total_tokens += summary.token_count
                    remaining_budget -= summary.token_count
                    summary_tokens += summary.token_count
                    context_items_used += 1
                    has_summaries = True
        
        # Add fresh tail messages
        for msg in fresh_tail_messages:
            msg_dict = self._build_message_dict(msg)
            if msg_dict:
                result_messages.append(msg_dict)
                total_tokens += msg.token_count
        
        # Sanitize tool use/result pairings
        result_messages = sanitize_tool_use_result_pairing(result_messages)
        
        # Check if we exceeded budget
        if total_tokens > effective_budget:
            warnings.append(f"Context exceeded budget: {total_tokens} > {effective_budget}")
        
        return AssembledContext(
            messages=result_messages,
            total_tokens=total_tokens,
            token_budget=self.config.token_budget,
            fresh_tail_tokens=fresh_tail_tokens,
            summary_tokens=summary_tokens,
            has_summaries=has_summaries,
            context_items_used=context_items_used,
            warnings=warnings
        )
    
    def _build_message_dict(self, msg: MessageRecord) -> Optional[Dict[str, Any]]:
        """Build a message dictionary from a message record.
        
        Args:
            msg: Message record
            
        Returns:
            Message dictionary or None if invalid
        """
        try:
            # Try to parse content as JSON (content blocks)
            content = json.loads(msg.content)
        except (json.JSONDecodeError, TypeError):
            # Treat as plain text
            content = msg.content
        
        result = {
            "role": msg.role.value if isinstance(msg.role, MessageRole) else msg.role,
            "content": content
        }
        
        # Add message parts if available
        if self._get_message_parts:
            parts = self._get_message_parts(msg.message_id)
            if parts:
                result = self._enrich_with_parts(result, parts)
        
        return result
    
    def _build_summary_dict(self, summary: SummaryRecord) -> Dict[str, Any]:
        """Build a summary message dictionary.
        
        Args:
            summary: Summary record
            
        Returns:
            Summary as system message dictionary
        """
        content_parts = [
            f"[LCM Summary - {summary.kind.value}]",
            summary.content
        ]
        
        # Add file references if any
        if summary.file_ids:
            file_refs = []
            for file_id in summary.file_ids[:5]:  # Limit to first 5
                if self._get_large_file:
                    file_record = self._get_large_file(file_id)
                    if file_record:
                        file_refs.append(f"[File: {file_record.file_name or file_id}]")
                else:
                    file_refs.append(f"[File: {file_id}]")
            
            if file_refs:
                content_parts.append(f"Referenced files: {', '.join(file_refs)}")
        
        # Add temporal context
        if summary.earliest_at and summary.latest_at:
            content_parts.append(
                f"Time range: {summary.earliest_at.isoformat()} to {summary.latest_at.isoformat()}"
            )
        
        return {
            "role": "system",
            "content": "\n\n".join(content_parts)
        }
    
    def _enrich_with_parts(
        self,
        msg_dict: Dict[str, Any],
        parts: List[MessagePartRecord]
    ) -> Dict[str, Any]:
        """Enrich message with parts data.
        
        Args:
            msg_dict: Base message dictionary
            parts: Message part records
            
        Returns:
            Enriched message dictionary
        """
        # If content is already a list, we may need to merge parts
        if isinstance(msg_dict.get("content"), list):
            return msg_dict
        
        # Build content blocks from parts
        content_blocks = []
        
        # Add text content if present
        if msg_dict.get("content"):
            content_blocks.append({
                "type": "text",
                "text": msg_dict["content"]
            })
        
        # Add tool blocks from parts
        for part in sorted(parts, key=lambda p: p.ordinal):
            if part.part_type.value == "tool":
                if part.tool_call_id:
                    # Tool use block
                    tool_block = {
                        "type": "tool_use",
                        "id": part.tool_call_id,
                        "name": part.tool_name or "unknown"
                    }
                    if part.tool_input:
                        try:
                            tool_block["input"] = json.loads(part.tool_input)
                        except (json.JSONDecodeError, TypeError):
                            tool_block["input"] = part.tool_input
                    content_blocks.append(tool_block)
                    
                    # Add tool result if available
                    if part.tool_output:
                        result_block = {
                            "type": "tool_result",
                            "tool_use_id": part.tool_call_id,
                            "content": part.tool_output
                        }
                        if part.tool_status:
                            result_block["status"] = part.tool_status
                        if part.tool_error:
                            result_block["is_error"] = True
                        content_blocks.append(result_block)
        
        if content_blocks:
            msg_dict["content"] = content_blocks
        
        return msg_dict
    
    def estimate_context_size(
        self,
        messages: List[MessageRecord],
        summaries: List[SummaryRecord],
        context_items: List[ContextItemRecord]
    ) -> Dict[str, int]:
        """Estimate context size without assembling.
        
        Args:
            messages: All message records
            summaries: Available summary records
            context_items: Context item ordering records
            
        Returns:
            Dict with size estimates
        """
        message_lookup = {m.message_id: m for m in messages}
        summary_lookup = {s.summary_id: s for s in summaries}
        
        # Calculate fresh tail
        sorted_messages = sorted(messages, key=lambda m: m.seq)
        fresh_tail = sorted_messages[-self.config.fresh_tail_count:] if len(sorted_messages) >= self.config.fresh_tail_count else sorted_messages
        fresh_tail_tokens = sum(m.token_count for m in fresh_tail)
        fresh_tail_ids = {m.message_id for m in fresh_tail}
        
        # Calculate summary tokens
        summary_tokens = sum(
            summary_lookup[item.summary_id].token_count
            for item in context_items
            if item.item_type == ContextItemType.SUMMARY and item.summary_id in summary_lookup
        )
        
        # Calculate message tokens (excluding fresh tail)
        message_tokens = sum(
            message_lookup[item.message_id].token_count
            for item in context_items
            if item.item_type == ContextItemType.MESSAGE 
            and item.message_id in message_lookup 
            and item.message_id not in fresh_tail_ids
        )
        
        # System prompt overhead
        system_overhead = estimate_tokens(LCM_RECALL_SYSTEM_PROMPT) if self.config.include_system_prompt else 0
        
        return {
            "fresh_tail_tokens": fresh_tail_tokens,
            "summary_tokens": summary_tokens,
            "message_tokens": message_tokens,
            "system_overhead": system_overhead,
            "total_estimated": fresh_tail_tokens + summary_tokens + message_tokens + system_overhead
        }


def create_assembler_from_config(
    lcm_config: LcmConfig,
    get_message_parts: Optional[Callable[[int], List[MessagePartRecord]]] = None,
    get_large_file: Optional[Callable[[str], Optional[LargeFileRecord]]] = None,
    log: Optional[Callable[[str], None]] = None
) -> ContextAssembler:
    """Create a ContextAssembler from LcmConfig.
    
    Args:
        lcm_config: LCM configuration
        get_message_parts: Function to get message parts
        get_large_file: Function to get large files
        log: Optional logging function
        
    Returns:
        Configured ContextAssembler
    """
    assembler_config = AssemblerConfig(
        token_budget=128000,  # Default model context window
        fresh_tail_count=lcm_config.fresh_tail_count,
        context_threshold=lcm_config.context_threshold,
        max_expand_tokens=lcm_config.max_expand_tokens,
        include_system_prompt=True
    )
    
    return ContextAssembler(
        config=assembler_config,
        get_message_parts=get_message_parts,
        get_large_file=get_large_file,
        log=log
    )


if __name__ == "__main__":
    # Demo usage
    from datetime import datetime
    
    # Create sample messages
    messages = [
        MessageRecord(
            message_id=1,
            conversation_id=1,
            seq=1,
            role=MessageRole.USER,
            content="Hello, how are you?",
            token_count=10,
            created_at=datetime.utcnow()
        ),
        MessageRecord(
            message_id=2,
            conversation_id=1,
            seq=2,
            role=MessageRole.ASSISTANT,
            content="I'm doing well, thank you!",
            token_count=12,
            created_at=datetime.utcnow()
        ),
        MessageRecord(
            message_id=3,
            conversation_id=1,
            seq=3,
            role=MessageRole.USER,
            content="Can you help me with Python?",
            token_count=15,
            created_at=datetime.utcnow()
        ),
    ]
    
    # Create sample summary
    summaries = [
        SummaryRecord(
            summary_id="sum_1",
            conversation_id=1,
            kind="leaf",
            depth=0,
            content="User greeted and asked for Python help",
            token_count=20,
            created_at=datetime.utcnow()
        )
    ]
    
    # Create context items
    context_items = [
        ContextItemRecord(
            conversation_id=1,
            ordinal=1,
            item_type=ContextItemType.SUMMARY,
            summary_id="sum_1",
            created_at=datetime.utcnow()
        )
    ]
    
    # Assemble context
    config = AssemblerConfig(token_budget=1000, fresh_tail_count=2)
    assembler = ContextAssembler(config)
    
    result = assembler.assemble(
        messages=messages,
        summaries=summaries,
        context_items=context_items,
        system_prompt="You are a helpful assistant."
    )
    
    print(f"Total tokens: {result.total_tokens}")
    print(f"Token budget: {result.token_budget}")
    print(f"Fresh tail tokens: {result.fresh_tail_tokens}")
    print(f"Has summaries: {result.has_summaries}")
    print(f"Context items used: {result.context_items_used}")
    print(f"Warnings: {result.warnings}")
    print(f"\nMessages ({len(result.messages)}):")
    for msg in result.messages:
        print(f"  [{msg['role']}]: {str(msg['content'])[:50]}...")
