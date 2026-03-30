#!/usr/bin/env python3
"""Compaction Engine for LCM (Lossless Context Management).

Implements the compaction workflow:
- Leaf pass: summarize raw messages into leaf summaries
- Condensed pass: summarize leaf summaries into condensed summaries
- Three-level escalation: normal → aggressive → fallback
- Fresh tail protection (preserve recent messages)
- Auth error detection (skip compaction on auth errors)
- Multiple rounds until under budget

Port of TypeScript compaction.ts from lossless-claw-enhanced.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Awaitable, Union

from estimate_tokens import estimate_tokens
from lcm_types import (
    MessageRecord,
    SummaryRecord,
    ContextItemRecord,
    SummaryKind,
    ContextItemType,
)
from summarize import (
    LcmProviderAuthError,
    strip_auth_errors,
    create_lcm_summarize_from_legacy_params,
    LegacyParams,
)
from large_files import (
    extract_file_ids_from_content,
    generate_exploration_summary,
    format_file_reference,
)


class CompactionLevel(str, Enum):
    """Compaction escalation levels."""
    NORMAL = "normal"
    AGGRESSIVE = "aggressive"
    FALLBACK = "fallback"


class CompactionPhase(str, Enum):
    """Compaction phases."""
    LEAF = "leaf"
    CONDENSED = "condensed"


@dataclass
class CompactionConfig:
    """Configuration for compaction engine."""
    token_budget: int = 128000
    context_threshold: float = 0.75
    fresh_tail_count: int = 32
    leaf_min_fanout: int = 8
    condensed_min_fanout: int = 4
    condensed_min_fanout_hard: int = 2
    leaf_chunk_tokens: int = 20000
    leaf_target_tokens: int = 1200
    condensed_target_tokens: int = 2000
    max_rounds: int = 10
    aggressive_ratio: float = 1.5  # Trigger aggressive when over budget by this ratio
    fallback_ratio: float = 2.0    # Trigger fallback when over budget by this ratio


@dataclass
class CompactionDecision:
    """Decision about whether and how to compact."""
    should_compact: bool
    reason: str
    level: CompactionLevel = CompactionLevel.NORMAL
    phase: CompactionPhase = CompactionPhase.LEAF
    current_tokens: int = 0
    target_tokens: int = 0
    fresh_tail_protected: int = 0


@dataclass
class CompactionStats:
    """Statistics from a compaction pass."""
    messages_compacted: int = 0
    summaries_created: int = 0
    tokens_before: int = 0
    tokens_after: int = 0
    tokens_saved: int = 0
    auth_errors_skipped: int = 0


@dataclass
class CompactionResult:
    """Result of compaction operation."""
    success: bool
    stats: CompactionStats
    new_summaries: List[SummaryRecord] = field(default_factory=list)
    new_context_items: List[ContextItemRecord] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    rounds_completed: int = 0
    final_tokens: int = 0
    final_level: CompactionLevel = CompactionLevel.NORMAL


class CompactionEngine:
    """Engine for compacting conversation context.
    
    Features:
    - Leaf pass: summarize raw messages
    - Condensed pass: summarize summaries
    - Three-level escalation (normal → aggressive → fallback)
    - Fresh tail protection
    - Auth error detection
    - Multiple rounds until under budget
    """
    
    def __init__(
        self,
        config: CompactionConfig,
        summarizer: Optional[Callable[[str, bool, Optional[Dict[str, Any]]], Awaitable[str]]] = None,
        log: Optional[Callable[[str], None]] = None
    ):
        """Initialize compaction engine.
        
        Args:
            config: Compaction configuration
            summarizer: Async summarizer function (text, aggressive, options) -> summary
            log: Optional logging function
        """
        self.config = config
        self._summarizer = summarizer
        self._log = log or print
    
    def evaluate(
        self,
        messages: List[MessageRecord],
        summaries: List[SummaryRecord],
        context_items: List[ContextItemRecord]
    ) -> CompactionDecision:
        """Evaluate whether compaction is needed.
        
        Args:
            messages: All message records
            summaries: Existing summary records
            context_items: Current context item ordering
            
        Returns:
            CompactionDecision with recommendation
        """
        if not messages:
            return CompactionDecision(
                should_compact=False,
                reason="No messages to compact",
                current_tokens=0,
                target_tokens=0
            )
        
        # Calculate total tokens
        total_tokens = sum(m.token_count for m in messages)
        
        # Add summary tokens
        summary_lookup = {s.summary_id: s for s in summaries}
        for item in context_items:
            if item.item_type == ContextItemType.SUMMARY:
                summary = summary_lookup.get(item.summary_id)
                if summary:
                    total_tokens += summary.token_count
        
        # Calculate effective budget
        effective_budget = int(self.config.token_budget * self.config.context_threshold)
        
        # Check if under budget
        if total_tokens <= effective_budget:
            return CompactionDecision(
                should_compact=False,
                reason=f"Under budget: {total_tokens} <= {effective_budget}",
                current_tokens=total_tokens,
                target_tokens=effective_budget,
                fresh_tail_protected=0
            )
        
        # Calculate fresh tail
        sorted_messages = sorted(messages, key=lambda m: m.seq)
        fresh_tail_count = min(self.config.fresh_tail_count, len(sorted_messages))
        fresh_tail_messages = sorted_messages[-fresh_tail_count:] if fresh_tail_count > 0 else []
        fresh_tail_tokens = sum(m.token_count for m in fresh_tail_messages)
        fresh_tail_ids = {m.message_id for m in fresh_tail_messages}
        
        # Calculate compactable tokens
        compactable_tokens = sum(
            m.token_count for m in messages 
            if m.message_id not in fresh_tail_ids
        )
        
        # Determine compaction level
        ratio = total_tokens / effective_budget if effective_budget > 0 else float('inf')
        
        if ratio >= self.config.fallback_ratio:
            level = CompactionLevel.FALLBACK
        elif ratio >= self.config.aggressive_ratio:
            level = CompactionLevel.AGGRESSIVE
        else:
            level = CompactionLevel.NORMAL
        
        # Determine phase
        # Use condensed phase if we have enough leaf summaries
        leaf_summaries = [s for s in summaries if s.kind == SummaryKind.LEAF]
        phase = CompactionPhase.LEAF
        
        if len(leaf_summaries) >= self.config.condensed_min_fanout:
            phase = CompactionPhase.CONDENSED
        
        return CompactionDecision(
            should_compact=True,
            reason=f"Over budget: {total_tokens} > {effective_budget} (ratio: {ratio:.2f})",
            level=level,
            phase=phase,
            current_tokens=total_tokens,
            target_tokens=effective_budget,
            fresh_tail_protected=fresh_tail_tokens
        )
    
    async def compact(
        self,
        conversation_id: int,
        messages: List[MessageRecord],
        summaries: List[SummaryRecord],
        context_items: List[ContextItemRecord]
    ) -> CompactionResult:
        """Perform compaction until under budget.
        
        Runs multiple rounds of compaction as needed:
        1. Leaf pass: compact messages into leaf summaries
        2. Condensed pass: compact leaf summaries into condensed summaries
        3. Escalate level if needed
        
        Args:
            conversation_id: The conversation ID
            messages: All message records
            summaries: Existing summary records
            context_items: Current context item ordering
            
        Returns:
            CompactionResult with new summaries and context items
        """
        all_new_summaries: List[SummaryRecord] = []
        all_new_context_items: List[ContextItemRecord] = []
        warnings: List[str] = []
        stats = CompactionStats(tokens_before=0, tokens_after=0)
        
        # Calculate initial tokens
        initial_tokens = self._calculate_total_tokens(messages, summaries, context_items)
        stats.tokens_before = initial_tokens
        current_tokens = initial_tokens
        current_level = CompactionLevel.NORMAL
        rounds_completed = 0
        auth_errors = 0
        
        effective_budget = int(self.config.token_budget * self.config.context_threshold)
        
        # Run compaction rounds
        for round_num in range(self.config.max_rounds):
            # Evaluate current state
            decision = self.evaluate(messages, summaries + all_new_summaries, context_items + all_new_context_items)
            
            if not decision.should_compact:
                break
            
            current_level = decision.level
            
            # Perform one round of compaction
            try:
                round_result = await self._compact_round(
                    conversation_id=conversation_id,
                    messages=messages,
                    summaries=summaries + all_new_summaries,
                    context_items=context_items + all_new_context_items,
                    level=decision.level,
                    phase=decision.phase
                )
                
                # Collect results
                all_new_summaries.extend(round_result["summaries"])
                all_new_context_items.extend(round_result["context_items"])
                auth_errors += round_result.get("auth_errors", 0)
                stats.messages_compacted += round_result.get("messages_compacted", 0)
                stats.summaries_created += len(round_result["summaries"])
                
                # Update current state for next round
                current_tokens = self._calculate_total_tokens(
                    messages, 
                    summaries + all_new_summaries, 
                    context_items + all_new_context_items
                )
                
                # Check if we're stuck (no progress)
                if current_tokens >= decision.current_tokens * 0.99:  # Less than 1% reduction
                    # Try escalating
                    if current_level == CompactionLevel.NORMAL:
                        current_level = CompactionLevel.AGGRESSIVE
                    elif current_level == CompactionLevel.AGGRESSIVE:
                        current_level = CompactionLevel.FALLBACK
                    else:
                        # Already at fallback, can't do more
                        warnings.append(f"Compaction stuck at {current_tokens} tokens after {round_num + 1} rounds")
                        break
                
                rounds_completed = round_num + 1
                
                # Check if under budget
                if current_tokens <= effective_budget:
                    break
                    
            except LcmProviderAuthError as e:
                # Auth error - stop compaction
                warnings.append(f"Compaction stopped due to auth error: {e}")
                auth_errors += 1
                break
            except Exception as e:
                warnings.append(f"Compaction round {round_num + 1} failed: {e}")
                break
        
        # Calculate final stats
        stats.tokens_after = current_tokens
        stats.tokens_saved = initial_tokens - current_tokens
        stats.auth_errors_skipped = auth_errors
        
        return CompactionResult(
            success=current_tokens <= effective_budget,
            stats=stats,
            new_summaries=all_new_summaries,
            new_context_items=all_new_context_items,
            warnings=warnings,
            rounds_completed=rounds_completed,
            final_tokens=current_tokens,
            final_level=current_level
        )
    
    async def _compact_round(
        self,
        conversation_id: int,
        messages: List[MessageRecord],
        summaries: List[SummaryRecord],
        context_items: List[ContextItemRecord],
        level: CompactionLevel,
        phase: CompactionPhase
    ) -> Dict[str, Any]:
        """Perform one round of compaction.
        
        Args:
            conversation_id: The conversation ID
            messages: All message records
            summaries: All summary records (existing + new)
            context_items: All context items
            level: Compaction level
            phase: Compaction phase
            
        Returns:
            Dict with summaries, context_items, messages_compacted, auth_errors
        """
        if phase == CompactionPhase.LEAF:
            return await self._leaf_pass(conversation_id, messages, summaries, context_items, level)
        else:
            return await self._condensed_pass(conversation_id, messages, summaries, context_items, level)
    
    async def _leaf_pass(
        self,
        conversation_id: int,
        messages: List[MessageRecord],
        summaries: List[SummaryRecord],
        context_items: List[ContextItemRecord],
        level: CompactionLevel
    ) -> Dict[str, Any]:
        """Leaf pass: compact raw messages into leaf summaries.
        
        Groups messages into chunks and summarizes each chunk.
        
        Args:
            conversation_id: The conversation ID
            messages: All message records
            summaries: Existing summary records
            context_items: Current context items
            level: Compaction level
            
        Returns:
            Dict with new summaries and context items
        """
        new_summaries: List[SummaryRecord] = []
        new_context_items: List[ContextItemRecord] = []
        messages_compacted = 0
        auth_errors = 0
        
        # Identify fresh tail to protect
        sorted_messages = sorted(messages, key=lambda m: m.seq)
        fresh_tail_count = min(self.config.fresh_tail_count, len(sorted_messages))
        fresh_tail_ids = {m.message_id for m in sorted_messages[-fresh_tail_count:]} if fresh_tail_count > 0 else set()
        
        # Identify already summarized messages
        summarized_message_ids = set()
        for item in context_items:
            if item.item_type == ContextItemType.SUMMARY:
                # Find the summary and its covered messages
                for s in summaries:
                    if s.summary_id == item.summary_id:
                        # Summary covers messages in its time range
                        pass  # We track by time range instead
        
        # Get messages eligible for compaction
        eligible_messages = [
            m for m in sorted_messages 
            if m.message_id not in fresh_tail_ids
        ]
        
        if not eligible_messages:
            return {
                "summaries": new_summaries,
                "context_items": new_context_items,
                "messages_compacted": messages_compacted,
                "auth_errors": auth_errors
            }
        
        # Group messages into chunks
        chunks = self._group_messages_into_chunks(
            eligible_messages, 
            self.config.leaf_chunk_tokens,
            self.config.leaf_min_fanout
        )
        
        # Process each chunk
        ordinal_start = len(context_items) + 1
        
        for chunk_idx, chunk in enumerate(chunks):
            if len(chunk) < self.config.leaf_min_fanout:
                continue
            
            # Build text for summarization
            chunk_text = self._build_chunk_text(chunk)
            
            # Check for auth errors in content
            if self._contains_auth_errors(chunk_text):
                auth_errors += 1
                continue
            
            # Summarize
            try:
                aggressive = level in (CompactionLevel.AGGRESSIVE, CompactionLevel.FALLBACK)
                summary_text = await self._summarize_chunk(
                    chunk_text, 
                    aggressive=aggressive,
                    max_tokens=self.config.leaf_target_tokens
                )
                
                # Strip any auth errors from summary
                summary_text = strip_auth_errors(summary_text)
                
                # Extract file IDs from chunk
                file_ids = self._extract_file_ids_from_messages(chunk)
                
                # Create summary record
                summary_id = self._generate_summary_id(conversation_id, "leaf", chunk_idx)
                summary_tokens = estimate_tokens(summary_text)
                
                summary = SummaryRecord(
                    summary_id=summary_id,
                    conversation_id=conversation_id,
                    kind=SummaryKind.LEAF,
                    depth=0,
                    content=summary_text,
                    token_count=summary_tokens,
                    file_ids=file_ids,
                    earliest_at=chunk[0].created_at if chunk else None,
                    latest_at=chunk[-1].created_at if chunk else None,
                    descendant_count=len(chunk),
                    descendant_token_count=sum(m.token_count for m in chunk),
                    source_message_token_count=sum(m.token_count for m in chunk),
                    model="unknown",
                    created_at=datetime.utcnow()
                )
                
                new_summaries.append(summary)
                
                # Create context item
                context_item = ContextItemRecord(
                    conversation_id=conversation_id,
                    ordinal=ordinal_start + chunk_idx,
                    item_type=ContextItemType.SUMMARY,
                    summary_id=summary_id,
                    created_at=datetime.utcnow()
                )
                
                new_context_items.append(context_item)
                messages_compacted += len(chunk)
                
            except LcmProviderAuthError:
                auth_errors += 1
                continue
            except Exception as e:
                self._log(f"Error summarizing chunk {chunk_idx}: {e}")
                continue
        
        return {
            "summaries": new_summaries,
            "context_items": new_context_items,
            "messages_compacted": messages_compacted,
            "auth_errors": auth_errors
        }
    
    async def _condensed_pass(
        self,
        conversation_id: int,
        messages: List[MessageRecord],
        summaries: List[SummaryRecord],
        context_items: List[ContextItemRecord],
        level: CompactionLevel
    ) -> Dict[str, Any]:
        """Condensed pass: compact leaf summaries into condensed summaries.
        
        Groups leaf summaries and summarizes them.
        
        Args:
            conversation_id: The conversation ID
            messages: All message records (for reference)
            summaries: All summary records
            context_items: Current context items
            level: Compaction level
            
        Returns:
            Dict with new summaries and context items
        """
        new_summaries: List[SummaryRecord] = []
        new_context_items: List[ContextItemRecord] = []
        auth_errors = 0
        
        # Get leaf summaries
        leaf_summaries = [s for s in summaries if s.kind == SummaryKind.LEAF]
        leaf_summaries.sort(key=lambda s: s.created_at)
        
        # Determine minimum fanout based on level
        if level == CompactionLevel.FALLBACK:
            min_fanout = self.config.condensed_min_fanout_hard
        else:
            min_fanout = self.config.condensed_min_fanout
        
        if len(leaf_summaries) < min_fanout:
            return {
                "summaries": new_summaries,
                "context_items": new_context_items,
                "messages_compacted": 0,
                "auth_errors": auth_errors
            }
        
        # Group leaf summaries
        groups = self._group_summaries_into_groups(
            leaf_summaries,
            min_fanout,
            level == CompactionLevel.FALLBACK
        )
        
        # Process each group
        ordinal_start = len(context_items) + 1
        
        for group_idx, group in enumerate(groups):
            if len(group) < min_fanout:
                continue
            
            # Build text for summarization
            group_text = self._build_summary_group_text(group)
            
            # Check for auth errors
            if self._contains_auth_errors(group_text):
                auth_errors += 1
                continue
            
            try:
                aggressive = level in (CompactionLevel.AGGRESSIVE, CompactionLevel.FALLBACK)
                summary_text = await self._summarize_chunk(
                    group_text,
                    aggressive=aggressive,
                    max_tokens=self.config.condensed_target_tokens
                )
                
                # Strip auth errors
                summary_text = strip_auth_errors(summary_text)
                
                # Merge file IDs
                file_ids = []
                for s in group:
                    file_ids.extend(s.file_ids or [])
                file_ids = list(dict.fromkeys(file_ids))  # Dedupe preserving order
                
                # Calculate depth
                max_depth = max((s.depth for s in group), default=0)
                
                # Create summary record
                summary_id = self._generate_summary_id(conversation_id, "condensed", group_idx)
                summary_tokens = estimate_tokens(summary_text)
                
                summary = SummaryRecord(
                    summary_id=summary_id,
                    conversation_id=conversation_id,
                    kind=SummaryKind.CONDENSED,
                    depth=max_depth + 1,
                    content=summary_text,
                    token_count=summary_tokens,
                    file_ids=file_ids,
                    earliest_at=min((s.earliest_at for s in group if s.earliest_at), default=None),
                    latest_at=max((s.latest_at for s in group if s.latest_at), default=None),
                    descendant_count=sum(s.descendant_count for s in group),
                    descendant_token_count=sum(s.descendant_token_count for s in group),
                    source_message_token_count=sum(s.source_message_token_count for s in group),
                    model="unknown",
                    created_at=datetime.utcnow()
                )
                
                new_summaries.append(summary)
                
                # Create context item
                context_item = ContextItemRecord(
                    conversation_id=conversation_id,
                    ordinal=ordinal_start + group_idx,
                    item_type=ContextItemType.SUMMARY,
                    summary_id=summary_id,
                    created_at=datetime.utcnow()
                )
                
                new_context_items.append(context_item)
                
            except LcmProviderAuthError:
                auth_errors += 1
                continue
            except Exception as e:
                self._log(f"Error summarizing group {group_idx}: {e}")
                continue
        
        return {
            "summaries": new_summaries,
            "context_items": new_context_items,
            "messages_compacted": 0,  # Condensed pass doesn't compact messages directly
            "auth_errors": auth_errors
        }
    
    async def _summarize_chunk(
        self,
        text: str,
        aggressive: bool = False,
        max_tokens: int = 1200
    ) -> str:
        """Summarize a chunk of text.
        
        Args:
            text: Text to summarize
            aggressive: Whether to use aggressive summarization
            max_tokens: Maximum tokens for summary
            
        Returns:
            Summarized text
            
        Raises:
            LcmProviderAuthError: If auth error during summarization
        """
        if self._summarizer:
            return await self._summarizer(
                text,
                aggressive=aggressive,
                options={"max_tokens": max_tokens}
            )
        
        # Fallback: simple truncation-based summarization
        return self._fallback_summarize(text, max_tokens, aggressive)
    
    def _fallback_summarize(
        self,
        text: str,
        max_tokens: int,
        aggressive: bool
    ) -> str:
        """Fallback summarization without LLM.
        
        Args:
            text: Text to summarize
            max_tokens: Maximum tokens
            aggressive: Whether to be aggressive
            
        Returns:
            Summarized text
        """
        current_tokens = estimate_tokens(text)
        
        if current_tokens <= max_tokens:
            return text
        
        # Truncate proportionally
        ratio = max_tokens / current_tokens
        lines = text.split('\n')
        
        if aggressive:
            # Take fewer lines
            target_lines = max(1, int(len(lines) * ratio * 0.5))
        else:
            target_lines = max(1, int(len(lines) * ratio))
        
        result = '\n'.join(lines[:target_lines])
        
        # Ensure we're under budget
        while estimate_tokens(result) > max_tokens and len(lines) > 1:
            target_lines -= 1
            result = '\n'.join(lines[:target_lines])
        
        return result
    
    def _group_messages_into_chunks(
        self,
        messages: List[MessageRecord],
        chunk_tokens: int,
        min_fanout: int
    ) -> List[List[MessageRecord]]:
        """Group messages into chunks for summarization.
        
        Groups messages such that each chunk:
        - Has at least min_fanout messages
        - Doesn't exceed chunk_tokens
        
        Args:
            messages: Messages to group
            chunk_tokens: Maximum tokens per chunk
            min_fanout: Minimum messages per chunk
            
        Returns:
            List of message chunks
        """
        if not messages:
            return []
        
        chunks: List[List[MessageRecord]] = []
        current_chunk: List[MessageRecord] = []
        current_tokens = 0
        
        for msg in messages:
            # Check if adding this message would exceed chunk size
            if current_chunk and current_tokens + msg.token_count > chunk_tokens:
                if len(current_chunk) >= min_fanout:
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_tokens = 0
            
            current_chunk.append(msg)
            current_tokens += msg.token_count
        
        # Handle remaining messages
        if current_chunk:
            if len(current_chunk) >= min_fanout:
                chunks.append(current_chunk)
            elif chunks:
                # Merge with last chunk if too small
                chunks[-1].extend(current_chunk)
        
        return chunks
    
    def _group_summaries_into_groups(
        self,
        summaries: List[SummaryRecord],
        min_fanout: int,
        aggressive: bool
    ) -> List[List[SummaryRecord]]:
        """Group summaries for condensed pass.
        
        Args:
            summaries: Summaries to group
            min_fanout: Minimum summaries per group
            aggressive: Whether to use aggressive grouping
            
        Returns:
            List of summary groups
        """
        if not summaries:
            return []
        
        # Target group size
        if aggressive:
            group_size = min_fanout
        else:
            group_size = max(min_fanout, len(summaries) // 4)
        
        groups: List[List[SummaryRecord]] = []
        
        for i in range(0, len(summaries), group_size):
            group = summaries[i:i + group_size]
            if len(group) >= min_fanout:
                groups.append(group)
            elif groups:
                # Merge with last group
                groups[-1].extend(group)
        
        return groups
    
    def _build_chunk_text(self, messages: List[MessageRecord]) -> str:
        """Build text from message chunk for summarization.
        
        Args:
            messages: Messages to format
            
        Returns:
            Formatted text
        """
        parts = []
        
        for msg in messages:
            role = msg.role.value if hasattr(msg.role, 'value') else str(msg.role)
            content = msg.content
            
            # Handle JSON content
            try:
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    # Extract text from content blocks
                    text_parts = []
                    for block in parsed:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                text_parts.append(block.get("text", ""))
                            elif block.get("type") == "tool_use":
                                text_parts.append(f"[Tool: {block.get('name', 'unknown')}]")
                            elif block.get("type") == "tool_result":
                                text_parts.append(f"[Tool Result]")
                        elif isinstance(block, str):
                            text_parts.append(block)
                    content = " ".join(text_parts)
                elif isinstance(parsed, str):
                    content = parsed
            except (json.JSONDecodeError, TypeError):
                pass
            
            parts.append(f"[{role}]: {content}")
        
        return "\n\n".join(parts)
    
    def _build_summary_group_text(self, summaries: List[SummaryRecord]) -> str:
        """Build text from summary group for condensed summarization.
        
        Args:
            summaries: Summaries to format
            
        Returns:
            Formatted text
        """
        parts = []
        
        for s in summaries:
            kind = s.kind.value if hasattr(s.kind, 'value') else str(s.kind)
            header = f"[{kind.upper()} SUMMARY]"
            
            if s.earliest_at and s.latest_at:
                header += f" ({s.earliest_at.isoformat()} to {s.latest_at.isoformat()})"
            
            parts.append(f"{header}\n{s.content}")
        
        return "\n\n---\n\n".join(parts)
    
    def _extract_file_ids_from_messages(self, messages: List[MessageRecord]) -> List[str]:
        """Extract file IDs from messages.
        
        Args:
            messages: Messages to extract from
            
        Returns:
            List of unique file IDs
        """
        file_ids = []
        
        for msg in messages:
            try:
                content = json.loads(msg.content)
                result = extract_file_ids_from_content(content)
                file_ids.extend(result.file_ids)
            except (json.JSONDecodeError, TypeError):
                pass
        
        return list(dict.fromkeys(file_ids))  # Dedupe preserving order
    
    def _contains_auth_errors(self, text: str) -> bool:
        """Check if text contains auth error patterns.
        
        Args:
            text: Text to check
            
        Returns:
            True if auth errors detected
        """
        # Import patterns from summarize module
        from summarize import AUTH_ERROR_PATTERNS
        import re
        
        combined_pattern = '|'.join(AUTH_ERROR_PATTERNS)
        return bool(re.search(combined_pattern, text, re.IGNORECASE))
    
    def _generate_summary_id(
        self,
        conversation_id: int,
        kind: str,
        index: int
    ) -> str:
        """Generate a unique summary ID.
        
        Args:
            conversation_id: Conversation ID
            kind: Summary kind (leaf/condensed)
            index: Index within this batch
            
        Returns:
            Unique summary ID
        """
        timestamp = datetime.utcnow().isoformat()
        hash_input = f"{conversation_id}_{kind}_{index}_{timestamp}"
        hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:12]
        return f"sum_{kind[:1]}_{hash_value}"
    
    def _calculate_total_tokens(
        self,
        messages: List[MessageRecord],
        summaries: List[SummaryRecord],
        context_items: List[ContextItemRecord]
    ) -> int:
        """Calculate total tokens in context.
        
        Args:
            messages: All message records
            summaries: All summary records
            context_items: All context items
            
        Returns:
            Total token count
        """
        message_lookup = {m.message_id: m for m in messages}
        summary_lookup = {s.summary_id: s for s in summaries}
        
        total = 0
        
        for item in context_items:
            if item.item_type == ContextItemType.MESSAGE:
                msg = message_lookup.get(item.message_id)
                if msg:
                    total += msg.token_count
            elif item.item_type == ContextItemType.SUMMARY:
                summary = summary_lookup.get(item.summary_id)
                if summary:
                    total += summary.token_count
        
        return total


def create_compaction_engine_from_config(
    lcm_config: "LcmConfig",
    summarizer: Optional[Callable[[str, bool, Optional[Dict[str, Any]]], Awaitable[str]]] = None,
    log: Optional[Callable[[str], None]] = None
) -> CompactionEngine:
    """Create a CompactionEngine from LcmConfig.
    
    Args:
        lcm_config: LCM configuration
        summarizer: Optional async summarizer function
        log: Optional logging function
        
    Returns:
        Configured CompactionEngine
    """
    config = CompactionConfig(
        token_budget=128000,  # Default model context window
        context_threshold=lcm_config.context_threshold,
        fresh_tail_count=lcm_config.fresh_tail_count,
        leaf_min_fanout=lcm_config.leaf_min_fanout,
        condensed_min_fanout=lcm_config.condensed_min_fanout,
        condensed_min_fanout_hard=lcm_config.condensed_min_fanout_hard,
        leaf_chunk_tokens=lcm_config.leaf_chunk_tokens,
        leaf_target_tokens=lcm_config.leaf_target_tokens,
        condensed_target_tokens=lcm_config.condensed_target_tokens,
        max_rounds=10,
        aggressive_ratio=1.5,
        fallback_ratio=2.0
    )
    
    return CompactionEngine(
        config=config,
        summarizer=summarizer,
        log=log
    )


if __name__ == "__main__":
    import asyncio
    from datetime import datetime
    
    async def demo():
        # Create sample messages
        messages = [
            MessageRecord(
                message_id=i,
                conversation_id=1,
                seq=i,
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i}: " + ("x" * 500),
                token_count=50 + i,
                created_at=datetime.utcnow()
            )
            for i in range(1, 51)
        ]
        
        # Create engine
        config = CompactionConfig(
            token_budget=1000,
            context_threshold=0.75,
            fresh_tail_count=5,
            leaf_min_fanout=3,
            leaf_chunk_tokens=300
        )
        
        engine = CompactionEngine(config)
        
        # Evaluate
        decision = engine.evaluate(messages, [], [])
        print(f"Decision: should_compact={decision.should_compact}")
        print(f"Reason: {decision.reason}")
        print(f"Level: {decision.level}")
        print(f"Phase: {decision.phase}")
        print(f"Current tokens: {decision.current_tokens}")
        print(f"Target tokens: {decision.target_tokens}")
        
        # Compact
        result = await engine.compact(1, messages, [], [])
        print(f"\nCompaction result:")
        print(f"Success: {result.success}")
        print(f"Rounds: {result.rounds_completed}")
        print(f"Final tokens: {result.final_tokens}")
        print(f"Tokens saved: {result.stats.tokens_saved}")
        print(f"Summaries created: {len(result.new_summaries)}")
        print(f"Warnings: {result.warnings}")
    
    asyncio.run(demo())
