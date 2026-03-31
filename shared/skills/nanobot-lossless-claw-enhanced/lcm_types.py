#!/usr/bin/env python3
"""Core types for LCM (Lossless Context Management).

Dataclasses and enums for the Python port of lossless-claw-enhanced.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Any, Dict, Callable, Awaitable, Protocol
from datetime import datetime
from enum import Enum


class MessageRole(str, Enum):
    """Message role in conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class MessagePartType(str, Enum):
    """Type of message part content block."""
    TEXT = "text"
    REASONING = "reasoning"
    TOOL = "tool"
    PATCH = "patch"
    FILE = "file"
    SUBTASK = "subtask"
    COMPACTION = "compaction"
    STEP_START = "step_start"
    STEP_FINISH = "step_finish"
    SNAPSHOT = "snapshot"
    AGENT = "agent"
    RETRY = "retry"


class SummaryKind(str, Enum):
    """Kind of summary in the DAG."""
    LEAF = "leaf"
    CONDENSED = "condensed"


class MemoryCategory(str, Enum):
    """Memory category classification (from memory-lancedb-pro)."""
    PROFILE = "profile"      # User profiles, identity
    PREFERENCES = "preferences"  # User preferences, settings
    ENTITIES = "entities"    # Named entities, objects
    EVENTS = "events"       # Event occurrences
    CASES = "cases"         # Case studies, examples
    PATTERNS = "patterns"   # Behavioral patterns, rules
    FACT = "fact"           # Factual knowledge
    DECISION = "decision"   # Decisions made
    OTHER = "other"         # Unclassified


class MemoryTier(str, Enum):
    """Memory tier based on importance/recency (from memory-lancedb-pro)."""
    PERIPHERAL = "peripheral"  # Low importance, can be forgotten
    WORKING = "working"        # Medium importance, active use
    CORE = "core"              # High importance, persistent


class ContextItemType(str, Enum):
    """Type of context item."""
    MESSAGE = "message"
    SUMMARY = "summary"


@dataclass
class CompletionResult:
    """Result from LLM completion call."""
    content: List[Dict[str, Any]]
    error: Optional[Dict[str, Any]] = None


@dataclass
class LcmDependencies:
    """Dependencies injected into LCM engine."""
    config: "LcmConfig"
    complete: Callable[..., Awaitable[CompletionResult]]
    log: Any  # Logger interface
    
    # Optional dependencies
    read_file: Optional[Callable[[str], str]] = None
    write_file: Optional[Callable[[str, str], None]] = None
    file_exists: Optional[Callable[[str], bool]] = None


    get_file_size: Optional[Callable[[str], int]] = None


    get_file_mtime: Optional[Callable[[str], float]] = None


@dataclass
class MessageRecord:
    """Database record for a message."""
    message_id: int
    conversation_id: int
    seq: int
    role: MessageRole
    content: str
    token_count: int
    created_at: datetime


@dataclass
class MessagePartRecord:
    """Database record for a message part."""
    part_id: str
    message_id: int
    session_id: str
    part_type: MessagePartType
    ordinal: int
    text_content: Optional[str] = None
    is_ignored: Optional[bool] = None
    is_synthetic: Optional[bool] = None
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_status: Optional[str] = None
    tool_input: Optional[str] = None
    tool_output: Optional[str] = None
    tool_error: Optional[str] = None
    tool_title: Optional[str] = None
    patch_hash: Optional[str] = None
    patch_files: Optional[str] = None
    file_mime: Optional[str] = None
    file_name: Optional[str] = None
    file_url: Optional[str] = None
    subtask_prompt: Optional[str] = None
    subtask_desc: Optional[str] = None
    subtask_agent: Optional[str] = None
    step_reason: Optional[str] = None
    step_cost: Optional[float] = None
    step_tokens_in: Optional[int] = None
    step_tokens_out: Optional[int] = None
    snapshot_hash: Optional[str] = None
    compaction_auto: Optional[bool] = None
    metadata: Optional[str] = None


@dataclass
class ConversationRecord:
    """Database record for a conversation."""
    conversation_id: int
    session_id: str
    session_key: Optional[str]
    title: Optional[str]
    bootstrapped_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime


@dataclass
class SummaryRecord:
    """Database record for a summary."""
    summary_id: str
    conversation_id: int
    kind: SummaryKind
    depth: int
    content: str
    token_count: int
    file_ids: List[str] = field(default_factory=list)
    earliest_at: Optional[datetime] = None
    latest_at: Optional[datetime] = None
    descendant_count: int = 0
    descendant_token_count: int = 0
    source_message_token_count: int = 0
    model: str = "unknown"
    created_at: Optional[datetime] = None
    # Memory lifecycle fields (from memory-lancedb-pro)
    category: Optional[MemoryCategory] = None
    tier: MemoryTier = MemoryTier.PERIPHERAL
    access_count: int = 0
    last_accessed_at: Optional[datetime] = None
    decay_score: float = 1.0  # Weibull decay score
    importance: float = 0.5   # 0.0 to 1.0
    # Scope isolation field (global/agent:/project:/user:)
    scope: str = "global"


@dataclass
class ContextItemRecord:
    """Database record for a context item (message or summary reference)."""
    conversation_id: int
    ordinal: int
    item_type: ContextItemType
    message_id: Optional[int] = None
    summary_id: Optional[str] = None
    created_at: Optional[datetime] = None


@dataclass
class LargeFileRecord:
    """Database record for a large externalized file."""
    file_id: str
    conversation_id: int
    storage_uri: str = ""
    file_name: Optional[str] = None
    mime_type: Optional[str] = None
    byte_size: Optional[int] = None
    exploration_summary: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    rank: Optional[int] = None


@dataclass
class SummarySearchInput:
    """Input for searching summaries."""
    query: str
    mode: str  # "regex" | "full_text"
    conversation_id: Optional[int] = None
    limit: int = 50


@dataclass
class SummarySearchResult:
    """Result from searching summaries."""
    summary_id: str
    conversation_id: int
    kind: SummaryKind
    snippet: str
    created_at: datetime
    rank: Optional[int] = None


# Protocol for summarization

class BaseSummarizer(Protocol):
    """Protocol for LLM summarization implementations."""
    
    async def __call__(
        self,
        text: str,
        aggressive: bool = False,
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """Summarize text using LLM.
        
        Args:
            text: Text to summarize
            aggressive: If True, use more aggressive summarization
            options: Additional options for summarization
            
        Returns:
            Summarized text
        """
        ...
