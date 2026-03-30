#!/usr/bin/env python3
"""Retrieval engine for LCM.

Provides search and retrieval capabilities:
- describe() - Lookup summaries and files by ID
- grep() - Regex/full-text search across messages and summaries
- expand() - Traverse summary hierarchy to get detailed context

Port of TypeScript retrieval.ts from lossless-claw-enhanced.
"""

import re
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Pattern

from estimate_tokens import estimate_tokens
from search.full_text_fallback import (
    contains_cjk,
    build_like_search_plan,
    create_fallback_snippet,
    should_use_fallback
)
from search.fts5_sanitize import sanitize_fts5_query
from lcm_types import (
    MessageRole,
    SummaryKind,
    ContextItemType,
    MessageRecord,
    MessagePartRecord,
    SummaryRecord,
    ContextItemRecord,
    LargeFileRecord,
    SummarySearchInput,
    SummarySearchResult,
)


@dataclass
class DescribeResult:
    """Result from describe() operation."""
    found: bool
    item_type: Optional[str] = None  # "summary", "file", "message"
    item_id: Optional[str] = None
    content: Optional[str] = None
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class GrepResult:
    """Result from grep() search operation."""
    matches: List[Dict[str, Any]]
    total_count: int
    query: str
    mode: str  # "regex" or "full_text"
    truncated: bool
    warnings: List[str] = field(default_factory=list)


@dataclass
class ExpandResult:
    """Result from expand() operation."""
    summary_id: str
    expanded_content: str
    token_count: int
    source_messages: List[Dict[str, Any]]
    depth: int
    descendant_summaries: List[str]
    warnings: List[str] = field(default_factory=list)


@dataclass
class RetrievalConfig:
    """Configuration for retrieval engine."""
    max_results: int = 50
    max_expand_depth: int = 5
    snippet_max_length: int = 200
    context_chars: int = 50


class RetrievalEngine:
    """Search and retrieval engine for LCM.
    
    Features:
    - describe() - Lookup summaries and files by ID
    - grep() - Regex/full-text search with CJK support
    - expand() - Traverse summary hierarchy
    """
    
    def __init__(
        self,
        config: RetrievalConfig,
        db_connection: Any,  # sqlite3.Connection
        log: Optional[Callable[[str], None]] = None
    ):
        """Initialize retrieval engine.
        
        Args:
            config: Retrieval configuration
            db_connection: SQLite database connection
            log: Optional logging function
        """
        self.config = config
        self.db = db_connection
        self._log = log or print
    
    def describe(
        self,
        item_id: str,
        include_children: bool = True,
        max_depth: int = 3
    ) -> DescribeResult:
        """Look up a summary or file by its ID.
        
        Args:
            item_id: ID of the summary or file to look up
            include_children: Whether to include child summary IDs
            max_depth: Maximum depth for child traversal
            
        Returns:
            DescribeResult with the item details
        """
        # Try to find as summary first
        summary = self._get_summary_by_id(item_id)
        if summary:
            return self._describe_summary(summary, include_children, max_depth)
        
        # Try to find as file
        file = self._get_file_by_id(item_id)
        if file:
            return self._describe_file(file)
        
        # Try to find as message
        message = self._get_message_by_id(item_id)
        if message:
            return self._describe_message(message)
        
        return DescribeResult(
            found=False,
            item_id=item_id,
            error=f"Item not found: {item_id}"
        )
    
    def _describe_summary(
        self,
        summary: SummaryRecord,
        include_children: bool,
        max_depth: int
    ) -> DescribeResult:
        """Create describe result for a summary."""
        children = []
        
        if include_children and max_depth > 0:
            children = self._get_child_summary_ids(summary.summary_id, max_depth)
        
        metadata = {
            "kind": summary.kind.value if hasattr(summary.kind, 'value') else summary.kind,
            "depth": summary.depth,
            "descendant_count": summary.descendant_count,
            "descendant_token_count": summary.descendant_token_count,
            "earliest_at": summary.earliest_at.isoformat() if summary.earliest_at else None,
            "latest_at": summary.latest_at.isoformat() if summary.latest_at else None,
            "model": summary.model,
            "file_ids": summary.file_ids[:10] if summary.file_ids else []
        }
        
        return DescribeResult(
            found=True,
            item_type="summary",
            item_id=summary.summary_id,
            content=summary.content,
            token_count=summary.token_count,
            metadata=metadata,
            children=children
        )
    
    def _describe_file(self, file: LargeFileRecord) -> DescribeResult:
        """Create describe result for a file."""
        metadata = {
            "file_name": file.file_name,
            "mime_type": file.mime_type,
            "byte_size": file.byte_size,
            "storage_uri": file.storage_uri,
            "created_at": file.created_at.isoformat() if file.created_at else None
        }
        
        content = file.exploration_summary or ""
        if not content and file.storage_uri:
            content = f"[File content stored at: {file.storage_uri}]"
        
        return DescribeResult(
            found=True,
            item_type="file",
            item_id=file.file_id,
            content=content,
            token_count=estimate_tokens(content),
            metadata=metadata
        )
    
    def _describe_message(self, message: MessageRecord) -> DescribeResult:
        """Create describe result for a message."""
        metadata = {
            "conversation_id": message.conversation_id,
            "seq": message.seq,
            "role": message.role.value if hasattr(message.role, 'value') else message.role,
            "created_at": message.created_at.isoformat() if message.created_at else None
        }
        
        return DescribeResult(
            found=True,
            item_type="message",
            item_id=str(message.message_id),
            content=message.content,
            token_count=message.token_count,
            metadata=metadata
        )
    
    def grep(
        self,
        query: str,
        mode: str = "full_text",
        conversation_id: Optional[int] = None,
        since: Optional[datetime] = None,
        before: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> GrepResult:
        """Search for content across messages and summaries.
        
        Args:
            query: Search query string
            mode: "regex" or "full_text"
            conversation_id: Optional conversation filter
            since: Optional start datetime filter
            before: Optional end datetime filter
            limit: Maximum results (default from config)
            
        Returns:
            GrepResult with matching items
        """
        limit = limit or self.config.max_results
        warnings = []
        matches = []
        
        # Determine if we should use LIKE fallback for CJK
        use_fallback = should_use_fallback(query) if mode == "full_text" else False
        
        if use_fallback:
            warnings.append("Using LIKE fallback for CJK content")
        
        # Search messages
        message_matches = self._search_messages(
            query=query,
            mode=mode,
            conversation_id=conversation_id,
            since=since,
            before=before,
            limit=limit,
            use_fallback=use_fallback
        )
        matches.extend(message_matches)
        
        # Search summaries if we have room
        remaining = limit - len(matches)
        if remaining > 0:
            summary_matches = self._search_summaries(
                query=query,
                mode=mode,
                conversation_id=conversation_id,
                limit=remaining,
                use_fallback=use_fallback
            )
            matches.extend(summary_matches)
        
        total_count = len(matches)
        truncated = total_count >= limit
        
        return GrepResult(
            matches=matches,
            total_count=total_count,
            query=query,
            mode=mode,
            truncated=truncated,
            warnings=warnings
        )
    
    def _search_messages(
        self,
        query: str,
        mode: str,
        conversation_id: Optional[int],
        since: Optional[datetime],
        before: Optional[datetime],
        limit: int,
        use_fallback: bool
    ) -> List[Dict[str, Any]]:
        """Search messages table."""
        matches = []
        
        try:
            if use_fallback:
                # Use LIKE-based search for CJK
                plan = build_like_search_plan("content", query)
                if not plan.where:
                    return []
                
                sql = f"""
                    SELECT message_id, conversation_id, role, content, created_at
                    FROM messages
                    WHERE {plan.where}
                """
                args = plan.args.copy()
                
            elif mode == "regex":
                # Regex search
                sql = """
                    SELECT message_id, conversation_id, role, content, created_at
                    FROM messages
                    WHERE content REGEXP ?
                """
                args = [query]
                
            else:
                # Standard LIKE search
                sql = """
                    SELECT message_id, conversation_id, role, content, created_at
                    FROM messages
                    WHERE content LIKE ?
                """
                args = [f"%{query}%"]
            
            # Add filters
            if conversation_id is not None:
                sql += " AND conversation_id = ?"
                args.append(conversation_id)
            
            if since is not None:
                sql += " AND created_at >= ?"
                args.append(since.isoformat())
            
            if before is not None:
                sql += " AND created_at <= ?"
                args.append(before.isoformat())
            
            sql += " ORDER BY created_at DESC LIMIT ?"
            args.append(limit)
            
            cursor = self.db.execute(sql, args)
            
            for row in cursor.fetchall():
                message_id, conv_id, role, content, created_at = row
                
                # Create snippet
                terms = [query] if use_fallback else []
                snippet = create_fallback_snippet(
                    content, 
                    terms,
                    self.config.snippet_max_length
                ) if use_fallback else self._create_snippet(content, query)
                
                matches.append({
                    "type": "message",
                    "message_id": message_id,
                    "conversation_id": conv_id,
                    "role": role,
                    "snippet": snippet,
                    "created_at": created_at
                })
                
        except Exception as e:
            self._log(f"[ERROR] Message search failed: {e}")
        
        return matches
    
    def _search_summaries(
        self,
        query: str,
        mode: str,
        conversation_id: Optional[int],
        limit: int,
        use_fallback: bool
    ) -> List[Dict[str, Any]]:
        """Search summaries table."""
        matches = []
        
        try:
            if use_fallback:
                plan = build_like_search_plan("content", query)
                if not plan.where:
                    return []
                
                sql = f"""
                    SELECT summary_id, conversation_id, kind, content, created_at
                    FROM summaries
                    WHERE {plan.where}
                """
                args = plan.args.copy()
                
            elif mode == "regex":
                sql = """
                    SELECT summary_id, conversation_id, kind, content, created_at
                    FROM summaries
                    WHERE content REGEXP ?
                """
                args = [query]
                
            else:
                sql = """
                    SELECT summary_id, conversation_id, kind, content, created_at
                    FROM summaries
                    WHERE content LIKE ?
                """
                args = [f"%{query}%"]
            
            if conversation_id is not None:
                sql += " AND conversation_id = ?"
                args.append(conversation_id)
            
            sql += " ORDER BY created_at DESC LIMIT ?"
            args.append(limit)
            
            cursor = self.db.execute(sql, args)
            
            for row in cursor.fetchall():
                summary_id, conv_id, kind, content, created_at = row
                
                terms = [query] if use_fallback else []
                snippet = create_fallback_snippet(
                    content, 
                    terms,
                    self.config.snippet_max_length
                ) if use_fallback else self._create_snippet(content, query)
                
                matches.append({
                    "type": "summary",
                    "summary_id": summary_id,
                    "conversation_id": conv_id,
                    "kind": kind,
                    "snippet": snippet,
                    "created_at": created_at
                })
                
        except Exception as e:
            self._log(f"[ERROR] Summary search failed: {e}")
        
        return matches
    
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
        warnings = []
        source_messages = []
        descendant_summaries = []
        expanded_parts = []
        total_tokens = 0
        
        # Get the summary
        summary = self._get_summary_by_id(summary_id)
        if not summary:
            return ExpandResult(
                summary_id=summary_id,
                expanded_content="",
                token_count=0,
                source_messages=[],
                depth=0,
                descendant_summaries=[],
                warnings=[f"Summary not found: {summary_id}"]
            )
        
        # Add summary content
        expanded_parts.append(f"[Summary {summary_id}]")
        expanded_parts.append(summary.content)
        total_tokens += summary.token_count
        
        # Get child summaries
        children = self._get_child_summaries(summary_id)
        for child in children:
            if total_tokens >= max_tokens:
                warnings.append(f"Token budget reached, truncating expansion")
                break
            
            descendant_summaries.append(child.summary_id)
            child_content = f"\n[Child Summary {child.summary_id}]\n{child.content}"
            child_tokens = estimate_tokens(child_content)
            
            if total_tokens + child_tokens <= max_tokens:
                expanded_parts.append(child_content)
                total_tokens += child_tokens
        
        # Get source messages
        messages = self._get_messages_for_summary(summary_id)
        for msg in messages:
            if total_tokens >= max_tokens:
                warnings.append(f"Token budget reached, truncating source messages")
                break
            
            msg_content = f"\n[Message {msg.message_id}] ({msg.role})\n{msg.content}"
            msg_tokens = estimate_tokens(msg_content)
            
            if total_tokens + msg_tokens <= max_tokens:
                expanded_parts.append(msg_content)
                source_messages.append({
                    "message_id": msg.message_id,
                    "role": msg.role.value if hasattr(msg.role, 'value') else msg.role,
                    "content": msg.content
                })
                total_tokens += msg_tokens
        
        # Include file references if requested
        if include_files and summary.file_ids:
            for file_id in summary.file_ids:
                file = self._get_file_by_id(file_id)
                if file:
                    file_content = f"\n[File {file_id}]\n{file.exploration_summary or '[No summary]'}"
                    file_tokens = estimate_tokens(file_content)
                    
                    if total_tokens + file_tokens <= max_tokens:
                        expanded_parts.append(file_content)
                        total_tokens += file_tokens
        
        expanded_content = "\n".join(expanded_parts)
        
        return ExpandResult(
            summary_id=summary_id,
            expanded_content=expanded_content,
            token_count=total_tokens,
            source_messages=source_messages,
            depth=summary.depth,
            descendant_summaries=descendant_summaries,
            warnings=warnings
        )
    
    def _create_snippet(self, content: str, query: str, max_len: int = 200) -> str:
        """Create a snippet with query highlighted.
        
        Args:
            content: Full content
            query: Search query
            max_len: Maximum snippet length
            
        Returns:
            Snippet with highlighted match
        """
        if not content or not query:
            return content[:max_len] if content else ""
        
        # Find query position (case-insensitive)
        pos = content.lower().find(query.lower())
        if pos == -1:
            return content[:max_len] + "..." if len(content) > max_len else content
        
        # Calculate snippet boundaries
        start = max(0, pos - self.config.context_chars)
        end = min(len(content), pos + max_len - self.config.context_chars)
        
        # Adjust to word boundaries
        if start > 0:
            space_pos = content.find(' ', start)
            if space_pos != -1 and space_pos < start + 20:
                start = space_pos + 1
        
        snippet = content[start:end]
        
        # Highlight the query
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        highlighted = pattern.sub(f"**{query}**", snippet)
        
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(content) else ""
        
        return f"{prefix}{highlighted}{suffix}"
    
    # Database access methods (to be implemented with actual DB)
    
    def _get_summary_by_id(self, summary_id: str) -> Optional[SummaryRecord]:
        """Get a summary by its ID."""
        try:
            cursor = self.db.execute(
                """
                SELECT summary_id, conversation_id, kind, depth, content, token_count,
                       file_ids, earliest_at, latest_at, descendant_count, 
                       descendant_token_count, source_message_token_count, model, created_at
                FROM summaries
                WHERE summary_id = ?
                """,
                (summary_id,)
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_summary(row)
        except Exception as e:
            self._log(f"[ERROR] Failed to get summary {summary_id}: {e}")
        return None
    
    def _get_file_by_id(self, file_id: str) -> Optional[LargeFileRecord]:
        """Get a large file by its ID."""
        try:
            cursor = self.db.execute(
                """
                SELECT file_id, conversation_id, file_name, mime_type, byte_size,
                       storage_uri, exploration_summary, created_at
                FROM large_files
                WHERE file_id = ?
                """,
                (file_id,)
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_file(row)
        except Exception as e:
            self._log(f"[ERROR] Failed to get file {file_id}: {e}")
        return None
    
    def _get_message_by_id(self, message_id: int) -> Optional[MessageRecord]:
        """Get a message by its ID."""
        try:
            cursor = self.db.execute(
                """
                SELECT message_id, conversation_id, seq, role, content, token_count, created_at
                FROM messages
                WHERE message_id = ?
                """,
                (message_id,)
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_message(row)
        except Exception as e:
            self._log(f"[ERROR] Failed to get message {message_id}: {e}")
        return None
    
    def _get_child_summary_ids(self, parent_id: str, max_depth: int) -> List[str]:
        """Get child summary IDs recursively."""
        children = []
        if max_depth <= 0:
            return children
        
        try:
            # Find summaries that are children of this one
            # This depends on how parent-child relationships are stored
            cursor = self.db.execute(
                """
                SELECT summary_id FROM summaries
                WHERE parent_summary_id = ?
                LIMIT 100
                """,
                (parent_id,)
            )
            for row in cursor.fetchall():
                children.append(row[0])
                if max_depth > 1:
                    children.extend(self._get_child_summary_ids(row[0], max_depth - 1))
        except Exception:
            pass  # Table might not have parent_summary_id column
        
        return children
    
    def _get_child_summaries(self, parent_id: str) -> List[SummaryRecord]:
        """Get child summary records."""
        summaries = []
        try:
            cursor = self.db.execute(
                """
                SELECT summary_id, conversation_id, kind, depth, content, token_count,
                       file_ids, earliest_at, latest_at, descendant_count, 
                       descendant_token_count, source_message_token_count, model, created_at
                FROM summaries
                WHERE parent_summary_id = ?
                ORDER BY depth DESC, created_at ASC
                """,
                (parent_id,)
            )
            for row in cursor.fetchall():
                summaries.append(self._row_to_summary(row))
        except Exception:
            pass
        return summaries
    
    def _get_messages_for_summary(self, summary_id: str) -> List[MessageRecord]:
        """Get source messages for a summary."""
        messages = []
        try:
            # This depends on how message-summary relationships are stored
            # Assuming there's a summary_messages or similar table
            cursor = self.db.execute(
                """
                SELECT m.message_id, m.conversation_id, m.seq, m.role, m.content, m.token_count, m.created_at
                FROM messages m
                INNER JOIN summary_messages sm ON m.message_id = sm.message_id
                WHERE sm.summary_id = ?
                ORDER BY m.seq ASC
                """,
                (summary_id,)
            )
            for row in cursor.fetchall():
                messages.append(self._row_to_message(row))
        except Exception:
            pass
        return messages
    
    # Row conversion helpers
    
    def _row_to_summary(self, row: tuple) -> SummaryRecord:
        """Convert database row to SummaryRecord."""
        file_ids = []
        if row[6]:  # file_ids column
            try:
                file_ids = json.loads(row[6]) if isinstance(row[6], str) else row[6]
            except (json.JSONDecodeError, TypeError):
                file_ids = []
        
        return SummaryRecord(
            summary_id=row[0],
            conversation_id=row[1],
            kind=SummaryKind(row[2]) if isinstance(row[2], str) else row[2],
            depth=row[3],
            content=row[4],
            token_count=row[5],
            file_ids=file_ids,
            earliest_at=datetime.fromisoformat(row[7]) if row[7] else None,
            latest_at=datetime.fromisoformat(row[8]) if row[8] else None,
            descendant_count=row[9] or 0,
            descendant_token_count=row[10] or 0,
            source_message_token_count=row[11] or 0,
            model=row[12] or "unknown",
            created_at=datetime.fromisoformat(row[13]) if row[13] else datetime.utcnow()
        )
    
    def _row_to_file(self, row: tuple) -> LargeFileRecord:
        """Convert database row to LargeFileRecord."""
        return LargeFileRecord(
            file_id=row[0],
            conversation_id=row[1],
            file_name=row[2],
            mime_type=row[3],
            byte_size=row[4],
            storage_uri=row[5],
            exploration_summary=row[6],
            created_at=datetime.fromisoformat(row[7]) if row[7] else datetime.utcnow()
        )
    
    def _row_to_message(self, row: tuple) -> MessageRecord:
        """Convert database row to MessageRecord."""
        return MessageRecord(
            message_id=row[0],
            conversation_id=row[1],
            seq=row[2],
            role=MessageRole(row[3]) if isinstance(row[3], str) else row[3],
            content=row[4],
            token_count=row[5],
            created_at=datetime.fromisoformat(row[6]) if row[6] else datetime.utcnow()
        )


def create_retrieval_engine(
    db_connection: Any,
    max_results: int = 50,
    log: Optional[Callable[[str], None]] = None
) -> RetrievalEngine:
    """Create a retrieval engine with default configuration.
    
    Args:
        db_connection: SQLite database connection
        max_results: Maximum search results
        log: Optional logging function
        
    Returns:
        Configured RetrievalEngine
    """
    config = RetrievalConfig(max_results=max_results)
    return RetrievalEngine(config, db_connection, log)


if __name__ == "__main__":
    # Demo usage
    import sqlite3
    import tempfile
    from pathlib import Path
    
    # Create in-memory database for demo
    db = sqlite3.connect(":memory:")
    
    # Create tables
    db.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            message_id INTEGER PRIMARY KEY,
            conversation_id INTEGER,
            seq INTEGER,
            role TEXT,
            content TEXT,
            token_count INTEGER,
            created_at TEXT
        )
    """)
    
    db.execute("""
        CREATE TABLE IF NOT EXISTS summaries (
            summary_id TEXT PRIMARY KEY,
            conversation_id INTEGER,
            kind TEXT,
            depth INTEGER,
            content TEXT,
            token_count INTEGER,
            file_ids TEXT,
            earliest_at TEXT,
            latest_at TEXT,
            descendant_count INTEGER,
            descendant_token_count INTEGER,
            source_message_token_count INTEGER,
            model TEXT,
            created_at TEXT
        )
    """)
    
    # Insert test data
    now = datetime.utcnow().isoformat()
    db.execute(
        "INSERT INTO messages VALUES (1, 1, 1, 'user', 'Hello, how are you?', 10, ?)",
        (now,)
    )
    db.execute(
        "INSERT INTO messages VALUES (2, 1, 2, 'assistant', 'I am doing well! Can I help you with Python?', 15, ?)",
        (now,)
    )
    db.execute(
        "INSERT INTO summaries VALUES ('sum_1', 1, 'leaf', 0, 'User greeted and asked about Python', 20, '[]', ?, ?, 0, 0, 25, 'unknown', ?)",
        (now, now, now)
    )
    db.commit()
    
    # Create engine and test
    engine = create_retrieval_engine(db)
    
    # Test describe
    result = engine.describe("sum_1")
    print(f"Describe result: found={result.found}, type={result.item_type}")
    print(f"Content: {result.content}")
    
    # Test grep
    search_result = engine.grep("Python", mode="full_text")
    print(f"\nGrep results: {search_result.total_count} matches")
    for match in search_result.matches:
        print(f"  - [{match['type']}] {match['snippet'][:50]}...")
    
    db.close()
