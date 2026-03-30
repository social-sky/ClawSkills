#!/usr/bin/env python3
"""Conversation Store for LCM.

Handles CRUD operations for conversations, messages, and message parts.
Port of TypeScript store/conversationStore.ts from lossless-claw-enhanced.
"""

import sqlite3
import json
import re
from datetime import datetime
from typing import Optional, List, Dict, Any

from ..lcm_types import (
    ConversationRecord,
    MessageRecord,
    MessagePartRecord,
    CreateConversationInput,
    CreateMessageInput,
    CreateMessagePartInput,
    MessageSearchInput,
    MessageSearchResult,
    MessageRole,
    MessagePartType,
)
from ..db.features import get_lcm_db_features
from ..search.fts5_sanitize import sanitize_fts5_query
from ..search.full_text_fallback import (
    contains_cjk,
    build_like_search_plan,
    create_fallback_snippet,
)


class ConversationStore:
    """Data access layer for conversations, messages, and message parts.
    
    Provides CRUD operations with proper transaction handling and
    FTS5 full-text search with LIKE fallback for CJK text.
    """
    
    def __init__(self, db: sqlite3.Connection):
        """Initialize the store with a database connection.
        
        Args:
            db: SQLite connection with proper configuration
        """
        self.db = db
        self._fts5_available = get_lcm_db_features(db).fts5_available
    
    # =========================================================================
    # Conversation Operations
    # =========================================================================
    
    def create_conversation(
        self,
        session_id: str,
        session_key: Optional[str] = None,
        title: Optional[str] = None
    ) -> int:
        """Create a new conversation.
        
        Args:
            session_id: Unique session identifier
            session_key: Optional session key for deduplication
            title: Optional conversation title
            
        Returns:
            The new conversation_id
        """
        now = datetime.utcnow().isoformat()
        
        cursor = self.db.execute(
            """
            INSERT INTO conversations (session_id, session_key, title, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, session_key, title, now, now)
        )
        
        return cursor.lastrowid
    
    def get_conversation_by_id(self, conversation_id: int) -> Optional[ConversationRecord]:
        """Get a conversation by ID.
        
        Args:
            conversation_id: The conversation ID
            
        Returns:
            ConversationRecord or None if not found
        """
        row = self.db.execute(
            """
            SELECT conversation_id, session_id, session_key, title, 
                   bootstrapped_at, created_at, updated_at
            FROM conversations
            WHERE conversation_id = ?
            """,
            (conversation_id,)
        ).fetchone()
        
        if not row:
            return None
        
        return self._row_to_conversation(row)
    
    def get_conversation_by_session_id(self, session_id: str) -> Optional[ConversationRecord]:
        """Get a conversation by session ID.
        
        Args:
            session_id: The session ID
            
        Returns:
            ConversationRecord or None if not found
        """
        row = self.db.execute(
            """
            SELECT conversation_id, session_id, session_key, title,
                   bootstrapped_at, created_at, updated_at
            FROM conversations
            WHERE session_id = ?
            """,
            (session_id,)
        ).fetchone()
        
        if not row:
            return None
        
        return self._row_to_conversation(row)
    
    def get_conversation_by_session_key(self, session_key: str) -> Optional[ConversationRecord]:
        """Get a conversation by session key.
        
        Args:
            session_key: The session key
            
        Returns:
            ConversationRecord or None if not found
        """
        row = self.db.execute(
            """
            SELECT conversation_id, session_id, session_key, title,
                   bootstrapped_at, created_at, updated_at
            FROM conversations
            WHERE session_key = ?
            """,
            (session_key,)
        ).fetchone()
        
        if not row:
            return None
        
        return self._row_to_conversation(row)
    
    def update_conversation(
        self,
        conversation_id: int,
        title: Optional[str] = None,
        bootstrapped_at: Optional[datetime] = None
    ) -> bool:
        """Update a conversation.
        
        Args:
            conversation_id: The conversation ID
            title: Optional new title
            bootstrapped_at: Optional bootstrap timestamp
            
        Returns:
            True if updated, False if not found
        """
        updates = []
        args = []
        
        if title is not None:
            updates.append("title = ?")
            args.append(title)
        
        if bootstrapped_at is not None:
            updates.append("bootstrapped_at = ?")
            args.append(bootstrapped_at.isoformat())
        
        if not updates:
            return False
        
        updates.append("updated_at = ?")
        args.append(datetime.utcnow().isoformat())
        args.append(conversation_id)
        
        cursor = self.db.execute(
            f"UPDATE conversations SET {', '.join(updates)} WHERE conversation_id = ?",
            args
        )
        
        return cursor.rowcount > 0
    
    def delete_conversation(self, conversation_id: int) -> bool:
        """Delete a conversation and all its messages.
        
        Args:
            conversation_id: The conversation ID
            
        Returns:
            True if deleted, False if not found
        """
        cursor = self.db.execute(
            "DELETE FROM conversations WHERE conversation_id = ?",
            (conversation_id,)
        )
        
        return cursor.rowcount > 0
    
    def list_conversations(
        self,
        limit: int = 50,
        offset: int = 0
    ) -> List[ConversationRecord]:
        """List conversations ordered by creation date.
        
        Args:
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of ConversationRecord
        """
        rows = self.db.execute(
            """
            SELECT conversation_id, session_id, session_key, title,
                   bootstrapped_at, created_at, updated_at
            FROM conversations
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset)
        ).fetchall()
        
        return [self._row_to_conversation(row) for row in rows]
    
    # =========================================================================
    # Message Operations
    # =========================================================================
    
    def create_message(
        self,
        conversation_id: int,
        seq: int,
        role: MessageRole,
        content: str,
        token_count: int
    ) -> int:
        """Create a new message.
        
        Args:
            conversation_id: The conversation ID
            seq: Sequence number within the conversation
            role: Message role (system, user, assistant, tool)
            content: Message content
            token_count: Estimated token count
            
        Returns:
            The new message_id
        """
        now = datetime.utcnow().isoformat()
        
        cursor = self.db.execute(
            """
            INSERT INTO messages (conversation_id, seq, role, content, token_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (conversation_id, seq, role.value, content, token_count, now)
        )
        
        message_id = cursor.lastrowid
        
        # Update conversation updated_at
        self.db.execute(
            "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
            (now, conversation_id)
        )
        
        # Add to FTS5 index if available
        if self._fts5_available:
            try:
                self.db.execute(
                    "INSERT INTO messages_fts(rowid, content) VALUES (?, ?)",
                    (message_id, content)
                )
            except sqlite3.OperationalError:
                pass  # FTS table might not exist
        
        return message_id
    
    def create_messages_bulk(
        self,
        messages: List[CreateMessageInput]
    ) -> List[int]:
        """Create multiple messages in a single transaction.
        
        Args:
            messages: List of CreateMessageInput
            
        Returns:
            List of new message_ids
        """
        if not messages:
            return []
        
        message_ids = []
        now = datetime.utcnow().isoformat()
        
        for msg in messages:
            cursor = self.db.execute(
                """
                INSERT INTO messages (conversation_id, seq, role, content, token_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (msg.conversation_id, msg.seq, msg.role.value, msg.content, msg.token_count, now)
            )
            message_id = cursor.lastrowid
            message_ids.append(message_id)
            
            # Add to FTS5 index if available
            if self._fts5_available:
                try:
                    self.db.execute(
                        "INSERT INTO messages_fts(rowid, content) VALUES (?, ?)",
                        (message_id, msg.content)
                    )
                except sqlite3.OperationalError:
                    pass
        
        # Update conversation updated_at for all affected conversations
        conv_ids = set(m.conversation_id for m in messages)
        for conv_id in conv_ids:
            self.db.execute(
                "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
                (now, conv_id)
            )
        
        return message_ids
    
    def get_message_by_id(self, message_id: int) -> Optional[MessageRecord]:
        """Get a message by ID.
        
        Args:
            message_id: The message ID
            
        Returns:
            MessageRecord or None if not found
        """
        row = self.db.execute(
            """
            SELECT message_id, conversation_id, seq, role, content, token_count, created_at
            FROM messages
            WHERE message_id = ?
            """,
            (message_id,)
        ).fetchone()
        
        if not row:
            return None
        
        return self._row_to_message(row)
    
    def get_message_by_seq(
        self,
        conversation_id: int,
        seq: int
    ) -> Optional[MessageRecord]:
        """Get a message by conversation ID and sequence number.
        
        Args:
            conversation_id: The conversation ID
            seq: Sequence number
            
        Returns:
            MessageRecord or None if not found
        """
        row = self.db.execute(
            """
            SELECT message_id, conversation_id, seq, role, content, token_count, created_at
            FROM messages
            WHERE conversation_id = ? AND seq = ?
            """,
            (conversation_id, seq)
        ).fetchone()
        
        if not row:
            return None
        
        return self._row_to_message(row)
    
    def get_messages_by_conversation(
        self,
        conversation_id: int,
        limit: Optional[int] = None,
        offset: int = 0,
        order: str = "ASC"
    ) -> List[MessageRecord]:
        """Get all messages for a conversation.
        
        Args:
            conversation_id: The conversation ID
            limit: Optional limit on results
            offset: Offset for pagination
            order: Sort order (ASC or DESC)
            
        Returns:
            List of MessageRecord
        """
        sql = f"""
            SELECT message_id, conversation_id, seq, role, content, token_count, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY seq {order}
        """
        
        args = [conversation_id]
        
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            args.extend([limit, offset])
        
        rows = self.db.execute(sql, args).fetchall()
        
        return [self._row_to_message(row) for row in rows]
    
    def get_messages_since(
        self,
        conversation_id: int,
        since_seq: int,
        limit: Optional[int] = None
    ) -> List[MessageRecord]:
        """Get messages since a sequence number.
        
        Args:
            conversation_id: The conversation ID
            since_seq: Starting sequence number (exclusive)
            limit: Optional limit on results
            
        Returns:
            List of MessageRecord
        """
        sql = """
            SELECT message_id, conversation_id, seq, role, content, token_count, created_at
            FROM messages
            WHERE conversation_id = ? AND seq > ?
            ORDER BY seq ASC
        """
        
        args = [conversation_id, since_seq]
        
        if limit is not None:
            sql += " LIMIT ?"
            args.append(limit)
        
        rows = self.db.execute(sql, args).fetchall()
        
        return [self._row_to_message(row) for row in rows]
    
    def get_latest_message(self, conversation_id: int) -> Optional[MessageRecord]:
        """Get the latest message for a conversation.
        
        Args:
            conversation_id: The conversation ID
            
        Returns:
            MessageRecord or None if no messages
        """
        row = self.db.execute(
            """
            SELECT message_id, conversation_id, seq, role, content, token_count, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY seq DESC
            LIMIT 1
            """,
            (conversation_id,)
        ).fetchone()
        
        if not row:
            return None
        
        return self._row_to_message(row)
    
    def get_message_count(self, conversation_id: int) -> int:
        """Get the count of messages for a conversation.
        
        Args:
            conversation_id: The conversation ID
            
        Returns:
            Message count
        """
        row = self.db.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
            (conversation_id,)
        ).fetchone()
        
        return row[0] if row else 0
    
    def get_message_token_sum(self, conversation_id: int) -> int:
        """Get the total token count for a conversation.
        
        Args:
            conversation_id: The conversation ID
            
        Returns:
            Total token count
        """
        row = self.db.execute(
            "SELECT COALESCE(SUM(token_count), 0) FROM messages WHERE conversation_id = ?",
            (conversation_id,)
        ).fetchone()
        
        return row[0] if row else 0
    
    def delete_message(self, message_id: int) -> bool:
        """Delete a message.
        
        Args:
            message_id: The message ID
            
        Returns:
            True if deleted, False if not found
        """
        # Remove from FTS5 index if available
        if self._fts5_available:
            try:
                self.db.execute(
                    "DELETE FROM messages_fts WHERE rowid = ?",
                    (message_id,)
                )
            except sqlite3.OperationalError:
                pass
        
        cursor = self.db.execute(
            "DELETE FROM messages WHERE message_id = ?",
            (message_id,)
        )
        
        return cursor.rowcount > 0
    
    def delete_messages_since(self, conversation_id: int, since_seq: int) -> int:
        """Delete messages since a sequence number.
        
        Args:
            conversation_id: The conversation ID
            since_seq: Starting sequence number (exclusive)
            
        Returns:
            Number of messages deleted
        """
        # Get message IDs to remove from FTS
        if self._fts5_available:
            rows = self.db.execute(
                "SELECT message_id FROM messages WHERE conversation_id = ? AND seq > ?",
                (conversation_id, since_seq)
            ).fetchall()
            
            for row in rows:
                try:
                    self.db.execute(
                        "DELETE FROM messages_fts WHERE rowid = ?",
                        (row[0],)
                    )
                except sqlite3.OperationalError:
                    pass
        
        cursor = self.db.execute(
            "DELETE FROM messages WHERE conversation_id = ? AND seq > ?",
            (conversation_id, since_seq)
        )
        
        return cursor.rowcount
    
    # =========================================================================
    # Message Part Operations
    # =========================================================================
    
    def create_message_part(
        self,
        message_id: int,
        session_id: str,
        part_type: MessagePartType,
        ordinal: int,
        text_content: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        tool_status: Optional[str] = None,
        tool_input: Optional[str] = None,
        tool_output: Optional[str] = None,
        tool_error: Optional[str] = None,
        tool_title: Optional[str] = None,
        patch_hash: Optional[str] = None,
        patch_files: Optional[str] = None,
        file_mime: Optional[str] = None,
        file_name: Optional[str] = None,
        file_url: Optional[str] = None,
        subtask_prompt: Optional[str] = None,
        subtask_desc: Optional[str] = None,
        subtask_agent: Optional[str] = None,
        step_reason: Optional[str] = None,
        step_cost: Optional[float] = None,
        step_tokens_in: Optional[int] = None,
        step_tokens_out: Optional[int] = None,
        snapshot_hash: Optional[str] = None,
        compaction_auto: Optional[bool] = None,
        metadata: Optional[str] = None,
        is_ignored: Optional[bool] = None,
        is_synthetic: Optional[bool] = None
    ) -> str:
        """Create a new message part.
        
        Args:
            message_id: Parent message ID
            session_id: Session ID
            part_type: Type of message part
            ordinal: Ordinal position within the message
            text_content: Text content for text/reasoning parts
            tool_*: Tool-related fields
            patch_*: Patch-related fields
            file_*: File-related fields
            subtask_*: Subtask-related fields
            step_*: Step-related fields
            snapshot_hash: Snapshot hash
            compaction_auto: Auto-compaction flag
            metadata: JSON metadata
            is_ignored: Whether this part should be ignored
            is_synthetic: Whether this part is synthetic
            
        Returns:
            The part_id (generated)
        """
        # Generate part_id
        part_id = f"{message_id}_{ordinal}"
        
        self.db.execute(
            """
            INSERT INTO message_parts (
                part_id, message_id, session_id, part_type, ordinal,
                text_content, is_ignored, is_synthetic,
                tool_call_id, tool_name, tool_status, tool_input, tool_output,
                tool_error, tool_title, patch_hash, patch_files,
                file_mime, file_name, file_url,
                subtask_prompt, subtask_desc, subtask_agent,
                step_reason, step_cost, step_tokens_in, step_tokens_out,
                snapshot_hash, compaction_auto, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                part_id, message_id, session_id, part_type.value, ordinal,
                text_content, is_ignored, is_synthetic,
                tool_call_id, tool_name, tool_status, tool_input, tool_output,
                tool_error, tool_title, patch_hash, patch_files,
                file_mime, file_name, file_url,
                subtask_prompt, subtask_desc, subtask_agent,
                step_reason, step_cost, step_tokens_in, step_tokens_out,
                snapshot_hash, compaction_auto, metadata
            )
        )
        
        return part_id
    
    def create_message_parts_bulk(
        self,
        message_id: int,
        parts: List[CreateMessagePartInput]
    ) -> List[str]:
        """Create multiple message parts in a single transaction.
        
        Args:
            message_id: Parent message ID
            parts: List of CreateMessagePartInput
            
        Returns:
            List of part_ids
        """
        if not parts:
            return []
        
        part_ids = []
        
        for part in parts:
            part_id = f"{message_id}_{part.ordinal}"
            
            self.db.execute(
                """
                INSERT INTO message_parts (
                    part_id, message_id, session_id, part_type, ordinal,
                    text_content, tool_call_id, tool_name, tool_input, tool_output, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    part_id, message_id, part.session_id, part.part_type.value, part.ordinal,
                    part.text_content, part.tool_call_id, part.tool_name,
                    part.tool_input, part.tool_output, part.metadata
                )
            )
            
            part_ids.append(part_id)
        
        return part_ids
    
    def get_message_parts(self, message_id: int) -> List[MessagePartRecord]:
        """Get all parts for a message.
        
        Args:
            message_id: The message ID
            
        Returns:
            List of MessagePartRecord
        """
        rows = self.db.execute(
            """
            SELECT part_id, message_id, session_id, part_type, ordinal,
                   text_content, is_ignored, is_synthetic,
                   tool_call_id, tool_name, tool_status, tool_input, tool_output,
                   tool_error, tool_title, patch_hash, patch_files,
                   file_mime, file_name, file_url,
                   subtask_prompt, subtask_desc, subtask_agent,
                   step_reason, step_cost, step_tokens_in, step_tokens_out,
                   snapshot_hash, compaction_auto, metadata
            FROM message_parts
            WHERE message_id = ?
            ORDER BY ordinal ASC
            """,
            (message_id,)
        ).fetchall()
        
        return [self._row_to_message_part(row) for row in rows]
    
    def get_message_part_by_ordinal(
        self,
        message_id: int,
        ordinal: int
    ) -> Optional[MessagePartRecord]:
        """Get a message part by ordinal.
        
        Args:
            message_id: The message ID
            ordinal: Ordinal position
            
        Returns:
            MessagePartRecord or None if not found
        """
        row = self.db.execute(
            """
            SELECT part_id, message_id, session_id, part_type, ordinal,
                   text_content, is_ignored, is_synthetic,
                   tool_call_id, tool_name, tool_status, tool_input, tool_output,
                   tool_error, tool_title, patch_hash, patch_files,
                   file_mime, file_name, file_url,
                   subtask_prompt, subtask_desc, subtask_agent,
                   step_reason, step_cost, step_tokens_in, step_tokens_out,
                   snapshot_hash, compaction_auto, metadata
            FROM message_parts
            WHERE message_id = ? AND ordinal = ?
            """,
            (message_id, ordinal)
        ).fetchone()
        
        if not row:
            return None
        
        return self._row_to_message_part(row)
    
    def delete_message_parts(self, message_id: int) -> int:
        """Delete all parts for a message.
        
        Args:
            message_id: The message ID
            
        Returns:
            Number of parts deleted
        """
        cursor = self.db.execute(
            "DELETE FROM message_parts WHERE message_id = ?",
            (message_id,)
        )
        
        return cursor.rowcount
    
    # =========================================================================
    # Search Operations
    # =========================================================================
    
    def search_messages(
        self,
        query: str,
        mode: str = "full_text",
        conversation_id: Optional[int] = None,
        since: Optional[datetime] = None,
        before: Optional[datetime] = None,
        limit: int = 50
    ) -> List[MessageSearchResult]:
        """Search messages with FTS5 or LIKE fallback.
        
        Args:
            query: Search query
            mode: Search mode ("full_text" or "regex")
            conversation_id: Optional filter by conversation
            since: Optional filter by created_at >= since
            before: Optional filter by created_at < before
            limit: Maximum number of results
            
        Returns:
            List of MessageSearchResult
        """
        if mode == "regex":
            return self._search_messages_regex(query, conversation_id, since, before, limit)
        
        # Use LIKE fallback for CJK queries
        if contains_cjk(query) or not self._fts5_available:
            return self._search_messages_like(query, conversation_id, since, before, limit)
        
        return self._search_messages_fts5(query, conversation_id, since, before, limit)
    
    def _search_messages_fts5(
        self,
        query: str,
        conversation_id: Optional[int],
        since: Optional[datetime],
        before: Optional[datetime],
        limit: int
    ) -> List[MessageSearchResult]:
        """Search messages using FTS5 full-text search."""
        sanitized = sanitize_fts5_query(query)
        
        if not sanitized:
            return []
        
        conditions = ["m.message_id IN (SELECT rowid FROM messages_fts WHERE messages_fts MATCH ?)"]
        args = [sanitized]
        
        if conversation_id is not None:
            conditions.append("m.conversation_id = ?")
            args.append(conversation_id)
        
        if since is not None:
            conditions.append("m.created_at >= ?")
            args.append(since.isoformat())
        
        if before is not None:
            conditions.append("m.created_at < ?")
            args.append(before.isoformat())
        
        sql = f"""
            SELECT m.message_id, m.conversation_id, m.role, m.content, m.created_at
            FROM messages m
            WHERE {' AND '.join(conditions)}
            ORDER BY m.created_at DESC
            LIMIT ?
        """
        args.append(limit)
        
        rows = self.db.execute(sql, args).fetchall()
        
        results = []
        for row in rows:
            snippet = self._create_snippet(row[3], query)
            results.append(MessageSearchResult(
                message_id=row[0],
                conversation_id=row[1],
                role=MessageRole(row[2]),
                snippet=snippet,
                created_at=self._parse_datetime(row[4]),
                rank=None
            ))
        
        return results
    
    def _search_messages_like(
        self,
        query: str,
        conversation_id: Optional[int],
        since: Optional[datetime],
        before: Optional[datetime],
        limit: int
    ) -> List[MessageSearchResult]:
        """Search messages using LIKE for CJK support."""
        plan = build_like_search_plan("m.content", query)
        
        if not plan.where:
            return []
        
        conditions = [plan.where]
        args = list(plan.args)
        
        if conversation_id is not None:
            conditions.append("m.conversation_id = ?")
            args.append(conversation_id)
        
        if since is not None:
            conditions.append("m.created_at >= ?")
            args.append(since.isoformat())
        
        if before is not None:
            conditions.append("m.created_at < ?")
            args.append(before.isoformat())
        
        sql = f"""
            SELECT m.message_id, m.conversation_id, m.role, m.content, m.created_at
            FROM messages m
            WHERE {' AND '.join(conditions)}
            ORDER BY m.created_at DESC
            LIMIT ?
        """
        args.append(limit)
        
        rows = self.db.execute(sql, args).fetchall()
        
        results = []
        for row in rows:
            snippet = create_fallback_snippet(row[3], plan.terms)
            results.append(MessageSearchResult(
                message_id=row[0],
                conversation_id=row[1],
                role=MessageRole(row[2]),
                snippet=snippet,
                created_at=self._parse_datetime(row[4]),
                rank=None
            ))
        
        return results
    
    def _search_messages_regex(
        self,
        query: str,
        conversation_id: Optional[int],
        since: Optional[datetime],
        before: Optional[datetime],
        limit: int
    ) -> List[MessageSearchResult]:
        """Search messages using regex pattern matching."""
        conditions = []
        args = []
        
        if conversation_id is not None:
            conditions.append("conversation_id = ?")
            args.append(conversation_id)
        
        if since is not None:
            conditions.append("created_at >= ?")
            args.append(since.isoformat())
        
        if before is not None:
            conditions.append("created_at < ?")
            args.append(before.isoformat())
        
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        
        sql = f"""
            SELECT message_id, conversation_id, role, content, created_at
            FROM messages
            {where_clause}
            ORDER BY created_at DESC
        """
        
        rows = self.db.execute(sql, args).fetchall()
        
        results = []
        try:
            pattern = re.compile(query, re.IGNORECASE)
        except re.error:
            return []
        
        for row in rows:
            if pattern.search(row[3]):
                snippet = self._create_snippet(row[3], query)
                results.append(MessageSearchResult(
                    message_id=row[0],
                    conversation_id=row[1],
                    role=MessageRole(row[2]),
                    snippet=snippet,
                    created_at=self._parse_datetime(row[4]),
                    rank=None
                ))
                
                if len(results) >= limit:
                    break
        
        return results
    
    # =========================================================================
    # Context Items Operations
    # =========================================================================
    
    def add_context_message(
        self,
        conversation_id: int,
        message_id: int,
        ordinal: int
    ) -> None:
        """Add a message to the context items.
        
        Args:
            conversation_id: The conversation ID
            message_id: The message ID
            ordinal: Ordinal position
        """
        now = datetime.utcnow().isoformat()
        
        self.db.execute(
            """
            INSERT OR REPLACE INTO context_items
            (conversation_id, ordinal, item_type, message_id, summary_id, created_at)
            VALUES (?, ?, 'message', ?, NULL, ?)
            """,
            (conversation_id, ordinal, message_id, now)
        )
    
    def add_context_summary(
        self,
        conversation_id: int,
        summary_id: str,
        ordinal: int
    ) -> None:
        """Add a summary to the context items.
        
        Args:
            conversation_id: The conversation ID
            summary_id: The summary ID
            ordinal: Ordinal position
        """
        now = datetime.utcnow().isoformat()
        
        self.db.execute(
            """
            INSERT OR REPLACE INTO context_items
            (conversation_id, ordinal, item_type, message_id, summary_id, created_at)
            VALUES (?, ?, 'summary', NULL, ?, ?)
            """,
            (conversation_id, ordinal, summary_id, now)
        )
    
    def get_context_items(
        self,
        conversation_id: int
    ) -> List[Dict[str, Any]]:
        """Get all context items for a conversation.
        
        Args:
            conversation_id: The conversation ID
            
        Returns:
            List of context item dicts
        """
        rows = self.db.execute(
            """
            SELECT conversation_id, ordinal, item_type, message_id, summary_id, created_at
            FROM context_items
            WHERE conversation_id = ?
            ORDER BY ordinal ASC
            """,
            (conversation_id,)
        ).fetchall()
        
        results = []
        for row in rows:
            results.append({
                "conversation_id": row[0],
                "ordinal": row[1],
                "item_type": row[2],
                "message_id": row[3],
                "summary_id": row[4],
                "created_at": self._parse_datetime(row[5])
            })
        
        return results
    
    def clear_context_items(self, conversation_id: int) -> int:
        """Clear all context items for a conversation.
        
        Args:
            conversation_id: The conversation ID
            
        Returns:
            Number of items cleared
        """
        cursor = self.db.execute(
            "DELETE FROM context_items WHERE conversation_id = ?",
            (conversation_id,)
        )
        
        return cursor.rowcount
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _row_to_conversation(self, row) -> ConversationRecord:
        """Convert a database row to ConversationRecord."""
        return ConversationRecord(
            conversation_id=row[0],
            session_id=row[1],
            session_key=row[2],
            title=row[3],
            bootstrapped_at=self._parse_datetime(row[4]),
            created_at=self._parse_datetime(row[5]),
            updated_at=self._parse_datetime(row[6])
        )
    
    def _row_to_message(self, row) -> MessageRecord:
        """Convert a database row to MessageRecord."""
        return MessageRecord(
            message_id=row[0],
            conversation_id=row[1],
            seq=row[2],
            role=MessageRole(row[3]),
            content=row[4],
            token_count=row[5],
            created_at=self._parse_datetime(row[6])
        )
    
    def _row_to_message_part(self, row) -> MessagePartRecord:
        """Convert a database row to MessagePartRecord."""
        return MessagePartRecord(
            part_id=row[0],
            message_id=row[1],
            session_id=row[2],
            part_type=MessagePartType(row[3]),
            ordinal=row[4],
            text_content=row[5],
            is_ignored=bool(row[6]) if row[6] is not None else None,
            is_synthetic=bool(row[7]) if row[7] is not None else None,
            tool_call_id=row[8],
            tool_name=row[9],
            tool_status=row[10],
            tool_input=row[11],
            tool_output=row[12],
            tool_error=row[13],
            tool_title=row[14],
            patch_hash=row[15],
            patch_files=row[16],
            file_mime=row[17],
            file_name=row[18],
            file_url=row[19],
            subtask_prompt=row[20],
            subtask_desc=row[21],
            subtask_agent=row[22],
            step_reason=row[23],
            step_cost=row[24],
            step_tokens_in=row[25],
            step_tokens_out=row[26],
            snapshot_hash=row[27],
            compaction_auto=bool(row[28]) if row[28] is not None else None,
            metadata=row[29]
        )
    
    def _parse_datetime(self, value: Optional[str]) -> Optional[datetime]:
        """Parse a datetime string."""
        if not value:
            return None
        
        try:
            # Try ISO format
            if "T" in value:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            else:
                # Try SQLite format
                return datetime.fromisoformat(value)
        except (ValueError, TypeError):
            return None
    
    def _create_snippet(
        self,
        content: str,
        query: str,
        max_len: int = 200,
        context_chars: int = 50
    ) -> str:
        """Create a snippet with query highlighted.
        
        Args:
            content: Full content
            query: Search query
            max_len: Maximum snippet length
            context_chars: Characters around match
            
        Returns:
            Snippet with **highlighted** query
        """
        # Find query position (case-insensitive)
        content_lower = content.lower()
        query_lower = query.lower()
        
        pos = content_lower.find(query_lower)
        if pos == -1:
            # Query not found, return start of content
            return content[:max_len] + "..." if len(content) > max_len else content
        
        # Calculate snippet boundaries
        start = max(0, pos - context_chars)
        end = min(len(content), pos + max_len - context_chars)
        
        # Adjust to not cut in the middle of a word
        if start > 0:
            space_pos = content.find(' ', start)
            if space_pos != -1 and space_pos < start + 20:
                start = space_pos + 1
        
        snippet = content[start:end]
        
        # Add ellipsis
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(content) else ""
        
        # Highlight query in snippet
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        highlighted = pattern.sub(r"**\g<0>**", snippet)
        
        return f"{prefix}{highlighted}{suffix}"
