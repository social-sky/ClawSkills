#!/usr/bin/env python3
"""Summary Store for LCM.

Handles CRUD operations for summaries and their DAG relationships.
Port of TypeScript store/summaryStore.ts from lossless-claw-enhanced.
"""

import json
import sqlite3
from datetime import datetime
from typing import Optional, List, Dict, Any

from ..lcm_types import (
    SummaryRecord,
    SummaryKind,
    ContextItemRecord,
    ContextItemType,
    MemoryTier,
    MemoryCategory,
)
from ..estimate_tokens import estimate_tokens


class SummaryStore:
    """Data access layer for summaries.
    
    Provides CRUD operations and DAG management for summaries
    including parent-child relationships and message linkages.
    """
    
    def __init__(self, db: sqlite3.Connection):
        """Initialize the store with a database connection.
        
        Args:
            db: SQLite connection with proper configuration
        """
        self.db = db
    
    # =========================================================================
    # CRUD Operations
    # =========================================================================
    
    def create_summary(
        self,
        summary_id: str,
        conversation_id: int,
        kind: SummaryKind,
        depth: int,
        content: str,
        token_count: int,
        file_ids: Optional[List[str]] = None,
        earliest_at: Optional[datetime] = None,
        latest_at: Optional[datetime] = None,
        descendant_count: int = 0,
        descendant_token_count: int = 0,
        source_message_token_count: int = 0,
        model: str = "unknown",
        category: Optional[MemoryCategory] = None,
        tier: MemoryTier = MemoryTier.PERIPHERAL,
        importance: float = 0.5,
        scope: str = "global"
    ) -> SummaryRecord:
        """Create a new summary.
        
        Args:
            summary_id: Unique summary identifier
            conversation_id: Parent conversation ID
            kind: Summary kind (leaf or condensed)
            depth: DAG depth level
            content: Summary content text
            token_count: Token count of content
            file_ids: Optional list of file IDs
            earliest_at: Earliest message timestamp
            latest_at: Latest message timestamp
            descendant_count: Number of descendants
            descendant_token_count: Token count of descendants
            source_message_token_count: Token count of source messages
            model: Model used for summarization
            category: Memory category classification
            tier: Memory tier (peripheral/working/core)
            importance: Importance score (0.0 to 1.0)
            scope: Scope isolation string
            
        Returns:
            The created SummaryRecord
        """
        now = datetime.utcnow().isoformat()
        
        # Serialize file_ids as JSON
        file_ids_json = json.dumps(file_ids or [])
        
        self.db.execute(
            """
            INSERT INTO summaries (
                summary_id, conversation_id, kind, depth, content, token_count,
                file_ids, earliest_at, latest_at, descendant_count,
                descendant_token_count, source_message_token_count,
                model, created_at, category, tier, importance, scope
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                summary_id, conversation_id, kind.value, depth, content, token_count,
                file_ids_json,
                earliest_at.isoformat() if earliest_at else None,
                latest_at.isoformat() if latest_at else None,
                descendant_count, descendant_token_count, source_message_token_count,
                model, now,
                category.value if category else None,
                tier.value, importance, scope
            )
        )
        
        return self.get_summary_by_id(summary_id)
    
    def get_summary_by_id(self, summary_id: str) -> Optional[SummaryRecord]:
        """Get a summary by ID.
        
        Args:
            summary_id: The summary ID
            
        Returns:
            SummaryRecord or None if not found
        """
        row = self.db.execute(
            """
            SELECT summary_id, conversation_id, kind, depth, content, token_count,
                   file_ids, earliest_at, latest_at, descendant_count,
                   descendant_token_count, source_message_token_count,
                   model, created_at, category, tier, importance, scope,
                   access_count, last_accessed_at, decay_score
            FROM summaries
            WHERE summary_id = ?
            """,
            (summary_id,)
        ).fetchone()
        
        if not row:
            return None
        
        return self._row_to_summary(row)
    
    def get_summaries_by_conversation(
        self,
        conversation_id: int,
        limit: int = 100,
        offset: int = 0
    ) -> List[SummaryRecord]:
        """Get all summaries for a conversation.
        
        Args:
            conversation_id: The conversation ID
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of SummaryRecord ordered by created_at
        """
        rows = self.db.execute(
            """
            SELECT summary_id, conversation_id, kind, depth, content, token_count,
                   file_ids, earliest_at, latest_at, descendant_count,
                   descendant_token_count, source_message_token_count,
                   model, created_at, category, tier, importance, scope,
                   access_count, last_accessed_at, decay_score
            FROM summaries
            WHERE conversation_id = ?
            ORDER BY created_at ASC
            LIMIT ? OFFSET ?
            """,
            (conversation_id, limit, offset)
        ).fetchall()
        
        return [self._row_to_summary(row) for row in rows]
    
    def update_summary(
        self,
        summary_id: str,
        content: Optional[str] = None,
        token_count: Optional[int] = None,
        tier: Optional[MemoryTier] = None,
        importance: Optional[float] = None,
        decay_score: Optional[float] = None,
        access_count: Optional[int] = None,
        last_accessed_at: Optional[datetime] = None
    ) -> bool:
        """Update a summary.
        
        Args:
            summary_id: The summary ID
            content: Optional new content
            token_count: Optional new token count
            tier: Optional new memory tier
            importance: Optional new importance score
            decay_score: Optional new decay score
            access_count: Optional new access count
            last_accessed_at: Optional new last accessed time
            
        Returns:
            True if updated, False if not found
        """
        updates = []
        args = []
        
        if content is not None:
            updates.append("content = ?")
            args.append(content)
        
        if token_count is not None:
            updates.append("token_count = ?")
            args.append(token_count)
        
        if tier is not None:
            updates.append("tier = ?")
            args.append(tier.value)
        
        if importance is not None:
            updates.append("importance = ?")
            args.append(importance)
        
        if decay_score is not None:
            updates.append("decay_score = ?")
            args.append(decay_score)
        
        if access_count is not None:
            updates.append("access_count = ?")
            args.append(access_count)
        
        if last_accessed_at is not None:
            updates.append("last_accessed_at = ?")
            args.append(last_accessed_at.isoformat())
        
        if not updates:
            return False
        
        args.append(summary_id)
        
        cursor = self.db.execute(
            f"UPDATE summaries SET {', '.join(updates)} WHERE summary_id = ?",
            args
        )
        
        return cursor.rowcount > 0
    
    def delete_summary(self, summary_id: str) -> bool:
        """Delete a summary.
        
        Args:
            summary_id: The summary ID
            
        Returns:
            True if deleted, False if not found
        """
        cursor = self.db.execute(
            "DELETE FROM summaries WHERE summary_id = ?",
            (summary_id,)
        )
        
        return cursor.rowcount > 0
    
    def increment_access_count(self, summary_id: str) -> None:
        """Increment the access count and update last accessed time.
        
        Args:
            summary_id: The summary ID
        """
        now = datetime.utcnow().isoformat()
        
        self.db.execute(
            """
            UPDATE summaries 
            SET access_count = access_count + 1, last_accessed_at = ?
            WHERE summary_id = ?
            """,
            (now, summary_id)
        )
    
    # =========================================================================
    # DAG Operations
    # =========================================================================
    
    def add_summary_parent(
        self,
        summary_id: str,
        parent_summary_id: str,
        ordinal: int = 0
    ) -> None:
        """Add a parent relationship to a summary.
        
        Args:
            summary_id: The child summary ID
            parent_summary_id: The parent summary ID
            ordinal: Ordinal position among siblings
        """
        self.db.execute(
            """
            INSERT OR REPLACE INTO summary_parents (summary_id, parent_summary_id, ordinal)
            VALUES (?, ?, ?)
            """,
            (summary_id, parent_summary_id, ordinal)
        )
    
    def get_parent_summaries(self, summary_id: str) -> List[SummaryRecord]:
        """Get parent summaries of a summary.
        
        Args:
            summary_id: The child summary ID
            
        Returns:
            List of parent SummaryRecord
        """
        rows = self.db.execute(
            """
            SELECT s.summary_id, s.conversation_id, s.kind, s.depth, s.content, s.token_count,
                   s.file_ids, s.earliest_at, s.latest_at, s.descendant_count,
                   s.descendant_token_count, s.source_message_token_count,
                   s.model, s.created_at, s.category, s.tier, s.importance, s.scope,
                   s.access_count, s.last_accessed_at, s.decay_score
            FROM summaries s
            INNER JOIN summary_parents sp ON s.summary_id = sp.parent_summary_id
            WHERE sp.summary_id = ?
            ORDER BY sp.ordinal ASC
            """,
            (summary_id,)
        ).fetchall()
        
        return [self._row_to_summary(row) for row in rows]
    
    def get_child_summaries(self, parent_summary_id: str) -> List[SummaryRecord]:
        """Get child summaries of a summary.
        
        Args:
            parent_summary_id: The parent summary ID
            
        Returns:
            List of child SummaryRecord
        """
        rows = self.db.execute(
            """
            SELECT s.summary_id, s.conversation_id, s.kind, s.depth, s.content, s.token_count,
                   s.file_ids, s.earliest_at, s.latest_at, s.descendant_count,
                   s.descendant_token_count, s.source_message_token_count,
                   s.model, s.created_at, s.category, s.tier, s.importance, s.scope,
                   s.access_count, s.last_accessed_at, s.decay_score
            FROM summaries s
            INNER JOIN summary_parents sp ON s.summary_id = sp.summary_id
            WHERE sp.parent_summary_id = ?
            ORDER BY sp.ordinal ASC
            """,
            (parent_summary_id,)
        ).fetchall()
        
        return [self._row_to_summary(row) for row in rows]
    
    def link_summary_to_messages(
        self,
        summary_id: str,
        message_ids: List[int]
    ) -> None:
        """Link a summary to its source messages.
        
        Args:
            summary_id: The summary ID
            message_ids: List of message IDs
        """
        for ordinal, msg_id in enumerate(message_ids):
            self.db.execute(
                """
                INSERT OR IGNORE INTO summary_messages (summary_id, message_id, ordinal)
                VALUES (?, ?, ?)
                """,
                (summary_id, msg_id, ordinal)
            )
    
    def get_messages_for_summary(
        self,
        summary_id: str
    ) -> List[Dict[str, Any]]:
        """Get source messages linked to a summary.
        
        Args:
            summary_id: The summary ID
            
        Returns:
            List of message info dicts with message_id, ordinal
        """
        rows = self.db.execute(
            """
            SELECT sm.message_id, sm.ordinal
            FROM summary_messages sm
            WHERE sm.summary_id = ?
            ORDER BY sm.ordinal ASC
            """,
            (summary_id,)
        ).fetchall()
        
        return [{"message_id": row[0], "ordinal": row[1]} for row in rows]
    
    # =========================================================================
    # Context Operations
    # =========================================================================
    
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
    
    def get_context_summaries(
        self,
        conversation_id: int
    ) -> List[SummaryRecord]:
        """Get all summary context items for a conversation.
        
        Args:
            conversation_id: The conversation ID
            
        Returns:
            List of SummaryRecord in ordinal order
        """
        rows = self.db.execute(
            """
            SELECT s.summary_id, s.conversation_id, s.kind, s.depth, s.content, s.token_count,
                   s.file_ids, s.earliest_at, s.latest_at, s.descendant_count,
                   s.descendant_token_count, s.source_message_token_count,
                   s.model, s.created_at, s.category, s.tier, s.importance, s.scope,
                   s.access_count, s.last_accessed_at, s.decay_score
            FROM summaries s
            INNER JOIN context_items ci ON s.summary_id = ci.summary_id
            WHERE ci.conversation_id = ? AND ci.item_type = 'summary'
            ORDER BY ci.ordinal ASC
            """,
            (conversation_id,)
        ).fetchall()
        
        return [self._row_to_summary(row) for row in rows]
    
    def clear_context_summaries(self, conversation_id: int) -> int:
        """Clear all summary context items for a conversation.
        
        Args:
            conversation_id: The conversation ID
            
        Returns:
            Number of items cleared
        """
        cursor = self.db.execute(
            "DELETE FROM context_items WHERE conversation_id = ? AND item_type = 'summary'",
            (conversation_id,)
        )
        
        return cursor.rowcount
    
    # =========================================================================
    # Stats Operations
    # =========================================================================
    
    def get_summary_stats(self, conversation_id: int) -> Dict[str, Any]:
        """Get summary statistics for a conversation.
        
        Args:
            conversation_id: The conversation ID
            
        Returns:
            Dict with summary statistics
        """
        # Total summaries
        total_row = self.db.execute(
            "SELECT COUNT(*) FROM summaries WHERE conversation_id = ?",
            (conversation_id,)
        ).fetchone()
        total = total_row[0] if total_row else 0
        
        # By kind
        kind_rows = self.db.execute(
            """
            SELECT kind, COUNT(*) as count 
            FROM summaries 
            WHERE conversation_id = ?
            GROUP BY kind
            """,
            (conversation_id,)
        ).fetchall()
        by_kind = {row[0]: row[1] for row in kind_rows}
        
        # By tier
        tier_rows = self.db.execute(
            """
            SELECT tier, COUNT(*) as count 
            FROM summaries 
            WHERE conversation_id = ?
            GROUP BY tier
            """,
            (conversation_id,)
        ).fetchall()
        by_tier = {row[0]: row[1] for row in tier_rows}
        
        # Total tokens
        token_row = self.db.execute(
            "SELECT COALESCE(SUM(token_count), 0) FROM summaries WHERE conversation_id = ?",
            (conversation_id,)
        ).fetchone()
        total_tokens = token_row[0] if token_row else 0
        
        # Max depth
        depth_row = self.db.execute(
            "SELECT COALESCE(MAX(depth), 0) FROM summaries WHERE conversation_id = ?",
            (conversation_id,)
        ).fetchone()
        max_depth = depth_row[0] if depth_row else 0
        
        return {
            "total_summaries": total,
            "by_kind": by_kind,
            "by_tier": by_tier,
            "total_tokens": total_tokens,
            "max_depth": max_depth
        }
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _row_to_summary(self, row: tuple) -> SummaryRecord:
        """Convert a database row to SummaryRecord.
        
        Args:
            row: Database row tuple
            
        Returns:
            SummaryRecord instance
        """
        # Parse file_ids JSON
        file_ids = []
        if len(row) > 6 and row[6]:
            try:
                file_ids = json.loads(row[6]) if isinstance(row[6], str) else row[6]
            except (json.JSONDecodeError, TypeError):
                file_ids = []
        
        # Parse enums
        kind = SummaryKind(row[2]) if isinstance(row[2], str) else row[2]
        
        # Parse memory tier
        tier = MemoryTier.PERIPHERAL
        if len(row) > 15 and row[15]:
            try:
                tier = MemoryTier(row[15])
            except (ValueError, TypeError):
                tier = MemoryTier.PERIPHERAL
        
        # Parse memory category
        category = None
        if len(row) > 14 and row[14]:
            try:
                category = MemoryCategory(row[14])
            except (ValueError, TypeError):
                category = None
        
        # Parse importance
        importance = 0.5
        if len(row) > 16 and row[16] is not None:
            importance = float(row[16])
        
        # Parse scope
        scope = "global"
        if len(row) > 17 and row[17]:
            scope = str(row[17])
        
        # Parse access count
        access_count = 0
        if len(row) > 18 and row[18] is not None:
            access_count = int(row[18])
        
        # Parse last accessed
        last_accessed_at = None
        if len(row) > 19 and row[19]:
            try:
                last_accessed_at = datetime.fromisoformat(row[19])
            except (ValueError, TypeError):
                last_accessed_at = None
        
        # Parse decay score
        decay_score = 1.0
        if len(row) > 20 and row[20] is not None:
            decay_score = float(row[20])
        
        # Parse created_at
        created_at = datetime.utcnow()
        if len(row) > 13 and row[13]:
            try:
                created_at = datetime.fromisoformat(row[13])
            except (ValueError, TypeError):
                created_at = datetime.utcnow()
        
        return SummaryRecord(
            summary_id=row[0],
            conversation_id=row[1],
            kind=kind,
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
            created_at=created_at,
            category=category,
            tier=tier,
            importance=importance,
            scope=scope,
            access_count=access_count,
            last_accessed_at=last_accessed_at,
            decay_score=decay_score
        )
