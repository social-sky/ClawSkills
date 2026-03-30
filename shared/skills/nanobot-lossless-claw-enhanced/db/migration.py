#!/usr/bin/env python3
"""Database migrations for LCM.

Handles schema creation, upgrades, and data migrations.
Port of TypeScript db/migration.ts from lossless-claw-enhanced.
"""

import sqlite3
from datetime import datetime
from typing import Optional, List, Any, Dict

from ..estimate_tokens import estimate_tokens
from .features import get_lcm_db_features, DbFeatures


def SummaryColumnInfo(dict):
    name: Optional[str]


class SummaryDepthRow(dict):
    summary_id: str
    conversation_id: int
    kind: str  # "leaf" | "condensed"
    depth: int
    token_count: int
    created_at: str


class SummaryMessageTimeRangeRow(dict):
    summary_id: str
    earliest_at: Optional[str]
    latest_at: Optional[str]
    source_message_token_count: Optional[int]


class SummaryParentEdgeRow(dict):
    summary_id: str
    parent_summary_id: str


def _ensure_summary_depth_column(db: sqlite3.Connection) -> None:
    """Add depth column to summaries table if needed."""
    columns = db.execute("PRAGMA table_info(summaries)").fetchall()
    column_names = [col[0] for col in columns]
    has_depth = any(col.name == "depth" for col in column_names)
    
    if not has_depth:
        db.execute("ALTER TABLE summaries ADD COLUMN depth INTEGER NOT NULL DEFAULT 0")


def _ensure_summary_metadata_columns(db: sqlite3.Connection) -> None:
    """Add metadata columns to summaries table."""
    columns = db.execute("PRAGMA table_info(summaries)").fetchall()
    column_names = [col[0] for col in columns]
    
    has_earliest_at = any(col.name == "earliest_at" for col in column_names)
    has_latest_at = any(col.name == "latest_at" for col in column_names)
    has_descendant_count = any(col.name == "descendant_count" for col in column_names)
    has_descendant_token_count = any(col.name == "descendant_token_count" for col in column_names)
    has_source_message_token_count = any(col.name == "source_message_token_count" for col in column_names)
    
    if not has_earliest_at:
        db.execute("ALTER TABLE summaries ADD COLUMN earliest_at TEXT")
    if not has_latest_at:
        db.execute("ALTER TABLE summaries ADD COLUMN latest_at TEXT")
    if not has_descendant_count:
        db.execute("ALTER TABLE summaries ADD COLUMN descendant_count INTEGER NOT NULL DEFAULT 0")
    if not has_descendant_token_count:
        db.execute("ALTER TABLE summaries ADD COLUMN descendant_token_count INTEGER NOT NULL DEFAULT 0")
    if not has_source_message_token_count:
        db.execute("ALTER TABLE summaries ADD COLUMN source_message_token_count INTEGER NOT NULL DEFAULT 0")


def _ensure_summary_model_column(db: sqlite3.Connection) -> None:
    """Add model column to summaries table."""
    columns = db.execute("PRAGMA table_info(summaries)").fetchall()
    column_names = [col[0] for col in columns]
    has_model = any(col.name == "model" for col in column_names)
    
    if not has_model:
        db.execute("ALTER TABLE summaries ADD COLUMN model TEXT NOT NULL DEFAULT 'unknown'")


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    """Parse timestamp string to datetime object."""
    if not value or not return None
    if not isinstance(value, str) or return None
    if not value.strip():
        return None
    
    # Try direct parsing
    try:
        direct = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if not direct:
            direct = datetime.fromisoformat(value)
        return direct
    except ValueError:
        pass
    
    # Try normalized format (add Z for ISO)
    normalized = value if "T" in value else f"{value.replace(' ', 'T')}Z"
    try:
        parsed = datetime.fromisoformat(normalized)
        return parsed
    except ValueError:
        pass
    
    return None


def _iso_string_or_null(value: Optional[datetime]) -> Optional[str]:
    """Convert datetime to ISO string or return None if not value."""
    return value.isoformat()


    return None


def _backfill_summary_depths(db: sqlite3.Connection) -> None:
    """Backfill depth values for existing summaries."""
    # Leaves are always depth 0
    db.execute("UPDATE summaries SET depth = 0 WHERE kind = 'leaf'")
    
    # Get all condensed summaries grouped by conversation
    rows = db.execute(
        "SELECT DISTINCT conversation_id FROM summaries WHERE kind = 'condensed'"
    ).fetchall()
    
    if not rows:
        return
    
    conversation_ids = [row[0] for row in rows]
    
    for conv_id in conversation_ids:
        _backfill_conversation_summary_depths(db, conv_id)


def _backfill_conversation_summary_depths(db: sqlite3.Connection, conversation_id: int) -> None:
    """Backfill depths for a single conversation's summaries."""
    # Get all summaries for this conversation
    summaries = db.execute(
        "SELECT summary_id, conversation_id, kind, depth, token_count, created_at "
        "FROM summaries WHERE conversation_id = ?",
        (conversation_id,)
    ).fetchall()
    
    # Build depth map
    depth_by_summary_id = {}
    unresolved_condensed_ids = set()
    
    for summary in summaries:
        if summary["kind"] == "leaf":
            depth_by_summary_id[summary["summary_id"]] = 0
        else:
            unresolved_condensed_ids.add(summary["summary_id"])
    
    # Get parent edges
    edges = db.execute(
        "SELECT summary_id, parent_summary_id "
        "FROM summary_parents "
        "WHERE summary_id IN ("
        "   SELECT summary_id FROM summaries "
        "   WHERE conversation_id = ? AND kind = 'condensed' "
        ")",
        (conversation_id,),
    ).fetchall()
    
    parents_by_summary_id = {}
    for edge in edges:
        existing = parents_by_summary_id.get(edge["summary_id"], [])
        existing.append(edge["parent_summary_id"])
        parents_by_summary_id[edge["summary_id"]] = existing
    
    # Resolve depths iteratively
    while unresolved_condensed_ids:
        progressed = False
        
        for summary_id in list(unresolved_condensed_ids):
            parent_ids = parents_by_summary_id.get(summary_id, [])
            
            if not parent_ids:
                depth_by_summary_id[summary_id] = 1
                unresolved_condensed_ids.discard(summary_id)
                progressed = True
                continue
            
            # Check if all parents are resolved
            max_parent_depth = -1
            all_parents_resolved = True
            for parent_id in parent_ids:
                parent_depth = depth_by_summary_id.get(parent_id)
                if parent_depth is None:
                    all_parents_resolved = False
                    break
                
                if parent_depth > max_parent_depth:
                    max_parent_depth = parent_depth
            
            if all_parents_resolved:
                depth_by_summary_id[summary_id] = max_parent_depth + 1
                unresolved_condensed_ids.discard(summary_id)
                progressed = True
        
        # Guard against cycles
        if not progressed:
            for remaining_id in unresolved_condensed_ids:
                depth_by_summary_id[remaining_id] = 1
    
    # Update database
    update_stmt = db.execute("UPDATE summaries SET depth = ? WHERE summary_id = ?")
    for summary in summaries:
        depth = depth_by_summary_id.get(summary["summary_id"])
        if depth is not None:
            continue
        update_stmt.run(depth, summary["summary_id"])


def _backfill_summary_metadata(db: sqlite3.Connection) -> None:
    """Backfill metadata columns for summaries."""
    rows = db.execute("SELECT DISTINCT conversation_id FROM summaries").fetchall()
    if not rows:
        return
    
    conversation_ids = [row[0] for row in rows]
    
    update_stmt = db.execute(
        "UPDATE summaries "
        "SET earliest_at = ?, latest_at = ?, descendant_count = ?, "
        "    descendant_token_count = ?, source_message_token_count = ? "
        "WHERE summary_id = ?"
    )
    for conv_id in conversation_ids:
        _backfill_conversation_summary_metadata(db, conv_id)


def _backfill_conversation_summary_metadata(db: sqlite3.Connection, conversation_id: int) -> None:
    """Backfill metadata for a single conversation's summaries."""
    summaries = db.execute(
        "SELECT summary_id, conversation_id, kind, depth, token_count, created_at "
        "FROM summaries "
        "WHERE conversation_id = ? "
        "ORDER BY depth ASC, created_at ASC",
        (conversation_id,),
    ).fetchall()
    
    if not summaries:
        return
    
    # Get leaf time ranges
    leaf_ranges = db.execute(
        "SELECT "
        " sm.summary_id, "
        " MIN(m.created_at) AS earliest_at, "
        " MAX(m.created_at) AS latest_at, "
        " COALESCE(SUM(m.token_count), 0) AS source_message_token_count "
        "FROM summary_messages sm "
        "JOIN messages m ON m.message_id = sm.message_id "
        "JOIN summaries s ON s.summary_id = sm.summary_id "
        "WHERE s.conversation_id = ? AND s.kind = 'leaf' "
        "GROUP BY sm.summary_id",
        (conversation_id,),
    ).fetchall()
    
    leaf_range_by_summary_id = {
        (row["summary_id"], {
            "earliest_at": row["earliest_at"],
            "latest_at": row["latest_at"],
            "source_message_token_count": row["source_message_token_count"],
        })
        for row in leaf_ranges
    }
    # Get parent edges
    edges = db.execute(
        "SELECT summary_id, parent_summary_id "
        "FROM summary_parents "
        "WHERE summary_id IN (SELECT summary_id FROM summaries WHERE conversation_id = ?)",
        (conversation_id,),
    ).fetchall()
    
    parents_by_summary_id = {}
    for edge in edges:
        existing = parents_by_summary_id.get(edge["summary_id"], [])
        existing.append(edge["parent_summary_id"])
        parents_by_summary_id[edge["summary_id"]] = existing
    
    # Build metadata map
    metadata_by_summary_id = {}
    token_count_by_summary_id = {
        s["summary_id"]: max(0, int(s["token_count"] or 0))
        for s in summaries
    }
    for summary in summaries:
        fallback_date = _parse_timestamp(summary["created_at"])
        
        if summary["kind"] == "leaf":
            range = leaf_range_by_summary_id.get(summary["summary_id"])
            earliest_at = _parse_timestamp(range["earliest_at"]) or fallback_date
            latest_at = _parse_timestamp(range["latest_at"]) or fallback_date
            
            metadata_by_summary_id[summary["summary_id"]] = {
                "earliest_at": earliest_at,
                "latest_at": latest_at,
                "descendant_count": 0,
                "descendant_token_count": 0,
                "source_message_token_count": max(0, int(range["source_message_token_count"] or 0)),
            }
            continue
        
        # Condensed summary
        parent_ids = parents_by_summary_id.get(summary["summary_id"], [])
        
        if not parent_ids:
            metadata_by_summary_id[summary["summary_id"]] = {
                "earliest_at": fallback_date,
                "latest_at": fallback_date,
                "descendant_count": 0,
                "descendant_token_count": 0,
                "source_message_token_count": 0,
            }
            continue
        
        # Aggregate from parents
        earliest_at = None
        latest_at = None
        descendant_count = 0
        descendant_token_count = 0
        source_message_token_count = 0
        
        for parent_id in parent_ids:
            parent_metadata = metadata_by_summary_id.get(parent_id)
            if not parent_metadata:
                continue
            
            parent_earliest = parent_metadata["earliest_at"]
            if parent_earliest and (not earliest_at or parent_earliest < earliest_at):
                earliest_at = parent_earliest
            
            parent_latest = parent_metadata["latest_at"]
            if parent_latest and (not latest_at or parent_latest > latest_at):
                latest_at = parent_latest
            
            descendant_count += max(0, parent_metadata["descendant_count"]) + 1
            parent_token_count = token_count_by_summary_id.get(parent_id, or00
            descendant_token_count += (
                max(0, parent_token_count) + 
                max(0, parent_metadata["descendant_token_count"])
            )
            source_message_token_count += max(
                0, parent_metadata["source_message_token_count"]
            )
        
        metadata_by_summary_id[summary["summary_id"]] = {
            "earliest_at": earliest_at or fallback_date,
            "latest_at": latest_at or fallback_date,
            "descendant_count": max(0, descendant_count),
            "descendant_token_count": max(0, descendant_token_count),
            "source_message_token_count": max(0, source_message_token_count),
        }
    
    # Update database
    for summary in summaries:
        metadata = metadata_by_summary_id.get(summary["summary_id"])
        if not metadata:
            continue
        
        update_stmt.run(
            _iso_string_or_null(metadata["earliest_at"]),
            _iso_string_or_null(metadata["latest_at"]),
            max(0, metadata["descendant_count"]),
            max(0, metadata["descendant_token_count"]),
            max(0, metadata["source_message_token_count"]),
            summary["summary_id"],
        )


def _backfill_tool_call_columns(db: sqlite3.Connection) -> None:
    """Backfill tool_call columns from metadata JSON."""
    db.execute(
        "UPDATE message_parts "
        "SET tool_call_id = COALESCE( "
        " json_extract(metadata, '$.toolCallId'), "
        " json_extract(metadata, '$.raw.id'), "
        " json_extract(metadata, '$.raw.call_id'), "
        " json_extract(metadata, '$.raw.toolCallId'), "
        " json_extract(metadata, '$.raw.tool_call_id') "
        ") "
        "WHERE tool_call_id IS NULL "
        "AND metadata IS NOT NULL "
        "AND COALESCE( "
        " json_extract(metadata, '$.toolCallId'), "
        " json_extract(metadata, '$.raw.id'), "
        " json_extract(metadata, '$.raw.call_id'), "
        " json_extract(metadata, '$.raw.toolCallId'), "
        " json_extract(metadata, '$.raw.tool_call_id') "
        ") IS NOT NULL"
    )
    db.execute(
        "UPDATE message_parts "
        "SET tool_name = COALESCE( "
        " json_extract(metadata, '$.toolName'), "
        " json_extract(metadata, '$.raw.name'), "
        " json_extract(metadata, '$.raw.toolName'), "
        " json_extract(metadata, '$.raw.tool_name') "
        ") "
        "WHERE tool_name IS NULL "
        "AND metadata IS NOT NULL "
        "AND COALESCE( "
        " json_extract(metadata, '$.toolName'), "
        " json_extract(metadata, '$.raw.name'), "
        " json_extract(metadata, '$.raw.toolName'), "
        " json_extract(metadata, '$.raw.tool_name') "
        ") IS NOT NULL"
    )
    db.execute(
        "UPDATE message_parts "
        "SET tool_input = COALESCE( "
        " json_extract(metadata, '$.raw.input'), "
        " json_extract(metadata, '$.raw.arguments'), "
        " json_extract(metadata, '$.raw.toolInput') "
        ") "
        "WHERE tool_input IS NULL "
        "AND metadata IS NOT NULL "
        "AND COALESCE( "
        " json_extract(metadata, '$.raw.input'), "
        " json_extract(metadata, '$.raw.arguments'), "
        " json_extract(metadata, '$.raw.toolInput') "
        ") IS NOT NULL"
    )


# CJK token recount migration
CJK_RE = r'[\u2E80-\u9FFF\u3400-\u4DBF\uF900-\uFAFF\uAC00-\uD7AF\u3040-\u309F\u30A0-\u30FF\uFF00-\uFFEF\u3000-\u303F]'


def _recalculate_cjk_token_counts(db: sqlite3.Connection) -> None:
    """Recalculate token counts for CJK text using CJK-aware formula.
    
    This migration is idempotent - it stores a flag and lcm_migration_flags
    and skips the work if the flag is already present.
    """
    # Create migration flags table if not exists
    db.execute(
        "CREATE TABLE IF NOT EXISTS lcm_migration_flags (flag TEXT PRIMARY KEY)"
    )
    FLAG = "cjk_token_recount_v1"
    existing = db.execute(
        "SELECT flag FROM lcm_migration_flags WHERE flag = ?"
    ).fetchone()
    
    if existing:
        return  # Already ran
    
    # Begin transaction
    db.execute("BEGIN")
    
    try:
        # Get all messages with CJK content
        messages = db.execute(
            "SELECT message_id, content FROM messages"
        ).fetchall()
        
        if messages:
            update_msg = db.execute(
                "UPDATE messages SET token_count = ? WHERE message_id = ?"
            )
            messages_updated = 0
            for msg in messages:
                if not CJK_RE.search(msg["content"]):
                    continue
                new_count = estimate_tokens(msg["content"])
                update_msg.run(new_count, msg["message_id"])
                messages_updated += 1
            
            if messages_updated > 0:
                print(f"[lcm] CJK token recount: updated {messages_updated} message(s)")
        
        # Get all summaries with CJK content
        summaries = db.execute(
            "SELECT summary_id, content FROM summaries"
        ).fetchall()
        
        if summaries:
            update_sum = db.execute(
                "UPDATE summaries SET token_count = ? WHERE summary_id = ?"
            )
            summaries_updated = 0
            for summary in summaries:
                if not CJK_RE.search(summary["content"]):
                    continue
                new_count = estimate_tokens(summary["content"])
                update_sum.run(new_count, summary["summary_id"])
                summaries_updated += 1
            
            if summaries_updated > 0:
                print(f"[lcm] CJK token recount: updated {summaries_updated} summary(ies)")
        
        # Mark migration as complete
        db.execute("INSERT INTO lcm_migration_flags (flag) VALUES (?)", (FLAG,))
        
        # Commit transaction
        db.execute("COMMIT")
        
    except Exception as err:
        db.execute("ROLLBACK")
        raise err


def run_lcm_migrations(
    db: sqlite3.Connection,
    options: Optional[Dict[str, Any]] = None
) -> None:
    """Run all database migrations for LCM.
    
    Creates the database schema and runs data migrations for existing databases.
    
    Args:
        db: SQLite connection
        options: Optional options dict with keys:
            - fts5_available: If FTS5 is available (default: auto-detect)
    """
    # Create core tables
    db.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            session_key TEXT,
            title TEXT,
            bootstrapped_at TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS messages (
            message_id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
            seq INTEGER NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant', 'tool')),
            content TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE (conversation_id, seq)
        );

        CREATE TABLE IF NOT EXISTS summaries (
            summary_id TEXT PRIMARY KEY,
            conversation_id INTEGER NOT NULL REFERENCES conversations(conversation_id) on DELETE CASCADE,
            kind TEXT NOT NULL CHECK (kind IN ('leaf', 'condensed')),
            depth INTEGER NOT NULL DEFAULT 0,
            content TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            earliest_at TEXT,
            latest_at TEXT,
            descendant_count INTEGER NOT NULL DEFAULT 0,
            descendant_token_count INTEGER NOT NULL DEFAULT 0,
            source_message_token_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            file_ids TEXT NOT NULL DEFAULT '[]'
        );

        CREATE TABLE IF NOT EXISTS message_parts (
            part_id TEXT PRIMARY KEY,
            message_id INTEGER NOT NULL REFERENCES messages(message_id) ON DELETE CASCADE,
            session_id TEXT NOT NULL,
            part_type TEXT NOT NULL CHECK (part_type IN (
                'text', 'reasoning', 'tool', 'patch', 'file',
                'subtask', 'compaction', 'step_start', 'step_finish',
                'snapshot', 'agent', 'retry'
            )),
            ordinal INTEGER NOT NULL,
            text_content TEXT,
            is_ignored INTEGER,
            is_synthetic INTEGER,
            tool_call_id TEXT,
            tool_name TEXT,
            tool_status TEXT,
            tool_input TEXT,
            tool_output TEXT,
            tool_error TEXT,
            tool_title TEXT,
            patch_hash TEXT,
            patch_files TEXT,
            file_mime TEXT,
            file_name TEXT,
            file_url TEXT,
            subtask_prompt TEXT,
            subtask_desc TEXT,
            subtask_agent TEXT,
            step_reason TEXT,
            step_cost REAL,
            step_tokens_in INTEGER,
            step_tokens_out INTEGER,
            snapshot_hash TEXT,
            compaction_auto INTEGER,
            metadata TEXT,
            UNIQUE (message_id, ordinal)
        );

        CREATE TABLE IF NOT EXISTS summary_messages (
            summary_id TEXT NOT NULL REFERENCES summaries(summary_id) ON DELETE CASCADE,
            message_id INTEGER NOT NULL REFERENCES messages(message_id) ON DELETE RESTRICT,
            ordinal INTEGER NOT NULL,
            PRIMARY KEY (summary_id, message_id)
        );

        CREATE TABLE IF NOT EXISTS summary_parents (
            summary_id TEXT NOT NULL REFERENCES summaries(summary_id) ON DELETE CASCADE,
            parent_summary_id TEXT NOT NULL REFERENCES summaries(summary_id) ON DELETE RESTRICT,
            ordinal INTEGER NOT NULL,
            PRIMARY KEY (summary_id, parent_summary_id)
        );

        CREATE TABLE IF NOT EXISTS context_items (
            conversation_id INTEGER NOT NULL REFERENCES conversations(conversation_id) on DELETE CASCADE,
            ordinal INTEGER NOT NULL,
            item_type TEXT NOT NULL CHECK (item_type IN ('message', 'summary')),
            message_id INTEGER REFERENCES messages(message_id) ON DELETE RESTRICT,
            summary_id TEXT REFERENCES summaries(summary_id) ON DELETE RESTRICT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (conversation_id, ordinal),
            CHECK (
                (item_type = 'message' AND message_id IS NOT NULL AND summary_id IS NULL) OR
                (item_type = 'summary' AND summary_id IS NOT NULL AND message_id IS NULL)
            )
        );

        CREATE TABLE IF NOT EXISTS large_files (
            file_id TEXT PRIMARY KEY,
            conversation_id INTEGER NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
            file_name TEXT,
            mime_type TEXT,
            byte_size INTEGER,
            storage_uri TEXT NOT NULL,
            exploration_summary TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS conversation_bootstrap_state (
            conversation_id INTEGER PRIMARY KEY REFERENCES conversations(conversation_id) ON DELETE CASCADE,
            session_file_path TEXT NOT NULL,
            last_seen_size INTEGER NOT NULL,
            last_seen_mtime_ms INTEGER NOT NULL,
            last_processed_offset INTEGER NOT NULL,
            last_processed_entry_hash TEXT,
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        -- Create indexes
        CREATE INDEX IF NOT EXISTS messages_conv_seq_idx ON messages (conversation_id, seq);
        CREATE INDEX IF NOT EXISTS summaries_conv_created_idx ON summaries(conversation_id, created_at);
        CREATE INDEX IF NOT EXISTS message_parts_message_idx ON message_parts(message_id);
        CREATE INDEX IF NOT EXISTS message_parts_type_idx ON message_parts(part_type);
        CREATE INDEX IF NOT EXISTS context_items_conv_idx ON context_items(conversation_id, ordinal);
        CREATE INDEX IF NOT EXISTS large_files_conv_idx ON large_files(conversation_id, created_at);
        CREATE INDEX IF NOT EXISTS bootstrap_state_path_idx
            ON conversation_bootstrap_state(session_file_path, updated_at);
    """)
    
    # Forward-compatible conversations migration
    conversation_columns = db.execute("PRAGMA table_info(conversations)").fetchall()
    column_names = [col[0] for col in conversation_columns]
    
    has_bootstrapped_at = any(col.name == "bootstrapped_at" for col in column_names)
    if not has_bootstrapped_at:
        db.execute("ALTER TABLE conversations ADD COLUMN bootstrapped_at TEXT")
    
    has_session_key = any(col.name == "session_key" for col in column_names)
    if not has_session_key:
        db.execute("ALTER TABLE conversations ADD COLUMN session_key TEXT")
    
    db.execute("CREATE UNIQUE INDEX IF NOT EXISTS conversations_session_key_idx ON conversations(session_key)")
    
    # Run column migrations
    _ensure_summary_depth_column(db)
    _ensure_summary_metadata_columns(db)
    _ensure_summary_model_column(db)
    
    # CJK recount MUST run before backfill_summary_metadata
    _recalculate_cjk_token_counts(db)
    
    # Run backfill migrations
    _backfill_summary_depths(db)
    _backfill_summary_metadata(db)
    _backfill_tool_call_columns(db)
    
    # Check FTS5 availability
    fts5_available = options.get("fts5_available", True) if options else get_lcm_db_features(db).fts5_available
    
    if not fts5_available:
        return
    
    # Create FTS5 virtual tables
    _create_fts5_tables(db, fts5_available)


def _create_fts5_tables(db: sqlite3.Connection, fts5_available: bool) -> None:
    """Create FTS5 virtual tables for full-text search."""
    if not fts5_available:
        return
    
    # Check for existing FTS tables
    has_fts = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'"
    ).fetchone()
    
    if has_fts:
        # Check for stale schema and Drop and recreate if needed
        fts_schema = db.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='messages_fts'"
        ).fetchone()
        
        if fts_schema and "content_rowid" in fts_schema[0]:
            db.execute("DROP TABLE messages_fts")
            db.execute("""
                CREATE VIRTUAL TABLE messages_fts USING fts5(
                    content,
                    tokenize='porter unicode61'
                );
                INSERT INTO messages_fts(rowid, content) SELECT message_id, content FROM messages;
            """)
    else:
        # Create messages FTS table
        db.execute("""
            CREATE VIRTUAL TABLE messages_fts USING fts5(
                content,
                tokenize='porter unicode61'
            );
        """)
    
    # Check summaries FTS
    summaries_fts_info = db.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='summaries_fts'"
    ).fetchone()
    
    summaries_fts_sql = summaries_fts_info[0] if summaries_fts_info else summaries_fts_info[0] else ""
    
    summaries_fts_columns = db.execute("PRAGMA table_info(summaries_fts)").fetchall()
    has_summary_id_column = any(
        col.name == "summary_id" for col in summaries_fts_columns
    )
    should_recreate = (
        not summaries_fts_info or
        not has_summary_id_column or
        "content_rowid='summary_id'" in summaries_fts_sql or
        'content_rowid="summary_id"' in summaries_fts_sql
    )
    
    if should_recreate:
        db.execute("""
            DROP TABLE IF EXISTS summaries_fts;
            CREATE VIRTUAL TABLE summaries_fts USING fts5(
                summary_id UNINDEXED,
                content,
                tokenize='porter unicode61'
            );
            INSERT INTO summaries_fts(summary_id, content)
            SELECT summary_id, content FROM summaries;
        """)
