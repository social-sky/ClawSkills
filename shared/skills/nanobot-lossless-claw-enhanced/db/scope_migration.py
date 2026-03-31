#!/usr/bin/env python3
"""Scope column migration for LCM database.

Adds scope column to summaries table for multi-scope isolation support.
"""

import sqlite3
from typing import Optional


def add_scope_column(db: sqlite3.Connection) -> bool:
    """Add scope column to summaries table if it doesn't exist.
    
    Args:
        db: SQLite database connection
        
    Returns:
        True if column was added, False if it already existed
    """
    try:
        # Check if column exists
        cursor = db.execute("PRAGMA table_info(summaries)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if "scope" not in columns:
            db.execute("ALTER TABLE summaries ADD COLUMN scope TEXT DEFAULT 'global'")
            db.commit()
            return True
        return False
        
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e).lower():
            return False
        raise


def add_memory_lifecycle_columns(db: sqlite3.Connection) -> bool:
    """Add memory lifecycle columns to summaries table if they don't exist.
    
    Adds: tier, category, access_count, last_accessed_at, decay_score, importance
    
    Args:
        db: SQLite database connection
        
    Returns:
        True if any column was added, False if all existed
    """
    columns_to_add = [
        ("tier", "TEXT DEFAULT 'peripheral'"),
        ("category", "TEXT"),
        ("access_count", "INTEGER DEFAULT 0"),
        ("last_accessed_at", "TEXT"),
        ("decay_score", "REAL DEFAULT 1.0"),
        ("importance", "REAL DEFAULT 0.5"),
    ]
    
    added = False
    
    try:
        # Check existing columns
        cursor = db.execute("PRAGMA table_info(summaries)")
        existing = [row[1] for row in cursor.fetchall()]
        
        for col_name, col_def in columns_to_add:
            if col_name not in existing:
                db.execute(f"ALTER TABLE summaries ADD COLUMN {col_name} {col_def}")
                added = True
        
        if added:
            db.commit()
        
        return added
        
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e).lower():
            return False
        raise


def run_all_migrations(db: sqlite3.Connection) -> dict:
    """Run all scope and lifecycle migrations.
    
    Args:
        db: SQLite database connection
        
    Returns:
        Dict with migration results
    """
    results = {
        "scope_added": False,
        "lifecycle_added": False,
    }
    
    results["scope_added"] = add_scope_column(db)
    results["lifecycle_added"] = add_memory_lifecycle_columns(db)
    
    return results


if __name__ == "__main__":
    # Test migration
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        db = sqlite3.connect(str(db_path))
        
        # Create test table without new columns
        db.execute("""
            CREATE TABLE summaries (
                summary_id TEXT PRIMARY KEY,
                conversation_id INTEGER,
                kind TEXT,
                depth INTEGER,
                content TEXT,
                token_count INTEGER
            )
        """)
        
        # Run migrations
        results = run_all_migrations(db)
        print(f"Migration results: {results}")
        
        # Verify columns exist
        cursor = db.execute("PRAGMA table_info(summaries)")
        columns = [row[1] for row in cursor.fetchall()]
        print(f"Columns after migration: {columns}")
        
        db.close()
