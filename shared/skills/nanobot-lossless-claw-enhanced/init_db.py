#!/usr/bin/env python3
"""Initialize LCM database for nanobot-lossless-claw-enhanced."""

import os
import sys
from pathlib import Path

# Add skill directory to path
skill_dir = Path(__file__).parent
sys.path.insert(0, str(skill_dir))

from db.config import resolve_lcm_config
from db.connection import create_database

def main():
    print("=== LCM Database Initialization ===\n")
    
    # Resolve configuration
    config = resolve_lcm_config()
    
    print(f"LCM Enabled: {config.enabled}")
    print(f"Database Path: {config.database_path}")
    print(f"Context Threshold: {config.context_threshold}")
    print(f"Fresh Tail Count: {config.fresh_tail_count}")
    print(f"Timezone: {config.timezone}")
    
    # Expand path
    db_path = Path(config.database_path).expanduser()
    print(f"\nExpanded DB Path: {db_path}")
    
    # Create parent directory if needed
    db_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Parent directory: {db_path.parent} (exists: {db_path.parent.exists()})")
    
    # Create database
    print("\nCreating database...")
    db = create_database(config)
    
    # Verify
    cursor = db.execute("PRAGMA journal_mode")
    journal_mode = cursor.fetchone()[0]
    print(f"Journal Mode: {journal_mode}")
    
    cursor = db.execute("PRAGMA foreign_keys")
    fk = cursor.fetchone()[0]
    print(f"Foreign Keys: {fk}")
    
    # Check if tables exist
    cursor = db.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"\nTables in database: {tables}")
    
    db.close()
    
    # Verify file exists
    if db_path.exists():
        print(f"\n✅ Database file created successfully: {db_path}")
        print(f"   File size: {db_path.stat().st_size} bytes")
    else:
        print(f"\n❌ Database file NOT found at: {db_path}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
