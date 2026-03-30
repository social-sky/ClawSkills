#!/usr/bin/env python3
"""Database connection management for LCM.

Provides utilities for creating and managing SQLite connections.
"""

import sqlite3
from pathlib import Path
from typing import Optional

from .config import LcmConfig


def create_database(config: LcmConfig) -> sqlite3.Connection:
    """Create SQLite connection with optimal settings.
    
    Args:
        config: LCM configuration with database_path
        
    Returns:
        Configured SQLite connection
    """
    # Ensure parent directory exists
    db_path = Path(config.database_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Connect with optimal settings
    db = sqlite3.connect(str(db_path))
    
    # Enable WAL mode for better concurrency
    db.execute("PRAGMA journal_mode=WAL")
    
    # Enable foreign key constraints
    db.execute("PRAGMA foreign_keys=ON")
    
    # Set busy timeout for concurrent access
    db.execute("PRAGMA busy_timeout=5000")
    
    return db


def create_in_memory_database() -> sqlite3.Connection:
    """Create in-memory SQLite database for testing.
    
    Returns:
        In-memory SQLite connection
    """
    db = sqlite3.connect(":memory:")
    
    # Enable foreign key constraints
    db.execute("PRAGMA foreign_keys=ON")
    
    return db
