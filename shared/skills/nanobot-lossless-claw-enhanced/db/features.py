#!/usr/bin/env python3
"""Database feature detection for LCM.

Detects SQLite features like FTS5 availability.
"""

import sqlite3
from dataclasses import dataclass
from typing import Dict


@dataclass
class DbFeatures:
    """Detected SQLite features."""
    fts5_available: bool


# Cache for feature detection results per connection
_feature_cache: Dict[int, DbFeatures] = {}


def get_lcm_db_features(db: sqlite3.Connection) -> DbFeatures:
    """Detect SQLite features exposed by the current runtime.
    
    The result is cached per connection object because the probe is
    runtime-specific, not database-file-specific.
    
    Args:
        db: SQLite connection to probe
        
    Returns:
        DbFeatures with detected feature flags
    """
    # Use id() as cache key since we can't use weak references in the same way as TS
    cache_key = id(db)
    
    if cache_key in _feature_cache:
        return _feature_cache[cache_key]
    
    fts5_available = _probe_fts5(db)
    
    features = DbFeatures(fts5_available=fts5_available)
    _feature_cache[cache_key] = features
    
    return features


def _probe_fts5(db: sqlite3.Connection) -> bool:
    """Probe for FTS5 availability.
    
    Args:
        db: SQLite connection
        
    Returns:
        True if FTS5 is available, False otherwise
    """
    cursor = db.cursor()
    try:
        # Try to create a temporary FTS5 table
        cursor.execute("DROP TABLE IF EXISTS temp.__lcm_fts5_probe")
        cursor.execute("CREATE VIRTUAL TABLE temp.__lcm_fts5_probe USING fts5(content)")
        cursor.execute("DROP TABLE temp.__lcm_fts5_probe")
        return True
    except sqlite3.OperationalError:
        # FTS5 not available
        try:
            cursor.execute("DROP TABLE IF EXISTS temp.__lcm_fts5_probe")
        except sqlite3.OperationalError:
            pass
        return False
    finally:
        cursor.close()
