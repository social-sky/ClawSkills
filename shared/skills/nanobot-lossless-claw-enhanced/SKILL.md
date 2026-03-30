# nanobot-lossless-claw-enhanced

Python port of LCM (Lossless Context Management) from https://github.com/win4r/lossless-claw-enhanced

## 授權 (License)

本專案是 [lossless-claw-enhanced](https://github.com/win4r/lossless-claw-enhanced) 的 Python 移植版。

原始專案授權：**MIT License**  
Copyright (c) 2026 Josh Lehman / Martian Engineering

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

## Overview

This is a complete Python port of the TypeScript LCM (Lossless Context Management) plugin from OpenClaw. It provides DAG-based summarization for efficient context management while preserving full conversation history.

## Features

- **DAG-based summarization**: Creates hierarchical summaries ( preserving full context
- **CJK-aware token estimation**: Accurate token counting for Chinese/Japanese/Korean text
- **SQLite storage**: Persistent storage with migrations
- **Full-text search**: FTS5 with CJK fallback
- **Tool pairing repair**: Ensures valid tool sequences

## Installation

```bash
pip install nanobot-lossless-claw-enhanced
```

Or clone and repository:
```bash
cd shared/skills/nanobot-lossless-claw-enhanced
python -m pytest tests/
```

## Quick Test

```bash
python -m pytest tests/test_estimate_tokens.py -v
```

## Test LcmConfig

```python
from nanobot_lossless_claw_enhanced.db.config import LcmConfig

config = LcmConfig()
print(config)
assert config.enabled == True
print(config.database_path)
print(config.context_threshold)
print(config.fresh_tail_count)
```

Output:
```
## Test DbFeatures

```python
from nanobot_lossless_claw_enhanced.db.features import get_lcm_db_features, create_in_memory_database

# Create in-memory database
db = create_in_memory_database()

# Test FTS5 detection
features = get_lcm_db_features(db)
print(features)
assert features.fts5_available == True
```

Output
```
## Test DbConnection

```python
import tempfile
from nanobot_lossless_claw_enhanced.db.connection import create_database
from nanobot_lossless_claw_enhanced.db.config import LcmConfig

# Create temp database file
with tempfile.TemporaryDirectory() as temp_dir:
    temp_db_path = temp_dir / "test_db.db"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create database
    config = LcmConfig(database_path=str(temp_db_path))
    db = create_database(config)
    
    # Test database creation
    assert db is not None
    
    # Verify WAL mode is enabled
    cursor = db.execute("PRAGMA journal_mode")
    result = cursor.fetchone()
    assert result[0] == "wal"
    
    # Verify foreign keys are enabled
    cursor = db.execute("PRAGMA foreign_keys")
    result = cursor.fetchone()
    assert result[0] == "ON"
    
    # Cleanup
    db.close()
    os.unlink(temp_db_path)


if __name__ == "__main__":
    pytest.main()
