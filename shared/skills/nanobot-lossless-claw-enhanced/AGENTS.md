# nanobot-lossless-claw-enhanced - AI Agent Guide

## Overview

Python port of LCM (Lossless Context Management) from [lossless-claw-enhanced](https://github.com/win4r/lossless-claw-enhanced). Provides DAG-based summarization for efficient context management while preserving full conversation history.

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

## Project Structure

```
nanobot-lossless-claw-enhanced/
├── estimate_tokens.py          # CJK-aware token estimation
├── session_patterns.py        # Glob pattern matching
├── lcm_types.py              # Core dataclasses and enums
├── transcript_repair.py       # Tool pairing repair
├── large_files.py            # Large file externalization
├── summarize.py              # LLM summarization
├── expansion_auth.py          # Expansion authorization
├── assembler.py              # Context assembly
├── retrieval.py              # Search/retrieval engine
├── compaction.py             # DAG compaction engine
├── engine.py                 # Main LCM engine
├── pyproject.toml            # Packaging
├── run_tests.py              # Test runner
├── SKILL.md                  # Skill definition
├── db/
│   ├── __init__.py
│   ├── config.py             # Configuration resolution
│   ├── features.py           # FTS5 detection
│   ├── connection.py         # Database connection
│   └── migration.py          # Schema migrations
├── search/
│   ├── __init__.py
│   ├── fts5_sanitize.py     # FTS5 query sanitization
│   └── full_text_fallback.py # LIKE search fallback
├── store/
│   ├── __init__.py
│   ├── conversation_store.py # Message storage
│   └── summary_store.py      # Summary storage
└── tests/
    ├── test_estimate_tokens.py
    ├── test_session_patterns.py
    ├── test_fts5_sanitize.py
    ├── test_full_text_fallback.py
    ├── test_expansion_auth.py
    ├── test_large_files.py
    ├── test_assembler.py
    ├── test_retrieval.py
    ├── test_compaction.py
    └── test_summarize.py
```

## Build/Lint/Test Commands

### Testing

```bash
# Run all tests
cd shared/skills/nanobot-lossless-claw-enhanced
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_estimate_tokens.py -v

# Run with coverage
python -m pytest tests/ --tb=short
```

### Code Quality

```bash
# Syntax check Python files
python -m py_compile estimate_tokens.py

# Check all modules
python -m py_compile *.py db/*.py search/*.py store/*.py
```

## Code Style Guidelines

### Python Style

```python
#!/usr/bin/env python3
"""Module docstring with description."""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Constants at top
SCRIPT_DIR = Path(__file__).parent
DEFAULT_PATH = "default/value"

def function_name(param: str, optional: Optional[int] = None) -> Dict[str, Any]:
    """Brief description of function.
    
    Args:
        param: Description of param
        optional: Optional parameter description
        
    Returns:
        Description of return value
    """
    pass

def main():
    parser = argparse.ArgumentParser(description="Tool description")
    parser.add_argument("--vault", "-v", default=DEFAULT_PATH, help="Help text")
    args = parser.parse_args()

if __name__ == "__main__":
    sys.exit(main())
```

**Key conventions:**
- Use `pathlib.Path` instead of `os.path` for file operations
- Type hints on all function signatures
- Docstrings for public functions
- UTF-8 encoding for file operations: `open(file, 'r', encoding='utf-8')`
- JSON with `ensure_ascii=False, indent=2`
- argparse for CLI tools

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Python files | snake_case | `estimate_tokens.py` |
| Python functions | snake_case | `estimate_tokens()` |
| Python classes | PascalCase | `LargeFileRecord` |
| Dataclass fields | snake_case | `conversation_id` |
| Constants | UPPER_SNAKE | `DEFAULT_THRESHOLD` |

### File Organization

```
skill-name/
├── SKILL.md           # Required
├── scripts/           # Optional: Executable code
├── references/        # Optional: Documentation
└── assets/            # Optional: Templates
```

---

_Guide for AI agents working on nanobot-lossless-claw-enhanced. Last updated: 2026-03-30_
