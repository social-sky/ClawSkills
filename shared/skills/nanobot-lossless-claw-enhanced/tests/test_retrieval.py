#!/usr/bin/env python3
"""Tests for RetrievalEngine.

Tests:
- describe() - Lookup summaries and files by ID
- grep() - Regex/full-text search
- expand() - Traverse summary hierarchy
"""

import os
import sys
import pytest
import sqlite3
import tempfile
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the modules
from estimate_tokens import estimate_tokens
from search.full_text_fallback import contains_cjk, build_like_search_plan
from search.fts5_sanitize import sanitize_fts5_query
from lcm_types import MessageRole, SummaryKind, ContextItemType
from lcm_types import (
    MessageRecord,
    SummaryRecord,
    ContextItemRecord,
    LargeFileRecord,
)
from db.config import resolve_lcm_config
from retrieval import (
    RetrievalEngine,
    RetrievalConfig,
    # describe, grep, expand are methods of RetrievalEngine, not top-level functions
)


# Test fixtures
@pytest.fixture
def temp_db():
    """Create temporary database for testing."""
    db = sqlite3.connect(":memory:")
    _create_tables(db)
    return db


def _create_tables(db):
    """Create necessary tables in the database."""
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
        CREATE TABLE IF not exists summaries (
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
    
    db.execute("""
        CREATE TABLE IF not exists large_files (
            file_id TEXT PRIMARY KEY,
            conversation_id INTEGER,
            file_name TEXT,
            mime_type TEXT,
            byte_size INTEGER,
            storage_uri TEXT,
            exploration_summary TEXT,
            created_at TEXT
        )
    """)
    
    db.execute("""
        CREATE TABLE IF not exists context_items (
            conversation_id INTEGER,
            ordinal INTEGER,
            item_type TEXT,
            message_id INTEGER,
            summary_id TEXT,
            created_at TEXT
        )
    """)
    
    db.execute("""
        CREATE TABLE IF not exists summary_hierarchy (
            parent_summary_id TEXT,
            child_summary_id TEXT,
            depth INTEGER,
            created_at TEXT
        )
    """)
    
    # Insert test data
    now = datetime.utcnow().isoformat()
    db.execute("INSERT INTO messages VALUES (1, 1, 'user', 'Hello, how are you?', 10, ?)", (now,))
    now = datetime.utcnow().isoformat()
    db.execute("INSERT INTO messages VALUES (2, 1, 2, 'assistant', 'I am doing well! Can I help you with Python?', 15, ?)", (now,))
    now = datetime.utcnow().isoformat()
    db.execute("INSERT INTO messages VALUES (3, 1, 3, 'user', 'Can you write a Python function?', 25, ?)", (now,))
    now = datetime.utcnow().isoformat()
    db.execute("INSERT INTO messages VALUES (4, 1, 4, 'assistant', json.dumps([{\"type\": \"text\", \"text\": \"Sure! Here is a Python function:\"}])", (now,))
    db.execute("INSERT INTO messages VALUES (5, 1, 5, 'user', json.dumps([{\"type\": \"tool_use\", \"id\": \"tool_1\", \"name\": \"python_eval\"}])", (now,))
    db.execute("INSERT INTO messages VALUES (6, 1, 6, 'user', json.dumps([{\"type\": \"tool_result\", \"tool_use_id\": \"tool_1\", \"content\": \"Result: 42\"}])", (now,))
    
    db.execute("INSERT INTO summaries VALUES ('sum_1', 1, 'leaf', 0, 'Conversation about Python: user greeted, asked for help, wrote a function.', 20, '[]', NULL, NULL, 0, 0, 0, 'unknown', ?)", (now,))
    db.execute("INSERT INTO summaries VALUES ('sum_2', 1, 'leaf', 0, 'Detailed discussion about Python functions.', 25, '[]', NULL, NULL, 0, 0, 0, 'unknown', ?)", (now,))
    db.execute("INSERT INTO summaries VALUES ('sum_3', 1, 'condensed', 1, 'Python functions covered: basic I/O, file handling, string manipulation. User greeted, assistant, discussed data structures. User questions about basic I/O operations, writing and reading files. Such assistant handled user requests for Python help including explaining code, providing examples, then answering questions.', 25, '[]', NULL, NULL, 0, 0, 0, 'unknown', ?)", (now,))
    db.execute("INSERT INTO summaries VALUES ('sum_4', 1, 'condensed', 2, 'Overview of all Python discussions: Topics include basic I/O, string manipulation, data structures, error handling. Testing. Such 25, '[]', NULL, NULL, 0, 0, 0, 'unknown', ?)", (now,))
    
    db.execute("INSERT INTO context_items VALUES (1, 1, 'summary', NULL, 'sum_1', ?)", (now,))
    db.execute("INSERT INTO context_items VALUES (2, 1, 'summary', NULL, 'sum_2', ?)", (now,))
    db.execute("INSERT INTO context_items VALUES (3, 1, 'summary', NULL, 'sum_3', ?)", (now,))
    db.execute("INSERT INTO context_items VALUES (4, 1, 'summary', NULL, 'sum_4', ?)", (now,))
    
    db.execute("INSERT INTO summary_hierarchy VALUES ('sum_1', 'sum_2', 1, ?)", (now,))
    db.execute("INSERT INTO summary_hierarchy VALUES ('sum_2', 'sum_3', 1, ?)", (now,))
    db.execute("INSERT INTO summary_hierarchy VALUES ('sum_3', 'sum_4', 1, ?)", (now,))
    
    db.commit()
    
    yield temp_db


class TestDescribe:
    """Test describe() operation."""
    
    def test_describe_summary(self, temp_db):
        """Test describe summary."""
        engine = RetrievalEngine(
            config=RetrievalConfig(),
            db_connection=temp_db
        )
        
        # Test finding a summary
        result = engine.describe("sum_1")
        assert result.found
        assert result.item_type == "summary"
        assert "Conversation about Python" in result.content
        
        # Test getting children
        result = engine.describe("sum_1", include_children=True)
        assert result.found
        assert "sum_2" in result.children
        
        # Test describing non-existent item
        result = engine.describe("nonexistent")
        assert not result.found
        assert result.error is not None
    
    def test_describe_file(self, temp_db):
        """Test describe file."""
        # Insert a file
        now = datetime.utcnow().isoformat()
        db.execute(
            "INSERT INTO large_files VALUES ('file_1', 1, 'test_file.txt', 'text/plain', 1000, 'db://file_1', 'Test file exploration summary', ?)",
            (now,)
        )
        
        engine = RetrievalEngine(
            config=RetrievalConfig(),
            db_connection=temp_db
        )
        
        result = engine.describe("file_1")
        assert result.found
        assert result.item_type == "file"
        assert "test_file.txt" in result.content


class TestGrep:
    """Test grep() operation."""
    
    def test_grep_full_text(self, temp_db):
        """Test full-text grep search."""
        engine = RetrievalEngine(
            config=RetrievalConfig(),
            db_connection=temp_db
        )
        
        # Test search
        result = engine.grep("Python", mode="full_text")
        assert result.mode == "full_text"
        assert result.total_count >= 3
        assert any("Python" in match["snippet"].lower() for match in result.matches)
        
    def test_grep_regex(self, temp_db):
        """Test regex grep search."""
        engine = RetrievalEngine(
            config=RetrievalConfig(),
            db_connection=temp_db
        )
        
        result = engine.grep(r"function\s+\w+", mode="regex")
        assert result.mode == "regex"
        assert result.total_count >= 2
        
    def test_grep_with_cjk(self, temp_db):
        """Test grep with CJK characters."""
        # Add CJK message
        db.execute(
            "INSERT INTO messages VALUES (10, 1, 10, 'user', '你好世界!这是一条测试消息。', 30, ?)",
            (now)
        )
        now = datetime.utcnow().isoformat()
        
        engine = RetrievalEngine(
            config=RetrievalConfig(),
            db_connection=temp_db
        )
        
        result = engine.grep("你好", mode="full_text")
        assert result.total_count >= 1
        
    def test_grep_pagination(self, temp_db):
        """Test grep pagination."""
        engine = RetrievalEngine(
            config=RetrievalConfig(max_results=2),
            db_connection=temp_db
        )
        
        result = engine.grep("Python", mode="full_text")
        assert len(result.matches) <= 2
        assert result.truncated


class TestExpand:
    """Test expand() operation."""
    
    def test_expand_summary(self, temp_db):
        """Test expanding a summary."""
        engine = RetrievalEngine(
            config=RetrievalConfig(),
            db_connection=temp_db
        )
        
        result = engine.expand("sum_4")
        assert result.found
        assert "Overview of all Python discussions" in result.expanded_content
        assert len(result.source_messages) >= 2
        
    def test_expand_nonexistent(self, temp_db):
        """Test expanding non-existent summary."""
        engine = RetrievalEngine(
            config=RetrievalConfig(),
            db_connection=temp_db
        )
        
        result = engine.expand("nonexistent")
        assert not result.found
        assert result.error is not None
        
    def test_expand_with_files(self, temp_db):
        """Test expand with file references."""
        # Insert a file reference in a summary
        db.execute(
            "UPDATE summaries SET file_ids = ? WHERE summary_id = ?", ('file_1', 'file_2')
        )
        
        engine = RetrievalEngine(
            config=RetrievalConfig(),
            db_connection=temp_db
        )
        
        result = engine.expand("sum_4")
        # Should find file references (though file_1 might not exist in current implementation)
        assert result.found
        assert result.item_type == "summary"
        assert "Conversation about Python" in result.content
        
        # Test getting children
        result_children = engine.describe("sum_1", include_children=True).children
        assert "sum_2" in result_children
        
        # Test describing non-existent item
        result = engine.describe("nonexistent")
        assert not result.found
        assert result.error is not None
    
    def test_describe_file(self, temp_db):
        """Test describe file."""
        engine = RetrievalEngine(
            config=RetrievalConfig(),
            db_connection=temp_db
        )
        
        result = engine.describe("file_1")
        assert result.found
        assert result.item_type == "file"
        assert "test_file.txt" in result.content


class TestGrep:
    """Test grep() operation."""
    
    def test_grep_full_text(self, temp_db):
        """Test full-text grep search."""
        engine = RetrievalEngine(
            config=RetrievalConfig(),
            db_connection=temp_db
        )
        
        # Test search
        result = engine.grep("Python", mode="full_text")
        assert result.mode == "full_text"
        assert result.total_count >= 3
        assert any("Python" in match["snippet"].lower() for match in result.matches)
        
    def test_grep_regex(self, temp_db):
        """Test regex grep search."""
        engine = RetrievalEngine(
            config=RetrievalConfig(),
            db_connection=temp_db
        )
        
        result = engine.grep(r"function\s+\w+", mode="regex")
        assert result.mode == "regex"
        assert result.total_count >= 2
        
    def test_grep_with_cjk(self, temp_db):
        """Test grep with CJK characters."""
        # Add CJK message
        now = datetime.utcnow().isoformat()
        db.execute(
            "INSERT INTO messages VALUES (10, 1, 10, 'user', '你好世界！這是一個測試消息。', 30, ?)",
            (now,)
        )
        
        engine = RetrievalEngine(
            config=RetrievalConfig(),
            db_connection=temp_db
        )
        
        result = engine.grep("你好", mode="full_text")
        assert result.total_count >= 1
        
    def test_grep_pagination(self, temp_db):
        """Test grep pagination."""
        engine = RetrievalEngine(
            config=RetrievalConfig(max_results=2),
            db_connection=temp_db
        )
        
        result = engine.grep("Python", mode="full_text")
        assert len(result.matches) <= 2
        assert result.truncated


class TestExpand:
    """Test expand() operation."""
    
    def test_expand_summary(self, temp_db):
        """Test expanding a summary."""
        engine = RetrievalEngine(
            config=RetrievalConfig(),
            db_connection=temp_db
        )
        
        result = engine.expand("sum_4")
        assert result.found
        assert "Overview of all Python discussions" in result.expanded_content
        assert len(result.source_messages) >= 2
        
    def test_expand_nonexistent(self, temp_db):
        """Test expanding non-existent summary."""
        engine = RetrievalEngine(
            config=RetrievalConfig(),
            db_connection=temp_db
        )
        
        result = engine.expand("nonexistent")
        assert not result.found
        assert result.error is not None
        
    def test_expand_with_files(self, temp_db):
        """Test expand with file references."""
        # Insert a file reference in a summary
        db.execute(
            "UPDATE summaries SET file_ids = ? WHERE summary_id = ?",
            ('file_1', 'file_2')
        )
        
        engine = RetrievalEngine(
            config=RetrievalConfig(),
            db_connection=temp_db
        )
        
        result = engine.expand("sum_4")
        # Should find file references (though file_1 might not exist in current implementation)
        assert result.found
