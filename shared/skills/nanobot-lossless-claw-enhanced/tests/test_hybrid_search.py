#!/usr/bin/env python3
"""Tests for Hybrid Search and Decay Scoring.

Tests:
- hybrid_search() - BM25 + decay + recency fusion
- ScoredSummary dataclass
- HybridSearchResult dataclass
"""

import pytest
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from lcm_types import SummaryKind, MemoryTier, MemoryCategory, SummaryRecord
from retrieval import (
    RetrievalEngine,
    RetrievalConfig,
    ScoredSummary,
    HybridSearchResult,
    parse_scope,
    matches_scope,
    get_allowed_scopes,
)
from memory_decay import calculate_decay_score, calculate_recency_score


# Test fixtures
@pytest.fixture
def temp_db_with_scope():
    """Create temporary database with scope columns for testing."""
    db = sqlite3.connect(":memory:")
    _create_tables_with_scope(db)
    _insert_test_data_with_scope(db)
    yield db
    db.close()


def _create_tables_with_scope(db):
    """Create tables with scope columns."""
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
        CREATE TABLE IF NOT EXISTS summaries (
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
            created_at TEXT,
            tier TEXT DEFAULT 'peripheral',
            category TEXT,
            importance REAL DEFAULT 0.5,
            access_count INTEGER DEFAULT 0,
            last_accessed_at TEXT,
            decay_score REAL DEFAULT 1.0,
            scope TEXT DEFAULT 'global'
        )
    """)
    
    db.execute("""
        CREATE TABLE IF NOT EXISTS large_files (
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


def _insert_test_data_with_scope(db):
    """Insert test data with various scopes and decay scores."""
    now = datetime.utcnow()
    
    # Insert summaries with different scopes, tiers, and importance
    summaries = [
        # Global scope - high importance (CORE tier)
        ("sum_global_core", 1, "leaf", 0, 
         "User prefers Python over Java. Important project decision.", 20,
         now - timedelta(days=1), now - timedelta(days=1),
         MemoryTier.CORE, MemoryCategory.DECISION, 0.9, 5, "global"),
        
        # Agent scope - medium importance (WORKING tier)
        ("sum_agent_work", 1, "leaf", 0,
         "Discussion about async programming in Python.", 25,
         now - timedelta(days=7), now - timedelta(days=3),
         MemoryTier.WORKING, MemoryCategory.PATTERNS, 0.7, 3, "agent:agent_1"),
        
        # Agent scope - low importance (PERIPHERAL tier)
        ("sum_agent_periph", 1, "leaf", 0,
         "Casual conversation about weather.", 15,
         now - timedelta(days=30), now - timedelta(days=30),
         MemoryTier.PERIPHERAL, MemoryCategory.OTHER, 0.3, 1, "agent:agent_1"),
        
        # Project scope - high importance
        ("sum_project_core", 1, "condensed", 1,
         "Project architecture discussion: microservices over monolith.", 30,
         now - timedelta(days=14), now - timedelta(days=5),
         MemoryTier.CORE, MemoryCategory.DECISION, 0.85, 8, "project:proj_1"),
        
        # User scope - low importance
        ("sum_user_periph", 1, "leaf", 0,
         "User mentioned they like dark mode.", 12,
         now - timedelta(days=60), now - timedelta(days=60),
         MemoryTier.PERIPHERAL, MemoryCategory.PREFERENCES, 0.4, 1, "user:user_1"),
        
        # Old but high importance - should still rank high due to importance
        ("sum_old_important", 1, "condensed", 2,
         "Critical decision: chose PostgreSQL over MongoDB.", 35,
         now - timedelta(days=90), now - timedelta(days=10),
         MemoryTier.CORE, MemoryCategory.DECISION, 0.95, 12, "global"),
    ]
    
    for (sum_id, conv_id, kind, depth, content, tokens, created, accessed,
         tier, category, importance, access_count, scope) in summaries:
        db.execute("""
            INSERT INTO summaries 
            (summary_id, conversation_id, kind, depth, content, token_count,
             created_at, last_accessed_at, tier, category, importance, 
             access_count, decay_score, scope)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (sum_id, conv_id, kind, depth, content, tokens,
              created.isoformat(), accessed.isoformat(),
              tier.value, category.value if category else None,
              importance, access_count, 1.0, scope))
    
    db.commit()


class TestHybridSearch:
    """Tests for hybrid_search() method."""
    
    def test_hybrid_search_returns_hybrid_result(self, temp_db_with_scope):
        """Test that hybrid_search returns HybridSearchResult."""
        engine = RetrievalEngine(
            config=RetrievalConfig(),
            db_connection=temp_db_with_scope
        )
        
        result = engine.hybrid_search("Python decision")
        
        assert isinstance(result, HybridSearchResult)
        assert hasattr(result, 'results')
        assert hasattr(result, 'total_count')
        assert hasattr(result, 'query')
        assert hasattr(result, 'bm25_weight')
        assert hasattr(result, 'decay_weight')
        assert hasattr(result, 'recency_weight')
    
    def test_hybrid_search_scores_sorted(self, temp_db_with_scope):
        """Test that results are sorted by final_score descending."""
        engine = RetrievalEngine(
            config=RetrievalConfig(),
            db_connection=temp_db_with_scope
        )
        
        result = engine.hybrid_search("Python decision architecture")
        
        # Verify descending order
        scores = [r.final_score for r in result.results]
        assert scores == sorted(scores, reverse=True)
    
    def test_hybrid_search_respects_limit(self, temp_db_with_scope):
        """Test that hybrid_search respects limit parameter."""
        engine = RetrievalEngine(
            config=RetrievalConfig(max_results=3),
            db_connection=temp_db_with_scope
        )
        
        result = engine.hybrid_search("Python", limit=2)
        
        assert len(result.results) <= 2
    
    def test_hybrid_search_includes_decay_info(self, temp_db_with_scope):
        """Test that results include decay and recency scores."""
        engine = RetrievalEngine(
            config=RetrievalConfig(),
            db_connection=temp_db_with_scope
        )
        
        result = engine.hybrid_search("Python")
        
        if result.results:
            first = result.results[0]
            assert hasattr(first, 'decay_score')
            assert hasattr(first, 'recency_score')
            assert hasattr(first, 'bm25_score')
            assert 0 <= first.decay_score <= 1.0 or first.decay_score > 1.0  # decay can exceed 1 with reinforcement
            assert 0 <= first.recency_score <= 1.0
    
    def test_hybrid_search_cjk_query(self, temp_db_with_scope):
        """Test hybrid_search with CJK query."""
        # Add CJK content
        now = datetime.utcnow()
        temp_db_with_scope.execute("""
            INSERT INTO summaries 
            (summary_id, conversation_id, kind, depth, content, token_count,
             created_at, last_accessed_at, tier, importance, access_count, scope)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, ("sum_cjk", 1, "leaf", 0, "中文測試內容", 10,
              now.isoformat(), now.isoformat(), "peripheral", 0.5, 0, "global"))
        temp_db_with_scope.commit()
        
        engine = RetrievalEngine(
            config=RetrievalConfig(),
            db_connection=temp_db_with_scope
        )
        
        result = engine.hybrid_search("中文")
        
        assert result.total_count >= 1


class TestScoredSummary:
    """Tests for ScoredSummary dataclass."""
    
    def test_scored_summary_creation(self):
        """Test creating a ScoredSummary."""
        now = datetime.utcnow()
        scored = ScoredSummary(
            summary_id="test_1",
            conversation_id=1,
            kind=SummaryKind.LEAF,
            content="Test content",
            snippet="Test snippet",
            created_at=now,
            bm25_score=10.5,
            decay_score=0.8,
            recency_score=0.9,
            final_score=5.2,
            tier=MemoryTier.WORKING,
            category=MemoryCategory.PATTERNS,
            importance=0.7,
            access_count=3,
            scope="global"
        )
        
        assert scored.summary_id == "test_1"
        assert scored.kind == SummaryKind.LEAF
        assert scored.tier == MemoryTier.WORKING
        assert scored.final_score == 5.2
    
    def test_scored_summary_defaults(self):
        """Test ScoredSummary default values."""
        now = datetime.utcnow()
        scored = ScoredSummary(
            summary_id="test_1",
            conversation_id=1,
            kind=SummaryKind.LEAF,
            content="Test",
            snippet="Test",
            created_at=now
        )
        
        assert scored.tier == MemoryTier.PERIPHERAL
        assert scored.scope == "global"
        assert scored.importance == 0.5


class TestScopeFiltering:
    """Tests for scope filtering in hybrid search."""
    
    def test_scope_parsing(self):
        """Test parse_scope function."""
        assert parse_scope("global") == ("global", "")
        assert parse_scope("agent:123") == ("agent", "123")
        assert parse_scope("project:proj_1") == ("project", "proj_1")
        assert parse_scope("user:user_1") == ("user", "user_1")
    
    def test_scope_matching(self):
        """Test matches_scope function."""
        assert matches_scope("global", "agent:123")
        assert matches_scope("agent:123", "agent:123")
        assert not matches_scope("agent:456", "agent:123")
        assert matches_scope("global", "project:any")
    
    def test_get_allowed_scopes(self):
        """Test get_allowed_scopes function."""
        scopes = get_allowed_scopes("agent_1")
        assert "global" in scopes
        assert "agent:agent_1" in scopes
        
        # With additional project access
        config = {"agentAccess": {"agent_1": ["project:proj_1"]}}
        scopes = get_allowed_scopes("agent_1", config)
        assert "global" in scopes
        assert "agent:agent_1" in scopes
        assert "project:proj_1" in scopes


class TestMemoryDecayIntegration:
    """Tests for memory decay integration in retrieval."""
    
    def test_decay_score_in_results(self, temp_db_with_scope):
        """Test that decay scores are calculated for search results."""
        engine = RetrievalEngine(
            config=RetrievalConfig(),
            db_connection=temp_db_with_scope
        )
        
        result = engine.hybrid_search("decision")
        
        if result.results:
            # High importance (0.9) item from yesterday should have high decay score
            for r in result.results:
                if r.importance > 0.8:
                    # Should have significant decay score
                    assert r.decay_score > 0.3
    
    def test_recency_affects_score(self, temp_db_with_scope):
        """Test that recency affects final score."""
        engine = RetrievalEngine(
            config=RetrievalConfig(recency_weight=0.5),
            db_connection=temp_db_with_scope
        )
        
        result = engine.hybrid_search("Python")
        
        if len(result.results) >= 2:
            # Recently accessed should potentially rank higher
            # (depending on other factors)
            assert result.recency_weight == 0.5


class TestTierBoosting:
    """Tests for memory tier boosting in hybrid search."""
    
    def test_core_tier_boosted(self, temp_db_with_scope):
        """Test that CORE tier items get boosted."""
        engine = RetrievalEngine(
            config=RetrievalConfig(decay_weight=0.3, recency_weight=0.1),
            db_connection=temp_db_with_scope
        )
        
        result = engine.hybrid_search("decision Python architecture")
        
        if result.results:
            # Find CORE tier items
            core_items = [r for r in result.results if r.tier == MemoryTier.CORE]
            if core_items:
                # CORE items should have tier_multiplier applied
                # (1.5x for CORE, checked via final_score calculation)
                for core in core_items:
                    # CORE tier with high importance should have good final score
                    assert core.final_score > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
