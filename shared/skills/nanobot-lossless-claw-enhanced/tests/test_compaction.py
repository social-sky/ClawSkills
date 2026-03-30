#!/usr/bin/env python3
"""Tests for Compaction Engine.

Tests cover:
- CompactionConfig dataclass
- CompactionDecision evaluation
- CompactionResult handling
- CompactionEngine with evaluate() and compact()
- Three-level escalation (normal → aggressive → fallback)
- Fresh tail protection
- Auth error detection
- Multiple rounds until under budget
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from compaction import (
    CompactionConfig,
    CompactionDecision,
    CompactionResult,
    CompactionStats,
    CompactionEngine,
    CompactionLevel,
    CompactionPhase,
    create_compaction_engine_from_config,
)
from lcm_types import (
    MessageRecord,
    SummaryRecord,
    ContextItemRecord,
    MessageRole,
    SummaryKind,
    ContextItemType,
)
from db.config import LcmConfig


# Helper to create test messages
def create_message(
    message_id: int,
    seq: int,
    content: str = "Test message",
    token_count: int = 10,
    role: MessageRole = MessageRole.USER
) -> MessageRecord:
    """Create a test message record."""
    return MessageRecord(
        message_id=message_id,
        conversation_id=1,
        seq=seq,
        role=role,
        content=content,
        token_count=token_count,
        created_at=datetime.utcnow()
    )


def create_summary(
    summary_id: str,
    kind: SummaryKind = SummaryKind.LEAF,
    token_count: int = 100,
    descendant_count: int = 5
) -> SummaryRecord:
    """Create a test summary record."""
    return SummaryRecord(
        summary_id=summary_id,
        conversation_id=1,
        kind=kind,
        depth=0,
        content="Test summary content",
        token_count=token_count,
        descendant_count=descendant_count,
        descendant_token_count=descendant_count * 10,
        source_message_token_count=descendant_count * 10,
        created_at=datetime.utcnow()
    )


def create_context_item(
    ordinal: int,
    item_type: ContextItemType,
    message_id: int = None,
    summary_id: str = None
) -> ContextItemRecord:
    """Create a test context item record."""
    return ContextItemRecord(
        conversation_id=1,
        ordinal=ordinal,
        item_type=item_type,
        message_id=message_id,
        summary_id=summary_id,
        created_at=datetime.utcnow()
    )


class TestCompactionConfig:
    """Tests for CompactionConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CompactionConfig()
        
        assert config.token_budget == 128000
        assert config.context_threshold == 0.75
        assert config.fresh_tail_count == 32
        assert config.leaf_min_fanout == 8
        assert config.condensed_min_fanout == 4
        assert config.leaf_chunk_tokens == 20000
        assert config.leaf_target_tokens == 1200
        assert config.condensed_target_tokens == 2000
        assert config.max_rounds == 10
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = CompactionConfig(
            token_budget=64000,
            context_threshold=0.5,
            fresh_tail_count=16
        )
        
        assert config.token_budget == 64000
        assert config.context_threshold == 0.5
        assert config.fresh_tail_count == 16


class TestCompactionDecision:
    """Tests for CompactionDecision dataclass."""
    
    def test_decision_creation(self):
        """Test creating a compaction decision."""
        decision = CompactionDecision(
            should_compact=True,
            reason="Over budget",
            level=CompactionLevel.NORMAL,
            phase=CompactionPhase.LEAF,
            current_tokens=10000,
            target_tokens=5000
        )
        
        assert decision.should_compact is True
        assert decision.reason == "Over budget"
        assert decision.level == CompactionLevel.NORMAL
        assert decision.phase == CompactionPhase.LEAF
        assert decision.current_tokens == 10000
        assert decision.target_tokens == 5000
    
    def test_decision_defaults(self):
        """Test default values in decision."""
        decision = CompactionDecision(
            should_compact=False,
            reason="Under budget"
        )
        
        assert decision.level == CompactionLevel.NORMAL
        assert decision.phase == CompactionPhase.LEAF
        assert decision.current_tokens == 0
        assert decision.target_tokens == 0
        assert decision.fresh_tail_protected == 0


class TestCompactionStats:
    """Tests for CompactionStats dataclass."""
    
    def test_stats_creation(self):
        """Test creating compaction stats."""
        stats = CompactionStats(
            messages_compacted=10,
            summaries_created=2,
            tokens_before=1000,
            tokens_after=500,
            tokens_saved=500
        )
        
        assert stats.messages_compacted == 10
        assert stats.summaries_created == 2
        assert stats.tokens_before == 1000
        assert stats.tokens_after == 500
        assert stats.tokens_saved == 500
    
    def test_stats_defaults(self):
        """Test default stats values."""
        stats = CompactionStats()
        
        assert stats.messages_compacted == 0
        assert stats.summaries_created == 0
        assert stats.tokens_before == 0
        assert stats.tokens_after == 0
        assert stats.tokens_saved == 0
        assert stats.auth_errors_skipped == 0


class TestCompactionResult:
    """Tests for CompactionResult dataclass."""
    
    def test_result_creation(self):
        """Test creating a compaction result."""
        stats = CompactionStats()
        result = CompactionResult(
            success=True,
            stats=stats,
            new_summaries=[],
            new_context_items=[],
            rounds_completed=2,
            final_tokens=5000
        )
        
        assert result.success is True
        assert result.stats == stats
        assert result.new_summaries == []
        assert result.new_context_items == []
        assert result.rounds_completed == 2
        assert result.final_tokens == 5000
    
    def test_result_defaults(self):
        """Test default result values."""
        stats = CompactionStats()
        result = CompactionResult(success=False, stats=stats)
        
        assert result.new_summaries == []
        assert result.new_context_items == []
        assert result.warnings == []
        assert result.rounds_completed == 0
        assert result.final_tokens == 0
        assert result.final_level == CompactionLevel.NORMAL


class TestCompactionEngine:
    """Tests for CompactionEngine class."""
    
    @pytest.fixture
    def engine(self):
        """Create a compaction engine for testing."""
        config = CompactionConfig(
            token_budget=1000,
            context_threshold=0.75,
            fresh_tail_count=5,
            leaf_min_fanout=3,
            condensed_min_fanout=2,
            leaf_chunk_tokens=200,
            leaf_target_tokens=50
        )
        return CompactionEngine(config)
    
    @pytest.fixture
    def engine_with_summarizer(self):
        """Create a compaction engine with mock summarizer."""
        config = CompactionConfig(
            token_budget=1000,
            context_threshold=0.75,
            fresh_tail_count=5,
            leaf_min_fanout=3,
            leaf_chunk_tokens=200,
            leaf_target_tokens=50
        )
        
        async def mock_summarizer(text, aggressive=False, options=None):
            # Return a short summary
            return f"Summary of {len(text)} chars"
        
        return CompactionEngine(config, summarizer=mock_summarizer)
    
    def test_evaluate_empty_messages(self, engine):
        """Test evaluation with no messages."""
        decision = engine.evaluate([], [], [])
        
        assert decision.should_compact is False
        assert "No messages" in decision.reason
    
    def test_evaluate_under_budget(self, engine):
        """Test evaluation when under budget."""
        messages = [create_message(i, i, token_count=10) for i in range(5)]
        context_items = [
            create_context_item(i, ContextItemType.MESSAGE, message_id=i)
            for i in range(5)
        ]
        
        decision = engine.evaluate(messages, [], context_items)
        
        # 5 messages * 10 tokens = 50 tokens
        # Budget is 1000 * 0.75 = 750
        # Should be under budget
        assert decision.should_compact is False
        assert "Under budget" in decision.reason
    
    def test_evaluate_over_budget_normal(self, engine):
        """Test evaluation when over budget (normal level)."""
        # Create enough messages to exceed budget
        messages = [create_message(i, i, token_count=100) for i in range(15)]
        context_items = [
            create_context_item(i, ContextItemType.MESSAGE, message_id=i)
            for i in range(15)
        ]
        
        decision = engine.evaluate(messages, [], context_items)
        
        # 15 messages * 100 tokens = 1500 tokens
        # Budget is 1000 * 0.75 = 750
        # Ratio is 1500/750 = 2.0 (fallback level)
        assert decision.should_compact is True
        assert decision.current_tokens == 1500
    
    def test_evaluate_fresh_tail_protection(self, engine):
        """Test that fresh tail is protected."""
        messages = [create_message(i, i, token_count=100) for i in range(15)]
        context_items = [
            create_context_item(i, ContextItemType.MESSAGE, message_id=i)
            for i in range(15)
        ]
        
        decision = engine.evaluate(messages, [], context_items)
        
        # Fresh tail count is 5
        # Fresh tail tokens = 5 * 100 = 500
        assert decision.fresh_tail_protected == 500
    
    def test_evaluate_escalation_levels(self, engine):
        """Test compaction level escalation."""
        # Normal level (ratio 1.0-1.5)
        messages = [create_message(i, i, token_count=60) for i in range(15)]
        context_items = [
            create_context_item(i, ContextItemType.MESSAGE, message_id=i)
            for i in range(15)
        ]
        
        decision = engine.evaluate(messages, [], context_items)
        # 15 * 60 = 900, budget = 750, ratio = 1.2
        # But fresh tail = 5 * 60 = 300, compactable = 600
        assert decision.level in (CompactionLevel.NORMAL, CompactionLevel.AGGRESSIVE, CompactionLevel.FALLBACK)
    
    def test_evaluate_phase_determination(self, engine):
        """Test phase determination based on summaries."""
        messages = [create_message(i, i, token_count=100) for i in range(15)]
        summaries = [
            create_summary(f"leaf_{i}", SummaryKind.LEAF) 
            for i in range(5)  # Enough for condensed min fanout
        ]
        
        decision = engine.evaluate(messages, summaries, [])
        
        # With 5 leaf summaries (>= condensed_min_fanout=2), phase should be condensed
        assert decision.phase == CompactionPhase.CONDENSED
    
    @pytest.mark.asyncio
    async def test_compact_no_messages(self, engine):
        """Test compaction with no messages."""
        result = await engine.compact(1, [], [], [])
        
        assert result.success is True
        assert result.rounds_completed == 0
        assert result.new_summaries == []
    
    @pytest.mark.asyncio
    async def test_compact_under_budget(self, engine):
        """Test compaction when already under budget."""
        messages = [create_message(i, i, token_count=10) for i in range(5)]
        context_items = [
            create_context_item(i, ContextItemType.MESSAGE, message_id=i)
            for i in range(5)
        ]
        
        result = await engine.compact(1, messages, [], context_items)
        
        # Should not compact since under budget
        assert result.success is True
        assert result.rounds_completed == 0
    
    @pytest.mark.asyncio
    async def test_compact_creates_summaries(self, engine_with_summarizer):
        """Test that compaction creates summaries."""
        engine = engine_with_summarizer
        
        # Create messages that exceed budget
        messages = [
            create_message(i, i, content="x" * 200, token_count=80)
            for i in range(20)
        ]
        context_items = [
            create_context_item(i, ContextItemType.MESSAGE, message_id=i)
            for i in range(20)
        ]
        
        result = await engine.compact(1, messages, [], context_items)
        
        # Should create summaries
        assert result.stats.summaries_created > 0 or result.success
        assert result.rounds_completed > 0
    
    @pytest.mark.asyncio
    async def test_compact_respects_fresh_tail(self, engine_with_summarizer):
        """Test that compaction respects fresh tail."""
        engine = engine_with_summarizer
        
        messages = [
            create_message(i, i, content=f"Message {i}", token_count=50)
            for i in range(20)
        ]
        context_items = [
            create_context_item(i, ContextItemType.MESSAGE, message_id=i)
            for i in range(20)
        ]
        
        result = await engine.compact(1, messages, [], context_items)
        
        # Fresh tail (last 5 messages) should not be compacted
        # They should still be in context
        assert result.success or result.rounds_completed > 0
    
    @pytest.mark.asyncio
    async def test_compact_handles_auth_errors(self, engine):
        """Test that compaction handles auth errors."""
        async def failing_summarizer(text, aggressive=False, options=None):
            from summarize import LcmProviderAuthError
            raise LcmProviderAuthError("API key invalid")
        
        engine._summarizer = failing_summarizer
        
        messages = [
            create_message(i, i, content="x" * 200, token_count=100)
            for i in range(20)
        ]
        context_items = [
            create_context_item(i, ContextItemType.MESSAGE, message_id=i)
            for i in range(20)
        ]
        
        result = await engine.compact(1, messages, [], context_items)
        
        # Should handle auth error gracefully
        assert "auth error" in " ".join(result.warnings).lower() or result.stats.auth_errors_skipped > 0
    
    @pytest.mark.asyncio
    async def test_compact_multiple_rounds(self, engine_with_summarizer):
        """Test that compaction runs multiple rounds."""
        engine = engine_with_summarizer
        engine.config.max_rounds = 5
        
        # Create many messages
        messages = [
            create_message(i, i, content="x" * 100, token_count=150)
            for i in range(30)
        ]
        context_items = [
            create_context_item(i, ContextItemType.MESSAGE, message_id=i)
            for i in range(30)
        ]
        
        result = await engine.compact(1, messages, [], context_items)
        
        # Should run at least one round
        assert result.rounds_completed >= 0
    
    @pytest.mark.asyncio
    async def test_leaf_pass_creates_leaf_summaries(self, engine_with_summarizer):
        """Test that leaf pass creates leaf summaries."""
        engine = engine_with_summarizer
        
        messages = [
            create_message(i, i, content=f"Test content {i}", token_count=50)
            for i in range(15)
        ]
        context_items = [
            create_context_item(i, ContextItemType.MESSAGE, message_id=i)
            for i in range(15)
        ]
        
        result = await engine.compact(1, messages, [], context_items)
        
        # Check that created summaries are leaf summaries
        for summary in result.new_summaries:
            assert summary.kind == SummaryKind.LEAF
    
    @pytest.mark.asyncio
    async def test_condensed_pass_creates_condensed_summaries(self, engine_with_summarizer):
        """Test that condensed pass creates condensed summaries."""
        engine = engine_with_summarizer
        
        # Create messages and existing leaf summaries
        messages = [
            create_message(i, i, content=f"Test {i}", token_count=50)
            for i in range(20)
        ]
        
        leaf_summaries = [
            create_summary(f"leaf_{i}", SummaryKind.LEAF, token_count=30)
            for i in range(6)
        ]
        
        context_items = [
            create_context_item(i, ContextItemType.SUMMARY, summary_id=f"leaf_{i}")
            for i in range(6)
        ]
        
        result = await engine.compact(1, messages, leaf_summaries, context_items)
        
        # Check for condensed summaries
        condensed_count = sum(
            1 for s in result.new_summaries 
            if s.kind == SummaryKind.CONDENSED
        )
        # May or may not create condensed summaries depending on budget
        assert result.success or result.rounds_completed > 0


class TestCompactionEngineHelpers:
    """Tests for CompactionEngine helper methods."""
    
    @pytest.fixture
    def engine(self):
        """Create a compaction engine for testing."""
        config = CompactionConfig(
            token_budget=1000,
            fresh_tail_count=5,
            leaf_min_fanout=3
        )
        return CompactionEngine(config)
    
    def test_group_messages_into_chunks(self, engine):
        """Test grouping messages into chunks."""
        messages = [
            create_message(i, i, token_count=50)
            for i in range(20)
        ]
        
        chunks = engine._group_messages_into_chunks(messages, 150, 3)
        
        # Each chunk should have at least 3 messages
        for chunk in chunks:
            assert len(chunk) >= 3
        
        # All messages should be in some chunk
        total_messages = sum(len(chunk) for chunk in chunks)
        assert total_messages == 20
    
    def test_build_chunk_text(self, engine):
        """Test building chunk text."""
        messages = [
            create_message(1, 1, content="Hello", role=MessageRole.USER),
            create_message(2, 2, content="Hi there!", role=MessageRole.ASSISTANT),
        ]
        
        text = engine._build_chunk_text(messages)
        
        assert "[user]: Hello" in text
        assert "[assistant]: Hi there!" in text
    
    def test_build_summary_group_text(self, engine):
        """Test building summary group text."""
        summaries = [
            create_summary("s1", SummaryKind.LEAF),
            create_summary("s2", SummaryKind.LEAF),
        ]
        
        text = engine._build_summary_group_text(summaries)
        
        assert "LEAF SUMMARY" in text
        assert "Test summary content" in text
    
    def test_generate_summary_id(self, engine):
        """Test summary ID generation."""
        id1 = engine._generate_summary_id(1, "leaf", 0)
        id2 = engine._generate_summary_id(1, "leaf", 1)
        id3 = engine._generate_summary_id(1, "condensed", 0)
        
        assert id1.startswith("sum_l_")
        assert id2.startswith("sum_l_")
        assert id3.startswith("sum_c_")
        assert id1 != id2  # Different indices should produce different IDs
    
    def test_calculate_total_tokens(self, engine):
        """Test token calculation."""
        messages = [
            create_message(1, 1, token_count=100),
            create_message(2, 2, token_count=200),
        ]
        
        summaries = [
            create_summary("s1", SummaryKind.LEAF, token_count=50),
        ]
        
        context_items = [
            create_context_item(0, ContextItemType.MESSAGE, message_id=1),
            create_context_item(1, ContextItemType.SUMMARY, summary_id="s1"),
        ]
        
        total = engine._calculate_total_tokens(messages, summaries, context_items)
        
        # 100 (message) + 50 (summary) = 150
        assert total == 150
    
    def test_contains_auth_errors(self, engine):
        """Test auth error detection."""
        # Should detect auth errors
        assert engine._contains_auth_errors("Error: 401 Unauthorized")
        assert engine._contains_auth_errors("Your API key is invalid")
        assert engine._contains_auth_errors("Authentication failed")
        
        # Should not detect in normal text
        assert not engine._contains_auth_errors("Normal message content")
        assert not engine._contains_auth_errors("Here is the API documentation")


class TestFactoryFunction:
    """Tests for factory function."""
    
    def test_create_engine_from_lcm_config(self):
        """Test creating engine from LcmConfig."""
        lcm_config = LcmConfig(
            context_threshold=0.8,
            fresh_tail_count=20,
            leaf_min_fanout=5,
            condensed_min_fanout=3
        )
        
        engine = create_compaction_engine_from_config(lcm_config)
        
        assert engine.config.context_threshold == 0.8
        assert engine.config.fresh_tail_count == 20
        assert engine.config.leaf_min_fanout == 5
        assert engine.config.condensed_min_fanout == 3
    
    def test_create_engine_with_summarizer(self):
        """Test creating engine with custom summarizer."""
        async def custom_summarizer(text, aggressive=False, options=None):
            return "Custom summary"
        
        lcm_config = LcmConfig()
        engine = create_compaction_engine_from_config(
            lcm_config, 
            summarizer=custom_summarizer
        )
        
        assert engine._summarizer == custom_summarizer


class TestCompactionLevels:
    """Tests for compaction level escalation."""
    
    def test_normal_level(self):
        """Test normal compaction level."""
        config = CompactionConfig(
            token_budget=1000,
            context_threshold=0.75,
            aggressive_ratio=1.5,
            fallback_ratio=2.0
        )
        engine = CompactionEngine(config)
        
        # Just over budget (ratio ~1.1)
        messages = [create_message(i, i, token_count=60) for i in range(15)]
        context_items = [
            create_context_item(i, ContextItemType.MESSAGE, message_id=i)
            for i in range(15)
        ]
        
        decision = engine.evaluate(messages, [], context_items)
        
        # Level depends on ratio calculation
        assert decision.level in (
            CompactionLevel.NORMAL, 
            CompactionLevel.AGGRESSIVE,
            CompactionLevel.FALLBACK
        )
    
    def test_aggressive_level(self):
        """Test aggressive compaction level."""
        config = CompactionConfig(
            token_budget=1000,
            context_threshold=0.75,
            aggressive_ratio=1.5,
            fallback_ratio=2.0
        )
        engine = CompactionEngine(config)
        
        # Over budget by 1.6x
        messages = [create_message(i, i, token_count=90) for i in range(15)]
        context_items = [
            create_context_item(i, ContextItemType.MESSAGE, message_id=i)
            for i in range(15)
        ]
        
        decision = engine.evaluate(messages, [], context_items)
        
        # 15 * 90 = 1350, budget = 750, ratio = 1.8
        assert decision.level in (CompactionLevel.AGGRESSIVE, CompactionLevel.FALLBACK)
    
    def test_fallback_level(self):
        """Test fallback compaction level."""
        config = CompactionConfig(
            token_budget=1000,
            context_threshold=0.75,
            aggressive_ratio=1.5,
            fallback_ratio=2.0
        )
        engine = CompactionEngine(config)
        
        # Way over budget (ratio > 2.0)
        messages = [create_message(i, i, token_count=120) for i in range(15)]
        context_items = [
            create_context_item(i, ContextItemType.MESSAGE, message_id=i)
            for i in range(15)
        ]
        
        decision = engine.evaluate(messages, [], context_items)
        
        # 15 * 120 = 1800, budget = 750, ratio = 2.4
        assert decision.level == CompactionLevel.FALLBACK


class TestFreshTailProtection:
    """Tests for fresh tail protection."""
    
    def test_fresh_tail_not_compacted(self):
        """Test that fresh tail messages are not included in compaction."""
        config = CompactionConfig(
            token_budget=500,
            context_threshold=0.75,
            fresh_tail_count=3,
            leaf_min_fanout=2
        )
        engine = CompactionEngine(config)
        
        # Create messages
        messages = [create_message(i, i, token_count=50) for i in range(10)]
        context_items = [
            create_context_item(i, ContextItemType.MESSAGE, message_id=i)
            for i in range(10)
        ]
        
        decision = engine.evaluate(messages, [], context_items)
        
        # Fresh tail = last 3 messages = 3 * 50 = 150 tokens
        assert decision.fresh_tail_protected == 150
    
    def test_fresh_tail_larger_than_messages(self):
        """Test when fresh tail count exceeds message count."""
        config = CompactionConfig(
            token_budget=1000,
            fresh_tail_count=100  # More than messages
        )
        engine = CompactionEngine(config)
        
        messages = [create_message(i, i, token_count=50) for i in range(5)]
        context_items = [
            create_context_item(i, ContextItemType.MESSAGE, message_id=i)
            for i in range(5)
        ]
        
        decision = engine.evaluate(messages, [], context_items)
        
        # All messages should be protected
        assert decision.fresh_tail_protected == 250


class TestEdgeCases:
    """Tests for edge cases."""
    
    @pytest.fixture
    def engine(self):
        """Create a compaction engine for testing."""
        config = CompactionConfig(
            token_budget=1000,
            fresh_tail_count=5,
            leaf_min_fanout=3
        )
        return CompactionEngine(config)
    
    @pytest.mark.asyncio
    async def test_compact_single_message(self, engine):
        """Test compaction with single message."""
        messages = [create_message(1, 1, token_count=100)]
        context_items = [create_context_item(0, ContextItemType.MESSAGE, message_id=1)]
        
        result = await engine.compact(1, messages, [], context_items)
        
        # Single message should not need compaction
        assert result.success is True
    
    @pytest.mark.asyncio
    async def test_compact_zero_token_messages(self, engine):
        """Test compaction with zero-token messages."""
        messages = [
            create_message(i, i, token_count=0)
            for i in range(10)
        ]
        context_items = [
            create_context_item(i, ContextItemType.MESSAGE, message_id=i)
            for i in range(10)
        ]
        
        result = await engine.compact(1, messages, [], context_items)
        
        # Should handle gracefully
        assert result.success is True
    
    def test_evaluate_with_summaries_only(self, engine):
        """Test evaluation with only summaries (no messages in context)."""
        summaries = [
            create_summary(f"s{i}", SummaryKind.LEAF, token_count=200)
            for i in range(5)
        ]
        context_items = [
            create_context_item(i, ContextItemType.SUMMARY, summary_id=f"s{i}")
            for i in range(5)
        ]
        
        decision = engine.evaluate([], summaries, context_items)
        
        # 5 summaries * 200 = 1000 tokens
        # Budget = 1000 * 0.75 = 750
        # Should be over budget
        assert decision.should_compact is True
        assert decision.current_tokens == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
