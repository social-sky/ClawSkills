#!/usr/bin/env python3
"""Tests for ContextAssembler.

Tests:
- Fresh tail protection
- Token budget constraints
- Summary inclusion
- System prompt handling
- Tool pairing repair integration
"""

import json
import pytest
from datetime import datetime
from dataclasses import dataclass

from typing import Dict, List, Any, Optional

import sys
import os

# Add parent directory to path
sys.path.insert(0, str(os.path.dirname(os.path.abspath(__file__))))


# Import the modules
from estimate_tokens import estimate_tokens
from transcript_repair import sanitize_tool_use_result_pairing
from lcm_types import MessageRole, ContextItemType
from lcm_types import (
    MessageRecord,
    SummaryRecord,
    ContextItemRecord,
    LargeFileRecord,
)
from db.config import LcmConfig, resolve_lcm_config
from assembler import (
    ContextAssembler,
    AssemblerConfig,
    create_assembler_from_config,
)


# Test fixtures
@pytest.fixture
def sample_messages():
    """Create sample messages for testing."""
    messages = []
    for i in range(10):
        messages.append(MessageRecord(
            message_id=i + 1,
            conversation_id=1,
            seq=i,
            role=MessageRole.USER,
            content=json.dumps([{"type": "text", "text": f"Message {i}"}]),
            token_count=20,
            created_at=datetime.utcnow()
        ))
    return messages


@pytest.fixture
def sample_summaries():
    """Create sample summaries for testing."""
    return [
        SummaryRecord(
            summary_id=f"sum_{i}",
            conversation_id=1,
            kind="leaf",
            depth=0,
            content=f"Summary of messages 0-{i}",
            token_count=15,
            created_at=datetime.utcnow()
        )
        for i in range(5, 10)
        ]


@pytest.fixture
def sample_context_items():
    """Create sample context items for testing."""
    return [
        ContextItemRecord(
            conversation_id=1,
            ordinal=i,
            item_type=ContextItemType.SUMMARY,
            summary_id=f"sum_{i}",
            created_at=datetime.utcnow()
        )
        for i in range(5, 10)
        ]


@pytest.fixture
def sample_messages_with_tools():
    """Create messages with tool calls for testing."""
    messages = []
    
    # Message with tool call
    tool_call_content = {
        "type": "tool_use",
        "id": "call_1",
        "name": "test_tool"
    }
    messages.append(MessageRecord(
        message_id=1,
        conversation_id=1,
        seq=1,
        role=MessageRole.ASSISTANT,
        content=json.dumps([tool_call_content]),
        token_count=20,
        created_at=datetime.utcnow()
    ))
    
    # Message with tool result
    tool_result_content = {
        "type": "tool_result",
        "tool_use_id": "call_1",
        "content": "Tool executed successfully"
    }
    messages.append(MessageRecord(
        message_id=2,
        conversation_id=1,
        seq=2,
        role=MessageRole.USER,
        content=json.dumps([tool_result_content]),
        token_count=15,
        created_at=datetime.utcnow()
    ))
    
    # Orphan tool result (should be removed by sanitizer)
    orphan_content = {
        "type": "tool_result",
        "tool_use_id": "orphan_call",
        "content": "Orphan result"
    }
    messages.append(MessageRecord(
        message_id=3,
        conversation_id=1,
        seq=3,
        role=MessageRole.USER,
        content=json.dumps([orphan_content]),
        token_count=10,
        created_at=datetime.utcnow()
    ))
    
    return messages


class TestEstimation:
    """Test token estimation functions."""
    
    def test_estimate_tokens_basic(self):
        assert estimate_tokens("hello world") == 3
        assert estimate_tokens("你好世界") >= 5
        
    def test_estimate_tokens_cjk(self):
        assert estimate_tokens("中文測試") >= 5
        assert estimate_tokens("日本語テスト") >= 5


class TestFreshTailProtection:
    """Test fresh tail protection."""
    
    def test_fresh_tail_basic(self, sample_messages):
        """Test that fresh tail is preserved always included."""
        config = AssemblerConfig(
            token_budget=1000,
            fresh_tail_count=3
        )
        assembler = ContextAssembler(config)
        
        result = assembler.assemble(
            messages=sample_messages,
            summaries=[],
            context_items=[],
        )
        
        # Should have fresh tail
        assert len(result.messages) == 3
        
        # Check that fresh tail messages are last
        fresh_roles = [msg["role"] for msg in result.messages]
        assert "user" in fresh_roles or "assistant" in fresh_roles
        
        # Should only have fresh tail
        assert len(result.messages) == 4


class TestMessageEnrichment:
    """Test message enrichment with parts."""
    
    def test_message_enrichment(self, sample_messages):
        """Test that messages are enriched with parts."""
        # Create mock parts function
        def get_parts(msg_id):
            if msg_id == 1:
                return [
                    type("mock_parts", []).append(
                        MessagePartRecord(
                            part_id="part_1",
                            message_id=1,
                            session_id="test",
                            part_type="text",
                            ordinal=0,
                            text_content="Additional context"
                        )
                    )
                ]
            return []
        
        config = AssemblerConfig(token_budget=1000, fresh_tail_count=5)
        assembler = ContextAssembler(config, get_message_parts=get_parts)
        
        result = assembler.assemble(
            messages=sample_messages,
            summaries=[],
            context_items=[]
        )
        
        # Check that parts were added to message
        if result.messages:
            assert "Additional context" in str(result.messages[0]["content"])


class TestEmptyContext:
    """Test with empty context."""
    
    def test_empty_context(self):
        """Test assembling with no messages or context."""
        config = AssemblerConfig()
        assembler = ContextAssembler(config)
        
        result = assembler.assemble(
            messages=[],
            summaries=[],
            context_items=[]
        )
        
        assert len(result.messages) == 0
        assert result.total_tokens == 0


if __name__ == "__main__":
    pytest.main([sys.argv[0]])
