#!/usr/bin/env python3
"""Tests for Enhanced Noise Filter with CJK and multilingual support.

Tests:
- CJK refusal/greeting/meta patterns
- Multilingual patterns (Spanish, French, German)
- Memory retrieval priority
- Adaptive retrieval triggers
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from noise_filter import (
    # Core functions
    is_greeting,
    is_meta_question,
    is_refusal,
    is_simple_confirmation,
    is_slash_command,
    is_cjk_content,
    should_force_memory_retrieval,
    is_noise_content,
    filter_noise_content,
    # Priority
    get_memory_retrieval_priority,
    # Constants
    CJK_REFUSAL_PATTERNS,
    CJK_GREETING_PATTERNS,
    CJK_META_PATTERNS,
    STRONG_MEMORY_TRIGGERS,
    ALL_REFUSAL_PATTERNS,
    ALL_GREETING_PATTERNS,
    ALL_META_PATTERNS,
)

# CJK-specific functions use the same is_* functions with CJK text
# The core functions (is_refusal, is_greeting, is_meta_question) 
# already handle CJK via ALL_*_PATTERNS which include CJK patterns
is_cjk_refusal = is_refusal
is_cjk_greeting = is_greeting
is_cjk_meta_question = is_meta_question


class TestCJKRefusal:
    """Tests for CJK refusal patterns."""
    
    @pytest.mark.parametrize("text,expected", [
        # Chinese
        ("我不知道", True),
        ("无法提供", True),
        ("抱歉，我不能告诉你", True),
        ("对不起，这信息无法获取", True),
        # Japanese
        ("すみません、できません", True),
        ("申し訳ありませんが", True),
        ("お答えできません", True),
        # Korean
        ("모르겠습니다", True),
        ("제공할 수 없습니다", True),
        ("죄송합니다", True),
        # Non-refusal
        ("我想知道更多信息", False),
        ("可以帮助我吗", False),
    ])
    def test_cjk_refusal(self, text, expected):
        """Test CJK refusal detection."""
        assert is_cjk_refusal(text) == expected
    
    def test_combined_refusal_includes_cjk(self):
        """Test that ALL_REFUSAL_PATTERNS includes CJK."""
        assert len(ALL_REFUSAL_PATTERNS) > len(CJK_REFUSAL_PATTERNS)


class TestCJKGreeting:
    """Tests for CJK greeting patterns."""
    
    @pytest.mark.parametrize("text,expected", [
        # Chinese
        ("你好", True),
        ("您好", True),
        ("嗨，你好", True),
        ("早上好", True),
        ("下午好", True),
        ("晚上好", True),
        # Japanese
        ("こんにちは", True),
        ("こんばんは", True),
        ("おはよう", True),
        # Korean
        ("안녕하세요", True),
        ("안녕", True),
        # Non-greeting
        ("今天天气很好", False),
        ("学习Python很有趣", False),
    ])
    def test_cjk_greeting(self, text, expected):
        """Test CJK greeting detection."""
        assert is_cjk_greeting(text) == expected


class TestCJKMetaQuestion:
    """Tests for CJK meta question patterns."""
    
    @pytest.mark.parametrize("text,expected", [
        # Chinese
        ("你还记得吗", True),
        ("你还记得上次我们讨论的内容吗", True),
        ("我记得之前你说的话", True),
        # Japanese
        ("覚えてますか", True),
        ("以前言ったよね", True),
        # Korean
        ("기억해요?", True),
        # Non-meta
        ("今天天气怎么样", False),
        ("你想吃什么", False),
    ])
    def test_cjk_meta_question(self, text, expected):
        """Test CJK meta question detection."""
        assert is_cjk_meta_question(text) == expected


class TestMultilingualRefusal:
    """Tests for multilingual refusal patterns."""
    
    @pytest.mark.parametrize("text,lang,expected", [
        # Spanish
        ("no lo sé", "es", True),
        ("lo siento, no puedo", "es", True),
        ("no tengo la información", "es", True),
        # French
        ("je ne sais pas", "fr", True),
        ("je suis désolé", "fr", True),
        ("je ne peux pas", "fr", True),
        # German
        ("ich weiß es nicht", "de", True),
        ("es tut mir leid", "de", True),
        ("ich kann nicht", "de", True),
    ])
    def test_multilingual_refusal(self, text, lang, expected):
        """Test multilingual refusal detection."""
        result = is_refusal(text)
        assert result == expected, f"Failed for {lang}: {text}"


class TestMultilingualGreeting:
    """Tests for multilingual greeting patterns."""
    
    @pytest.mark.parametrize("text,expected", [
        # Spanish
        ("hola", True),
        ("buenos días", True),
        ("buenas tardes", True),
        # French
        ("bonjour", True),
        ("salut", True),
        ("bonsoir", True),
        # German
        ("hallo", True),
        ("guten morgen", True),
        ("guten tag", True),
    ])
    def test_multilingual_greeting(self, text, expected):
        """Test multilingual greeting detection."""
        assert is_greeting(text) == expected


class TestMemoryRetrievalPriority:
    """Tests for memory retrieval priority scoring."""
    
    @pytest.mark.parametrize("text,min_priority", [
        # Strong triggers - priority 2
        ("Don't forget what I said earlier", 2),
        ("Please remember to check the docs", 2),
        ("As we discussed before", 2),
        ("别忘了", 2),
        ("请记住", 2),
        # Regular triggers - priority 1
        ("Do you remember this?", 1),
        ("Previously we talked about", 1),
        ("Earlier I mentioned", 1),
        ("你还记得吗", 1),
        # Non-triggering - priority 0
        ("Tell me about Python", 0),
        ("What is machine learning?", 0),
        ("今天天气很好", 0),
    ])
    def test_memory_priority(self, text, min_priority):
        """Test memory retrieval priority scoring."""
        priority = get_memory_retrieval_priority(text)
        assert priority >= min_priority, f"Text: {text}, Expected >= {min_priority}, Got {priority}"
    
    def test_strong_triggers_higher_priority(self):
        """Test that strong triggers have higher priority than regular."""
        strong_text = "Don't forget what I said"
        regular_text = "Do you remember this?"
        
        strong_priority = get_memory_retrieval_priority(strong_text)
        regular_priority = get_memory_retrieval_priority(regular_text)
        
        assert strong_priority > regular_priority


class TestShouldForceMemoryRetrieval:
    """Tests for should_force_memory_retrieval function."""
    
    @pytest.mark.parametrize("text,expected", [
        # English triggers
        ("Do you remember the project?", True),
        ("As I mentioned earlier...", True),
        ("Please remember this", True),
        ("Recall the meeting notes", True),
        # Short text below min_length
        ("Remember", False),
        ("OK", False),
        # Non-triggering
        ("Tell me about your day", False),
        ("What's the weather?", False),
    ])
    def test_should_force_memory_retrieval(self, text, expected):
        """Test should_force_memory_retrieval detection."""
        assert should_force_memory_retrieval(text) == expected


class TestCJKContentDetection:
    """Tests for CJK content detection."""
    
    @pytest.mark.parametrize("text,expected", [
        ("你好世界", True),
        ("日本語テスト", True),
        ("안녕하세요", True),
        ("hello world", False),
        ("123456", False),
    ])
    def test_is_cjk_content(self, text, expected):
        """Test CJK content detection."""
        assert is_cjk_content(text) == expected


class TestCombinedPatterns:
    """Tests for combined pattern matching."""
    
    def test_combined_greeting_includes_all(self):
        """Test ALL_GREETING_PATTERNS combines English + CJK + multilingual."""
        assert len(ALL_GREETING_PATTERNS) >= len(CJK_GREETING_PATTERNS)
        assert len(ALL_GREETING_PATTERNS) >= len([
            r"^(hi|hello|hey|greetings|howdy|yo)\b",
        ])
    
    def test_combined_meta_includes_cjk(self):
        """Test ALL_META_PATTERNS includes CJK."""
        assert len(ALL_META_PATTERNS) >= len(CJK_META_PATTERNS)


class TestNoiseContent:
    """Tests for the main is_noise_content function."""
    
    @pytest.mark.parametrize("text,is_assistant,expected", [
        # Greetings (non-assistant)
        ("Hello!", False, True),
        ("你好", False, True),
        # Refusals (any)
        ("I don't know", False, True),
        ("I cannot help", True, True),
        ("我不知道", False, True),
        # Meta questions
        ("Do you remember?", False, True),
        ("你还记得吗", False, True),
        # Simple confirmations (non-assistant)
        ("yes", False, True),
        ("ok", False, True),
        # Slash commands
        ("/help", False, True),
        # Non-noise
        ("Tell me about Python", False, False),
        ("What is machine learning?", False, False),
        ("How do I write a Python function?", False, False),
    ])
    def test_is_noise_content(self, text, is_assistant, expected):
        """Test comprehensive noise content detection."""
        is_noise, reason = is_noise_content(text, is_assistant)
        assert is_noise == expected, f"Text: {text}, Reason: {reason}"


class TestFilterNoiseContent:
    """Tests for filter_noise_content function."""
    
    @pytest.mark.parametrize("text,expected", [
        # Should filter
        ("Hello!", ""),
        ("I don't know", ""),
        ("/help", ""),
        ("yes", ""),
        # Should pass through
        ("Tell me about Python", "Tell me about Python"),
        ("What is the meaning of life?", "What is the meaning of life?"),
    ])
    def test_filter_noise_content(self, text, expected):
        """Test noise content filtering."""
        result = filter_noise_content(text)
        assert result == expected


class TestCJKIntegration:
    """Integration tests for CJK noise filtering."""
    
    def test_is_greeting_multilingual(self):
        """Test is_greeting works for multiple languages."""
        # English
        assert is_greeting("Hello there!")
        # CJK
        assert is_greeting("你好")
        assert is_greeting("안녕하세요")
    
    def test_is_refusal_multilingual(self):
        """Test is_refusal works for multiple languages."""
        # English
        assert is_refusal("I don't know")
        # CJK
        assert is_refusal("我不知道")
        assert is_refusal("すみません")
    
    def test_noise_content_cjk(self):
        """Test comprehensive noise detection including CJK."""
        # CJK greeting
        is_noise, reason = is_noise_content("你好")
        assert is_noise
        assert "greeting" in reason.lower() if reason else True
        
        # CJK refusal
        is_noise, reason = is_noise_content("抱歉，我不能告诉你")
        assert is_noise
        assert "refusal" in reason.lower() if reason else True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
