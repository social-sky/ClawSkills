#!/usr/bin/env python3
"""Unit tests for full_text_fallback.py."""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from search.full_text_fallback import (
    LikeSearchPlan,
    contains_cjk,
    build_like_search_plan,
    create_fallback_snippet,
    estimate_cjk_ratio,
    should_use_fallback,
    _extract_cjk_terms,
    _extract_non_cjk_terms
)


class TestContainsCjk:
    """Tests for contains_cjk function."""

    def test_empty_string(self):
        """Empty string should return False."""
        assert contains_cjk("") is False

    def test_none_input(self):
        """None should return False."""
        assert contains_cjk(None) is False

    def test_ascii_only(self):
        """ASCII-only text should return False."""
        assert contains_cjk("Hello World") is False
        assert contains_cjk("python programming 123") is False

    def test_chinese_characters(self):
        """Chinese characters should be detected."""
        assert contains_cjk("中文") is True
        assert contains_cjk("繁體字") is True
        assert contains_cjk("简体字") is True

    def test_japanese_characters(self):
        """Japanese characters should be detected."""
        assert contains_cjk("日本語") is True  # Kanji
        assert contains_cjk("ひらがな") is True  # Hiragana
        assert contains_cjk("カタカナ") is True  # Katakana

    def test_korean_characters(self):
        """Korean characters should be detected."""
        assert contains_cjk("한국어") is True
        assert contains_cjk("조선말") is True

    def test_mixed_content(self):
        """Mixed content with CJK should return True."""
        assert contains_cjk("Hello 你好 World") is True
        assert contains_cjk("Python 學習") is True
        assert contains_cjk("Learn 日本語 now") is True

    def test_single_cjk_char(self):
        """Single CJK character should be detected."""
        assert contains_cjk("中") is True
        assert contains_cjk("あ") is True
        assert contains_cjk("한") is True


class TestExtractCjkTerms:
    """Tests for _extract_cjk_terms function."""

    def test_empty_string(self):
        """Empty string should return empty list."""
        assert _extract_cjk_terms("") == []

    def test_no_cjk(self):
        """Non-CJK text should return empty list."""
        assert _extract_cjk_terms("Hello World") == []

    def test_single_cjk_term(self):
        """Single CJK phrase should be extracted."""
        assert _extract_cjk_terms("中文") == ["中文"]

    def test_multiple_cjk_terms(self):
        """Multiple CJK terms separated by non-CJK should be extracted."""
        result = _extract_cjk_terms("學習 python 程式設計")
        assert "學習" in result
        assert "程式設計" in result

    def test_min_length_filter(self):
        """Minimum length should filter short terms."""
        result = _extract_cjk_terms("中文字", min_length=2)
        assert "中文字" in result
        
        result = _extract_cjk_terms("中", min_length=2)
        assert result == []


class TestExtractNonCjkTerms:
    """Tests for _extract_non_cjk_terms function."""

    def test_empty_string(self):
        """Empty string should return empty list."""
        assert _extract_non_cjk_terms("") == []

    def test_only_cjk(self):
        """CJK-only text should return empty list."""
        assert _extract_non_cjk_terms("中文測試") == []

    def test_single_word(self):
        """Single word should be extracted."""
        assert _extract_non_cjk_terms("python") == ["python"]

    def test_multiple_words(self):
        """Multiple words should be extracted."""
        result = _extract_non_cjk_terms("python programming")
        assert "python" in result
        assert "programming" in result

    def test_min_length_filter(self):
        """Short words should be filtered."""
        result = _extract_non_cjk_terms("a an the python", min_length=2)
        assert "python" in result
        assert "a" not in result


class TestBuildLikeSearchPlan:
    """Tests for build_like_search_plan function."""

    def test_empty_query(self):
        """Empty query should return empty plan."""
        plan = build_like_search_plan("content", "")
        assert plan.terms == []
        assert plan.where == ""
        assert plan.args == []

    def test_whitespace_query(self):
        """Whitespace-only query should return empty plan."""
        plan = build_like_search_plan("content", "   ")
        assert plan.terms == []

    def test_ascii_query(self):
        """ASCII query should create proper LIKE plan."""
        plan = build_like_search_plan("content", "python")
        assert "python" in plan.terms
        assert "LIKE" in plan.where
        assert len(plan.args) == 1
        assert "%python%" in plan.args

    def test_cjk_query(self):
        """CJK query should create proper LIKE plan."""
        plan = build_like_search_plan("content", "中文")
        assert "中文" in plan.terms
        assert "LIKE" in plan.where
        assert "%中文%" in plan.args

    def test_mixed_query(self):
        """Mixed CJK and ASCII query should extract both."""
        plan = build_like_search_plan("content", "python 學習")
        assert "python" in plan.terms
        assert "學習" in plan.terms
        assert len(plan.args) == 2

    def test_or_mode(self):
        """OR mode should use OR between conditions."""
        plan = build_like_search_plan("content", "python 學習", use_or=True)
        assert " OR " in plan.where

    def test_and_mode(self):
        """AND mode should use AND between conditions."""
        plan = build_like_search_plan("content", "python 學習", use_or=False)
        assert " AND " in plan.where

    def test_like_escaping(self):
        """Special LIKE characters should be escaped."""
        plan = build_like_search_plan("content", "100% test_data")
        # % and _ should be escaped
        assert "\\%" in plan.args[0]
        assert "\\_" in plan.args[1]

    def test_custom_column(self):
        """Custom column name should be used."""
        plan = build_like_search_plan("title", "test")
        assert "title" in plan.where


class TestCreateFallbackSnippet:
    """Tests for create_fallback_snippet function."""

    def test_empty_content(self):
        """Empty content should return empty string."""
        assert create_fallback_snippet("", ["test"]) == ""

    def test_empty_terms(self):
        """Empty terms should return empty string."""
        assert create_fallback_snippet("content", []) == ""

    def test_no_match(self):
        """No match should return empty string."""
        assert create_fallback_snippet("Hello World", ["xyz"]) == ""

    def test_simple_match(self):
        """Simple match should highlight term."""
        result = create_fallback_snippet(
            "Hello World",
            ["World"],
            max_len=200
        )
        assert "**World**" in result

    def test_cjk_match(self):
        """CJK match should highlight term."""
        result = create_fallback_snippet(
            "這是一段關於程式設計的文章",
            ["程式"],
            max_len=100
        )
        assert "**程式**" in result

    def test_case_insensitive_match(self):
        """Match should be case-insensitive."""
        result = create_fallback_snippet(
            "Python Programming",
            ["python"],
            max_len=100
        )
        assert "**Python**" in result

    def test_multiple_terms(self):
        """Multiple terms should all be highlighted."""
        result = create_fallback_snippet(
            "Python is great for programming",
            ["python", "great"],
            max_len=200
        )
        assert "**Python**" in result
        assert "**great**" in result

    def test_truncation_with_ellipsis(self):
        """Long content should be truncated with ellipsis."""
        long_content = "A" * 100 + "TARGET" + "B" * 100
        result = create_fallback_snippet(
            long_content,
            ["TARGET"],
            max_len=50
        )
        assert "..." in result
        assert "**TARGET**" in result

    def test_custom_marker(self):
        """Custom marker should be used."""
        result = create_fallback_snippet(
            "Hello World",
            ["World"],
            marker="[["
        )
        assert "[[World]]" in result

    def test_context_window(self):
        """Context window should include surrounding text."""
        content = "prefix " + "X" * 100 + " TARGET " + "Y" * 100 + " suffix"
        result = create_fallback_snippet(
            content,
            ["TARGET"],
            context_chars=20,
            max_len=100
        )
        # Should include some context before/after TARGET
        assert "**TARGET**" in result


class TestEstimateCjkRatio:
    """Tests for estimate_cjk_ratio function."""

    def test_empty_string(self):
        """Empty string should return 0.0."""
        assert estimate_cjk_ratio("") == 0.0

    def test_no_cjk(self):
        """Non-CJK text should return 0.0."""
        assert estimate_cjk_ratio("Hello World") == 0.0

    def test_all_cjk(self):
        """All CJK text should return 1.0."""
        assert estimate_cjk_ratio("中文測試") == 1.0

    def test_mixed_content(self):
        """Mixed content should calculate correct ratio."""
        # "你好" = 2 CJK, "Hello" = 5 ASCII
        ratio = estimate_cjk_ratio("Hello你好")
        assert 0.0 < ratio < 1.0
        assert abs(ratio - 2/7) < 0.01  # 2 CJK out of 7 total


class TestShouldUseFallback:
    """Tests for should_use_fallback function."""

    def test_ascii_only(self):
        """ASCII-only should not use fallback."""
        assert should_use_fallback("python programming") is False

    def test_cjk_detected(self):
        """CJK text should use fallback."""
        assert should_use_fallback("中文") is True

    def test_custom_threshold(self):
        """Custom threshold should affect decision."""
        # 2 CJK out of 10 = 0.2 ratio
        text = "Hello你好World"
        assert should_use_fallback(text, threshold=0.1) is True
        assert should_use_fallback(text, threshold=0.5) is False


class TestLikeSearchPlanDataclass:
    """Tests for LikeSearchPlan dataclass."""

    def test_default_values(self):
        """Default values should be empty."""
        plan = LikeSearchPlan()
        assert plan.terms == []
        assert plan.where == ""
        assert plan.args == []

    def test_custom_values(self):
        """Custom values should be stored."""
        plan = LikeSearchPlan(
            terms=["test"],
            where="content LIKE ?",
            args=["%test%"]
        )
        assert plan.terms == ["test"]
        assert plan.where == "content LIKE ?"
        assert plan.args == ["%test%"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
