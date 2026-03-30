#!/usr/bin/env python3
"""Test suite for estimate_tokens module.

Tests CJK-aware token estimation with comprehensive coverage:
- ASCII text
- CJK characters (Chinese, Japanese, Korean)
- Emoji and supplementary characters
- Mixed content
- Edge cases
"""

import sys
import unittest
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from estimate_tokens import (
    estimate_tokens,
    estimate_tokens_precise,
    is_cjk_char,
    is_supplementary_char,
)


class TestIsCjkChar(unittest.TestCase):
    """Test CJK character detection."""

    def test_chinese_characters(self):
        """Test Chinese characters are detected as CJK."""
        self.assertTrue(is_cjk_char('你'))
        self.assertTrue(is_cjk_char('好'))
        self.assertTrue(is_cjk_char('世'))
        self.assertTrue(is_cjk_char('界'))

    def test_japanese_hiragana(self):
        """Test Japanese Hiragana are detected as CJK."""
        self.assertTrue(is_cjk_char('あ'))
        self.assertTrue(is_cjk_char('い'))
        self.assertTrue(is_cjk_char('う'))

    def test_japanese_katakana(self):
        """Test Japanese Katakana are detected as CJK."""
        self.assertTrue(is_cjk_char('ア'))
        self.assertTrue(is_cjk_char('イ'))
        self.assertTrue(is_cjk_char('ウ'))

    def test_korean_hangul(self):
        """Test Korean Hangul are detected as CJK."""
        self.assertTrue(is_cjk_char('한'))
        self.assertTrue(is_cjk_char('국'))
        self.assertTrue(is_cjk_char('어'))

    def test_ascii_not_cjk(self):
        """Test ASCII characters are not CJK."""
        self.assertFalse(is_cjk_char('a'))
        self.assertFalse(is_cjk_char('Z'))
        self.assertFalse(is_cjk_char('0'))
        self.assertFalse(is_cjk_char(' '))


class TestIsSupplementaryChar(unittest.TestCase):
    """Test supplementary character detection."""

    def test_emoji_are_supplementary(self):
        """Test emoji are supplementary characters."""
        self.assertTrue(is_supplementary_char('🎉'))
        self.assertTrue(is_supplementary_char('🎊'))
        self.assertTrue(is_supplementary_char('🎁'))

    def test_ascii_not_supplementary(self):
        """Test ASCII are not supplementary."""
        self.assertFalse(is_supplementary_char('a'))
        self.assertFalse(is_supplementary_char('Z'))

    def test_cjk_not_supplementary(self):
        """Test basic CJK are not supplementary."""
        self.assertFalse(is_supplementary_char('你'))
        self.assertFalse(is_supplementary_char('あ'))


class TestEstimateTokens(unittest.TestCase):
    """Test token estimation."""

    def test_empty_string(self):
        """Test empty string returns 0 tokens."""
        self.assertEqual(estimate_tokens(""), 0)

    def test_none_input(self):
        """Test None input returns 0 tokens."""
        self.assertEqual(estimate_tokens(None), 0)

    def test_ascii_only(self):
        """Test ASCII text estimation (0.25 tokens/char)."""
        # "hello" = 5 chars * 0.25 = 1.25 -> 2 tokens (rounded up)
        self.assertEqual(estimate_tokens("hello"), 2)
        # "hi" = 2 chars * 0.25 = 0.5 -> 1 token
        self.assertEqual(estimate_tokens("hi"), 1)
        # "test" = 4 chars * 0.25 = 1.0 -> 1 token
        self.assertEqual(estimate_tokens("test"), 1)

    def test_ascii_longer_text(self):
        """Test longer ASCII text."""
        # "Hello, World!" = 13 chars * 0.25 = 3.25 -> 4 tokens
        self.assertEqual(estimate_tokens("Hello, World!"), 4)

    def test_chinese_only(self):
        """Test Chinese text estimation (1.5 tokens/char)."""
        # "你好" = 2 chars * 1.5 = 3 tokens
        self.assertEqual(estimate_tokens("你好"), 3)
        # "你好世界" = 4 chars * 1.5 = 6 tokens
        self.assertEqual(estimate_tokens("你好世界"), 6)

    def test_japanese_only(self):
        """Test Japanese text estimation (1.5 tokens/char)."""
        # "こんにちは" = 5 chars * 1.5 = 7.5 -> 8 tokens
        self.assertEqual(estimate_tokens("こんにちは"), 8)

    def test_korean_only(self):
        """Test Korean text estimation (1.5 tokens/char)."""
        # "안녕하세요" = 5 chars * 1.5 = 7.5 -> 8 tokens
        self.assertEqual(estimate_tokens("안녕하세요"), 8)

    def test_emoji_only(self):
        """Test emoji estimation (2.0 tokens/char)."""
        # "🎉" = 1 char * 2.0 = 2 tokens
        self.assertEqual(estimate_tokens("🎉"), 2)
        # "🎉🎊🎁" = 3 chars * 2.0 = 6 tokens
        self.assertEqual(estimate_tokens("🎉🎊🎁"), 6)

    def test_mixed_ascii_and_chinese(self):
        """Test mixed ASCII and Chinese text."""
        # "Hello 你好" = "Hello " (6*0.25=1.5) + "你好" (2*1.5=3) = 4.5 -> 5 tokens
        self.assertEqual(estimate_tokens("Hello 你好"), 5)

    def test_mixed_all_types(self):
        """Test mixed ASCII, CJK, and emoji."""
        # "Hi 你好 🎉" = "Hi " (3*0.25=0.75) + "你好" (2*1.5=3) + " " (1*0.25=0.25) + "🎉" (1*2.0=2)
        # = 0.75 + 3 + 0.25 + 2 = 6.0 -> 6 tokens
        self.assertEqual(estimate_tokens("Hi 你好 🎉"), 6)


class TestEstimateTokensPrecise(unittest.TestCase):
    """Test precise token estimation."""

    def test_ascii_precise(self):
        """Test ASCII precise estimation."""
        # "hello" = 5 * 0.25 = 1.25
        self.assertAlmostEqual(estimate_tokens_precise("hello"), 1.25)

    def test_chinese_precise(self):
        """Test Chinese precise estimation."""
        # "你好" = 2 * 1.5 = 3.0
        self.assertAlmostEqual(estimate_tokens_precise("你好"), 3.0)

    def test_emoji_precise(self):
        """Test emoji precise estimation."""
        # "🎉" = 1 * 2.0 = 2.0
        self.assertAlmostEqual(estimate_tokens_precise("🎉"), 2.0)

    def test_empty_returns_zero(self):
        """Test empty string returns 0.0."""
        self.assertEqual(estimate_tokens_precise(""), 0.0)
        self.assertEqual(estimate_tokens_precise(None), 0.0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and special scenarios."""

    def test_whitespace_only(self):
        """Test whitespace-only text."""
        # 4 spaces = 4 * 0.25 = 1.0 -> 1 token
        self.assertEqual(estimate_tokens("    "), 1)

    def test_newlines(self):
        """Test newlines in text."""
        # "\n\n" = 2 * 0.25 = 0.5 -> 1 token
        self.assertEqual(estimate_tokens("\n\n"), 1)

    def test_numbers(self):
        """Test numeric characters."""
        # "12345" = 5 * 0.25 = 1.25 -> 2 tokens
        self.assertEqual(estimate_tokens("12345"), 2)

    def test_special_ascii_chars(self):
        """Test special ASCII characters."""
        # "!@#$%" = 5 * 0.25 = 1.25 -> 2 tokens
        self.assertEqual(estimate_tokens("!@#$%"), 2)

    def test_single_char_ascii(self):
        """Test single ASCII character."""
        # "a" = 1 * 0.25 = 0.25 -> 1 token
        self.assertEqual(estimate_tokens("a"), 1)

    def test_single_char_cjk(self):
        """Test single CJK character."""
        # "你" = 1 * 1.5 = 1.5 -> 2 tokens
        self.assertEqual(estimate_tokens("你"), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
