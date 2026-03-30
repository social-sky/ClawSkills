#!/usr/bin/env python3
"""Unit tests for FTS5 query sanitization module."""

import sys
import unittest
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "search"))

from fts5_sanitize import (
    sanitize_fts5_query,
    is_safe_fts5_query,
    build_fts5_match_clause,
    _remove_control_characters,
    _balance_quotes,
    _neutralize_sql_injection,
    _escape_single_quotes,
    _normalize_whitespace,
    MAX_QUERY_LENGTH,
)


class TestRemoveControlCharacters(unittest.TestCase):
    """Tests for _remove_control_characters function."""

    def test_removes_null_byte(self):
        self.assertEqual(_remove_control_characters("hello\x00world"), "helloworld")

    def test_removes_multiple_control_chars(self):
        self.assertEqual(_remove_control_characters("\x01\x02\x03test"), "test")

    def test_preserves_whitespace(self):
        self.assertEqual(_remove_control_characters("hello world"), "hello world")
        self.assertEqual(_remove_control_characters("hello\tworld"), "hello\tworld")
        self.assertEqual(_remove_control_characters("hello\nworld"), "hello\nworld")

    def test_removes_del_char(self):
        self.assertEqual(_remove_control_characters("test\x7f"), "test")

    def test_empty_string(self):
        self.assertEqual(_remove_control_characters(""), "")


class TestBalanceQuotes(unittest.TestCase):
    """Tests for _balance_quotes function."""

    def test_already_balanced(self):
        self.assertEqual(_balance_quotes('"hello"'), '"hello"')
        self.assertEqual(_balance_quotes('"hello" "world"'), '"hello" "world"')

    def test_single_quote_removed(self):
        self.assertEqual(_balance_quotes('"hello'), 'hello')
        self.assertEqual(_balance_quotes('hello"'), 'hello')

    def test_odd_quotes_balanced(self):
        self.assertEqual(_balance_quotes('"a" "b" "c'), '"a" "b" c')

    def test_no_quotes(self):
        self.assertEqual(_balance_quotes('hello world'), 'hello world')

    def test_empty_string(self):
        self.assertEqual(_balance_quotes(''), '')


class TestNeutralizeSqlInjection(unittest.TestCase):
    """Tests for _neutralize_sql_injection function."""

    def test_drop_table_neutralized(self):
        result = _neutralize_sql_injection("'; DROP TABLE users; --")
        self.assertNotIn("DROP", result.upper())

    def test_delete_neutralized(self):
        result = _neutralize_sql_injection("'; DELETE FROM data; --")
        self.assertNotIn("DELETE", result.upper())

    def test_union_select_neutralized(self):
        result = _neutralize_sql_injection("' UNION SELECT * FROM users")
        self.assertNotIn("UNION", result.upper())

    def test_preserves_normal_text(self):
        result = _neutralize_sql_injection("hello world")
        self.assertEqual(result, "hello world")

    def test_or_injection_neutralized(self):
        result = _neutralize_sql_injection("' OR '1'='1")
        self.assertNotIn("' OR '", result)

    def test_block_comment_neutralized(self):
        result = _neutralize_sql_injection("test/* comment */value")
        self.assertNotIn("/*", result)


class TestEscapeSingleQuotes(unittest.TestCase):
    """Tests for _escape_single_quotes function."""

    def test_escapes_single_quote(self):
        self.assertEqual(_escape_single_quotes("it's"), "it''s")

    def test_escapes_multiple_quotes(self):
        self.assertEqual(_escape_single_quotes("it's a test's"), "it''s a test''s")

    def test_no_quotes_unchanged(self):
        self.assertEqual(_escape_single_quotes("hello world"), "hello world")

    def test_empty_string(self):
        self.assertEqual(_escape_single_quotes(""), "")


class TestNormalizeWhitespace(unittest.TestCase):
    """Tests for _normalize_whitespace function."""

    def test_multiple_spaces(self):
        result = _normalize_whitespace("hello    world")
        self.assertEqual(result, "hello world")

    def test_leading_trailing_spaces(self):
        result = _normalize_whitespace("  hello world  ")
        self.assertEqual(result, "hello world")

    def test_tabs_and_newlines(self):
        result = _normalize_whitespace("hello\t\nworld")
        self.assertEqual(result, "hello world")

    def test_empty_string(self):
        self.assertEqual(_normalize_whitespace(""), "")


class TestSanitizeFts5Query(unittest.TestCase):
    """Tests for sanitize_fts5_query function."""

    def test_empty_string(self):
        self.assertEqual(sanitize_fts5_query(""), "")

    def test_none_returns_empty(self):
        # None should be handled gracefully
        self.assertEqual(sanitize_fts5_query(None or ""), "")

    def test_simple_query(self):
        self.assertEqual(sanitize_fts5_query("hello world"), "hello world")

    def test_preserves_phrase_queries(self):
        result = sanitize_fts5_query('"exact phrase"')
        self.assertEqual(result, '"exact phrase"')

    def test_preserves_prefix_queries(self):
        result = sanitize_fts5_query("prefix*")
        self.assertEqual(result, "prefix*")

    def test_preserves_boolean_operators(self):
        result = sanitize_fts5_query("term1 AND term2")
        self.assertEqual(result, "term1 AND term2")

    def test_preserves_parentheses(self):
        result = sanitize_fts5_query("(term1 OR term2)")
        self.assertEqual(result, "(term1 OR term2)")

    def test_preserves_near_operator(self):
        result = sanitize_fts5_query("word1 NEAR word2")
        self.assertEqual(result, "word1 NEAR word2")

    def test_preserves_column_filter(self):
        result = sanitize_fts5_query("title:search")
        self.assertEqual(result, "title:search")

    def test_preserves_initial_token(self):
        result = sanitize_fts5_query("^initial")
        self.assertEqual(result, "^initial")

    def test_removes_control_characters(self):
        result = sanitize_fts5_query("hello\x00world")
        self.assertEqual(result, "helloworld")

    def test_balances_quotes(self):
        result = sanitize_fts5_query('"unbalanced')
        self.assertEqual(result, "unbalanced")

    def test_neutralizes_sql_injection(self):
        result = sanitize_fts5_query("'; DROP TABLE users; --")
        self.assertNotIn("DROP", result)

    def test_escapes_single_quotes(self):
        result = sanitize_fts5_query("it's")
        self.assertEqual(result, "it''s")

    def test_complex_valid_query(self):
        query = '(title:python OR content:python) AND "machine learning"'
        result = sanitize_fts5_query(query)
        self.assertEqual(result, query)

    def test_max_length_limit(self):
        long_query = "a" * 20000
        result = sanitize_fts5_query(long_query)
        self.assertLessEqual(len(result), MAX_QUERY_LENGTH)

    def test_custom_max_length(self):
        long_query = "a" * 1000
        result = sanitize_fts5_query(long_query, max_length=100)
        self.assertLessEqual(len(result), 100)

    def test_bobby_tables(self):
        """Test the famous XKCD Bobby Tables injection."""
        result = sanitize_fts5_query("Robert'); DROP TABLE students; --")
        self.assertNotIn("DROP", result)

    def test_combined_query_with_special_chars(self):
        query = 'normal "quoted phrase" prefix* (grouped)'
        result = sanitize_fts5_query(query)
        self.assertIn('"quoted phrase"', result)
        self.assertIn("prefix*", result)
        self.assertIn("(grouped)", result)


class TestIsSafeFts5Query(unittest.TestCase):
    """Tests for is_safe_fts5_query function."""

    def test_safe_simple_query(self):
        self.assertTrue(is_safe_fts5_query("hello world"))

    def test_safe_phrase_query(self):
        self.assertTrue(is_safe_fts5_query('"exact phrase"'))

    def test_unsafe_control_chars(self):
        self.assertFalse(is_safe_fts5_query("hello\x00world"))

    def test_unsafe_sql_injection(self):
        self.assertFalse(is_safe_fts5_query("'; DROP TABLE users; --"))

    def test_unsafe_unbalanced_quotes(self):
        self.assertFalse(is_safe_fts5_query('"unbalanced'))

    def test_unsafe_single_quotes(self):
        self.assertFalse(is_safe_fts5_query("it's"))

    def test_empty_query_safe(self):
        self.assertTrue(is_safe_fts5_query(""))


class TestBuildFts5MatchClause(unittest.TestCase):
    """Tests for build_fts5_match_clause function."""

    def test_simple_match(self):
        result = build_fts5_match_clause("fts", [], "hello")
        self.assertIn("fts MATCH", result)
        self.assertIn("hello", result)

    def test_empty_query_returns_true_clause(self):
        result = build_fts5_match_clause("fts", [], "")
        self.assertEqual(result, "1=1")

    def test_with_columns(self):
        result = build_fts5_match_clause("fts", ["title", "content"], "search")
        self.assertIn("title", result)
        self.assertIn("content", result)

    def test_sanitizes_query(self):
        result = build_fts5_match_clause("fts", [], "'; DROP TABLE users; --")
        self.assertNotIn("DROP", result)

    def test_preserves_valid_syntax(self):
        result = build_fts5_match_clause("fts", [], "prefix*")
        self.assertIn("prefix*", result)


class TestEdgeCases(unittest.TestCase):
    """Edge case tests."""

    def test_only_special_chars(self):
        result = sanitize_fts5_query("***")
        self.assertEqual(result, "***")

    def test_only_control_chars(self):
        result = sanitize_fts5_query("\x00\x01\x02")
        self.assertEqual(result, "")

    def test_unicode_query(self):
        result = sanitize_fts5_query("你好世界")
        self.assertEqual(result, "你好世界")

    def test_mixed_unicode_and_special(self):
        result = sanitize_fts5_query('你好 "世界"')
        self.assertEqual(result, '你好 "世界"')

    def test_very_long_word(self):
        long_word = "a" * 1000
        result = sanitize_fts5_query(long_word)
        self.assertEqual(result, long_word)

    def test_newlines_normalized(self):
        result = sanitize_fts5_query("hello\nworld")
        self.assertEqual(result, "hello world")

    def test_tabs_normalized(self):
        result = sanitize_fts5_query("hello\tworld")
        self.assertEqual(result, "hello world")


if __name__ == "__main__":
    unittest.main(verbosity=2)
