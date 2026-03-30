#!/usr/bin/env python3
"""Tests for session_patterns module."""

import re
import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from session_patterns import compile_session_patterns, matches_session_pattern


class TestCompileSessionPatterns:
    """Tests for compile_session_patterns function."""
    
    def test_empty_patterns(self):
        """Empty pattern list should return empty list."""
        result = compile_session_patterns([])
        assert result == []
    
    def test_single_exact_pattern(self):
        """Single exact match pattern should compile correctly."""
        patterns = compile_session_patterns(["agent:main"])
        assert len(patterns) == 1
        assert isinstance(patterns[0], type(re.compile("")))
    
    def test_multiple_patterns(self):
        """Multiple patterns should compile to multiple regexes."""
        patterns = compile_session_patterns(["agent:*", "sisyphus:*"])
        assert len(patterns) == 2
    
    def test_wildcard_pattern(self):
        """Wildcard * should compile to match non-colon characters."""
        patterns = compile_session_patterns(["agent:*"])
        assert len(patterns) == 1
        # Should match "agent:main" but not "agent:main:sub"
        assert patterns[0].match("agent:main") is not None
        assert patterns[0].match("agent:123") is not None
    
    def test_double_wildcard(self):
        """Double wildcard should compile correctly."""
        patterns = compile_session_patterns(["*:*"])
        assert len(patterns) == 1


class TestMatchesSessionPattern:
    """Tests for matches_session_pattern function."""
    
    def test_empty_patterns_no_match(self):
        """Empty pattern list should never match."""
        assert not matches_session_pattern("agent:main", [])
        assert not matches_session_pattern("", [])
    
    def test_exact_match(self):
        """Exact pattern should match exact session key."""
        patterns = compile_session_patterns(["agent:main"])
        assert matches_session_pattern("agent:main", patterns)
        assert not matches_session_pattern("agent:other", patterns)
    
    def test_single_wildcard_match(self):
        """Single * should match segment but not cross colon."""
        patterns = compile_session_patterns(["agent:*"])
        
        # Should match
        assert matches_session_pattern("agent:main", patterns)
        assert matches_session_pattern("agent:123", patterns)
        assert matches_session_pattern("agent:abc123", patterns)
        
        # Should NOT match - * doesn't cross colon
        assert not matches_session_pattern("agent:main:sub", patterns)
        assert not matches_session_pattern("sisyphus:main", patterns)
    
    def test_multiple_segment_wildcards(self):
        """Multiple wildcards in different segments."""
        patterns = compile_session_patterns(["*:*"])
        
        assert matches_session_pattern("agent:main", patterns)
        assert matches_session_pattern("sisyphus:123", patterns)
        # * doesn't cross colon, so this won't match
        assert not matches_session_pattern("agent:main:sub", patterns)
    
    def test_no_match(self):
        """Pattern that doesn't match session key."""
        patterns = compile_session_patterns(["sisyphus:*"])
        
        assert not matches_session_pattern("agent:main", patterns)
        assert not matches_session_pattern("oracle:123", patterns)
        assert matches_session_pattern("sisyphus:main", patterns)
    
    def test_multiple_patterns_any_match(self):
        """Should return True if ANY pattern matches."""
        patterns = compile_session_patterns(["agent:*", "sisyphus:*", "oracle:*"])
        
        assert matches_session_pattern("agent:main", patterns)
        assert matches_session_pattern("sisyphus:123", patterns)
        assert matches_session_pattern("oracle:456", patterns)
        assert not matches_session_pattern("prometheus:main", patterns)
    
    def test_complex_session_key(self):
        """Complex session keys with multiple segments."""
        patterns = compile_session_patterns(["agent:*:subagent:*"])
        
        assert matches_session_pattern("agent:123:subagent:456", patterns)
        assert matches_session_pattern("agent:main:subagent:sub", patterns)
        assert not matches_session_pattern("agent:main", patterns)
        assert not matches_session_pattern("agent:123:subagent:456:extra", patterns)
    
    def test_partial_wildcard(self):
        """Wildcard within segment."""
        patterns = compile_session_patterns(["agent:main*"])
        
        assert matches_session_pattern("agent:main", patterns)
        assert matches_session_pattern("agent:main123", patterns)
        assert not matches_session_pattern("agent:other", patterns)
    
    def test_question_mark_wildcard(self):
        """Question mark should match single non-colon character."""
        patterns = compile_session_patterns(["agent:?"])
        
        assert matches_session_pattern("agent:a", patterns)
        assert matches_session_pattern("agent:1", patterns)
        assert not matches_session_pattern("agent:ab", patterns)
        assert not matches_session_pattern("agent:", patterns)
    
    def test_empty_session_key(self):
        """Empty session key handling."""
        patterns = compile_session_patterns(["agent:*"])
        assert not matches_session_pattern("", patterns)
    
    def test_pattern_with_special_chars(self):
        """Patterns with regex special characters."""
        patterns = compile_session_patterns(["agent:test.value"])
        
        # Literal dot should match
        assert matches_session_pattern("agent:test.value", patterns)
        assert not matches_session_pattern("agent:testXvalue", patterns)


class TestEdgeCases:
    """Edge case tests."""
    
    def test_wildcard_at_end(self):
        """Wildcard at end of pattern."""
        patterns = compile_session_patterns(["agent:*"])
        
        assert matches_session_pattern("agent:abc", patterns)
        assert matches_session_pattern("agent:xyz", patterns)
    
    def test_wildcard_at_start(self):
        """Wildcard at start of pattern."""
        patterns = compile_session_patterns(["*:main"])
        
        assert matches_session_pattern("agent:main", patterns)
        assert matches_session_pattern("sisyphus:main", patterns)
        assert not matches_session_pattern("agent:other", patterns)
    
    def test_consecutive_wildcards(self):
        """Multiple consecutive wildcards."""
        patterns = compile_session_patterns(["agent:**"])
        
        # ** should work as two separate wildcards
        assert matches_session_pattern("agent:ab", patterns)
    
    def test_pattern_longer_than_key(self):
        """Pattern longer than session key."""
        patterns = compile_session_patterns(["agent:*:*:*"])
        
        assert not matches_session_pattern("agent:main", patterns)
        assert not matches_session_pattern("agent:main:sub", patterns)
        assert matches_session_pattern("agent:a:b:c", patterns)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
