#!/usr/bin/env python3
"""LIKE-based search fallback for CJK text.

FTS5's unicode61 tokenizer doesn't handle CJK (Chinese, Japanese, Korean) well.
This module provides a simple LIKE-based fallback with substring matching.

Usage:
    from search.full_text_fallback import (
        contains_cjk,
        build_like_search_plan,
        create_fallback_snippet
    )
    
    if contains_cjk(query):
        plan = build_like_search_plan("content", query)
        # Use plan.where and plan.args in SQL query
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple


# Unicode ranges for CJK characters
CJK_RANGES = [
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs
    (0x3400, 0x4DBF),    # CJK Unified Ideographs Extension A
    (0x20000, 0x2A6DF),  # CJK Unified Ideographs Extension B
    (0x2A700, 0x2B73F),  # CJK Unified Ideographs Extension C
    (0x2B740, 0x2B81F),  # CJK Unified Ideographs Extension D
    (0x2B820, 0x2CEAF),  # CJK Unified Ideographs Extension E
    (0x2CEB0, 0x2EBEF),  # CJK Unified Ideographs Extension F
    (0x30000, 0x3134F),  # CJK Unified Ideographs Extension G
    (0xF900, 0xFAFF),    # CJK Compatibility Ideographs
    (0x2F800, 0x2FA1F),  # CJK Compatibility Ideographs Supplement
    (0x3040, 0x309F),    # Hiragana
    (0x30A0, 0x30FF),    # Katakana
    (0x31F0, 0x31FF),    # Katakana Phonetic Extensions
    (0xAC00, 0xD7AF),    # Hangul Syllables
    (0x1100, 0x11FF),    # Hangul Jamo
    (0x3130, 0x318F),    # Hangul Compatibility Jamo
    (0xA960, 0xA97F),    # Hangul Jamo Extended-A
    (0xD7B0, 0xD7FF),    # Hangul Jamo Extended-B
]

# Pre-compile regex for performance
_CJK_PATTERN = None


def _get_cjk_pattern() -> re.Pattern:
    """Get or create compiled CJK detection pattern."""
    global _CJK_PATTERN
    if _CJK_PATTERN is None:
        # Build character class for BMP characters
        bmp_ranges = []
        for start, end in CJK_RANGES:
            if end <= 0xFFFF:  # BMP characters only for regex
                bmp_ranges.append(f"{chr(start)}-{chr(end)}")
        
        pattern = f"[{''.join(bmp_ranges)}]"
        _CJK_PATTERN = re.compile(pattern)
    return _CJK_PATTERN


@dataclass
class LikeSearchPlan:
    """Plan for LIKE-based search with CJK support.
    
    Attributes:
        terms: List of search terms extracted from query
        where: SQL WHERE clause with LIKE conditions
        args: Parameter values for the WHERE clause
    """
    terms: List[str] = field(default_factory=list)
    where: str = ""
    args: List[str] = field(default_factory=list)


def contains_cjk(text: str) -> bool:
    """Check if text contains CJK (Chinese, Japanese, Korean) characters.
    
    Args:
        text: Text to check for CJK characters
        
    Returns:
        True if any CJK character is found, False otherwise
        
    Examples:
        >>> contains_cjk("Hello World")
        False
        >>> contains_cjk("中文測試")
        True
        >>> contains_cjk("日本語テスト")
        True
        >>> contains_cjk("한국어 테스트")
        True
        >>> contains_cjk("Hello 你好 World")
        True
    """
    if not text:
        return False
    
    # Quick check for supplementary planes (non-BMP)
    for char in text:
        code = ord(char)
        for start, end in CJK_RANGES:
            if start <= code <= end:
                return True
    
    return False


def _extract_cjk_terms(text: str, min_length: int = 1) -> List[str]:
    """Extract CJK terms from text.
    
    For CJK languages, we extract individual characters and short phrases
    since word boundaries are not space-delimited.
    
    Args:
        text: Input text
        min_length: Minimum term length (default 1 for CJK)
        
    Returns:
        List of CJK terms
    """
    terms = []
    current_cjk = []
    
    for char in text:
        if contains_cjk(char):
            current_cjk.append(char)
        else:
            # End of CJK sequence
            if current_cjk:
                term = ''.join(current_cjk)
                if len(term) >= min_length:
                    terms.append(term)
                current_cjk = []
    
    # Don't forget trailing CJK
    if current_cjk:
        term = ''.join(current_cjk)
        if len(term) >= min_length:
            terms.append(term)
    
    return terms


def _extract_non_cjk_terms(text: str, min_length: int = 2) -> List[str]:
    """Extract non-CJK terms from text (space-delimited words).
    
    Args:
        text: Input text
        min_length: Minimum term length
        
    Returns:
        List of non-CJK terms
    """
    # Split on whitespace and common punctuation
    # Using explicit punctuation chars for Python compatibility
    words = re.split(r'[\s.,!?;:\'"()\[\]{}<>@#$%^&*+=|\\/~`-]+', text)
    
    terms = []
    for word in words:
        word = word.strip()
        if len(word) >= min_length and not contains_cjk(word):
            terms.append(word.lower())
    
    return terms


def build_like_search_plan(
    column: str,
    query: str,
    cjk_min_length: int = 1,
    non_cjk_min_length: int = 2,
    use_or: bool = True
) -> LikeSearchPlan:
    """Build LIKE-based search plan for a query that may contain CJK text.
    
    Creates SQL WHERE clause with LIKE conditions for each search term.
    CJK terms are matched as substrings, while non-CJK terms are matched
    as whole words (case-insensitive).
    
    Args:
        column: Database column name to search
        query: Search query string
        cjk_min_length: Minimum length for CJK terms (default 1)
        non_cjk_min_length: Minimum length for non-CJK terms (default 2)
        use_or: If True, use OR between terms; if False, use AND
        
    Returns:
        LikeSearchPlan with terms, WHERE clause, and args
        
    Examples:
        >>> plan = build_like_search_plan("content", "python 學習")
        >>> plan.terms
        ['python', '學習']
        >>> 'LIKE' in plan.where
        True
        >>> len(plan.args)
        2
    """
    if not query or not query.strip():
        return LikeSearchPlan(terms=[], where="", args=[])
    
    # Escape special SQL LIKE characters
    def escape_like(s: str) -> str:
        """Escape % and _ in LIKE pattern."""
        return s.replace('%', '\\%').replace('_', '\\_')
    
    # Extract CJK and non-CJK terms
    cjk_terms = _extract_cjk_terms(query, cjk_min_length)
    non_cjk_terms = _extract_non_cjk_terms(query, non_cjk_min_length)
    
    # Combine all terms
    all_terms = cjk_terms + non_cjk_terms
    
    if not all_terms:
        return LikeSearchPlan(terms=[], where="", args=[])
    
    # Build LIKE conditions
    conditions = []
    args = []
    
    for term in all_terms:
        escaped = escape_like(term)
        # Use case-insensitive LIKE with % wildcards
        conditions.append(f"{column} LIKE ? ESCAPE '\\'")
        args.append(f"%{escaped}%")
    
    # Join with OR or AND
    operator = " OR " if use_or else " AND "
    where_clause = f"({operator.join(conditions)})" if len(conditions) > 1 else conditions[0]
    
    return LikeSearchPlan(
        terms=all_terms,
        where=where_clause,
        args=args
    )


def create_fallback_snippet(
    content: str,
    terms: List[str],
    max_len: int = 200,
    context_chars: int = 50,
    marker: str = "**"
) -> str:
    """Create a snippet with matched terms highlighted.
    
    Finds the first occurrence of any term and creates a context window
    around it, highlighting all matching terms within the snippet.
    
    Args:
        content: Full text content
        terms: List of search terms to highlight
        max_len: Maximum snippet length (default 200)
        context_chars: Characters to show before/after match (default 50)
        marker: Marker for highlighting (default "**" for markdown bold)
        
    Returns:
        Snippet with highlighted terms, or empty string if no matches
        
    Examples:
        >>> create_fallback_snippet(
        ...     "這是一段關於程式設計的文章",
        ...     ["程式"],
        ...     max_len=100
        ... )
        '這是一段關於**程式**設計的文章'
        
        >>> create_fallback_snippet(
        ...     "Python is a programming language",
        ...     ["python"],
        ...     max_len=50
        ... )
        '**Python** is a programming language'
    """
    if not content or not terms:
        return ""
    
    # Find the first match position
    first_match_pos = -1
    content_lower = content.lower()
    
    for term in terms:
        # Case-insensitive search for position
        pos = content_lower.find(term.lower())
        if pos != -1:
            if first_match_pos == -1 or pos < first_match_pos:
                first_match_pos = pos
    
    if first_match_pos == -1:
        return ""
    
    # Calculate snippet boundaries
    start = max(0, first_match_pos - context_chars)
    end = min(len(content), first_match_pos + max_len - context_chars)
    
    # Adjust to not cut in the middle of a word at the start
    if start > 0:
        # Find next space after start position
        space_pos = content.find(' ', start)
        if space_pos != -1 and space_pos < start + 20:
            start = space_pos + 1
    
    # Extract snippet
    snippet = content[start:end]
    
    # Add ellipsis if truncated
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(content) else ""
    
    # Highlight all terms in the snippet
    highlighted = snippet
    for term in terms:
        # Case-insensitive replacement while preserving original case
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        highlighted = pattern.sub(f"{marker}\\g<0>{marker}", highlighted)
    
    return f"{prefix}{highlighted}{suffix}"


def estimate_cjk_ratio(text: str) -> float:
    """Estimate the ratio of CJK characters in text.
    
    Useful for determining whether to use FTS5 or LIKE fallback.
    
    Args:
        text: Text to analyze
        
    Returns:
        Ratio of CJK characters (0.0 to 1.0)
        
    Examples:
        >>> estimate_cjk_ratio("Hello World")
        0.0
        >>> estimate_cjk_ratio("中文測試")
        1.0
        >>> estimate_cjk_ratio("Hello 你好 World")
        0.25
    """
    if not text:
        return 0.0
    
    cjk_count = 0
    total = len(text)
    
    for char in text:
        if contains_cjk(char):
            cjk_count += 1
    
    return cjk_count / total if total > 0 else 0.0


def should_use_fallback(text: str, threshold: float = 0.3) -> bool:
    """Determine if LIKE fallback should be used instead of FTS5.
    
    Args:
        text: Query text
        threshold: CJK ratio threshold (default 0.3)
        
    Returns:
        True if LIKE fallback should be used
        
    Examples:
        >>> should_use_fallback("python programming")
        False
        >>> should_use_fallback("程式設計")
        True
        >>> should_use_fallback("python 程式")
        True
    """
    return contains_cjk(text) or estimate_cjk_ratio(text) >= threshold


if __name__ == "__main__":
    # Demo / quick test
    import sys
    
    test_queries = [
        "python programming",
        "中文搜尋",
        "日本語テスト",
        "한국어 검색",
        "python 學習",
        "machine learning 機器學習"
    ]
    
    print("CJK Detection Demo")
    print("=" * 50)
    
    for query in test_queries:
        has_cjk = contains_cjk(query)
        ratio = estimate_cjk_ratio(query)
        use_fallback = should_use_fallback(query)
        
        print(f"\nQuery: {query}")
        print(f"  Contains CJK: {has_cjk}")
        print(f"  CJK Ratio: {ratio:.2%}")
        print(f"  Use Fallback: {use_fallback}")
        
        if has_cjk:
            plan = build_like_search_plan("content", query)
            print(f"  Terms: {plan.terms}")
            print(f"  WHERE: {plan.where}")
            
            snippet = create_fallback_snippet(
                f"This is a test document about {query} and related topics.",
                plan.terms
            )
            print(f"  Snippet: {snippet}")
    
    sys.exit(0)
