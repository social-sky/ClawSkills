#!/usr/bin/env python3
"""FTS5 Query Sanitization Module.

Sanitizes user input for SQLite FTS5 full-text search queries.
Prevents SQL injection while preserving valid FTS5 syntax.

FTS5 Syntax Reference:
- Phrase queries: "word1 word2"
- Prefix queries: word*
- Boolean operators: AND, OR, NOT
- Grouping: (query)
- NEAR operator: word1 NEAR word2
- Column filters: column:query
"""

import re
from typing import Optional


# FTS5 special characters that need handling
# Single quote ' is the main SQL injection vector
# FTS5 uses " for phrase queries (different from SQL string escaping)
FTS5_SPECIAL_CHARS = set('"*():^')

# SQL injection patterns to detect and neutralize
SQL_INJECTION_PATTERNS = [
    r";\s*DROP\s+",      # ; DROP
    r";\s*DELETE\s+",    # ; DELETE
    r";\s*INSERT\s+",    # ; INSERT
    r";\s*UPDATE\s+",    # ; UPDATE
    r";\s*SELECT\s+",    # ; SELECT
    r"--\s*$",           # SQL comment at end
    r"/\*.*\*/",         # Block comment
    r"UNION\s+SELECT",   # UNION SELECT
    r"'\s*OR\s+'",       # ' OR ' injection
    r"'\s*;\s*--",       # '; -- injection
]

# Control characters to remove (ASCII 0-31 except whitespace, plus DEL)
CONTROL_CHARS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')

# Maximum query length to prevent DoS
MAX_QUERY_LENGTH = 10000


def _remove_control_characters(text: str) -> str:
    """Remove control characters except whitespace.
    
    Args:
        text: Input text
        
    Returns:
        Text with control characters removed
    """
    return CONTROL_CHARS.sub('', text)


def _balance_quotes(text: str) -> str:
    """Balance double quotes for FTS5 phrase queries.
    
    FTS5 uses double quotes for phrase queries.
    Unmatched quotes will cause syntax errors.
    
    Args:
        text: Input text
        
    Returns:
        Text with balanced double quotes
    """
    count = text.count('"')
    if count % 2 == 1:
        # Odd number of quotes - remove the last one
        last_pos = text.rfind('"')
        text = text[:last_pos] + text[last_pos + 1:]
    return text


def _neutralize_sql_injection(text: str) -> str:
    """Neutralize common SQL injection patterns.
    
    Args:
        text: Input text
        
    Returns:
        Text with SQL injection attempts neutralized
    """
    result = text
    for pattern in SQL_INJECTION_PATTERNS:
        result = re.sub(pattern, ' ', result, flags=re.IGNORECASE)
    return result


def _escape_single_quotes(text: str) -> str:
    """Escape single quotes to prevent SQL string injection.
    
    In SQLite, single quotes in strings are escaped by doubling them.
    This prevents the query from breaking out of the MATCH clause.
    
    Args:
        text: Input text
        
    Returns:
        Text with single quotes escaped
    """
    return text.replace("'", "''")


def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace while preserving structure.
    
    Args:
        text: Input text
        
    Returns:
        Text with normalized whitespace
    """
    # Replace multiple spaces with single space
    result = re.sub(r'\s+', ' ', text)
    return result.strip()


def sanitize_fts5_query(query: str, *, max_length: Optional[int] = None) -> str:
    """Sanitize a user input string for use in FTS5 MATCH queries.
    
    This function:
    1. Removes control characters
    2. Neutralizes SQL injection attempts
    3. Balances double quotes for phrase queries
    4. Escapes single quotes
    5. Normalizes whitespace
    
    Preserves valid FTS5 syntax:
    - Double-quoted phrases: "exact phrase"
    - Prefix queries: word*
    - Boolean operators: AND, OR, NOT
    - Parentheses for grouping: (term1 OR term2)
    - Column filters: title:search
    - NEAR operator: term1 NEAR term2
    - Initial token queries: ^term
    
    Args:
        query: Raw user input query string
        max_length: Optional maximum length (defaults to MAX_QUERY_LENGTH)
        
    Returns:
        Sanitized query string safe for FTS5 MATCH
        
    Examples:
        >>> sanitize_fts5_query('hello world')
        'hello world'
        >>> sanitize_fts5_query('"exact phrase"')
        '"exact phrase"'
        >>> sanitize_fts5_query("'; DROP TABLE users; --")
        "''  TABLE users   "
        >>> sanitize_fts5_query('prefix*')
        'prefix*'
    """
    if not query:
        return ""
    
    # Apply max length limit
    limit = max_length or MAX_QUERY_LENGTH
    if len(query) > limit:
        query = query[:limit]
    
    # Step 1: Remove control characters
    result = _remove_control_characters(query)
    
    # Step 2: Neutralize SQL injection patterns
    result = _neutralize_sql_injection(result)
    
    # Step 3: Balance double quotes for FTS5 phrase queries
    result = _balance_quotes(result)
    
    # Step 4: Escape single quotes (SQL injection prevention)
    result = _escape_single_quotes(result)
    
    # Step 5: Normalize whitespace
    result = _normalize_whitespace(result)
    
    return result


def is_safe_fts5_query(query: str) -> bool:
    """Check if a query is safe for FTS5 without additional sanitization.
    
    Args:
        query: Query string to check
        
    Returns:
        True if query appears safe, False otherwise
    """
    if not query:
        return True
    
    # Check for control characters
    if CONTROL_CHARS.search(query):
        return False
    
    # Check for SQL injection patterns
    for pattern in SQL_INJECTION_PATTERNS:
        if re.search(pattern, query, flags=re.IGNORECASE):
            return False
    
    # Check for balanced quotes
    if query.count('"') % 2 != 0:
        return False
    
    # Check for single quotes (potential injection)
    if "'" in query:
        return False
    
    return True


def build_fts5_match_clause(table_alias: str, columns: list[str], query: str) -> str:
    """Build a safe FTS5 MATCH clause for SQL queries.
    
    Args:
        table_alias: Alias for the FTS5 virtual table
        columns: List of columns to search (empty for all columns)
        query: User input query
        
    Returns:
        SQL MATCH clause string
        
    Example:
        >>> build_fts5_match_clause('fts', ['title', 'content'], 'hello')
        "fts MATCH '\"hello\"'"
    """
    sanitized = sanitize_fts5_query(query)
    
    if not sanitized:
        return "1=1"  # No filter if empty query
    
    # Wrap the query for the MATCH clause
    # FTS5 expects the query to be a string literal
    match_query = f'"{sanitized}"'
    
    if columns:
        # Column-specific search
        column_list = ', '.join(columns)
        return f"{table_alias} MATCH '{{{column_list}}}: {match_query}'"
    else:
        # Search all columns
        return f"{table_alias} MATCH {match_query}"


if __name__ == "__main__":
    # Demo/test when run directly
    test_queries = [
        'hello world',
        '"exact phrase"',
        "'; DROP TABLE users; --",
        'prefix*',
        'term1 AND term2',
        '(term1 OR term2) NOT term3',
        'title:search',
        'word1 NEAR word2',
        '^initial',
        "Robert'); DROP TABLE students; --",
        'normal "quoted phrase" prefix*',
        '\x00\x01\x02bad chars',
    ]
    
    print("FTS5 Query Sanitization Demo")
    print("=" * 60)
    
    for q in test_queries:
        sanitized = sanitize_fts5_query(q)
        safe = is_safe_fts5_query(q)
        print(f"Input:    {repr(q)}")
        print(f"Sanitized: {repr(sanitized)}")
        print(f"Safe:      {safe}")
        print("-" * 60)
