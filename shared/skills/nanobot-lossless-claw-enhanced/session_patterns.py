#!/usr/bin/env python3
"""Session key pattern matching using glob patterns.

Provides glob-style pattern matching for session keys where colons
are delimiters and * should not match across them.

Session key format: "agent:main" or "agent:123:subagent:456"
"""

import re
import fnmatch
from typing import List, Pattern


def compile_session_patterns(patterns: List[str]) -> List[Pattern]:
    """Convert glob patterns to compiled regex patterns.
    
    Transforms glob patterns where:
    - * matches any characters EXCEPT colon
    - ? matches any single character EXCEPT colon
    - Standard glob syntax otherwise
    
    Args:
        patterns: List of glob pattern strings
        
    Returns:
        List of compiled regex patterns
    """
    compiled = []
    for pattern in patterns:
        # Use fnmatch to get the base regex
        regex_str = fnmatch.translate(pattern)
        
        # Modify so * and ? don't match colons
        # fnmatch.translate converts:
        #   * -> .* (we want [^:]*)
        #   ? -> .  (we want [^:])
        # But we need to be careful with escaping and existing patterns
        
        # Replace .* (from *) with [^:]* except when already escaped
        # Replace single . (from ?) with [^:] but be careful with \. escapes
        
        # Remove the trailing \Z marker temporarily
        regex_str = regex_str.rstrip('\\Z')
        regex_str = regex_str.rstrip('\\Z(?ms)')  # Some versions include flags
        
        # Replace .* (from glob *) with [^:]*
        # We need to be careful not to replace \. (literal dot)
        result = []
        i = 0
        while i < len(regex_str):
            if i < len(regex_str) - 1 and regex_str[i] == '.' and regex_str[i+1] == '*':
                # Check if it's escaped (preceded by odd number of backslashes)
                num_backslashes = 0
                j = i - 1
                while j >= 0 and regex_str[j] == '\\':
                    num_backslashes += 1
                    j -= 1
                
                if num_backslashes % 2 == 0:
                    # Not escaped, replace with [^:]*
                    result.append('[^:]*')
                    i += 2
                else:
                    # Escaped, keep as is
                    result.append(regex_str[i])
                    i += 1
            elif regex_str[i] == '.' and (i == 0 or regex_str[i-1] != '\\'):
                # Check if this is a lone . from glob ?
                # Don't replace if it's followed by * (already handled)
                if i < len(regex_str) - 1 and regex_str[i+1] == '*':
                    result.append(regex_str[i])
                    i += 1
                else:
                    # Check for escaping
                    num_backslashes = 0
                    j = i - 1
                    while j >= 0 and regex_str[j] == '\\':
                        num_backslashes += 1
                        j -= 1
                    
                    if num_backslashes % 2 == 0:
                        result.append('[^:]')
                        i += 1
                    else:
                        result.append(regex_str[i])
                        i += 1
            else:
                result.append(regex_str[i])
                i += 1
        
        modified_regex = ''.join(result) + '\\Z'
        
        try:
            compiled.append(re.compile(modified_regex))
        except re.error:
            # If modification fails, fall back to original
            compiled.append(re.compile(fnmatch.translate(pattern)))
    
    return compiled


def matches_session_pattern(session_key: str, patterns: List[Pattern]) -> bool:
    """Check if a session key matches any of the compiled patterns.
    
    Args:
        session_key: Session key string (e.g., "agent:main", "agent:123:subagent:456")
        patterns: List of compiled regex patterns from compile_session_patterns
        
    Returns:
        True if session key matches any pattern, False otherwise
    """
    if not patterns:
        return False
    
    for pattern in patterns:
        if pattern.match(session_key):
            return True
    
    return False
