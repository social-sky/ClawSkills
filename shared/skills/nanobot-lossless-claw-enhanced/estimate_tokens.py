#!/usr/bin/env python3
"""CJK-aware token estimation module.

Estimates token count for text with proper handling of:
- ASCII characters (0.25 tokens/char)
- CJK characters (1.5 tokens/char)
- Supplementary characters/emoji (2.0 tokens/char)
"""

import unicodedata
from typing import Union


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
    # Japanese
    (0x3040, 0x309F),    # Hiragana
    (0x30A0, 0x30FF),    # Katakana
    (0x31F0, 0x31FF),    # Katakana Phonetic Extensions
    # Korean
    (0xAC00, 0xD7AF),    # Hangul Syllables
    (0x1100, 0x11FF),    # Hangul Jamo
    (0x3130, 0x318F),    # Hangul Compatibility Jamo
    (0xA960, 0xA97F),    # Hangul Jamo Extended-A
    (0xD7B0, 0xD7FF),    # Hangul Jamo Extended-B
]


def is_cjk_char(char: str) -> bool:
    """Check if a character is a CJK character.
    
    Args:
        char: A single character string
        
    Returns:
        True if the character is CJK, False otherwise
    """
    code_point = ord(char)
    for start, end in CJK_RANGES:
        if start <= code_point <= end:
            return True
    return False


def is_supplementary_char(char: str) -> bool:
    """Check if a character is a supplementary character (emoji, etc.).
    
    Supplementary characters are those with code points >= 0x10000.
    This includes most emoji and other extended Unicode characters.
    
    Args:
        char: A single character string
        
    Returns:
        True if the character is supplementary, False otherwise
    """
    return ord(char) >= 0x10000


def estimate_tokens(text: Union[str, None]) -> int:
    """Estimate the number of tokens in text with CJK awareness.
    
    Token estimation rules:
    - ASCII characters: 0.25 tokens per character (4 chars = 1 token)
    - CJK characters: 1.5 tokens per character
    - Supplementary characters (emoji): 2.0 tokens per character
    
    Args:
        text: Input text string, or None
        
    Returns:
        Estimated token count as integer (rounded up)
        
    Examples:
        >>> estimate_tokens("hello")
        2
        >>> estimate_tokens("你好")
        3
        >>> estimate_tokens("🎉")
        2
    """
    if text is None or len(text) == 0:
        return 0
    
    total_tokens = 0.0
    
    for char in text:
        if is_supplementary_char(char):
            # Emoji and other supplementary characters
            total_tokens += 2.0
        elif is_cjk_char(char):
            # CJK characters
            total_tokens += 1.5
        else:
            # ASCII and other BMP characters
            total_tokens += 0.25
    
    # Round up to nearest integer
    return int(total_tokens + 0.999999999) if total_tokens > 0 else 0


def estimate_tokens_precise(text: Union[str, None]) -> float:
    """Estimate tokens with floating point precision.
    
    Same as estimate_tokens but returns float for more precise calculations.
    
    Args:
        text: Input text string, or None
        
    Returns:
        Estimated token count as float
    """
    if text is None or len(text) == 0:
        return 0.0
    
    total_tokens = 0.0
    
    for char in text:
        if is_supplementary_char(char):
            total_tokens += 2.0
        elif is_cjk_char(char):
            total_tokens += 1.5
        else:
            total_tokens += 0.25
    
    return total_tokens


if __name__ == "__main__":
    # Demo usage
    test_cases = [
        "Hello, World!",
        "你好世界",
        "こんにちは",
        "안녕하세요",
        "🎉🎊🎁",
        "Hello 你好 🎉",
        "",
        None,
    ]
    
    print("Token Estimation Demo:")
    print("-" * 50)
    for text in test_cases:
        tokens = estimate_tokens(text)
        precise = estimate_tokens_precise(text)
        print(f"'{text}' -> {tokens} tokens ({precise:.2f} precise)")
