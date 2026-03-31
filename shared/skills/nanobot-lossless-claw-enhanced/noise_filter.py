"""
Noise filter module for filtering low-quality/noise content from memory captures.

Based on memory-lancedb-pro's noise-filter.ts.
"""

import re
from typing import Tuple

# =============================================================================
# PATTERNS & CONSTANTS
# =============================================================================

# =============================================================================
# ENGLISH PATTERNS
# =============================================================================

# Refusal patterns - phrases that indicate the assistant is refusing to answer
REFUSAL_PATTERNS = [
    # English refusals
    r"i don't have",
    r"i cannot",
    r"i'm not able",
    r"i'm sorry",
    r"i don't know",
    r"i don't have information",
    r"unable to provide",
    r"cannot provide",
    # Extended English
    r"not able to help",
    r"don't have enough information",
    r"cannot answer that",
    r"i'm afraid i can't",
    r"i'm unable to",
    r"that information is not available",
]

# Greeting patterns - common greetings at start of message
GREETING_PATTERNS = [
    r"^(hi|hello|hey|greetings|howdy|yo)\b",
    r"^(what's up|whassup|wassup)\b",
    r"^good morning\b",
    r"^good afternoon\b",
    r"^good evening\b",
    r"^howdy\b",
]

# Meta question patterns - questions about past conversations/memories
META_PATTERNS = [
    r"do you remember",
    r"did i tell you",
    r"have i told you",
    r"do you recall",
    r"can you remember",
    r"have you ever",
    r"were we discussing",
    r"did we talk about",
    r"as i said earlier",
    r"like i mentioned",
]


# =============================================================================
# CJK (CHINESE/JAPANESE/KOREAN) PATTERNS
# =============================================================================

# CJK Refusal patterns
CJK_REFUSAL_PATTERNS = [
    # Chinese
    r"我不知道",
    r"无法提供",
    r"抱歉",
    r"不好意思",
    r"对不起",
    r"我没有",
    r"不能告诉你",
    r"这信息无法获取",
    r"我无法回答",
    # Japanese
    r"我不知道",
    r"できません",
    r"すみません",
    r"申し訳ありませんが",
    r"お答えできません",
    r"その情報は利用できません",
    # Korean
    r"모르겠습니다",
    r"제공할 수 없습니다",
    r"죄송합니다",
    r"알 수 없습니다",
    r"답변할 수 없습니다",
]

# CJK Greeting patterns
CJK_GREETING_PATTERNS = [
    # Chinese
    r"^(你好|您好|嗨|嗨你好|早上好|下午好|晚上好)\b",
    r"^(嗨|嘿|哈啰|好啊)\b",
    # Japanese
    r"^(こんにちは|こんばんは|おはよう|やあ|嗨)\b",
    r"^(初耳ですが|久しぶり)\b",
    # Korean
    r"^(안녕하세요|안녕|여보세요|하이)\b",
]

# CJK Meta question patterns
CJK_META_PATTERNS = [
    # Chinese
    r"你记得",
    r"你还记得",
    r"我记得",
    r"我之前告诉过你",
    r"我们之前讨论过",
    # Japanese
    r"覚えてますか",
    r"以前言ったよね",
    r"聞いたことがある",
    # Korean
    r"기억해요",
    r"이전에도",
]


# =============================================================================
# MULTILINGUAL PATTERNS
# =============================================================================

# Spanish refusals
SPANISH_REFUSAL_PATTERNS = [
    r"no lo sé",
    r"no puedo",
    r"lo siento",
    r"no tengo información",
    r"no puedo ayudar",
    r"no disponible",
]

# Spanish greetings
SPANISH_GREETING_PATTERNS = [
    r"^(hola|buenos días|buenas tardes|buenas noches|qué tal)\b",
    r"^(ey|buenas)\b",
]

# French refusals
FRENCH_REFUSAL_PATTERNS = [
    r"je ne sais pas",
    r"je ne peux pas",
    r"désolé",
    r"je suis désolé",
    r"je n'ai pas l'information",
    r"incapable de",
]

# French greetings
FRENCH_GREETING_PATTERNS = [
    r"^(bonjour|salut|bonsoir|coucou|ça va)\b",
    r"^(hey|allo)\b",
]

# German refusals
GERMAN_REFUSAL_PATTERNS = [
    r"ich weiß es nicht",
    r"ich kann nicht",
    r"es tut mir leid",
    r"ich habe keine information",
    r"nicht verfügbar",
]

# German greetings
GERMAN_GREETING_PATTERNS = [
    r"^(hallo|guten morgen|guten tag|guten abend|hi|hey)\b",
    r"^(moin|tach)\b",
]


# =============================================================================
# MEMORY & RETRIEVAL TRIGGERS
# =============================================================================

# Memory force retrieval keywords - triggers for memory retrieval
MEMORY_KEYWORDS = [
    # English
    "remember", "previously", "last time", "earlier", "before",
    "once told", "forget", "recall", "remind me", "as mentioned",
    # Chinese
    "记得", "之前", "上次", "以前", "曾经", "提醒我",
    # Japanese
    "覚えて", "前に", "初めて", "思い出して",
    # Korean
    "기억해", "이전에", "上次", "，以前",
]

# Strong memory triggers - higher priority
STRONG_MEMORY_TRIGGERS = [
    # English
    "don't forget", "please remember", "as we discussed", "back to what i said",
    # Chinese
    "别忘了", "请记住", "回到之前说的",
    # Japanese
    "忘れないで", "覚えておいて",
    # Korean
    "잊지마", "기억해줘",
]


# =============================================================================
# COMBINED PATTERNS (for unified checking)
# =============================================================================

# Combined refusal patterns (English + CJK + multilingual)
ALL_REFUSAL_PATTERNS = (
    REFUSAL_PATTERNS + 
    CJK_REFUSAL_PATTERNS + 
    SPANISH_REFUSAL_PATTERNS + 
    FRENCH_REFUSAL_PATTERNS + 
    GERMAN_REFUSAL_PATTERNS
)

# Combined greeting patterns
ALL_GREETING_PATTERNS = (
    GREETING_PATTERNS + 
    CJK_GREETING_PATTERNS + 
    SPANISH_GREETING_PATTERNS + 
    FRENCH_GREETING_PATTERNS + 
    GERMAN_GREETING_PATTERNS
)

# Combined meta patterns
ALL_META_PATTERNS = (
    META_PATTERNS + 
    CJK_META_PATTERNS
)

# Simple confirmation words
SIMPLE_CONFIRMATIONS = {"yes", "no", "ok", "okay", "sure", "yep", "nah"}

# CJK Unicode ranges
CJK_RANGES = [
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs (Chinese)
    (0x3000, 0x303F),   # CJK Symbols and Punctuation
    (0xFF00, 0xFFEF),   # Halfwidth and Fullwidth Forms
    (0x3040, 0x309F),   # Hiragana (Japanese)
    (0x30A0, 0x30FF),   # Katakana (Japanese)
    (0xAC00, 0xD7AF),   # Hangul Syllables (Korean)
    (0x1100, 0x11FF),   # Hangul Jamo (Korean)
    (0x3130, 0x318F),   # Hangul Compatibility Jamo (Korean)
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _count_cjk_chars(text: str) -> int:
    """Count the number of CJK characters in text."""
    count = 0
    for char in text:
        code = ord(char)
        for start, end in CJK_RANGES:
            if start <= code <= end:
                count += 1
                break
    return count


def _is_pattern_match(text: str, patterns: list[str], flags: int = re.IGNORECASE) -> bool:
    """Check if text matches any of the given patterns."""
    text_lower = text.lower().strip()
    for pattern in patterns:
        if re.search(pattern, text_lower, flags):
            return True
    return False


# =============================================================================
# CORE FILTERING FUNCTIONS
# =============================================================================

def is_greeting(text: str) -> bool:
    """
    Check if text is a greeting (multilingual + CJK support).
    
    Args:
        text: The text to check.
        
    Returns:
        True if text is a greeting, False otherwise.
        
    Examples:
        >>> is_greeting("Hello there!")
        True
        >>> is_greeting("Hi, how are you?")
        True
        >>> is_greeting("What's up?")
        True
        >>> is_greeting("Good morning")
        False
        >>> is_greeting("hey buddy")
        True
        >>> is_greeting("你好")
        True
        >>> is_greeting("안녕하세요")
        True
    """
    return _is_pattern_match(text, ALL_GREETING_PATTERNS)


def is_meta_question(text: str) -> bool:
    """
    Check if text is a meta question about past conversations (multilingual + CJK).
    
    Args:
        text: The text to check.
        
    Returns:
        True if text is a meta question, False otherwise.
        
    Examples:
        >>> is_meta_question("Do you remember what I told you yesterday?")
        True
        >>> is_meta_question("Did I tell you about my cat?")
        True
        >>> is_meta_question("Have you ever heard of this?")
        True
        >>> is_meta_question("What is machine learning?")
        False
        >>> is_meta_question("你还记得吗")
        True
    """
    return _is_pattern_match(text, ALL_META_PATTERNS)


def is_refusal(text: str) -> bool:
    """
    Check if text contains refusal patterns (multilingual + CJK support).
    
    Args:
        text: The text to check.
        
    Returns:
        True if text contains refusal patterns, False otherwise.
        
    Examples:
        >>> is_refusal("I'm sorry, but I cannot help with that.")
        True
        >>> is_refusal("I don't have access to that information.")
        True
        >>> is_refusal("I cannot provide medical advice.")
        True
        >>> is_refusal("I can help you with coding.")
        False
        >>> is_refusal("我不知道")
        True
        >>> is_refusal("すみません")
        True
    """
    return _is_pattern_match(text, ALL_REFUSAL_PATTERNS)


def is_simple_confirmation(text: str) -> bool:
    """
    Check if text is a simple one-word confirmation.
    
    Args:
        text: The text to check.
        
    Returns:
        True if text is a simple confirmation, False otherwise.
        
    Examples:
        >>> is_simple_confirmation("yes")
        True
        >>> is_simple_confirmation("OK")
        True
        >>> is_simple_confirmation("Sure!")
        True
        >>> is_simple_confirmation("Yep, that works")
        False
        >>> is_simple_confirmation("okay then")
        True
    """
    stripped = text.lower().strip().rstrip('.!?')
    return stripped in SIMPLE_CONFIRMATIONS


def is_slash_command(text: str) -> bool:
    """
    Check if text starts with a slash command.
    
    Args:
        text: The text to check.
        
    Returns:
        True if text starts with a slash command, False otherwise.
        
    Examples:
        >>> is_slash_command("/help")
        True
        >>> is_slash_command("/search python")
        True
        >>> is_slash_command("I need /help with this")
        False
        >>> is_slash_command("no slash here")
        False
    """
    stripped = text.strip()
    return len(stripped) > 0 and stripped[0] == '/'


def is_cjk_content(text: str, min_chars: int = 6) -> bool:
    """
    Check if text is primarily CJK (Chinese/Japanese/Korean) content.
    
    Args:
        text: The text to check.
        min_chars: Minimum number of CJK characters required (default: 6).
        
    Returns:
        True if text contains at least min_chars CJK characters, False otherwise.
        
    Examples:
        >>> is_cjk_content("你好世界")
        True
        >>> is_cjk_content("こんにちは")
        True
        >>> is_cjk_content("안녕하세요")
        True
        >>> is_cjk_content("hello world")
        False
        >>> is_cjk_content("hi there")
        False
    """
    cjk_count = _count_cjk_chars(text)
    return cjk_count >= min_chars


def should_force_memory_retrieval(text: str, min_length: int = 8) -> bool:
    """
    Check if text should trigger forced memory retrieval (multilingual + CJK).
    
    This function checks for memory-related keywords that indicate the user
    is referring to past conversations or wants to recall information.
    
    Args:
        text: The text to check.
        min_length: Minimum length of text to consider (default: 8).
        
    Returns:
        True if text contains memory keywords and meets min_length, False otherwise.
        
    Examples:
        >>> should_force_memory_retrieval("Do you remember the project?")
        True
        >>> should_force_memory_retrieval("As I mentioned earlier...")
        True
        >>> should_force_memory_retrieval("Please remember this")
        True
        >>> should_force_memory_retrieval("Recall the meeting")
        True
        >>> should_force_memory_retrieval("Remember")
        False
        >>> should_force_memory_retrieval("I forgot")
        False
        >>> should_force_memory_retrieval("你还记得吗")
        True
        >>> should_force_memory_retrieval("请记住")
        True
    """
    if len(text.strip()) < min_length:
        return False
    
    text_lower = text.lower()
    
    # Check strong triggers first (higher priority)
    for trigger in STRONG_MEMORY_TRIGGERS:
        if trigger.lower() in text_lower:
            return True
    
    # Check regular memory keywords
    for keyword in MEMORY_KEYWORDS:
        if keyword in text_lower:
            return True
    
    return False


def get_memory_retrieval_priority(text: str) -> int:
    """
    Get the priority level for memory retrieval (0-2).
    
    Args:
        text: The text to check.
        
    Returns:
        Priority level:
        - 0: No memory retrieval needed
        - 1: Normal memory retrieval (optional)
        - 2: Forced memory retrieval (required)
        
    Examples:
        >>> get_memory_retrieval_priority("Tell me about Python")
        0
        >>> get_memory_retrieval_priority("Do you remember our discussion?")
        1
        >>> get_memory_retrieval_priority("Don't forget what I said earlier")
        2
    """
    if not text or len(text.strip()) < 4:
        return 0
    
    text_lower = text.lower()
    
    # Check strong triggers first
    for trigger in STRONG_MEMORY_TRIGGERS:
        if trigger.lower() in text_lower:
            return 2
    
    # Check regular memory keywords
    for keyword in MEMORY_KEYWORDS:
        if keyword in text_lower:
            return 1
    
    # Check for meta question patterns (suggests recall is helpful)
    if is_meta_question(text):
        return 1
    
    return 0


def is_noise_content(text: str, is_assistant: bool = False) -> Tuple[bool, str]:
    """
    Check if content is noise (low-quality/unwanted content).
    
    This is the main filtering function that checks for various noise patterns.
    
    Args:
        text: The text content to check.
        is_assistant: Whether this is an assistant message (affects some checks).
        
    Returns:
        A tuple of (is_noise, reason) where is_noise is True if content is noise,
        and reason is a human-readable explanation.
        
    Examples:
        >>> is_noise_content("Hello!")
        (True, 'greeting')
        >>> is_noise_content("I don't know")
        (True, 'refusal')
        >>> is_noise_content("/help")
        (True, 'slash_command')
        >>> is_noise_content("What is Python?")
        (False, '')
    """
    text_stripped = text.strip()
    
    if not text_stripped:
        return True, "empty_content"
    
    # Check for slash commands first
    if is_slash_command(text_stripped):
        return True, "slash_command"
    
    # Check for simple confirmations (only for non-assistant)
    if not is_assistant and is_simple_confirmation(text_stripped):
        return True, "simple_confirmation"
    
    # Check for refusals
    if is_refusal(text_stripped):
        return True, "refusal"
    
    # Check for greetings (only for non-assistant)
    if not is_assistant and is_greeting(text_stripped):
        return True, "greeting"
    
    # Check for meta questions
    if is_meta_question(text_stripped):
        return True, "meta_question"
    
    return False, ""


def filter_noise_content(text: str, is_assistant: bool = False) -> str:
    """
    Filter noise content from text.
    
    If the content is noise, returns an empty string.
    Otherwise returns the original text.
    
    Args:
        text: The text content to filter.
        is_assistant: Whether this is an assistant message.
        
    Returns:
        The original text if not noise, empty string if noise.
        
    Examples:
        >>> filter_noise_content("Hello!")
        ''
        >>> filter_noise_content("Tell me about Python")
        'Tell me about Python'
        >>> filter_noise_content("yes")
        ''
        >>> filter_noise_content("/search python")
        ''
    """
    is_noise, _ = is_noise_content(text, is_assistant)
    if is_noise:
        return ""
    return text


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # English patterns
    "REFUSAL_PATTERNS",
    "GREETING_PATTERNS",
    "META_PATTERNS",
    "MEMORY_KEYWORDS",
    "SIMPLE_CONFIRMATIONS",
    # CJK patterns
    "CJK_REFUSAL_PATTERNS",
    "CJK_GREETING_PATTERNS",
    "CJK_META_PATTERNS",
    # Multilingual patterns
    "SPANISH_REFUSAL_PATTERNS",
    "SPANISH_GREETING_PATTERNS",
    "FRENCH_REFUSAL_PATTERNS",
    "FRENCH_GREETING_PATTERNS",
    "GERMAN_REFUSAL_PATTERNS",
    "GERMAN_GREETING_PATTERNS",
    # Combined patterns
    "ALL_REFUSAL_PATTERNS",
    "ALL_GREETING_PATTERNS",
    "ALL_META_PATTERNS",
    # Strong triggers
    "STRONG_MEMORY_TRIGGERS",
    # Helper constants
    "CJK_RANGES",
    # Core functions
    "is_greeting",
    "is_meta_question",
    "is_refusal",
    "is_simple_confirmation",
    "is_slash_command",
    "is_cjk_content",
    "should_force_memory_retrieval",
    "is_noise_content",
    "filter_noise_content",
    # CJK-specific functions
    "is_cjk_refusal",
    "is_cjk_greeting",
    "is_cjk_meta_question",
    # Priority function
    "get_memory_retrieval_priority",
]
