"""
Unit tests for noise_filter module.
"""

import pytest
from noise_filter import (
    # Functions
    is_greeting,
    is_meta_question,
    is_refusal,
    is_simple_confirmation,
    is_slash_command,
    is_cjk_content,
    should_force_memory_retrieval,
    is_noise_content,
    filter_noise_content,
    # Constants
    REFUSAL_PATTERNS,
    GREETING_PATTERNS,
    META_PATTERNS,
    MEMORY_KEYWORDS,
)


class TestIsGreeting:
    """Tests for is_greeting function."""
    
    def test_simple_greetings(self):
        assert is_greeting("Hello!") is True
        assert is_greeting("Hi there!") is True
        assert is_greeting("Hey!") is True
        assert is_greeting("Greetings!") is True
        assert is_greeting("Howdy!") is True
        assert is_greeting("Yo!") is True
    
    def test_greeting_variants(self):
        assert is_greeting("What's up?") is True
        assert is_greeting("whassup") is True
        assert is_greeting("Wassup!") is True
    
    def test_case_insensitive(self):
        assert is_greeting("HELLO!") is True
        assert is_greeting("hi there") is True
        assert is_greeting("HeY!") is True
    
    def test_not_greeting(self):
        assert is_greeting("Good morning") is False
        assert is_greeting("What's the weather?") is False
        assert is_greeting("hey buddy") is True  # 'hey' is a greeting
    
    def test_greeting_in_context(self):
        assert is_greeting("Hello! How are you?") is True


class TestIsMetaQuestion:
    """Tests for is_meta_question function."""
    
    def test_do_you_remember(self):
        assert is_meta_question("Do you remember what I told you?") is True
        assert is_meta_question("do you remember the project?") is True
    
    def test_did_i_tell_you(self):
        assert is_meta_question("Did I tell you about my cat?") is True
        assert is_meta_question("DID I TELL YOU") is True
    
    def test_have_i_told_you(self):
        assert is_meta_question("Have I told you about this?") is True
    
    def test_do_you_recall(self):
        assert is_meta_question("Do you recall that meeting?") is True
    
    def test_can_you_remember(self):
        assert is_meta_question("Can you remember what I said?") is True
    
    def test_have_you_ever(self):
        assert is_meta_question("Have you ever worked with Python?") is True
    
    def test_not_meta_question(self):
        assert is_meta_question("Do you know Python?") is False
        assert is_meta_question("What is machine learning?") is False


class TestIsRefusal:
    """Tests for is_refusal function."""
    
    def test_i_dont_have(self):
        assert is_refusal("I don't have access to that.") is True
        assert is_refusal("I don't have information about that.") is True
    
    def test_i_cannot(self):
        assert is_refusal("I cannot help with that.") is True
        assert is_refusal("I CANNOT do that.") is True
    
    def test_im_not_able(self):
        assert is_refusal("I'm not able to provide that.") is True
    
    def test_im_sorry(self):
        assert is_refusal("I'm sorry, but I can't help.") is True
    
    def test_i_dont_know(self):
        assert is_refusal("I don't know the answer.") is True
    
    def test_unable_to_provide(self):
        assert is_refusal("I'm unable to provide medical advice.") is True
    
    def test_cannot_provide(self):
        assert is_refusal("I cannot provide legal counsel.") is True
    
    def test_not_refusal(self):
        assert is_refusal("I can help you with coding.") is False
        assert is_refusal("I know Python well.") is False


class TestIsSimpleConfirmation:
    """Tests for is_simple_confirmation function."""
    
    @pytest.mark.parametrize("word", ["yes", "no", "ok", "okay", "sure", "yep", "nah"])
    def test_simple_confirmations(self, word):
        assert is_simple_confirmation(word) is True
    
    @pytest.mark.parametrize("word", ["YES", "No", "OK", "Sure!"])
    def test_case_and_punctuation(self, word):
        assert is_simple_confirmation(word) is True
    
    def test_with_period(self):
        assert is_simple_confirmation("yes.") is True
        assert is_simple_confirmation("ok?") is True
    
    def test_not_simple_confirmation(self):
        assert is_simple_confirmation("yep, that works") is False
        assert is_simple_confirmation("sure thing") is False
        assert is_simple_confirmation("maybe") is False


class TestIsSlashCommand:
    """Tests for is_slash_command function."""
    
    def test_simple_slash_commands(self):
        assert is_slash_command("/help") is True
        assert is_slash_command("/search python") is True
        assert is_slash_command("/quit") is True
    
    def test_with_leading_space(self):
        assert is_slash_command("  /help") is True
    
    def test_not_slash_command(self):
        assert is_slash_command("I need /help with this") is False
        assert is_slash_command("no slash here") is False
        assert is_slash_command("") is False
    
    def test_only_slash(self):
        assert is_slash_command("/") is True


class TestIsCjkContent:
    """Tests for is_cjk_content function."""
    
    def test_chinese(self):
        # "你好世界很好" = 6 CJK chars, meets default min_chars=6
        assert is_cjk_content("你好世界很好") is True
        assert is_cjk_content("中文", min_chars=2) is True
    
    def test_japanese(self):
        # "今日はいい天気" = 6+ CJK chars
        assert is_cjk_content("今日はいい天気") is True
        assert is_cjk_content("東京", min_chars=2) is True
    
    def test_korean(self):
        # "안녕하세요 여러분" = 6+ CJK chars
        assert is_cjk_content("안녕하세요 여러분") is True
        assert is_cjk_content("한국어", min_chars=2) is True
    
    def test_english(self):
        assert is_cjk_content("hello world") is False
    
    def test_mixed(self):
        # "hello 世界 good" = 2 CJK chars - need min_chars=2
        assert is_cjk_content("hello 世界 good", min_chars=2) is True
    
    def test_min_chars(self):
        assert is_cjk_content("你好", min_chars=3) is False
        assert is_cjk_content("你好世界", min_chars=10) is False
    
    def test_empty_string(self):
        assert is_cjk_content("") is False


class TestShouldForceMemoryRetrieval:
    """Tests for should_force_memory_retrieval function."""
    
    def test_remember_keyword(self):
        assert should_force_memory_retrieval("Do you remember?") is True
        assert should_force_memory_retrieval("Please remember this.") is True
    
    def test_previously(self):
        assert should_force_memory_retrieval("As I mentioned previously...") is True
    
    def test_last_time(self):
        assert should_force_memory_retrieval("Last time we talked about...") is True
    
    def test_earlier(self):
        assert should_force_memory_retrieval("Earlier you mentioned...") is True
    
    def test_before(self):
        assert should_force_memory_retrieval("Before we continue...") is True
    
    def test_once_told(self):
        assert should_force_memory_retrieval("You once told me...") is True
    
    def test_forget(self):
        assert should_force_memory_retrieval("I will not forget.") is True
    
    def test_recall(self):
        assert should_force_memory_retrieval("Recall the meeting notes.") is True
    
    def test_remind_me(self):
        assert should_force_memory_retrieval("Remind me to buy groceries.") is True
    
    def test_min_length(self):
        # Both "Remember" and "Remembered" contain keyword "remember"
        assert should_force_memory_retrieval("Remember", min_length=8) is True
        assert should_force_memory_retrieval("Remembered", min_length=8) is True
        # Below min_length returns False
        assert should_force_memory_retrieval("Remember", min_length=9) is False
    
    def test_short_text(self):
        assert should_force_memory_retrieval("hi", min_length=8) is False


class TestIsNoiseContent:
    """Tests for is_noise_content function."""
    
    def test_empty_content(self):
        assert is_noise_content("") == (True, "empty_content")
        assert is_noise_content("   ") == (True, "empty_content")
    
    def test_greeting(self):
        assert is_noise_content("Hello!") == (True, "greeting")
        assert is_noise_content("Hi there!", is_assistant=True) == (False, "")  # assistant greetings not filtered
    
    def test_slash_command(self):
        assert is_noise_content("/help") == (True, "slash_command")
    
    def test_simple_confirmation(self):
        assert is_noise_content("yes") == (True, "simple_confirmation")
        assert is_noise_content("yes", is_assistant=True) == (False, "")  # assistant confirmations not filtered
    
    def test_refusal(self):
        assert is_noise_content("I don't know.") == (True, "refusal")
    
    def test_meta_question(self):
        assert is_noise_content("Do you remember?") == (True, "meta_question")
    
    def test_normal_content(self):
        assert is_noise_content("What is Python?") == (False, "")
        assert is_noise_content("Tell me about machine learning.") == (False, "")


class TestFilterNoiseContent:
    """Tests for filter_noise_content function."""
    
    def test_filter_greeting(self):
        assert filter_noise_content("Hello!") == ""
    
    def test_pass_through(self):
        assert filter_noise_content("Tell me about Python") == "Tell me about Python"
    
    def test_filter_slash_command(self):
        assert filter_noise_content("/search python") == ""
    
    def test_filter_refusal(self):
        assert filter_noise_content("I don't know.") == ""
    
    def test_assistant_mode(self):
        # Assistant messages don't filter greetings/confirmations
        assert filter_noise_content("yes", is_assistant=True) == "yes"
        assert filter_noise_content("Hello!", is_assistant=True) == "Hello!"


class TestConstants:
    """Tests for module constants."""
    
    def test_refusal_patterns_exist(self):
        assert len(REFUSAL_PATTERNS) > 0
        assert all(isinstance(p, str) for p in REFUSAL_PATTERNS)
    
    def test_greeting_patterns_exist(self):
        assert len(GREETING_PATTERNS) > 0
        assert all(isinstance(p, str) for p in GREETING_PATTERNS)
    
    def test_meta_patterns_exist(self):
        assert len(META_PATTERNS) > 0
        assert all(isinstance(p, str) for p in META_PATTERNS)
    
    def test_memory_keywords_exist(self):
        assert len(MEMORY_KEYWORDS) > 0
        assert all(isinstance(k, str) for k in MEMORY_KEYWORDS)
        assert "remember" in MEMORY_KEYWORDS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
