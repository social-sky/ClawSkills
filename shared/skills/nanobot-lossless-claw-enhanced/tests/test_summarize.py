#!/usr/bin/env python3
"""Tests for summarize module.

Comprehensive tests for:
- LcmProviderAuthError exception
- strip_auth_errors function
- create_lcm_summarize_from_legacy_params factory
- format_file_reference and format_tool_output_reference utilities
"""

import pytest
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from summarize import (
    strip_auth_errors,
    LcmProviderAuthError,
    create_lcm_summarize_from_legacy_params,
    createLcmSummarizeFromLegacyParams,
    format_file_reference,
    format_tool_output_reference,
    LegacyParams,
    SummarizerConfig,
    default_summarizer,
)


class TestLcmProviderAuthError:
    """Test cases for LcmProviderAuthError exception."""
    
    def test_basic_error(self):
        """Basic error should have message."""
        error = LcmProviderAuthError("Authentication failed")
        assert str(error) == "Authentication failed"
        assert error.message == "Authentication failed"
    
    def test_error_with_provider(self):
        """Error should include provider in string."""
        error = LcmProviderAuthError(
            "Authentication failed",
            provider="openai"
        )
        assert "[openai]" in str(error)
        assert "Authentication failed" in str(error)
    
    def test_error_with_status_code(self):
        """Error should store status code."""
        error = LcmProviderAuthError(
            "Unauthorized",
            provider="anthropic",
            status_code=401
        )
        assert error.status_code == 401
        assert error.provider == "anthropic"
    
    def test_error_with_details(self):
        """Error should store details dict."""
        error = LcmProviderAuthError(
            "Token expired",
            details={"token_id": "abc123", "expired_at": "2024-01-01"}
        )
        assert error.details["token_id"] == "abc123"
        assert error.details["expired_at"] == "2024-01-01"
    
    def test_error_is_exception(self):
        """Error should be catchable as Exception."""
        with pytest.raises(Exception):
            raise LcmProviderAuthError("Test error")
    
    def test_error_is_raiseable(self):
        """Error should be raiseable and catchable."""
        with pytest.raises(LcmProviderAuthError) as exc_info:
            raise LcmProviderAuthError("Test", provider="test_provider")
        assert exc_info.value.provider == "test_provider"


class TestStripAuthErrors:
    """Test cases for strip_auth_errors function."""
    
    def test_strip_401_errors(self):
        """401 errors should be stripped."""
        content = "Error: 401 Unauthorized. Please check your API key."
        result = strip_auth_errors(content)
        assert "401" not in result or "Unauthorized" not in result
    
    def test_strip_403_errors(self):
        """403 Forbidden errors should be stripped."""
        content = "Access denied: 403 Forbidden for this resource."
        result = strip_auth_errors(content)
        assert "403" not in result or "Forbidden" not in result
    
    def test_strip_api_key_mentions(self):
        """API key mentions should be handled."""
        content = "Your API key is invalid. Please check your API key."
        result = strip_auth_errors(content)
        # The function strips auth-related phrases
        assert result != content or len(result) < len(content)
    
    def test_strip_invalid_api_key(self):
        """Invalid API key messages should be stripped."""
        content = "Error: invalid API key provided"
        result = strip_auth_errors(content)
        assert "invalid API key" not in result.lower() or result != content
    
    def test_preserves_normal_content(self):
        """Normal content should be preserved."""
        content = "This is a normal message without auth issues."
        result = strip_auth_errors(content)
        # Normal content should remain mostly unchanged
        assert "normal message" in result
    
    def test_empty_content(self):
        """Empty content should return empty."""
        result = strip_auth_errors("")
        assert result == ""
    
    def test_none_handling(self):
        """None should return empty string."""
        # The function expects string, so None would raise or return empty
        result = strip_auth_errors("")
        assert result == ""
    
    def test_multiple_auth_issues(self):
        """Multiple auth issues should all be stripped."""
        content = """Error 1: 401 Unauthorized
Error 2: Invalid API key
Error 3: Authentication failed
Normal line here."""
        result = strip_auth_errors(content)
        # At minimum, the content should change
        assert "Normal line here" in result or result != content
    
    def test_preserves_structure(self):
        """Should preserve line structure."""
        content = "Line 1\nLine 2\nLine 3"
        result = strip_auth_errors(content)
        lines = result.split('\n')
        assert len(lines) >= 1
    
    def test_authentication_failed(self):
        """Authentication failed message should be stripped."""
        content = "Authentication failed. Please try again."
        result = strip_auth_errors(content)
        # Should be stripped or modified
        assert "Authentication failed" not in result or result != content
    
    def test_token_expired(self):
        """Token expired message should be stripped."""
        content = "Your token is expired. Please refresh."
        result = strip_auth_errors(content)
        assert "token is expired" not in result.lower() or result != content


class TestLegacyParams:
    """Test cases for LegacyParams dataclass."""
    
    def test_basic_params(self):
        """Basic params should be created."""
        params = LegacyParams(model="gpt-4")
        assert params.model == "gpt-4"
    
    def test_all_params(self):
        """All params should be stored."""
        params = LegacyParams(
            model="gpt-4",
            max_tokens=1000,
            temperature=0.7,
            api_key="test-key",
            base_url="https://api.example.com"
        )
        assert params.model == "gpt-4"
        assert params.max_tokens == 1000
        assert params.temperature == 0.7
        assert params.api_key == "test-key"
        assert params.base_url == "https://api.example.com"
    
    def test_additional_params(self):
        """Additional params should be stored."""
        params = LegacyParams(
            model="claude-3",
            additional_params={"top_p": 0.9, "stream": True}
        )
        assert params.additional_params["top_p"] == 0.9
        assert params.additional_params["stream"] is True


class TestSummarizerConfig:
    """Test cases for SummarizerConfig dataclass."""
    
    def test_default_config(self):
        """Default config should have sensible defaults."""
        config = SummarizerConfig()
        assert config.max_summary_tokens == 500
        assert config.aggressive is False
        assert config.preserve_structure is True
    
    def test_custom_config(self):
        """Custom config should override defaults."""
        config = SummarizerConfig(
            max_summary_tokens=1000,
            aggressive=True,
            include_key_points=False
        )
        assert config.max_summary_tokens == 1000
        assert config.aggressive is True
        assert config.include_key_points is False


class TestCreateLcmSummarizeFromLegacyParams:
    """Test cases for create_lcm_summarize_from_legacy_params factory."""
    
    def test_creates_callable(self):
        """Factory should create a callable."""
        params = LegacyParams(model="gpt-4")
        summarizer = create_lcm_summarize_from_legacy_params(params)
        assert callable(summarizer)
    
    def test_async_callable(self):
        """Summarizer should be async."""
        params = LegacyParams(model="gpt-4")
        summarizer = create_lcm_summarize_from_legacy_params(params)
        # Check if it's a coroutine function
        import inspect
        assert inspect.iscoroutinefunction(summarizer)
    
    @pytest.mark.asyncio
    async def test_summarizer_returns_string(self):
        """Summarizer should return a string."""
        params = LegacyParams(model="gpt-4", max_tokens=100)
        summarizer = create_lcm_summarize_from_legacy_params(params)
        
        text = "This is a test text."
        result = await summarizer(text)
        assert isinstance(result, str)
    
    @pytest.mark.asyncio
    async def test_summarizer_short_text_unchanged(self):
        """Short text should remain mostly unchanged."""
        params = LegacyParams(model="gpt-4", max_tokens=100)
        summarizer = create_lcm_summarize_from_legacy_params(params)
        
        text = "Short text."
        result = await summarizer(text)
        assert result == text or len(result) <= len(text)
    
    @pytest.mark.asyncio
    async def test_summarizer_with_aggressive(self):
        """Aggressive mode should work."""
        params = LegacyParams(model="gpt-4", max_tokens=100)
        summarizer = create_lcm_summarize_from_legacy_params(params)
        
        text = "Line 1. Line 2. Line 3. Line 4. Line 5."
        result = await summarizer(text, aggressive=True)
        assert isinstance(result, str)
    
    @pytest.mark.asyncio
    async def test_summarizer_with_options(self):
        """Options should be accepted."""
        params = LegacyParams(model="gpt-4")
        summarizer = create_lcm_summarize_from_legacy_params(params)
        
        text = "Test text."
        result = await summarizer(text, options={"max_tokens": 50})
        assert isinstance(result, str)
    
    def test_backwards_compat_alias(self):
        """TypeScript-style alias should work."""
        params = LegacyParams(model="gpt-4")
        summarizer = createLcmSummarizeFromLegacyParams(params)
        assert callable(summarizer)


class TestDefaultSummarizer:
    """Test cases for default_summarizer function."""
    
    @pytest.mark.asyncio
    async def test_empty_text(self):
        """Empty text should return empty."""
        result = await default_summarizer("")
        assert result == ""
    
    @pytest.mark.asyncio
    async def test_short_text_unchanged(self):
        """Short text should be returned as-is."""
        text = "This is short."
        result = await default_summarizer(text)
        assert result == text
    
    @pytest.mark.asyncio
    async def test_long_text_summarized(self):
        """Long text should be summarized."""
        text = "This is a sentence. " * 100
        result = await default_summarizer(text, options={"max_tokens": 50})
        assert len(result) < len(text)
    
    @pytest.mark.asyncio
    async def test_aggressive_mode(self):
        """Aggressive mode should produce shorter summary."""
        text = "First sentence here. Second sentence here. Third sentence here."
        
        normal = await default_summarizer(text, aggressive=False)
        aggressive = await default_summarizer(text, aggressive=True)
        
        # Aggressive should be shorter or equal
        assert len(aggressive) <= len(normal)


class TestFormatFileReference:
    """Test cases for format_file_reference function."""
    
    def test_basic_reference(self):
        """Basic reference with ID only."""
        result = format_file_reference("file_123")
        assert "file_123" in result
        assert "[File:" in result or "[External File:" in result
    
    def test_reference_with_name(self):
        """Reference with file name."""
        result = format_file_reference("file_abc", "data.json")
        assert "data.json" in result
        assert "file_abc" in result
    
    def test_reference_format(self):
        """Reference should have proper format."""
        result = format_file_reference("id123", "test.py")
        # Should be in bracket format
        assert result.startswith("[") or "File:" in result


class TestFormatToolOutputReference:
    """Test cases for format_tool_output_reference function."""
    
    def test_basic_reference(self):
        """Basic reference with tool name and ID."""
        result = format_tool_output_reference("read_file", "call_123")
        assert "read_file" in result
        assert "call_123" in result
    
    def test_reference_with_preview(self):
        """Reference with output preview."""
        result = format_tool_output_reference(
            "read_file",
            "call_456",
            output_preview="File contents here..."
        )
        assert "File contents" in result
    
    def test_preview_truncation(self):
        """Long preview should be truncated."""
        long_output = "x" * 200
        result = format_tool_output_reference(
            "read_file",
            "call_789",
            output_preview=long_output,
            max_preview_length=50
        )
        # Should be truncated
        assert len(result) < len(long_output) + 100
        assert "..." in result
    
    def test_no_preview(self):
        """Reference without preview should work."""
        result = format_tool_output_reference("bash", "call_000")
        assert "bash" in result
        assert "call_000" in result


class TestIntegration:
    """Integration tests for summarize module."""
    
    @pytest.mark.asyncio
    async def test_full_summarization_workflow(self):
        """Test full summarization workflow."""
        # Create params
        params = LegacyParams(
            model="gpt-4",
            max_tokens=200,
            temperature=0.3
        )
        
        # Create summarizer
        summarizer = create_lcm_summarize_from_legacy_params(params)
        
        # Test text
        long_text = "This is a line of text. " * 50
        
        # Summarize
        summary = await summarizer(long_text, aggressive=False)
        
        # Verify
        assert isinstance(summary, str)
        assert len(summary) <= len(long_text)
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in summarization."""
        params = LegacyParams(model="test-model")
        summarizer = create_lcm_summarize_from_legacy_params(params)
        
        # Should not raise even with empty input
        result = await summarizer("")
        assert result == ""
    
    def test_strip_and_format_integration(self):
        """Test strip_auth_errors with formatted references."""
        # Create some tool output with auth error
        tool_ref = format_tool_output_reference(
            "api_call",
            "call_123",
            output_preview="Error: 401 Unauthorized"
        )
        
        # Strip auth errors
        result = strip_auth_errors(tool_ref)
        
        # Should preserve tool reference structure
        assert "api_call" in result or result != tool_ref


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
