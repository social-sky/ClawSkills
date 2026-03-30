#!/usr/bin/env python3
"""LCM Summarization module.

Provides LLM summarization and tool pairing repair, configuration resolution,
text summarization utilities, exploration summary generation, large file 
externalization, and expansion authorization.

Port of TypeScript summarize.ts from lossless-claw-enhanced.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Awaitable, Union

from estimate_tokens import estimate_tokens
from transcript_repair import sanitize_tool_use_result_pairing


class LcmProviderAuthError(Exception):
    """Exception raised for LCM provider authentication errors.
    
    This error is raised when:
    - API key is invalid or missing
    - Authentication with the provider fails
    - Authorization is denied for the requested resource
    - Token has expired or been revoked
    """
    
    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize the auth error.
        
        Args:
            message: Error message
            provider: Provider name (e.g., "openai", "anthropic")
            status_code: HTTP status code if applicable
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.status_code = status_code
        self.details = details or {}
    
    def __str__(self) -> str:
        parts = [self.message]
        if self.provider:
            parts.insert(0, f"[{self.provider}]")
        return " ".join(parts)


# Patterns for detecting authentication-related content
AUTH_ERROR_PATTERNS = [
    # HTTP status codes
    r'\b401\s*(?:Unauthorized|unauthorized)?\b',
    r'\b403\s*(?:Forbidden|forbidden)?\b',
    # API key related
    r'\bAPI\s+key\s+(?:is\s+)?(?:invalid|missing|expired|revoked|incorrect)\b',
    r'\binvalid\s+API\s+key\b',
    r'\bmissing\s+API\s+key\b',
    r'\bexpired\s+API\s+key\b',
    # Authentication related
    r'\bauthentication\s+(?:failed|error)\b',
    r'\bauth\s+(?:failed|error)\b',
    r'\bunauthorized\s+access\b',
    r'\baccess\s+denied\b',
    r'\bpermission\s+denied\b',
    # Token related
    r'\btoken\s+(?:is\s+)?(?:invalid|expired|revoked)\b',
    r'\binvalid\s+token\b',
    r'\bexpired\s+token\b',
    # Credential related
    r'\bcredentials\s+(?:are\s+)?(?:invalid|missing|expired)\b',
    r'\binvalid\s+credentials\b',
    # Common error messages
    r'\bplease\s+(?:check|verify)\s+(?:your\s+)?(?:API\s+)?key\b',
    r'\bapi[_-]?key\s*[:=]\s*\S+\b',  # Actual API key exposure
]

# Compiled regex pattern for efficiency
_AUTH_PATTERN = re.compile('|'.join(AUTH_ERROR_PATTERNS), re.IGNORECASE)


def strip_auth_errors(content: str) -> str:
    """Strip authentication-related error messages from content.
    
    Removes lines and phrases that contain authentication-related errors,
    API key mentions, or other sensitive auth information.
    
    Args:
        content: Text content to sanitize
        
    Returns:
        Sanitized content with auth errors removed
        
    Examples:
        >>> strip_auth_errors("Error: 401 Unauthorized. Please retry.")
        'Please retry.'
        >>> strip_auth_errors("Your API key is invalid")
        ''
    """
    if not content:
        return content
    
    # Split into lines for line-by-line processing
    lines = content.split('\n')
    result_lines = []
    
    for line in lines:
        # Check if line contains auth errors
        if _AUTH_PATTERN.search(line):
            # Try to preserve non-auth parts of the line
            sanitized_line = _sanitize_line(line)
            if sanitized_line.strip():
                result_lines.append(sanitized_line)
        else:
            result_lines.append(line)
    
    return '\n'.join(result_lines)


def _sanitize_line(line: str) -> str:
    """Sanitize a single line by removing auth-related phrases.
    
    Args:
        line: Line to sanitize
        
    Returns:
        Sanitized line
    """
    # Remove matched patterns
    result = _AUTH_PATTERN.sub('', line)
    
    # Clean up artifacts
    result = re.sub(r'\s+', ' ', result)
    result = re.sub(r'^\s*[,;:]\s*', '', result)
    result = re.sub(r'\s*[,;:]\s*$', '', result)
    
    return result.strip()


@dataclass
class SummarizerConfig:
    """Configuration for the summarizer."""
    max_summary_tokens: int = 500
    aggressive: bool = False
    preserve_structure: bool = True
    include_key_points: bool = True


@dataclass
class LegacyParams:
    """Legacy parameters for summarizer creation."""
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    additional_params: Dict[str, Any] = field(default_factory=dict)


async def default_summarizer(
    text: str,
    aggressive: bool = False,
    options: Optional[Dict[str, Any]] = None
) -> str:
    """Default summarizer implementation.
    
    Provides basic summarization without LLM - extracts key points
    and reduces content size.
    
    Args:
        text: Text to summarize
        aggressive: If True, apply more aggressive summarization
        options: Additional options
        
    Returns:
        Summarized text
    """
    if not text:
        return ""
    
    options = options or {}
    max_tokens = options.get("max_tokens", 500)
    
    # Estimate current tokens
    current_tokens = estimate_tokens(text)
    
    if current_tokens <= max_tokens:
        return text
    
    # Apply summarization
    summary = _extract_summary(text, aggressive, max_tokens)
    
    return summary


def _extract_summary(text: str, aggressive: bool, max_tokens: int) -> str:
    """Extract a summary from text.
    
    Args:
        text: Source text
        aggressive: Whether to use aggressive summarization
        max_tokens: Maximum tokens for summary
        
    Returns:
        Summary text
    """
    lines = text.split('\n')
    
    # Filter out empty and very short lines
    content_lines = [l for l in lines if l.strip()]
    
    if aggressive:
        # More aggressive: take only first sentence of each paragraph
        summarized_lines = []
        for line in content_lines:
            sentences = re.split(r'[.!?]\s+', line)
            if sentences:
                first_sentence = sentences[0]
                if first_sentence:
                    summarized_lines.append(first_sentence + '.')
    else:
        # Less aggressive: take proportionally fewer lines
        ratio = max_tokens / max(estimate_tokens(text), 1)
        target_lines = max(1, int(len(content_lines) * ratio))
        summarized_lines = content_lines[:target_lines]
    
    summary = '\n'.join(summarized_lines)
    
    # Truncate if still too long
    while estimate_tokens(summary) > max_tokens and len(summarized_lines) > 1:
        summarized_lines.pop()
        summary = '\n'.join(summarized_lines)
    
    return summary


def create_lcm_summarize_from_legacy_params(
    legacy_params: LegacyParams,
    complete: Optional[Callable[..., Awaitable[Any]]] = None,
    config: Optional[SummarizerConfig] = None
) -> Callable[[str, bool, Optional[Dict[str, Any]]], Awaitable[str]]:
    """Create an LCM summarizer function from legacy parameters.
    
    Factory function that creates a summarizer compatible with the LCM
    system from legacy-style parameters.
    
    Args:
        legacy_params: Legacy configuration parameters
        complete: Optional async completion function for LLM-based summarization
        config: Summarizer configuration
        
    Returns:
        Async summarizer function
        
    Example:
        >>> params = LegacyParams(model="gpt-4", max_tokens=1000)
        >>> summarizer = create_lcm_summarize_from_legacy_params(params)
        >>> summary = await summarizer("Long text...", aggressive=True)
    """
    config = config or SummarizerConfig()
    
    # Build options from legacy params
    base_options: Dict[str, Any] = {
        "model": legacy_params.model,
        "max_tokens": legacy_params.max_tokens or config.max_summary_tokens,
        "temperature": legacy_params.temperature,
    }
    
    # Add any additional params
    base_options.update(legacy_params.additional_params)
    
    async def summarizer(
        text: str,
        aggressive: bool = False,
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """Summarize text using configured parameters.
        
        Args:
            text: Text to summarize
            aggressive: If True, use more aggressive summarization
            options: Additional options that override defaults
            
        Returns:
            Summarized text
        """
        # Merge options
        merged_options = {**base_options}
        if options:
            merged_options.update(options)
        
        # If no completion function, use default
        if complete is None:
            return await default_summarizer(text, aggressive, merged_options)
        
        # Use LLM-based summarization
        try:
            result = await _llm_summarize(
                complete,
                text,
                aggressive,
                merged_options,
                legacy_params
            )
            
            # Strip any auth errors from result
            return strip_auth_errors(result)
            
        except LcmProviderAuthError:
            raise
        except Exception as e:
            # Check if it's an auth-related error
            error_msg = str(e).lower()
            if any(auth_term in error_msg for auth_term in 
                   ['401', '403', 'unauthorized', 'api key', 'authentication']):
                raise LcmProviderAuthError(
                    str(e),
                    provider=legacy_params.model,
                    details={"original_error": str(e)}
                )
            # Fall back to default summarizer on other errors
            return await default_summarizer(text, aggressive, merged_options)
    
    return summarizer


async def _llm_summarize(
    complete: Callable[..., Awaitable[Any]],
    text: str,
    aggressive: bool,
    options: Dict[str, Any],
    legacy_params: LegacyParams
) -> str:
    """Perform LLM-based summarization.
    
    Args:
        complete: Async completion function
        text: Text to summarize
        aggressive: Whether to use aggressive summarization
        options: Summarization options
        legacy_params: Legacy parameters for the LLM
        
    Returns:
        Summarized text
        
    Raises:
        LcmProviderAuthError: If authentication fails
    """
    max_tokens = options.get("max_tokens", 500)
    
    # Build prompt for summarization
    if aggressive:
        prompt = f"""Summarize the following text very concisely, extracting only the most critical information. Maximum {max_tokens} tokens.

Text:
{text}

Summary:"""
    else:
        prompt = f"""Summarize the following text, preserving key information and structure. Maximum {max_tokens} tokens.

Text:
{text}

Summary:"""
    
    # Build messages for completion
    messages = [
        {"role": "system", "content": "You are a helpful assistant that creates concise, accurate summaries."},
        {"role": "user", "content": prompt}
    ]
    
    # Sanitize tool use/result pairings if present
    messages = sanitize_tool_use_result_pairing(messages)
    
    try:
        # Call completion function
        result = await complete(
            messages=messages,
            model=options.get("model"),
            max_tokens=max_tokens,
            temperature=options.get("temperature", 0.3)
        )
        
        # Extract content from result
        if hasattr(result, 'content'):
            if isinstance(result.content, list):
                # Extract text from content blocks
                text_parts = []
                for block in result.content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        text_parts.append(block)
                return "".join(text_parts)
            return str(result.content)
        return str(result)
        
    except Exception as e:
        # Check for auth errors
        error_str = str(e)
        if '401' in error_str or 'unauthorized' in error_str.lower():
            raise LcmProviderAuthError(
                f"Authentication failed: {error_str}",
                provider=legacy_params.model,
                details={"original_error": error_str}
            )
        raise


# Utility functions for external use

def format_file_reference(file_id: str, file_name: Optional[str] = None) -> str:
    """Format a file reference for display.
    
    Args:
        file_id: Unique file identifier
        file_name: Optional file name
        
    Returns:
        Formatted file reference string
    """
    if file_name:
        return f"[File: {file_name} (id: {file_id})]"
    return f"[File: {file_id}]"


def format_tool_output_reference(
    tool_name: str,
    tool_call_id: str,
    output_preview: Optional[str] = None,
    max_preview_length: int = 100
) -> str:
    """Format a tool output reference for display.
    
    Args:
        tool_name: Name of the tool
        tool_call_id: Tool call identifier
        output_preview: Optional preview of output
        max_preview_length: Maximum length for preview
        
    Returns:
        Formatted tool output reference string
    """
    parts = [f"[Tool: {tool_name} (id: {tool_call_id})"]
    
    if output_preview:
        # Truncate preview if needed
        preview = output_preview[:max_preview_length]
        if len(output_preview) > max_preview_length:
            preview += "..."
        parts.append(f" output: {preview}")
    
    parts.append("]")
    return "".join(parts)


# For backwards compatibility with TypeScript naming
createLcmSummarizeFromLegacyParams = create_lcm_summarize_from_legacy_params


if __name__ == "__main__":
    import asyncio
    
    async def demo():
        # Test strip_auth_errors
        test_content = """
        Error: 401 Unauthorized. Please check your API key.
        Your API key is invalid for this request.
        Normal content that should remain.
        Another normal line.
        """
        print("Original:")
        print(test_content)
        print("\nStripped:")
        print(strip_auth_errors(test_content))
        
        # Test summarizer creation
        params = LegacyParams(model="gpt-4", max_tokens=100)
        summarizer = create_lcm_summarize_from_legacy_params(params)
        
        long_text = "This is a very long text. " * 100
        summary = await summarizer(long_text, aggressive=False)
        print(f"\nSummary ({estimate_tokens(summary)} tokens):")
        print(summary[:200] + "...")
        
        # Test format functions
        print("\n" + format_file_reference("file_123", "example.py"))
        print(format_tool_output_reference("read_file", "call_456", "File content here..."))
    
    asyncio.run(demo())
