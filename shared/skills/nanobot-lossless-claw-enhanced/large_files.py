#!/usr/bin/env python3
"""Large file externalization module for LCM.

Handles externalization of large files to SQLite database for efficient storage.
Files exceeding the token threshold are stored separately with exploration summaries.

Key features:
- File ID extraction from message content
- Exploration summary generation
- Token threshold checking
- Large file record management
"""

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Callable, Awaitable
from pathlib import Path

from estimate_tokens import estimate_tokens
from lcm_types import LargeFileRecord


# MIME type to extension mapping
_MIME_TO_EXT = {
    "application/json": "json",
    "text/csv": "csv",
    "text/plain": "txt",
    "text/html": "html",
    "text/css": "css",
    "text/javascript": "js",
    "application/javascript": "js",
    "application/xml": "xml",
    "text/xml": "xml",
    "application/pdf": "pdf",
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/gif": "gif",
    "image/webp": "webp",
    "image/svg+xml": "svg",
    "application/zip": "zip",
    "application/x-tar": "tar",
    "application/gzip": "gz",
}


def extension_from_name_or_mime(
    name: Optional[str] = None,
    mime: Optional[str] = None
) -> Optional[str]:
    """Extract file extension from filename or MIME type.
    
    Args:
        name: Optional filename (e.g., "document.txt")
        mime: Optional MIME type (e.g., "application/json")
        
    Returns:
        Lowercase extension string (e.g., "txt", "json") or None
        
    Examples:
        >>> extension_from_name_or_mime("test.py")
        'py'
        >>> extension_from_name_or_mime("document.md")
        'md'
        >>> extension_from_name_or_mime(None, "application/json")
        'json'
        >>> extension_from_name_or_mime("no_extension")
        >>> extension_from_name_or_mime(None, "text/plain")
        'txt'
    """
    # Try name first
    if name:
        if isinstance(name, str) and "." in name:
            ext = name.rsplit(".", 1)[-1].lower()
            if ext:
                return ext
    
    # Try MIME type
    if mime and mime in _MIME_TO_EXT:
        return _MIME_TO_EXT[mime]
    
    return None


# Default token threshold for large files
DEFAULT_LARGE_FILE_TOKEN_THRESHOLD = 25000


@dataclass
class FileExtractionResult:
    """Result of extracting file references from content."""
    file_ids: List[str]
    file_blocks: List[Dict[str, Any]]
    total_file_tokens: int


@dataclass
class ExplorationSummaryResult:
    """Result of generating exploration summary."""
    file_id: str
    summary: str
    token_count: int
    method: str  # "llm" or "exploration"


def generate_file_id(content: str, file_name: Optional[str] = None) -> str:
    """Generate a unique file ID based on content hash.
    
    Args:
        content: File content
        file_name: Optional file name for additional uniqueness
        
    Returns:
        Unique file ID string
    """
    hasher = hashlib.sha256()
    hasher.update(content.encode('utf-8', errors='replace'))
    if file_name:
        hasher.update(file_name.encode('utf-8', errors='replace'))
    # Add timestamp for uniqueness
    hasher.update(str(datetime.utcnow().timestamp()).encode('utf-8'))
    return f"file_{hasher.hexdigest()[:16]}"


def extract_file_ids_from_content(content: Any) -> FileExtractionResult:
    """Extract file IDs and file blocks from message content.
    
    Supports various content formats:
    - List of content blocks
    - String content (no files)
    - Dict with nested content
    
    Args:
        content: Message content (list, string, or dict)
        
    Returns:
        FileExtractionResult with extracted file information
    """
    file_ids: List[str] = []
    file_blocks: List[Dict[str, Any]] = []
    total_file_tokens = 0
    
    if content is None:
        return FileExtractionResult(file_ids, file_blocks, total_file_tokens)
    
    # Handle string content (no files)
    if isinstance(content, str):
        return FileExtractionResult(file_ids, file_blocks, total_file_tokens)
    
    # Handle list of content blocks
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            
            block_type = block.get("type", "")
            
            # Check for file blocks
            if block_type == "file":
                file_id = block.get("file_id") or block.get("id")
                if file_id:
                    file_ids.append(str(file_id))
                    file_blocks.append(block)
                    # Estimate tokens from file content if available
                    file_content = block.get("content", "") or block.get("text", "")
                    total_file_tokens += estimate_tokens(file_content)
            
            # Check for image blocks (treated as files)
            elif block_type in ("image", "image_file"):
                file_id = block.get("file_id") or block.get("id")
                if file_id:
                    file_ids.append(str(file_id))
                    file_blocks.append(block)
            
            # Check for document blocks
            elif block_type == "document":
                file_id = block.get("file_id") or block.get("id")
                if file_id:
                    file_ids.append(str(file_id))
                    file_blocks.append(block)
                    doc_content = block.get("content", "") or block.get("text", "")
                    total_file_tokens += estimate_tokens(doc_content)
    
    return FileExtractionResult(file_ids, file_blocks, total_file_tokens)


def is_large_file(
    content: str,
    token_threshold: int = DEFAULT_LARGE_FILE_TOKEN_THRESHOLD
) -> bool:
    """Check if file content exceeds the large file threshold.
    
    Args:
        content: File content to check
        token_threshold: Token count threshold
        
    Returns:
        True if content exceeds threshold, False otherwise
    """
    token_count = estimate_tokens(content)
    return token_count >= token_threshold


def create_large_file_record(
    conversation_id: int,
    content: str,
    file_name: Optional[str] = None,
    mime_type: Optional[str] = None,
    storage_uri: Optional[str] = None,
    exploration_summary: Optional[str] = None
) -> LargeFileRecord:
    """Create a large file record for database storage.
    
    Args:
        conversation_id: Conversation ID
        content: File content
        file_name: Original file name
        mime_type: MIME type of file
        storage_uri: URI where file is stored
        exploration_summary: Summary of file exploration
        
    Returns:
        LargeFileRecord instance
    """
    file_id = generate_file_id(content, file_name)
    
    return LargeFileRecord(
        file_id=file_id,
        conversation_id=conversation_id,
        file_name=file_name,
        mime_type=mime_type,
        byte_size=len(content.encode('utf-8', errors='replace')),
        storage_uri=storage_uri or f"db://{file_id}",
        exploration_summary=exploration_summary,
        created_at=datetime.utcnow()
    )


async def generate_exploration_summary(
    content: str,
    file_name: Optional[str] = None,
    summarizer: Optional[Callable[[str, bool, Optional[Dict[str, Any]]], Awaitable[str]]] = None,
    max_summary_tokens: int = 500
) -> ExplorationSummaryResult:
    """Generate an exploration summary for large file content.
    
    Uses LLM summarization if available, otherwise creates a basic exploration summary.
    
    Args:
        content: File content to summarize
        file_name: Optional file name for context
        summarizer: Optional async summarizer function
        max_summary_tokens: Maximum tokens for summary
        
    Returns:
        ExplorationSummaryResult with summary details
    """
    file_id = generate_file_id(content, file_name)
    
    # If summarizer is available, use it
    if summarizer is not None:
        try:
            summary = await summarizer(
                content,
                aggressive=True,
                options={"max_tokens": max_summary_tokens}
            )
            return ExplorationSummaryResult(
                file_id=file_id,
                summary=summary,
                token_count=estimate_tokens(summary),
                method="llm"
            )
        except Exception:
            # Fall through to basic exploration
            pass
    
    # Generate basic exploration summary
    summary = _generate_basic_exploration_summary(content, file_name)
    
    return ExplorationSummaryResult(
        file_id=file_id,
        summary=summary,
        token_count=estimate_tokens(summary),
        method="exploration"
    )


def _generate_basic_exploration_summary(
    content: str,
    file_name: Optional[str] = None
) -> str:
    """Generate a basic exploration summary without LLM.
    
    Creates a summary based on:
    - File structure analysis
    - Content patterns
    - Key statistics
    
    Args:
        content: File content
        file_name: Optional file name
        
    Returns:
        Basic exploration summary string
    """
    lines = content.split('\n')
    total_lines = len(lines)
    total_chars = len(content)
    total_tokens = estimate_tokens(content)
    
    # Analyze content type
    content_type = _detect_content_type(content)
    
    # Extract key patterns
    patterns = _extract_key_patterns(content, content_type)
    
    # Build summary
    parts = []
    
    if file_name:
        parts.append(f"File: {file_name}")
    
    parts.append(f"Type: {content_type}")
    parts.append(f"Size: {total_chars} chars, {total_lines} lines, ~{total_tokens} tokens")
    
    if patterns:
        parts.append(f"Key elements: {', '.join(patterns[:10])}")
    
    # Add first few meaningful lines as preview
    preview_lines = []
    for line in lines[:20]:
        stripped = line.strip()
        if stripped and not stripped.startswith(('#', '//', '/*', '*')):
            preview_lines.append(stripped)
        if len(preview_lines) >= 5:
            break
    
    if preview_lines:
        preview = ' '.join(preview_lines)[:200]
        parts.append(f"Preview: {preview}...")
    
    return '\n'.join(parts)


def _detect_content_type(content: str) -> str:
    """Detect the type of content based on patterns.
    
    Args:
        content: File content
        
    Returns:
        Detected content type string
    """
    # Check for JSON
    if content.strip().startswith('{') or content.strip().startswith('['):
        try:
            json.loads(content)
            return "JSON"
        except json.JSONDecodeError:
            pass
    
    # Check for Python
    if re.search(r'^\s*(def |class |import |from |if __name__)', content, re.MULTILINE):
        return "Python"
    
    # Check for JavaScript/TypeScript
    if re.search(r'\b(function|const|let|var|=>|async|await)\b', content):
        if re.search(r':\s*(string|number|boolean|any)\b', content):
            return "TypeScript"
        return "JavaScript"
    
    # Check for HTML
    if re.search(r'<(!DOCTYPE|html|head|body|div|span)', content, re.IGNORECASE):
        return "HTML"
    
    # Check for CSS
    if re.search(r'[.#]?\w+\s*\{[^}]*:\s*[^;]+;', content):
        return "CSS"
    
    # Check for Markdown
    if re.search(r'^#+\s+\w|^[-*+]\s+\w|^\|.*\|$', content, re.MULTILINE):
        return "Markdown"
    
    # Check for YAML
    if re.search(r'^\w+:\s*$', content, re.MULTILINE):
        return "YAML"
    
    # Check for SQL
    if re.search(r'\b(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER)\b', content, re.IGNORECASE):
        return "SQL"
    
    # Check for shell script
    if re.search(r'^#!/bin/(bash|sh|zsh)', content) or re.search(r'^\$\s+\w', content, re.MULTILINE):
        return "Shell"
    
    return "Text"


def _extract_key_patterns(content: str, content_type: str) -> List[str]:
    """Extract key patterns from content based on type.
    
    Args:
        content: File content
        content_type: Detected content type
        
    Returns:
        List of extracted patterns
    """
    patterns = []
    
    if content_type == "Python":
        # Extract function and class names
        patterns.extend(re.findall(r'def\s+(\w+)', content))
        patterns.extend(re.findall(r'class\s+(\w+)', content))
    
    elif content_type in ("JavaScript", "TypeScript"):
        # Extract function names
        patterns.extend(re.findall(r'function\s+(\w+)', content))
        patterns.extend(re.findall(r'const\s+(\w+)\s*=', content))
        patterns.extend(re.findall(r'(\w+)\s*=\s*(?:async\s*)?\(', content))
    
    elif content_type == "JSON":
        # Extract top-level keys
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                patterns.extend(list(data.keys())[:20])
            elif isinstance(data, list) and data:
                if isinstance(data[0], dict):
                    patterns.extend(list(data[0].keys())[:20])
        except json.JSONDecodeError:
            pass
    
    elif content_type == "Markdown":
        # Extract headers
        patterns.extend(re.findall(r'^#+\s+(.+)$', content, re.MULTILINE))
    
    elif content_type == "SQL":
        # Extract table names
        patterns.extend(re.findall(r'\bFROM\s+(\w+)', content, re.IGNORECASE))
        patterns.extend(re.findall(r'\bINTO\s+(\w+)', content, re.IGNORECASE))
        patterns.extend(re.findall(r'\bTABLE\s+(\w+)', content, re.IGNORECASE))
    
    return list(dict.fromkeys(patterns))  # Remove duplicates while preserving order


def format_file_reference(file_id: str, file_name: Optional[str] = None) -> str:
    """Format a file reference for display in summaries.
    
    Creates a human-readable reference to an externalized file.
    
    Args:
        file_id: Unique file identifier
        file_name: Optional original file name
        
    Returns:
        Formatted file reference string
        
    Example:
        >>> format_file_reference("file_abc123", "data.json")
        '[External File: data.json (id: file_abc123)]'
    """
    if file_name:
        return f"[External File: {file_name} (id: {file_id})]"
    return f"[External File: {file_id}]"


def format_tool_output_reference(
    tool_name: str,
    tool_call_id: str,
    output_preview: Optional[str] = None,
    max_preview_length: int = 100
) -> str:
    """Format a tool output reference for display in summaries.
    
    Creates a human-readable reference to tool output that has been
    externalized or summarized.
    
    Args:
        tool_name: Name of the tool that was called
        tool_call_id: Unique tool call identifier
        output_preview: Optional preview of the output
        max_preview_length: Maximum length for preview text
        
    Returns:
        Formatted tool output reference string
        
    Example:
        >>> format_tool_output_reference("read_file", "call_123", "File contents...")
        '[Tool Output: read_file (id: call_123) preview: File contents...]'
    """
    parts = [f"[Tool Output: {tool_name} (id: {tool_call_id})"]
    
    if output_preview:
        # Clean and truncate preview
        preview = output_preview.replace('\n', ' ').strip()
        if len(preview) > max_preview_length:
            preview = preview[:max_preview_length] + "..."
        parts.append(f" preview: {preview}")
    
    parts.append("]")
    return "".join(parts)


class LargeFileManager:
    """Manager for large file operations.
    
    Handles:
    - File externalization
    - Summary generation
    - Database storage coordination
    """
    
    def __init__(
        self,
        token_threshold: int = DEFAULT_LARGE_FILE_TOKEN_THRESHOLD,
        summarizer: Optional[Callable[[str, bool, Optional[Dict[str, Any]]], Awaitable[str]]] = None
    ):
        """Initialize large file manager.
        
        Args:
            token_threshold: Token threshold for large files
            summarizer: Optional async summarizer function
        """
        self.token_threshold = token_threshold
        self.summarizer = summarizer
    
    def should_externalize(self, content: str) -> bool:
        """Check if content should be externalized.
        
        Args:
            content: File content to check
            
        Returns:
            True if content should be externalized
        """
        return is_large_file(content, self.token_threshold)
    
    async def externalize_file(
        self,
        conversation_id: int,
        content: str,
        file_name: Optional[str] = None,
        mime_type: Optional[str] = None
    ) -> Tuple[LargeFileRecord, ExplorationSummaryResult]:
        """Externalize a large file with exploration summary.
        
        Args:
            conversation_id: Conversation ID
            content: File content
            file_name: Original file name
            mime_type: MIME type
            
        Returns:
            Tuple of (LargeFileRecord, ExplorationSummaryResult)
        """
        # Generate exploration summary
        summary_result = await generate_exploration_summary(
            content,
            file_name,
            self.summarizer
        )
        
        # Create file record
        record = create_large_file_record(
            conversation_id=conversation_id,
            content=content,
            file_name=file_name,
            mime_type=mime_type,
            exploration_summary=summary_result.summary
        )
        
        return record, summary_result
    
    def extract_files_from_messages(
        self,
        messages: List[Dict[str, Any]]
    ) -> Dict[str, FileExtractionResult]:
        """Extract all file references from messages.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Dict mapping message index to FileExtractionResult
        """
        results = {}
        
        for idx, message in enumerate(messages):
            content = message.get("content")
            result = extract_file_ids_from_content(content)
            if result.file_ids:
                results[idx] = result
        
        return results


if __name__ == "__main__":
    # Demo usage
    import asyncio
    
    async def demo():
        # Test large file detection
        large_content = "x" * 100000  # Large content
        print(f"Is large file: {is_large_file(large_content)}")
        
        # Test file ID extraction
        content_with_file = [
            {"type": "text", "text": "Here is a file:"},
            {"type": "file", "file_id": "file_123", "content": "File content"}
        ]
        result = extract_file_ids_from_content(content_with_file)
        print(f"Extracted file IDs: {result.file_ids}")
        print(f"File tokens: {result.total_file_tokens}")
        
        # Test exploration summary
        code_content = '''
def hello_world():
    """Say hello to the world."""
    print("Hello, World!")

class Greeter:
    def greet(self, name):
        return f"Hello, {name}!"
'''
        summary = await generate_exploration_summary(code_content, "example.py")
        print(f"\nExploration summary:")
        print(summary.summary)
    
    asyncio.run(demo())
