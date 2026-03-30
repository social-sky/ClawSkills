#!/usr/bin/env python3
"""Tests for large_files module."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from large_files import (
    extract_file_ids_from_content,
    extension_from_name_or_mime,
    generate_exploration_summary,
)


class TestLargeFiles:
    """Test cases for large file externalization."""
    
    def test_extract_file_ids_empty(self):
        """Empty content should return empty list."""
        result = extract_file_ids_from_content("")
        assert result.file_ids == []
    
    def test_extract_file_ids_single(self):
        """Single file ID should be extracted."""
        content = [{"type": "file", "file_id": "file_abc123", "name": "test.pdf"}]
        result = extract_file_ids_from_content(content)
        assert result.file_ids == ["file_abc123"]
    
    def test_extract_file_ids_multiple(self):
        """Multiple file IDs should be extracted."""
        content = [
            {"type": "file", "file_id": "file_abc", "name": "test.pdf"},
            {"type": "file", "file_id": "file_def456", "name": "data.json"}
        ]
        result = extract_file_ids_from_content(content)
        assert set(result.file_ids) == {"file_abc", "file_def456"}
    
    def test_extension_from_name(self):
        """Extension from filename."""
        assert extension_from_name_or_mime("test.py") == "py"
        assert extension_from_name_or_mime("document.md") == "md"
        assert extension_from_name_or_mime("data.json") == "json"
        assert extension_from_name_or_mime("no_extension") is None
    
    def test_extension_from_mime(self):
        """Extension from MIME type."""
        assert extension_from_name_or_mime(None, "application/json") == "json"
        assert extension_from_name_or_mime(None, "text/csv") == "csv"
        assert extension_from_name_or_mime(None, "text/plain") == "txt"
    
    @pytest.mark.asyncio
    async def test_generate_exploration_summary_short(self):
        """Short text summary."""
        content = "This is a short text file for testing."
        result = await generate_exploration_summary(content)
        assert "text" in result.summary.lower()
        assert len(result.summary) > 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
