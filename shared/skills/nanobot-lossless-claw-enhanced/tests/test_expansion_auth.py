#!/usr/bin/env python3
"""Tests for expansion_auth module."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from expansion_auth import (
    create_delegated_expansion_grant,
    resolve_delegated_expansion_grant_id,
    revoke_delegated_expansion_grant_for_session,
)


class TestExpansionAuth:
    """Test cases for expansion authorization."""
    
    def test_create_grant(self):
        """Creating a grant should return an ID."""
        grant_id = create_delegated_expansion_grant("session:main", conversation_id=1)
        assert grant_id is not None
        assert len(grant_id) > 10
    
    def test_resolve_grant(self):
        """Resolving a grant should return the ID."""
        grant_id = create_delegated_expansion_grant("session:main", conversation_id=1)
        resolved = resolve_delegated_expansion_grant_id("session:main")
        assert resolved == grant_id
    
    def test_resolve_grant_not_found(self):
        """Resolving non-existent grant should return None."""
        result = resolve_delegated_expansion_grant_id("session:unknown")
        assert result is None
    
    def test_revoke_grant(self):
        """Revoking a grant should remove it."""
        create_delegated_expansion_grant("session:main", conversation_id=1)
        revoke_delegated_expansion_grant_for_session("session:main")
        result = resolve_delegated_expansion_grant_id("session:main")
        assert result is None
    
    def test_multiple_sessions(self):
        """Multiple sessions should have independent grants."""
        grant1 = create_delegated_expansion_grant("session:one", conversation_id=1)
        grant2 = create_delegated_expansion_grant("session:two", conversation_id=2)
        
        assert resolve_delegated_expansion_grant_id("session:one") == grant1
        assert resolve_delegated_expansion_grant_id("session:two") == grant2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
