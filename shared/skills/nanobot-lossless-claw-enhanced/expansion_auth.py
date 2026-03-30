#!/usr/bin/env python3
"""LCM Expansion Authorization module.

Provides delegated expansion grant management for the LCM system.
Handles session-based expansion authorization tokens.

Port of TypeScript expansion-auth.ts from lossless-claw-enhanced.
"""

import secrets
import threading
from typing import Dict, Optional


# In-memory store for expansion grants
# Thread-safe dictionary for session -> grant_id mapping
_expansion_grants: Dict[str, str] = {}
_grants_lock = threading.Lock()


def create_delegated_expansion_grant(
    session_id: str,
    conversation_id: int,
    grant_ttl_seconds: int = 3600
) -> str:
    """Create a delegated expansion grant for a session.
    
    Args:
        session_id: The session identifier (e.g., "session:main")
        conversation_id: The conversation ID this grant is for
        grant_ttl_seconds: Time-to-live for the grant in seconds (default: 1 hour)
        
    Returns:
        The generated grant ID
        
    Examples:
        >>> grant_id = create_delegated_expansion_grant("session:main", conversation_id=1)
        >>> len(grant_id) > 10
        True
    """
    # Generate a secure random grant ID
    grant_id = f"grant_{secrets.token_urlsafe(24)}"
    
    with _grants_lock:
        _expansion_grants[session_id] = grant_id
    
    return grant_id


def resolve_delegated_expansion_grant_id(session_id: str) -> Optional[str]:
    """Resolve the grant ID for a session.
    
    Args:
        session_id: The session identifier to look up
        
    Returns:
        The grant ID if found, None otherwise
        
    Examples:
        >>> grant_id = create_delegated_expansion_grant("session:test", conversation_id=1)
        >>> resolve_delegated_expansion_grant_id("session:test") == grant_id
        True
        >>> resolve_delegated_expansion_grant_id("session:unknown") is None
        True
    """
    with _grants_lock:
        return _expansion_grants.get(session_id)


def revoke_delegated_expansion_grant_for_session(session_id: str) -> None:
    """Revoke (delete) the expansion grant for a session.
    
    Args:
        session_id: The session identifier to revoke
        
    Examples:
        >>> grant_id = create_delegated_expansion_grant("session:revoke", conversation_id=1)
        >>> revoke_delegated_expansion_grant_for_session("session:revoke")
        >>> resolve_delegated_expansion_grant_id("session:revoke") is None
        True
    """
    with _grants_lock:
        _expansion_grants.pop(session_id, None)


def list_active_expansion_grants() -> Dict[str, str]:
    """List all active expansion grants.
    
    Returns:
        Dictionary mapping session_ids to grant_ids
        
    Note:
        This is primarily for debugging/testing purposes.
        In production, prefer individual lookups for performance.
    """
    with _grants_lock:
        return dict(_expansion_grants)


def clear_all_expansion_grants() -> None:
    """Clear all expansion grants.
    
    Warning:
        This is primarily for testing purposes.
        Use with caution in production environments.
    """
    with _grants_lock:
        _expansion_grants.clear()


# For backwards compatibility with TypeScript naming
createDelegatedExpansionGrant = create_delegated_expansion_grant
resolveDelegatedExpansionGrantId = resolve_delegated_expansion_grant_id
revokeDelegatedExpansionGrantForSession = revoke_delegated_expansion_grant_for_session


if __name__ == "__main__":
    # Demo/test code
    print("Expansion Authorization Demo")
    print("-" * 40)
    
    # Test basic operations
    grant1 = create_delegated_expansion_grant("session:one", conversation_id=1)
    print(f"Created grant for session:one -> {grant1}")
    
    grant2 = create_delegated_expansion_grant("session:two", conversation_id=2)
    print(f"Created grant for session:two -> {grant2}")
    
    # Resolve
    resolved1 = resolve_delegated_expansion_grant_id("session:one")
    print(f"Resolved session:one -> {resolved1}")
    print(f"Match: {grant1 == resolved1}")
    
    # Revoke
    revoke_delegated_expansion_grant_for_session("session:one")
    resolved1 = resolve_delegated_expansion_grant_id("session:one")
    print(f"After revoke, session:one -> {resolved1}")
    
    # List all
    all_grants = list_active_expansion_grants()
    print(f"Active grants: {all_grants}")
    
    # Cleanup
    clear_all_expansion_grants()
    print("All grants cleared")
