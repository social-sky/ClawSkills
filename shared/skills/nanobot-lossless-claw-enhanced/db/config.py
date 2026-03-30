#!/usr/bin/env python3
"""LCM Configuration module.

Provides configuration resolution with 3-tier precedence:
1. Environment variables (highest priority)
2. Plugin configuration
3. Hardcoded defaults (lowest priority)
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Any
from pathlib import Path


@dataclass
class LcmConfig:
    """LCM configuration with all settings.
    
    Configuration precedence (highest to lowest):
    1. Environment variables
    2. Plugin configuration object
    3. Default values
    """
    enabled: bool = True
    database_path: str = ""
    ignore_session_patterns: List[str] = field(default_factory=list)
    stateless_session_patterns: List[str] = field(default_factory=list)
    skip_stateless_sessions: bool = True
    context_threshold: float = 0.75
    fresh_tail_count: int = 32
    leaf_min_fanout: int = 8
    condensed_min_fanout: int = 4
    condensed_min_fanout_hard: int = 2
    incremental_max_depth: int = 0
    leaf_chunk_tokens: int = 20000
    leaf_target_tokens: int = 1200
    condensed_target_tokens: int = 2000
    max_expand_tokens: int = 4000
    large_file_token_threshold: int = 25000
    summary_provider: str = ""
    summary_model: str = ""
    large_file_summary_provider: str = ""
    large_file_summary_model: str = ""
    expansion_provider: str = ""
    expansion_model: str = ""
    autocompact_disabled: bool = False
    timezone: str = "UTC"
    prune_heartbeat_ok: bool = False


def _to_number(value: Any) -> Optional[float]:
    """Safely coerce value to a finite number."""
    if isinstance(value, (int, float)) and value == value:  # Check for NaN
        return float(value)
    if isinstance(value, str):
        try:
            result = float(value)
            if result == result:  # Check for NaN
                return result
        except (ValueError, TypeError):
            pass
    return None


def _to_bool(value: Any) -> Optional[bool]:
    """Safely coerce value to boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lower = value.lower().strip()
        if lower == "true":
            return True
        if lower == "false":
            return False
    return None


def _to_str(value: Any) -> Optional[str]:
    """Safely coerce value to trimmed non-empty string."""
    if isinstance(value, str):
        trimmed = value.strip()
        return trimmed if trimmed else None
    return None


def _to_str_array(value: Any) -> Optional[List[str]]:
    """Coerce value to trimmed string array."""
    if isinstance(value, list):
        result = []
        for item in value:
            s = _to_str(item)
            if s:
                result.append(s)
        return result if result else None
    
    single = _to_str(value)
    if single:
        return [s.strip() for s in single.split(",") if s.strip()]
    return None


def resolve_lcm_config(
    env: Optional[dict] = None,
    plugin_config: Optional[dict] = None
) -> LcmConfig:
    """Resolve LCM configuration with 3-tier precedence.
    
    Args:
        env: Environment variables dict (defaults to os.environ)
        plugin_config: Plugin configuration object
        
    Returns:
        Resolved LcmConfig instance
    """
    if env is None:
        env = dict(os.environ)
    
    pc = plugin_config or {}
    
    def get_env(key: str) -> Optional[str]:
        return env.get(key)
    
    def get_number(key: str, default: float) -> float:
        val = get_env(key)
        if val is not None:
            parsed = _to_number(val)
            if parsed is not None:
                return parsed
        return _to_number(pc.get(key.lower())) or default
    
    def get_int(key: str, default: int) -> int:
        return int(get_number(key, default))
    
    def get_bool(key: str, default: bool) -> bool:
        val = get_env(key)
        if val is not None:
            parsed = _to_bool(val)
            if parsed is not None:
                return parsed
        return _to_bool(pc.get(key.lower())) or default
    
    def get_str(key: str, default: str) -> str:
        val = get_env(key)
        if val is not None:
            return val.strip()
        return _to_str(pc.get(key.lower())) or default
    
    def get_str_list(key: str) -> List[str]:
        val = get_env(key)
        if val is not None:
            return [s.strip() for s in val.split(",") if s.strip()]
        result = _to_str_array(pc.get(key.lower().replace("_", "")))
        return result if result is not None else []
    
    # Default database path
    default_db_path = str(Path.home() / ".openclaw" / "lcm.db")
    
    return LcmConfig(
        enabled=get_bool("LCM_ENABLED", True),
        database_path=get_str("LCM_DATABASE_PATH", 
                             _to_str(pc.get("dbpath")) or 
                             _to_str(pc.get("databasepath")) or 
                             default_db_path),
        ignore_session_patterns=get_str_list("LCM_IGNORE_SESSION_PATTERNS"),
        stateless_session_patterns=get_str_list("LCM_STATELESS_SESSION_PATTERNS"),
        skip_stateless_sessions=get_bool("LCM_SKIP_STATELESS_SESSIONS", True),
        context_threshold=get_number("LCM_CONTEXT_THRESHOLD", 0.75),
        fresh_tail_count=get_int("LCM_FRESH_TAIL_COUNT", 32),
        leaf_min_fanout=get_int("LCM_LEAF_MIN_FANOUT", 8),
        condensed_min_fanout=get_int("LCM_CONDENSED_MIN_FANOUT", 4),
        condensed_min_fanout_hard=get_int("LCM_CONDENSED_MIN_FANOUT_HARD", 2),
        incremental_max_depth=get_int("LCM_INCREMENTAL_MAX_DEPTH", 0),
        leaf_chunk_tokens=get_int("LCM_LEAF_CHUNK_TOKENS", 20000),
        leaf_target_tokens=get_int("LCM_LEAF_TARGET_TOKENS", 1200),
        condensed_target_tokens=get_int("LCM_CONDENSED_TARGET_TOKENS", 2000),
        max_expand_tokens=get_int("LCM_MAX_EXPAND_TOKENS", 4000),
        large_file_token_threshold=get_int("LCM_LARGE_FILE_TOKEN_THRESHOLD", 25000),
        summary_provider=get_str("LCM_SUMMARY_PROVIDER", ""),
        summary_model=get_str("LCM_SUMMARY_MODEL", ""),
        large_file_summary_provider=get_str("LCM_LARGE_FILE_SUMMARY_PROVIDER", ""),
        large_file_summary_model=get_str("LCM_LARGE_FILE_SUMMARY_MODEL", ""),
        expansion_provider=get_str("LCM_EXPANSION_PROVIDER", ""),
        expansion_model=get_str("LCM_EXPANSION_MODEL", ""),
        autocompact_disabled=get_bool("LCM_AUTOCOMPACT_DISABLED", False),
        timezone=get_str("TZ", 
                        _to_str(pc.get("timezone")) or 
                        _get_system_timezone()),
        prune_heartbeat_ok=get_bool("LCM_PRUNE_HEARTBEAT_OK", False),
    )


def _get_system_timezone() -> str:
    """Get system timezone."""
    try:
        import datetime
        if hasattr(datetime.datetime, 'now'):
            return datetime.datetime.now().astimezone().tzinfo.tzname[0]
    except (ImportError, AttributeError):
        pass
    return "UTC"
