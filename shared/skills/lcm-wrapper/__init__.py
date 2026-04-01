"""LCM Wrapper - Decorator/Hook integration for nanobot.

Provides non-invasive LCM integration via decorators and hooks.
"""

from .hook import (
    LcmHook,
    LcmHookConfig,
    HookContext,
    lcm_hook,
    lcm_tool,
    quick_start,
    quick_ingest,
    quick_assemble,
)

__all__ = [
    "LcmHook",
    "LcmHookConfig", 
    "HookContext",
    "lcm_hook",
    "lcm_tool",
    "quick_start",
    "quick_ingest",
    "quick_assemble",
]
