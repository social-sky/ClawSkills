#!/usr/bin/env python3
"""Transcript repair module for sanitizing tool use/result pairings.

Ensures every tool_result has a matching tool_use with tool_call_id,
and removes orphaned tool_results or tool_uses without results.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple


def extract_tool_call_id(block: Dict[str, Any]) -> Optional[str]:
    """Extract tool_call_id from various tool block formats.
    
    Supports formats:
    - tool_use: {"type": "tool_use", "id": "xxx", "name": "..."}
    - tool_call: {"type": "tool_call", "id": "xxx", "function": {...}}
    - function_call: {"id": "xxx", "function": {...}}
    - tool_result: {"type": "tool_result", "tool_use_id": "xxx"}
    
    Args:
        block: Tool block dictionary
        
    Returns:
        Tool call ID if found, None otherwise
    """
    # Direct id field (tool_use, tool_call)
    if "id" in block:
        return str(block["id"])
    
    # tool_use_id field (tool_result)
    if "tool_use_id" in block:
        return str(block["tool_use_id"])
    
    # tool_call_id field
    if "tool_call_id" in block:
        return str(block["tool_call_id"])
    
    return None


def is_tool_use_block(block: Dict[str, Any]) -> bool:
    """Check if a block is a tool use/call block (not a result).
    
    Args:
        block: Block to check
        
    Returns:
        True if block is a tool use/call, False otherwise
    """
    block_type = block.get("type", "")
    
    # Known tool use types
    if block_type in ("tool_use", "tool_call", "function_call"):
        return True
    
    # Legacy function_call format (no type field but has function)
    if "function" in block and "id" in block and "type" not in block:
        return True
    
    return False


def is_tool_result_block(block: Dict[str, Any]) -> bool:
    """Check if a block is a tool result block.
    
    Args:
        block: Block to check
        
    Returns:
        True if block is a tool result, False otherwise
    """
    block_type = block.get("type", "")
    
    # Known tool result types
    if block_type in ("tool_result", "function_call_output"):
        return True
    
    # Check for tool_use_id which indicates a result
    if "tool_use_id" in block or "tool_call_id" in block:
        return True
    
    return False


def extract_tool_uses_from_message(message: Dict[str, Any]) -> List[Tuple[int, str]]:
    """Extract all tool use IDs from a message.
    
    Args:
        message: Message dictionary
        
    Returns:
        List of (index_in_content, tool_call_id) tuples
    """
    tool_uses = []
    
    # Check content blocks
    content = message.get("content", [])
    if isinstance(content, list):
        for idx, block in enumerate(content):
            if isinstance(block, dict) and is_tool_use_block(block):
                tool_id = extract_tool_call_id(block)
                if tool_id:
                    tool_uses.append((idx, tool_id))
    
    # Check for tool_calls field (OpenAI format)
    tool_calls = message.get("tool_calls", [])
    if isinstance(tool_calls, list):
        for idx, call in enumerate(tool_calls):
            if isinstance(call, dict):
                tool_id = extract_tool_call_id(call)
                if tool_id:
                    tool_uses.append((idx + 1000, tool_id))  # Offset to distinguish from content
    
    # Check for function_call field (legacy)
    function_call = message.get("function_call")
    if function_call and isinstance(function_call, dict):
        tool_id = extract_tool_call_id(function_call)
        if tool_id:
            tool_uses.append((0, tool_id))
    
    return tool_uses


def extract_tool_results_from_message(message: Dict[str, Any]) -> List[Tuple[int, str]]:
    """Extract all tool result IDs from a message.
    
    Args:
        message: Message dictionary
        
    Returns:
        List of (index_in_content, tool_call_id) tuples
    """
    tool_results = []
    
    # Check content blocks
    content = message.get("content", [])
    if isinstance(content, list):
        for idx, block in enumerate(content):
            if isinstance(block, dict) and is_tool_result_block(block):
                tool_id = extract_tool_call_id(block)
                if tool_id:
                    tool_results.append((idx, tool_id))
    
    # Check for role="tool" message format (OpenAI)
    if message.get("role") == "tool":
        tool_id = message.get("tool_call_id")
        if tool_id:
            tool_results.append((0, str(tool_id)))
    
    return tool_results


def sanitize_tool_use_result_pairing(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sanitize messages to ensure proper tool use/result pairing.
    
    Removes:
    - Orphaned tool_results (results without matching tool_uses)
    - tool_uses without results
    
    Preserves message order.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        Sanitized list of messages with proper tool pairing
    """
    if not messages:
        return []
    
    # First pass: collect all tool use IDs and result IDs
    all_tool_uses: Set[str] = set()
    all_tool_results: Set[str] = set()
    
    for message in messages:
        for _, tool_id in extract_tool_uses_from_message(message):
            all_tool_uses.add(tool_id)
        for _, tool_id in extract_tool_results_from_message(message):
            all_tool_results.add(tool_id)
    
    # Find valid pairs (both use and result exist)
    valid_pairs = all_tool_uses & all_tool_results
    
    # Second pass: filter messages
    result_messages = []
    
    for message in messages:
        # Create a copy to avoid modifying original
        new_message = message.copy()
        
        # Process content blocks
        content = message.get("content", [])
        if isinstance(content, list):
            new_content = []
            for idx, block in enumerate(content):
                if not isinstance(block, dict):
                    new_content.append(block)
                    continue
                
                # Keep non-tool blocks
                if not is_tool_use_block(block) and not is_tool_result_block(block):
                    new_content.append(block)
                    continue
                
                tool_id = extract_tool_call_id(block)
                
                # Keep only if it's a valid pair
                if tool_id and tool_id in valid_pairs:
                    new_content.append(block)
            
            new_message["content"] = new_content
        
        # Process tool_calls field (OpenAI format)
        tool_calls = message.get("tool_calls", [])
        if isinstance(tool_calls, list):
            new_tool_calls = []
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                tool_id = extract_tool_call_id(call)
                if tool_id and tool_id in valid_pairs:
                    new_tool_calls.append(call)
            if new_tool_calls:
                new_message["tool_calls"] = new_tool_calls
            elif "tool_calls" in new_message:
                del new_message["tool_calls"]
        
        # Process function_call field (legacy)
        function_call = message.get("function_call")
        if function_call and isinstance(function_call, dict):
            tool_id = extract_tool_call_id(function_call)
            if tool_id and tool_id not in valid_pairs:
                if "function_call" in new_message:
                    del new_message["function_call"]
        
        # Handle role="tool" messages (OpenAI format)
        if message.get("role") == "tool":
            tool_id = message.get("tool_call_id")
            if tool_id and str(tool_id) not in valid_pairs:
                continue  # Skip this message entirely
        
        # Check if message has any content left
        final_content = new_message.get("content", [])
        has_tool_calls = bool(new_message.get("tool_calls"))
        has_function_call = bool(new_message.get("function_call"))
        
        # Keep message if it has text content or tool blocks
        if isinstance(final_content, list):
            if final_content or has_tool_calls or has_function_call or message.get("role") != "tool":
                result_messages.append(new_message)
        elif final_content:  # String content
            result_messages.append(new_message)
        elif has_tool_calls or has_function_call:
            result_messages.append(new_message)
    
    return result_messages


def repair_transcript(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Alias for sanitize_tool_use_result_pairing for backwards compatibility."""
    return sanitize_tool_use_result_pairing(messages)


if __name__ == "__main__":
    # Simple test
    import json
    
    test_messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check that."},
                {"type": "tool_use", "id": "call_1", "name": "get_weather"}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": "Sunny"}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "orphan_1", "content": "Orphaned"}
            ]
        }
    ]
    
    print("Original:")
    print(json.dumps(test_messages, indent=2))
    print("\nRepaired:")
    print(json.dumps(sanitize_tool_use_result_pairing(test_messages), indent=2))
