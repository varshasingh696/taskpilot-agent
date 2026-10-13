"""Recover tool calls that a small local model wrote as plain text.

Some local models answer with `Finish: {"summary": ...}` or a bare
`{"name": "finish", "arguments": {...}}` in the message body instead of using the
structured tool-call channel. We parse those into normal tool calls. They still go
through the same harness (allowlist, approval, discovery and verification gates),
so recovery never bypasses a safety rule.
"""
from __future__ import annotations

import json
import re

from .tools import TOOL_NAMES

_PREFIX = re.compile(r"\b(" + "|".join(sorted(TOOL_NAMES)) + r")\b\s*[:(]?\s*", re.IGNORECASE)


def recover_tool_calls(text: str) -> list[dict]:
    if not text:
        return []
    decoder = json.JSONDecoder()

    # Form 1: {"name": "<tool>", "arguments": {...}}
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
        except ValueError:
            continue
        if isinstance(obj, dict) and obj.get("name") in TOOL_NAMES:
            arguments = obj.get("arguments", obj.get("parameters"))
            if isinstance(arguments, dict):
                return [{"name": obj["name"], "arguments": arguments}]

    # Form 2: "<tool>: {...}"  (for example  Finish: {"summary": "...", "evidence": "..."})
    for match in _PREFIX.finditer(text):
        rest = text[match.end():]
        if not rest.startswith("{"):
            continue
        try:
            obj, _ = decoder.raw_decode(rest)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return [{"name": match.group(1).lower(), "arguments": obj}]
    return []
