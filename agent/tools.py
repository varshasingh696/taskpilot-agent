"""Generic tools. None of them knows anything about invoices."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

MAX_OUTPUT_CHARS = 2000


@dataclass
class ToolResult:
    content: str
    ok: bool = True            # did the call succeed?
    transient: bool = False    # failure worth an automatic retry (5xx, timeout, connection)
    changed_state: bool = False  # a write succeeded (used by the verification gate)


def _clip(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + f"\n...[truncated, {len(text) - MAX_OUTPUT_CHARS} more characters]"


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


TOOL_SPECS = [
    _tool(
        "list_files",
        "List files and folders in a workspace directory. Folders end with '/'.",
        {"directory": {"type": "string", "description": "Relative directory, '.' for the workspace root"}},
        [],
    ),
    _tool(
        "read_file",
        "Read a text file from the workspace.",
        {"path": {"type": "string", "description": "Relative path of the file"}},
        ["path"],
    ),
    _tool(
        "http_request",
        "Call the company's internal system over HTTP. GET reads data; POST/PUT/PATCH/DELETE change it and need user approval.",
        {
            "method": {"type": "string", "description": "GET, POST, PUT, PATCH or DELETE"},
            "url": {"type": "string", "description": "Full URL, must start with the system's base URL"},
            "json_body": {"type": "object", "description": "JSON body for POST/PUT/PATCH"},
        },
        ["method", "url"],
    ),
    _tool(
        "remember",
        "Store a fact in working memory so it stays available in later steps.",
        {"key": {"type": "string"}, "value": {"type": "string"}},
        ["key", "value"],
    ),
    _tool(
        "ask_user",
        "Ask the human a question when the task is ambiguous, information is missing, or you cannot proceed safely.",
        {"question": {"type": "string"}},
        ["question"],
    ),
    _tool(
        "finish",
        "End the task. Provide whether it succeeded, a short summary and concrete evidence.",
        {
            "success": {
                "type": "boolean",
                "description": "true only if the user's goal was actually achieved and verified; false if it could not be done",
            },
            "summary": {"type": "string", "description": "Short summary of what was done"},
            "evidence": {"type": "string", "description": "Concrete data observed that proves the outcome"},
        },
        ["success", "summary", "evidence"],
    ),
]

TOOL_NAMES = {spec["function"]["name"] for spec in TOOL_SPECS}


class Toolbox:
    def __init__(self, workspace, base_url: str, ask_fn):
        self.workspace = Path(workspace).resolve()
        self.base_url = base_url.rstrip("/")
        self.ask_fn = ask_fn
        self.memory: dict[str, str] = {}

    # ---- dispatcher -------------------------------------------------------
    def call(self, name: str, args) -> ToolResult:
        if name not in TOOL_NAMES or name == "finish":
            return ToolResult(f"Unknown tool '{name}'. Available: {sorted(TOOL_NAMES)}", ok=False)
        if isinstance(args, str):  # small models sometimes send arguments as a JSON string
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return ToolResult("Arguments must be a JSON object.", ok=False)
        if not isinstance(args, dict):
            args = {}
        try:
            return getattr(self, f"tool_{name}")(**args)
        except TypeError as exc:
            return ToolResult(f"Bad arguments for {name}: {exc}", ok=False)
        except Exception as exc:  # never let a tool crash the agent loop
            return ToolResult(f"{name} failed: {type(exc).__name__}: {exc}", ok=False)

    # ---- helpers ----------------------------------------------------------
    def _safe_path(self, relative: str) -> Path:
        path = (self.workspace / relative).resolve()
        if path != self.workspace and self.workspace not in path.parents:
            raise ValueError("path escapes the workspace")
        return path

    # ---- tools ------------------------------------------------------------
    def tool_list_files(self, directory: str = ".") -> ToolResult:
        path = self._safe_path(directory)
        if not path.is_dir():
            return ToolResult(f"Not a directory: {directory}", ok=False)
        names = sorted(p.name + ("/" if p.is_dir() else "") for p in path.iterdir())
        return ToolResult("\n".join(names) or "(empty directory)")

    def tool_read_file(self, path: str) -> ToolResult:
        file_path = self._safe_path(path)
        if not file_path.is_file():
            return ToolResult(f"File not found: {path}", ok=False)
        return ToolResult(_clip(file_path.read_text(encoding="utf-8", errors="replace")))

    def tool_http_request(self, method: str, url: str, json_body=None) -> ToolResult:
        method = str(method).upper()
        if method not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
            return ToolResult(f"Unsupported method: {method}", ok=False)
        if not (url == self.base_url or url.startswith(self.base_url + "/")):
            return ToolResult(f"URL not allowed. Only {self.base_url} may be called.", ok=False)
        if isinstance(json_body, str):
            try:
                json_body = json.loads(json_body)
            except json.JSONDecodeError:
                return ToolResult("json_body must be a JSON object, not a string.", ok=False)
        data = json.dumps(json_body).encode() if json_body is not None else None
        request = urllib.request.Request(
            url, data=data, method=method, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                status, body = response.status, response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            status, body = exc.code, exc.read().decode("utf-8", "replace")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            return ToolResult(f"Connection error: {exc}", ok=False, transient=True)
        ok = 200 <= status < 300
        if status in (404, 405):
            body += f"\nHint: do not guess paths. GET {self.base_url}/api/help lists the valid endpoints."
        return ToolResult(
            f"HTTP {status}\n{_clip(body)}",
            ok=ok,
            transient=status >= 500 or status == 429,
            changed_state=ok and method != "GET",
        )

    def tool_remember(self, key: str, value: str) -> ToolResult:
        self.memory[str(key)] = str(value)
        return ToolResult(f"Stored '{key}'.")

    def tool_ask_user(self, question: str) -> ToolResult:
        answer = self.ask_fn(question)
        return ToolResult(f"User answered: {answer}")
