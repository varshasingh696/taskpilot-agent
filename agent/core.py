"""The agent loop plus the safety/reliability harness around it.

The LLM decides *what* to do. This harness decides what is *allowed* and
*reliable*, and none of it depends on the task:

* discovery gate     - no writes until the agent has read the system (GET) at least once
* approval gate      - state-changing HTTP calls need a human "yes"
* automatic retries  - transient failures (5xx, timeouts) are retried with backoff
* repeat guard       - an identical call can't be made again and again
* verification gate  - after a write, `finish(success=true)` is refused until the
                       agent has read state back
* working memory     - facts the agent stored are re-injected into every prompt
* text-call recovery - tool calls a small model wrote as plain text are parsed (same gates apply)
* run log            - every step is written to runs/*.jsonl as evidence
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

from .prompts import build_system_prompt
from .recover import recover_tool_calls
from .tools import TOOL_SPECS, ToolResult, Toolbox


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def _short(obj, limit: int = 160) -> str:
    text = json.dumps(obj, default=str)
    return text if len(text) <= limit else text[:limit] + "..."


class Agent:
    def __init__(
        self,
        llm,
        toolbox: Toolbox,
        approve,
        max_steps: int = 15,
        max_retries: int = 3,
        backoff: float = 0.5,
        log_dir: str = "runs",
        say=print,
    ):
        self.llm = llm
        self.toolbox = toolbox
        self.approve = approve
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.backoff = backoff
        self.log_dir = Path(log_dir)
        self.say = say
        self.log_file: Path | None = None

    # ---- public -----------------------------------------------------------
    def run(self, task: str) -> dict:
        self.messages: list[dict] = [{"role": "user", "content": task}]
        self.unverified = False
        self.has_read_system = False
        self.seen: Counter = Counter()
        nudges = 0
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"run-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
        self._log("task", task=task)

        for step in range(1, self.max_steps + 1):
            system = build_system_prompt(self.toolbox.base_url, self.toolbox.memory)
            reply = self.llm.chat(system, self.messages, TOOL_SPECS)
            calls = reply["tool_calls"]
            if not calls:
                recovered = recover_tool_calls(reply["content"])
                if recovered:
                    calls = recovered
                    self._log("recovered_tool_call", step=step, calls=recovered)
                    self.say(f"[step {step}] (recovered a tool call the model wrote as text)")

            assistant = {"role": "assistant", "content": reply["content"]}
            if calls:
                assistant["tool_calls"] = [
                    {"function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls
                ]
            self.messages.append(assistant)

            if not calls:
                nudges += 1
                self._log("no_tool_call", step=step, content=reply["content"])
                self.say(f"[step {step}] (model answered in plain text instead of calling a tool) {reply['content'][:300]}")
                if nudges >= 3:
                    return self._end(
                        False, "The model stopped calling tools without finishing.", reply["content"][:500], step
                    )
                self.messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You replied without calling a tool. Do not answer in plain text. If you already have "
                            "the answer, call the finish tool now: put the direct answer to the user's question in "
                            "`summary` and the data you observed in `evidence`. Otherwise call the next tool."
                        ),
                    }
                )
                continue

            nudges = 0  # the model is calling tools again
            for call in calls:
                result, final = self._handle(call, step)
                self.messages.append({"role": "tool", "tool_name": call["name"], "content": result.content})
                if final:
                    return final

        return self._end(False, f"Stopped after {self.max_steps} steps without finishing.", "", self.max_steps)

    # ---- one tool call ----------------------------------------------------
    def _handle(self, call: dict, step: int):
        name, args = call["name"], call["arguments"]
        if not isinstance(args, dict):
            args = {}
        self.say(f"[step {step}] {name}({_short(args)})")
        self._log("tool_call", step=step, name=name, args=args)

        if name == "finish":
            return self._handle_finish(args, step)

        key = name + json.dumps(args, sort_keys=True, default=str)
        self.seen[key] += 1
        if self.seen[key] > 2:
            return self._observe(
                step,
                ToolResult(
                    "You already made this exact call twice, so you already have its result above. Use it: "
                    "if you can answer now, call finish; otherwise try a different call or ask_user.",
                    ok=False,
                ),
            ), None

        if self._is_side_effect(name, args) and not self.has_read_system:
            self._log("discovery_gate", step=step)
            return self._observe(
                step,
                ToolResult(
                    "BLOCKED: you have not looked at the system yet. Before changing anything, call "
                    f"http_request GET {self.toolbox.base_url}/api/help to learn the real endpoints and "
                    "field formats, then retry.",
                    ok=False,
                ),
            ), None

        if self._is_side_effect(name, args):
            description = f"{str(args.get('method')).upper()} {args.get('url')}  body={_short(args.get('json_body'))}"
            if not self.approve(description):
                self._log("approval_denied", step=step, action=description)
                return self._observe(
                    step,
                    ToolResult(
                        "DENIED: the user did not approve this action. Do not retry it; "
                        "use ask_user or finish with success=false.",
                        ok=False,
                    ),
                ), None
            self._log("approval_granted", step=step, action=description)

        result = self._execute_with_retry(name, args, step)

        if result.changed_state:
            self.unverified = True
            self.seen.clear()  # progress was made, so identical reads are legitimate again
        elif name == "http_request" and result.ok and str(args.get("method", "")).upper() == "GET":
            self.unverified = False
            self.has_read_system = True
        return self._observe(step, result), None

    def _execute_with_retry(self, name: str, args: dict, step: int) -> ToolResult:
        result = self.toolbox.call(name, args)
        for attempt in range(1, self.max_retries):
            if not result.transient:
                break
            delay = self.backoff * 2 ** (attempt - 1)
            self.say(f"    transient failure, retry {attempt}/{self.max_retries - 1} in {delay:.1f}s")
            self._log("retry", step=step, attempt=attempt, reason=result.content[:200])
            time.sleep(delay)
            result = self.toolbox.call(name, args)
        return result

    def _handle_finish(self, args: dict, step: int):
        if "success" not in args:
            self._log("finish_refused", step=step, reason="success_not_stated")
            return ToolResult(
                "REFUSED: finish needs an explicit success value. Use success=true only if the user's goal was "
                "actually achieved and verified; use success=false if it could not be done.",
                ok=False,
            ), None
        success = _as_bool(args["success"])
        summary = str(args.get("summary", ""))
        evidence = str(args.get("evidence", ""))
        if success and self.unverified:
            refusal = ToolResult(
                "REFUSED: you changed state but have not verified it. Read it back (GET) and confirm "
                "the stored data matches your intent, then call finish again.",
                ok=False,
            )
            self._log("finish_refused", step=step, reason="unverified_write")
            return refusal, None
        if success and not evidence.strip():
            self._log("finish_refused", step=step, reason="no_evidence")
            return ToolResult("REFUSED: provide concrete evidence of completion.", ok=False), None
        return ToolResult("Done."), self._end(success, summary, evidence, step)

    # ---- helpers ----------------------------------------------------------
    @staticmethod
    def _is_side_effect(name: str, args: dict) -> bool:
        return name == "http_request" and str(args.get("method", "GET")).upper() != "GET"

    def _observe(self, step: int, result: ToolResult) -> ToolResult:
        self.say(f"    -> {'ok' if result.ok else 'FAILED'}: {result.content[:200].replace(chr(10), ' ')}")
        self._log("observation", step=step, ok=result.ok, content=result.content)
        return result

    def _end(self, success: bool, summary: str, evidence: str, steps: int) -> dict:
        outcome = {
            "success": success,
            "summary": summary,
            "evidence": evidence,
            "steps": steps,
            "log_file": str(self.log_file),
        }
        self._log("finish", **outcome)
        return outcome

    def _log(self, kind: str, **data) -> None:
        if self.log_file is None:
            return
        entry = {"t": round(time.time(), 2), "kind": kind, **data}
        with self.log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, default=str) + "\n")
