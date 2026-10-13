"""Tests for the harness. A scripted fake LLM replaces Ollama so the reliability
behaviour (retry, error recovery, verification gate, approval, safety) is tested
deterministically against the real mock app."""
import json
import tempfile
import threading
import unittest
from pathlib import Path

from agent.core import Agent
from agent.tools import Toolbox
from mock_app.server import make_server

DATA = Path(__file__).resolve().parent.parent / "data"


def call(name, **arguments):
    return {"name": name, "arguments": arguments}


class FakeLLM:
    def __init__(self, script):
        self.script = list(script)
        self.index = 0

    def chat(self, system, messages, tools):
        item = self.script[self.index]
        self.index += 1
        if isinstance(item, str):  # a plain-text reply with no structured tool call
            return {"content": item, "tool_calls": []}
        return {"content": "", "tool_calls": [item] if item else []}


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.server = make_server(0, flaky=True)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.logs = tempfile.mkdtemp()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def make_agent(self, script, approve=lambda d: True, **kw):
        toolbox = Toolbox(DATA, self.base, lambda q: "n/a")
        return Agent(FakeLLM(script), toolbox, approve, backoff=0.01, log_dir=self.logs, say=lambda *_: None, **kw)

    def log_kinds(self, outcome):
        lines = Path(outcome["log_file"]).read_text().splitlines()
        return [json.loads(line)["kind"] for line in lines]

    def test_full_flow_with_retry_error_recovery_and_verification(self):
        records = f"{self.base}/api/records"
        good = {"vendor": "Acme Corp", "invoice_number": "ACME-1041", "amount": 48500.0, "due_date": "2026-10-27"}
        bad = dict(good, due_date="27 October 2026")
        script = [
            call("list_files", directory="inbox"),
            call("read_file", path="inbox/acme_inv_a.txt"),
            call("read_file", path="inbox/acme_inv_b.txt"),
            call("remember", key="latest_acme", value="ACME-1041"),
            call("http_request", method="GET", url=f"{self.base}/api/help"),
            call("http_request", method="POST", url=records, json_body=bad),    # 503 -> retry -> 400
            call("http_request", method="POST", url=records, json_body=good),   # 201
            call("finish", success=True, summary="entered", evidence="n/a"),    # refused: unverified
            call("http_request", method="GET", url=records),                     # verify
            call("finish", success=True, summary="entered", evidence="record ACME-1041 present"),
        ]
        outcome = self.make_agent(script).run("enter latest Acme invoice")
        self.assertTrue(outcome["success"])
        self.assertEqual(outcome["steps"], 10)
        stored = self.server.state.records
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0]["amount"], 48500.0)
        self.assertEqual(stored[0]["due_date"], "2026-10-27")
        kinds = self.log_kinds(outcome)
        self.assertIn("retry", kinds)
        self.assertIn("finish_refused", kinds)

    def test_denied_approval_blocks_the_write(self):
        body = {"vendor": "Acme Corp", "invoice_number": "X", "amount": 1, "due_date": "2026-10-27"}
        script = [
            call("http_request", method="GET", url=f"{self.base}/api/help"),
            call("http_request", method="POST", url=f"{self.base}/api/records", json_body=body),
            call("finish", success=False, summary="user denied", evidence="no record created"),
        ]
        outcome = self.make_agent(script, approve=lambda d: False).run("enter invoice")
        self.assertFalse(outcome["success"])
        self.assertEqual(self.server.state.post_attempts, 0)
        self.assertEqual(self.server.state.records, [])

    def test_discovery_gate_blocks_write_before_reading_the_system(self):
        asked = []
        body = {"vendor": "Acme Corp", "invoice_number": "X", "amount": 1, "due_date": "2026-10-27"}
        script = [
            call("http_request", method="POST", url=f"{self.base}/api/invoices", json_body=body),  # guessed, no discovery
            call("finish", success=False, summary="stopped", evidence="blocked"),
        ]
        outcome = self.make_agent(script, approve=lambda d: asked.append(d) or True).run("enter invoice")
        self.assertEqual(asked, [])  # the human was never bothered with a blind write
        self.assertEqual(self.server.state.post_attempts, 0)
        self.assertIn("discovery_gate", self.log_kinds(outcome))

    def test_plain_text_reply_is_nudged_and_recovers(self):
        script = [None, call("finish", success=False, summary="x", evidence="y")]
        outcome = self.make_agent(script).run("anything")
        self.assertEqual(outcome["summary"], "x")
        self.assertEqual(outcome["steps"], 2)

    def test_nudge_counter_resets_after_a_tool_call(self):
        script = [None, None, call("list_files", directory="."), None, None,
                  call("finish", success=False, summary="done", evidence="e")]
        outcome = self.make_agent(script).run("anything")
        self.assertEqual(outcome["summary"], "done")

    def test_tool_call_written_as_text_is_recovered(self):
        text = 'Finish: {"success": true, "summary": "2 invoices, total 87250", "evidence": "records read back"}'
        outcome = self.make_agent([text]).run("count invoices")
        self.assertTrue(outcome["success"])
        self.assertEqual(outcome["steps"], 1)
        self.assertEqual(outcome["summary"], "2 invoices, total 87250")

    def test_json_style_text_call_is_recovered(self):
        text = 'Done. {"name": "finish", "arguments": {"success": true, "summary": "s", "evidence": "e"}}'
        self.assertTrue(self.make_agent([text]).run("x")["success"])

    def test_ordinary_prose_is_not_mistaken_for_a_tool_call(self):
        script = ["I will finish the task soon.", call("finish", success=False, summary="s", evidence="e")]
        self.assertEqual(self.make_agent(script).run("x")["steps"], 2)

    def test_recovered_write_still_hits_the_gates(self):
        text = f'http_request {{"method": "POST", "url": "{self.base}/api/records", "json_body": {{}}}}'
        script = [text, call("finish", success=False, summary="s", evidence="e")]
        outcome = self.make_agent(script).run("x")
        self.assertEqual(self.server.state.post_attempts, 0)  # blocked by the discovery gate
        self.assertIn("discovery_gate", self.log_kinds(outcome))

    def test_finish_without_explicit_success_is_refused(self):
        script = [
            call("finish", summary="no initech invoice", evidence="none"),   # success missing -> refused
            call("finish", success=False, summary="no initech invoice", evidence="none"),
        ]
        outcome = self.make_agent(script).run("x")
        self.assertFalse(outcome["success"])
        self.assertEqual(outcome["steps"], 2)
        self.assertIn("finish_refused", self.log_kinds(outcome))

    def test_model_that_never_calls_tools_is_stopped(self):
        outcome = self.make_agent([None, None, None]).run("anything")
        self.assertFalse(outcome["success"])

    def test_repeated_identical_call_is_blocked(self):
        same = call("list_files", directory="inbox")
        script = [same, same, same, call("finish", success=False, summary="stuck", evidence="loop")]
        agent = self.make_agent(script)
        outcome = agent.run("loop")
        observations = [json.loads(l) for l in Path(outcome["log_file"]).read_text().splitlines()]
        blocked = [o for o in observations if o["kind"] == "observation" and "exact call twice" in o["content"]]
        self.assertEqual(len(blocked), 1)

    def test_tool_safety(self):
        tb = Toolbox(DATA, self.base, lambda q: "")
        escape = tb.call("read_file", {"path": "../run_agent.py"})
        self.assertFalse(escape.ok)
        self.assertIn("escapes", escape.content)
        outside = tb.call("http_request", {"method": "GET", "url": "http://example.com"})
        self.assertFalse(outside.ok)
        self.assertIn("not allowed", outside.content)
        self.assertFalse(tb.call("delete_everything", {}).ok)


if __name__ == "__main__":
    unittest.main()
