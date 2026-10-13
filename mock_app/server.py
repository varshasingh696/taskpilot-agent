"""Mock internal finance system (standard library only).

It plays the role of "our internal system" from the assignment. It has two
deliberate behaviours that make the agent's reliability logic observable:

* the very first POST fails with HTTP 503 (a transient outage) -> tests retries
* strict validation (due_date must be YYYY-MM-DD, amount must be a plain number)
  -> the agent has to read the error and fix its own input
"""
from __future__ import annotations

import argparse
import html
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

API_HELP = {
    "name": "NorthStar Finance System (mock)",
    "endpoints": [
        {
            "method": "GET",
            "path": "/api/records",
            "description": "List all invoice records entered so far.",
        },
        {
            "method": "POST",
            "path": "/api/records",
            "description": "Create one invoice record.",
            "json_body": {
                "vendor": "string, vendor name",
                "invoice_number": "string, must be unique",
                "amount": "number, plain value without currency symbol or commas, e.g. 1234.5",
                "due_date": "string, format YYYY-MM-DD",
            },
        },
    ],
}


class State:
    def __init__(self, flaky: bool):
        self.records: list[dict] = []
        self.lock = threading.Lock()
        self.flaky = flaky
        self.post_attempts = 0


def validate(data) -> str | None:
    if not isinstance(data, dict):
        return "Body must be a JSON object."
    for field in ("vendor", "invoice_number", "amount", "due_date"):
        if field not in data:
            return f"Missing field: {field}"
    if not isinstance(data["vendor"], str) or not data["vendor"].strip():
        return "vendor must be a non-empty string."
    if not isinstance(data["invoice_number"], str) or not data["invoice_number"].strip():
        return "invoice_number must be a non-empty string."
    amount = data["amount"]
    if isinstance(amount, bool) or not isinstance(amount, (int, float)) or amount <= 0:
        return "amount must be a positive number (no currency symbol, no commas)."
    if not isinstance(data["due_date"], str) or not DATE_RE.match(data["due_date"]):
        return "due_date must be in YYYY-MM-DD format."
    return None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # keep the console quiet
        pass

    def _send(self, code: int, payload, content_type: str = "application/json"):
        body = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        state: State = self.server.state
        path = self.path.split("?")[0]
        if path == "/api/health":
            self._send(200, {"status": "ok"})
        elif path == "/api/help":
            self._send(200, API_HELP)
        elif path == "/api/records":
            with state.lock:
                self._send(200, {"records": list(state.records)})
        elif path == "/":
            with state.lock:
                rows = "".join(
                    f"<tr><td>{r['id']}</td><td>{html.escape(r['vendor'])}</td>"
                    f"<td>{html.escape(r['invoice_number'])}</td><td>{r['amount']}</td>"
                    f"<td>{html.escape(r['due_date'])}</td></tr>"
                    for r in state.records
                )
            page = (
                "<html><body><h2>NorthStar Finance System (mock)</h2>"
                "<table border=1 cellpadding=6><tr><th>ID</th><th>Vendor</th>"
                "<th>Invoice No</th><th>Amount</th><th>Due date</th></tr>"
                f"{rows}</table></body></html>"
            )
            self._send(200, page, "text/html")
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        state: State = self.server.state
        if self.path.split("?")[0] != "/api/records":
            self._send(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        with state.lock:
            state.post_attempts += 1
            if state.flaky and state.post_attempts == 1:
                self._send(503, {"error": "Service temporarily unavailable. Try again."})
                return
        try:
            data = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"error": "Body is not valid JSON."})
            return
        error = validate(data)
        if error:
            self._send(400, {"error": error})
            return
        with state.lock:
            if any(r["invoice_number"] == data["invoice_number"] for r in state.records):
                self._send(409, {"error": f"Invoice {data['invoice_number']} already exists."})
                return
            record = {
                "id": len(state.records) + 1,
                "vendor": data["vendor"],
                "invoice_number": data["invoice_number"],
                "amount": data["amount"],
                "due_date": data["due_date"],
            }
            state.records.append(record)
        self._send(201, {"record": record})


def make_server(port: int = 8000, flaky: bool = True) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.state = State(flaky)
    return server


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the mock internal finance system")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-flaky", action="store_true", help="disable the simulated first-request outage")
    args = parser.parse_args()
    srv = make_server(args.port, flaky=not args.no_flaky)
    print(f"Mock system running at http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
