# Autonomous AI Task Worker (local, Ollama)

A small agent that takes a plain-English task and completes it against a company
system: it explores files and an internal API, extracts what it needs, performs the
action, **reads the result back to verify it**, and returns evidence.

Everything runs locally. No cloud API keys and no real credentials are used.

- GitHub: `<https://github.com/varshasingh696/taskpilot-agentk>`
- Demo video: `<https://drive.google.com/file/d/1hJvdJKfVTOtHSvzPCUhi_U0AvQfv1hJZ/view?usp=sharing>`

## Quick start

Requirements: Python 3.10+, [Ollama](https://ollama.com).

```bash
# 1. model (any Ollama model with tool-calling support works)
ollama pull qwen2.5:7b

# 2. python deps
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. terminal A: the mock company system
python -m mock_app.server

# 4. terminal B: run the agent
python run_agent.py "Find the latest invoice from Acme Corp, extract the amount and due date, enter it into our internal system, and tell me once it is done."
```

Open http://127.0.0.1:8000 to watch the internal system's table update.
Use `--model qwen3:8b` / `llama3.1:8b` to switch models, `--auto-approve` to skip the approval prompt.

### Try different tasks (same code, nothing changed)

```bash
python run_agent.py "Enter all invoices from Globex Industries into our internal system."
python run_agent.py "How many invoices are in our internal system and what is their total amount?"
python run_agent.py "Enter the latest invoice from Initech into our system."   # no such file -> agent should ask or report
```

### Tests

```bash
python -m unittest -v tests.test_agent
```

The tests use a scripted fake LLM against the real mock app, so they check the
harness (retry, error recovery, verification gate, approval, safety) deterministically.
They do not test model quality; that depends on the Ollama model you use.

## Architecture

```
                 +-----------------------------+
 user task ----> |  Agent loop (agent/core.py) | <---- working memory (re-injected each step)
                 +--------------+--------------+
                                | asks LLM for next tool call
                                v
                        Ollama (local LLM)
                                |
                                v
                 +-----------------------------+
                 |  Harness around every call  |
                 |  - repeat guard             |
                 |  - approval gate (writes)   |
                 |  - retry w/ backoff (5xx)   |
                 |  - verification gate        |
                 |  - run log (runs/*.jsonl)   |
                 +--------------+--------------+
                                v
        list_files  read_file  http_request  remember  ask_user  finish
                                |
                                v
              data/ workspace      Mock internal system (mock_app/)
              (invoice files)      GET/POST /api/records, /api/help
```

Loop: observe -> decide -> act -> observe, until `finish`, a step limit, or a stop condition.

## Key design decisions

1. **The model decides, the harness enforces.** The LLM plans and picks tools. Safety and
   reliability rules live in plain code, so a weak local model cannot skip them.
2. **Generic tools, no invoice logic.** The tools are file read/list, HTTP, memory, ask-user and finish.
   The system prompt says nothing about invoices. Changing the task does not change the code (generalization).
3. **API discovery, enforced.** The agent reads `/api/help` instead of having endpoints hard-coded, and a discovery gate blocks any write until the system has been read at least once (small models tend to guess endpoints). 404/405 responses also carry a hint to read the help.
4. **Verification gate.** After a successful write, `finish(success=true)` is refused until the agent has read state back.
   The evidence in the final answer is the data it actually read back.
5. **Two kinds of failure, two responses.** Transient errors (5xx, timeouts) are retried automatically with exponential backoff.
   Client errors (4xx) are returned to the model so it can fix its own input (for example date format).
6. **Human approval for side effects.** Every non-GET request needs a yes/no. If denied, the agent is told not to retry.
7. **Working memory.** Facts stored with `remember` are re-injected into the system prompt each step, so they survive long contexts.
8. **Standard library for the mock app and HTTP.** The only dependency is the `ollama` client, which keeps setup short.
9. **Mock system with deliberate faults.** The first POST returns 503 and validation is strict, so reliability behaviour can be seen in the demo.

## Models, APIs, frameworks used

- Ollama (local) with `qwen2.5:7b` by default; tested options: `qwen3:8b`, `llama3.1:8b` (`<update after you test>`)
- `ollama` Python client
- Python standard library (`http.server`, `urllib`, `unittest`)
- No agent framework, no cloud API, no external services
- AI coding assistance was used while writing this project (`<say which tools you used>`)

## Assumptions

- The "internal system" is a mock REST API, and invoices are text files in `data/inbox/`.
- "Latest invoice" means the most recent invoice date inside the document, not the file name or modified time.
- Currency is kept implicit (INR); the mock system stores a plain number.
- One task at a time, one user, local machine.

## Known limitations

- No browser/GUI control. The agent works through files and HTTP only.
- Verification checks that the stored record exists and is readable; it does not independently re-check extracted values against the source document.
- Retrying a POST after a connection error can in theory double-submit; the mock system rejects duplicate invoice numbers, but a real system would need idempotency keys.
- Small local models can pick wrong tools or produce malformed arguments. The repeat guard, step limit and error feedback reduce this but do not remove it.
- Only text invoices are supported (no PDF/OCR).
- Context is the full message history, with no summarization, so very long tasks would exceed it.

## What I would build next

- Browser tool (Playwright) so the same loop can operate real web UIs.
- PDF/image invoice extraction.
- Independent verification: re-extract from the source and compare with the stored record.
- Idempotency keys, per-tool permission policies and an audit trail.
- Context summarization and a small evaluation suite that scores tasks automatically across models.
