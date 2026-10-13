"""CLI entry point:  python run_agent.py "your task in plain English" """
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

from agent.core import Agent
from agent.llm import OllamaLLM
from agent.tools import Toolbox

ROOT = Path(__file__).parent


def check_mock_app(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(base_url + "/api/health", timeout=3):
            return True
    except Exception:
        return False


def check_ollama(llm: OllamaLLM, model: str) -> str | None:
    """Return an error message, or None if Ollama is reachable and the model is pulled."""
    try:
        names = [m.model for m in llm.client.list().models]
    except Exception as exc:
        return f"Cannot reach Ollama ({exc}). Start it with:  ollama serve"
    if model not in names and f"{model}:latest" not in names:
        return f"Model '{model}' is not pulled. Run:  ollama pull {model}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Autonomous AI task worker (Ollama)")
    parser.add_argument("task", help="the task, in plain English")
    parser.add_argument("--model", default="qwen2.5:7b", help="Ollama model with tool-calling support")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="internal system base URL")
    parser.add_argument("--ollama-host", default=None, help="e.g. http://localhost:11434")
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--auto-approve", action="store_true", help="skip the human approval prompt (demo only)")
    args = parser.parse_args()

    llm = OllamaLLM(args.model, host=args.ollama_host)
    problem = check_ollama(llm, args.model)
    if problem:
        print(problem)
        return 2
    if not check_mock_app(args.base_url):
        print(f"Mock company app not reachable at {args.base_url}. Start it in another terminal:\n"
              "  python -m mock_app.server")
        return 2

    def approve(description: str) -> bool:
        if args.auto_approve:
            print(f"    (auto-approved) {description}")
            return True
        return input(f"\n  APPROVAL NEEDED -> {description}\n  Allow? [y/N] ").strip().lower() in {"y", "yes"}

    def ask_user(question: str) -> str:
        return input(f"\n  AGENT ASKS: {question}\n  > ")

    toolbox = Toolbox(ROOT / "data", args.base_url, ask_user)
    agent = Agent(llm, toolbox, approve, max_steps=args.max_steps, log_dir=str(ROOT / "runs"))

    print(f"Task: {args.task}\nModel: {args.model}\n")
    outcome = agent.run(args.task)

    print("\n" + "=" * 60)
    print("SUCCESS" if outcome["success"] else "NOT COMPLETED")
    print(f"Summary : {outcome['summary']}")
    print(f"Evidence: {outcome['evidence']}")
    print(f"Steps   : {outcome['steps']}   Log: {outcome['log_file']}")
    return 0 if outcome["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
