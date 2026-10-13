"""Thin wrapper around Ollama so the agent core never imports a model library."""
from __future__ import annotations


class OllamaLLM:
    def __init__(self, model: str, host: str | None = None, temperature: float = 0.0, num_ctx: int = 8192):
        import ollama  # imported lazily so unit tests run without Ollama installed

        self.model = model
        self.client = ollama.Client(host=host) if host else ollama.Client()
        self.options = {"temperature": temperature, "num_ctx": num_ctx}

    def chat(self, system: str, messages: list[dict], tools: list[dict]) -> dict:
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "system", "content": system}] + messages,
            tools=tools,
            options=self.options,
        )
        message = response.message
        calls = [
            {"name": c.function.name, "arguments": dict(c.function.arguments or {})}
            for c in (message.tool_calls or [])
        ]
        return {"content": message.content or "", "tool_calls": calls}
