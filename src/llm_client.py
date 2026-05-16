"""Small OpenAI-compatible LLM client for student answer generation."""

from __future__ import annotations

import os
import time


class LLMClient:
    """Generate text with the official OpenAI client or a dry-run placeholder."""

    def __init__(self, model: str | None = None, dry_run: bool = False) -> None:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass

        self.dry_run = dry_run
        self.model = model or os.getenv("MODEL_NAME") or "gpt-4o-mini"
        self._client = None
        self.timeout = float(os.getenv("OPENAI_TIMEOUT", "60"))

        if self.dry_run:
            return

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is missing. Use --dry_run for a local test, "
                "or set OPENAI_API_KEY before running real LLM generation."
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("The openai package is missing. Run: pip install -r requirements.txt") from exc

        base_url = os.getenv("OPENAI_BASE_URL") or None
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=self.timeout)

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 256) -> str:
        """Return generated text for a prompt."""
        if self.dry_run:
            return self._dry_response(prompt)

        assert self._client is not None
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content or ""
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(1 + attempt)

        raise RuntimeError(f"LLM generation failed after 3 attempts: {last_error}") from last_error

    def _dry_response(self, prompt: str) -> str:
        """Return a deterministic placeholder answer based on the rendered prompt."""
        lower = prompt.lower()
        prefix = "I think " if "experienced science teacher" in lower else ""
        if "low science student" in lower or "low student" in lower:
            return f"{prefix}maybe it happens because of the thing in the question, but I am not really sure."
        if "medium science student" in lower or "medium student" in lower:
            return f"{prefix}it is related to the main science idea, but I might be missing the exact mechanism."
        if "high science student" in lower or "high student" in lower:
            return f"{prefix}the key mechanism explains the observation, so the answer should connect the cause to the result."
        return "This is a deterministic dry-run student answer."
