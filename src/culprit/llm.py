"""
LLM access for the judge baselines, via the Anthropic Messages API.

The judge arm runs on Claude Haiku 4.5 (`claude-haiku-4-5`) -- the cheapest
current model, which is the right call for a classification-shaped workload
that runs thousands of times. Thinking is left off: the judge returns a small
JSON verdict, and paying for reasoning tokens on every span would multiply the
cost of the arm without changing what it can see.

Three properties this layer is built around:

1. **Cost is tracked and hard-capped.** Token usage is converted to dollars at
   the model's published rate and accumulated; `budget_usd` raises before a
   sweep can run away. A benchmark that quietly drains an API balance is not a
   benchmark anyone can re-run.

2. **Responses are cached on disk**, keyed by model plus the exact prompt.
   Re-running an experiment after a scoring bug -- which happens constantly --
   then costs nothing and returns byte-identical judgements, so a change in
   the results is always a change in the code and never in the sampling.

3. **A mock backend exists** so the judge pipeline runs, and is tested, with
   no key and no spend. It is labelled as mock and its numbers are never
   reported.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import time
from dataclasses import dataclass
from typing import Any

CACHE_DIR = pathlib.Path(__file__).resolve().parents[2] / ".llm-cache"

# Published per-MTok rates. Used to price usage locally; the API does not
# return a dollar figure.
PRICING: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-opus-5": (5.00, 25.00),
}
DEFAULT_MODEL = "claude-haiku-4-5"


class BudgetExceeded(RuntimeError):
    pass


def load_api_key(env_path: str | pathlib.Path | None = None) -> str | None:
    """Environment first, then a local .env. Returns None when neither has it."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key.strip()
    path = pathlib.Path(env_path or pathlib.Path(__file__).resolve().parents[2] / ".env")
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip().startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip().strip("'\"")
    return None


def price(model: str, in_tokens: int, out_tokens: int) -> float:
    pin, pout = PRICING.get(model, PRICING[DEFAULT_MODEL])
    return (in_tokens / 1e6) * pin + (out_tokens / 1e6) * pout


@dataclass
class Reply:
    text: str
    usd: float = 0.0
    cached: bool = False
    model: str = ""
    in_tokens: int = 0
    out_tokens: int = 0


@dataclass
class LLMClient:
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    budget_usd: float = 2.0
    max_tokens: int = 256
    use_cache: bool = True
    spent: float = 0.0
    n_calls: int = 0
    n_cached: int = 0
    in_tokens: int = 0
    out_tokens: int = 0

    def __post_init__(self) -> None:
        self.api_key = self.api_key or load_api_key()
        self._client: Any = None
        CACHE_DIR.mkdir(exist_ok=True)

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _sdk(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def _cache_path(self, prompt: str) -> pathlib.Path:
        h = hashlib.sha256(f"{self.model}|{self.max_tokens}|{prompt}".encode()).hexdigest()[:32]
        return CACHE_DIR / f"{h}.json"

    def count_tokens(self, prompt: str, system: str | None = None) -> int:
        """Exact input-token count, for costing a sweep before committing to it."""
        r = self._sdk().messages.count_tokens(
            model=self.model,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
        )
        return int(r.input_tokens)

    def complete(self, prompt: str, system: str | None = None) -> Reply:
        full = f"{system or ''}\n---\n{prompt}"
        cache = self._cache_path(full)
        if self.use_cache and cache.exists():
            d = json.loads(cache.read_text())
            self.n_cached += 1
            return Reply(d["text"], 0.0, cached=True, model=self.model)

        if not self.api_key:
            raise RuntimeError(
                "no ANTHROPIC_API_KEY found (env or culprit/.env) -- the judge arm needs one"
            )
        if self.spent >= self.budget_usd:
            raise BudgetExceeded(f"spent ${self.spent:.4f} of ${self.budget_usd:.2f}")

        import anthropic

        last: Exception | None = None
        for attempt in range(5):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                }
                if system:
                    kwargs["system"] = system
                msg = self._sdk().messages.create(**kwargs)
                text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
                usd = price(self.model, msg.usage.input_tokens, msg.usage.output_tokens)
                self.spent += usd
                self.n_calls += 1
                self.in_tokens += msg.usage.input_tokens
                self.out_tokens += msg.usage.output_tokens
                if self.use_cache:
                    cache.write_text(json.dumps({"text": text, "usd": usd}))
                return Reply(text, usd, False, self.model, msg.usage.input_tokens, msg.usage.output_tokens)
            except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError) as exc:
                last = exc
                time.sleep(2.0 * (attempt + 1))
            except anthropic.APIStatusError as exc:  # 400/404 -- not worth retrying
                raise RuntimeError(f"Anthropic API error {exc.status_code}: {exc}") from exc
        raise RuntimeError(f"LLM call failed after retries: {last}")


@dataclass
class MockLLM:
    """
    Deterministic stand-in. Always picks the last tool span it is offered.

    Its purpose is to exercise the judge code paths in tests, not to stand in
    for a model. Results produced with it are never reported.
    """

    model: str = "mock"
    spent: float = 0.0
    n_calls: int = 0
    n_cached: int = 0
    available: bool = True

    def complete(self, prompt: str, system: str | None = None) -> Reply:
        self.n_calls += 1
        ids = [ln.split("]")[0].lstrip("[") for ln in prompt.splitlines() if ln.startswith("[")]
        return Reply(
            json.dumps({"span_id": ids[-1] if ids else "unknown", "confidence": 0.5,
                        "wrong": True, "why": "mock"}),
            0.0, model="mock",
        )
