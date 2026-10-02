"""LLM back-ends used by the Multi-Response Sampling module.

The external entity 'LLM Model Pool' of the DFD is represented by the
`LLMClient` interface.  Three implementations are provided:

* MockLLMClient      - offline *simulator* (default).  NOT a language model.
* AnthropicClient    - real API calls (needs `pip install anthropic` + API key).
* OllamaClient       - local models over HTTP (standard library only).

IMPORTANT: results from the mock backend validate the *pipeline*; they say
nothing about how real LLMs behave.  The mock derives its answer distribution
from the dataset's human votes / references, so it can never be used as evidence.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .config import Config, ConfigError
from .data_collection import CLASSIFICATION, Item


class LLMUnavailable(RuntimeError):
    """LLM Model Pool could not be reached (STD event: queryFailed)."""


@dataclass(frozen=True)
class GenRequest:
    item_id: str
    prompt: str
    sample_idx: int
    temperature: float
    max_tokens: int


class LLMClient(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate(self, req: GenRequest) -> str:
        """Return one raw text response, or raise LLMUnavailable."""


# ----------------------------------------------------------------------------
# Mock simulator
# ----------------------------------------------------------------------------
def stable_int(*parts) -> int:
    """Deterministic hash -> int (Python's built-in hash() is salted per run)."""
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16], 16)


class MockLLMClient(LLMClient):
    """Simulates an LLM so the pipeline can run offline and reproducibly.

    Classification items with human votes: answers are sampled from the human
    distribution sharpened by a per-model exponent `gamma` (gamma > 1 makes a
    model more confident than the crowd - the overconfidence effect studied by
    Lu et al. 2025).  Open items: the reference answer, sometimes phrased
    differently or replaced by another item's answer.
    """
    GAMMAS = {"model-A": 2.5, "model-B": 1.5, "model-C": 1.0}
    OPEN_ACCURACY = {"model-A": 0.95, "model-B": 0.90, "model-C": 0.85}
    STYLES = ["{a}", "{a}.", "The answer is {a}", "I think {a}", "{A}"]

    def __init__(self, model_name: str, items: dict[str, Item], seed: int):
        super().__init__(model_name)
        self.items, self.seed = items, seed
        self.references = sorted({i.reference for i in items.values() if i.reference})

    def generate(self, req: GenRequest) -> str:
        rng = random.Random(stable_int(self.seed, self.model_name, req.item_id, req.sample_idx))
        item = self.items[req.item_id]
        style = rng.choice(self.STYLES)
        if item.task_type == CLASSIFICATION:
            dist = item.human_distribution()
            if dist is None:                       # no votes: near-deterministic on option 0
                weights = [0.9] + [0.1 / (len(item.options) - 1)] * (len(item.options) - 1)
            else:
                g = self.GAMMAS.get(self.model_name, 1.5)
                weights = [(p + 0.02) ** g for p in dist]
            ans = rng.choices(item.options, weights=weights, k=1)[0]
        else:
            ref = item.reference or "unknown"
            acc = self.OPEN_ACCURACY.get(self.model_name, 0.9)
            if rng.random() < acc:
                ans = ref
                if " " in ref and rng.random() < 0.3:   # e.g. 'William Shakespeare' -> 'Shakespeare'
                    ans = ref.split()[-1]
            else:
                others = [r for r in self.references if r != ref] or ["unknown"]
                ans = rng.choice(others)
        return style.format(a=ans, A=ans.upper())


# ----------------------------------------------------------------------------
# Real back-ends (not exercised in this milestone's test run: they need
# network access / credentials / a local server)
# ----------------------------------------------------------------------------
class AnthropicClient(LLMClient):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        try:
            import anthropic
        except ImportError as exc:
            raise ConfigError("backend 'anthropic' needs: pip install anthropic") from exc
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ConfigError("set the ANTHROPIC_API_KEY environment variable")
        self._anthropic = anthropic
        self._client = anthropic.Anthropic()

    def generate(self, req: GenRequest) -> str:
        try:
            msg = self._client.messages.create(
                model=self.model_name, max_tokens=req.max_tokens,
                temperature=req.temperature,
                messages=[{"role": "user", "content": req.prompt}])
        except self._anthropic.APIError as exc:
            raise LLMUnavailable(f"{self.model_name}: {exc}") from exc
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


class OllamaClient(LLMClient):
    URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")

    def generate(self, req: GenRequest) -> str:
        body = json.dumps({"model": self.model_name, "prompt": req.prompt, "stream": False,
                           "options": {"temperature": req.temperature,
                                       "num_predict": req.max_tokens}}).encode()
        http_req = urllib.request.Request(self.URL, data=body,
                                          headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(http_req, timeout=120) as resp:
                return json.loads(resp.read())["response"]
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            raise LLMUnavailable(f"{self.model_name}: {exc}") from exc


def build_clients(cfg: Config, items: list[Item]) -> dict[str, LLMClient]:
    """Create one client per model name in cfg.models."""
    by_id = {i.item_id: i for i in items}
    clients: dict[str, LLMClient] = {}
    for m in cfg.models:
        if cfg.backend == "mock":
            clients[m] = MockLLMClient(m, by_id, cfg.seed)
        elif cfg.backend == "anthropic":
            clients[m] = AnthropicClient(m)
        else:
            clients[m] = OllamaClient(m)
    return clients
