"""Central configuration for the LLM Disagreement Analysis pipeline.

All tunable parameters live here so that every module receives the same,
validated settings (STD state: CONFIGURING).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Project root = parent of src/.  No absolute paths are hard-coded anywhere.
ROOT = Path(__file__).resolve().parents[1]


class ConfigError(ValueError):
    """Raised when settings are invalid (STD guard: [settings invalid])."""


@dataclass
class Config:
    # ---- Data Collection ----
    data_path: Path = ROOT / "data" / "sample_dataset.csv"
    limit: int | None = None              # use only the first N items (debugging)

    # ---- Multi-Response Sampling ----
    backend: str = "mock"                 # mock | anthropic | ollama
    models: list[str] = field(default_factory=lambda: ["model-A", "model-B", "model-C"])
    n_samples: int = 5                    # responses per (prompt, model)
    temperature: float = 1.0              # must be > 0 so repeated sampling can differ
    max_tokens: int = 64
    max_retries: int = 3
    use_cache: bool = True
    seed: int = 42                        # only used by the mock backend

    # ---- Disagreement Detection ----
    agreement_threshold: float = 0.75     # agreement < threshold  => item is flagged
    jaccard_threshold: float = 0.5        # open-ended answers: token-overlap clustering
    human_agreement_threshold: float = 0.6   # humans "disagree" below this top-label share
    llm_overconfidence_threshold: float = 0.8  # LLM "confident" at/above this agreement

    # ---- Reporting ----
    output_dir: Path = ROOT / "outputs"
    top_k_uncertain: int = 10

    def validate(self) -> None:
        errs = []
        if self.backend not in {"mock", "anthropic", "ollama"}:
            errs.append(f"backend must be mock|anthropic|ollama, got '{self.backend}'")
        if not self.models:
            errs.append("at least one model is required")
        if self.n_samples < 2:
            errs.append("n_samples must be >= 2 (disagreement needs several responses)")
        if self.temperature <= 0:
            errs.append("temperature must be > 0, otherwise repeated samples are identical")
        for name in ("agreement_threshold", "human_agreement_threshold",
                     "llm_overconfidence_threshold", "jaccard_threshold"):
            v = getattr(self, name)
            if not 0 < v <= 1:
                errs.append(f"{name} must be in (0, 1], got {v}")
        if self.max_retries < 1:
            errs.append("max_retries must be >= 1")
        if errs:
            raise ConfigError("; ".join(errs))
