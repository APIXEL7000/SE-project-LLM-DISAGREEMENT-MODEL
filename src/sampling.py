"""Multi-Response Sampling Module  (DFD Process P2, STD state QUERYING).

For every prompt, every model is asked `n_samples` times at temperature > 0.
Repeating the query exposes the model's *internal* uncertainty, which a single
response hides (Nan et al. 2025; Xiong et al. 2024; Wang et al. 2022).

Responses are cached in a JSONL file so an interrupted run can be resumed
without paying for the same queries again.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import pandas as pd

from .config import Config
from .data_collection import Item, build_prompt
from .llm_clients import GenRequest, LLMClient, LLMUnavailable

log = logging.getLogger(__name__)


def _cache_key(cfg: Config, model: str, req: GenRequest) -> str:
    raw = f"{cfg.backend}|{model}|{req.item_id}|{req.sample_idx}|{req.temperature}|{req.prompt}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _load_cache(path: Path) -> dict[str, str]:
    cache: dict[str, str] = {}
    if path.is_file():
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    cache[rec["key"]] = rec["raw_text"]
                except (json.JSONDecodeError, KeyError):
                    continue        # ignore a half-written last line
    return cache


def _generate_with_retry(client: LLMClient, req: GenRequest, retries: int) -> str:
    for attempt in range(1, retries + 1):
        try:
            return client.generate(req)
        except LLMUnavailable as exc:
            log.warning("attempt %d/%d failed for %s: %s", attempt, retries, client.model_name, exc)
            if attempt == retries:
                raise
            time.sleep(0.5 * 2 ** (attempt - 1))       # exponential back-off
    raise AssertionError("unreachable")


def sample_responses(items: list[Item], clients: dict[str, LLMClient], cfg: Config,
                     cache_path: Path) -> pd.DataFrame:
    """Return a DataFrame: item_id, model, sample_idx, raw_text.

    Raises LLMUnavailable if a request still fails after `max_retries`
    (the STD's `queryFailed [LLM Model Pool unavailable]` event).
    """
    cache = _load_cache(cache_path) if cfg.use_cache else {}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    rows, n_cached, n_new = [], 0, 0
    with open(cache_path, "a", encoding="utf-8") as cache_fh:
        for item in items:
            prompt = build_prompt(item)
            for model, client in clients.items():
                for k in range(cfg.n_samples):
                    req = GenRequest(item.item_id, prompt, k, cfg.temperature, cfg.max_tokens)
                    key = _cache_key(cfg, model, req)
                    if key in cache:
                        raw, n_cached = cache[key], n_cached + 1
                    else:
                        raw = _generate_with_retry(client, req, cfg.max_retries)
                        n_new += 1
                        if cfg.use_cache:
                            cache_fh.write(json.dumps({"key": key, "item_id": item.item_id,
                                                       "model": model, "sample_idx": k,
                                                       "raw_text": raw}) + "\n")
                            cache_fh.flush()
                    rows.append({"item_id": item.item_id, "model": model,
                                 "sample_idx": k, "raw_text": raw})
    log.info("sampling done: %d new, %d from cache, %d total", n_new, n_cached, len(rows))
    return pd.DataFrame(rows)
