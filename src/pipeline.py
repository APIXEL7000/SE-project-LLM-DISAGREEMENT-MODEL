"""End-to-end pipeline.  Log lines use the state names of the project's
State Transition Diagram so the run can be followed on the diagram."""
from __future__ import annotations

import logging
from pathlib import Path

from .config import Config
from .data_collection import load_dataset
from .disagreement import analyze_all, summarize
from .llm_clients import build_clients
from .normalization import normalize_responses
from .reporting import generate_report
from .sampling import sample_responses

log = logging.getLogger("pipeline")


def run_pipeline(cfg: Config) -> dict:
    log.info("[STATE] AWAITING_INPUT -> CONFIGURING")
    cfg.validate()
    items = load_dataset(cfg.data_path, cfg.limit)            # raises DatasetError
    n_cls = sum(i.task_type == "classification" for i in items)
    log.info("[STATE] DATASET_READY  (%d items: %d classification, %d open; %d with human votes)",
             len(items), n_cls, len(items) - n_cls, sum(i.human_votes is not None for i in items))

    log.info("[STATE] QUERYING  (%d models x %d samples x %d prompts)",
             len(cfg.models), cfg.n_samples, len(items))
    clients = build_clients(cfg, items)
    cache = Path(cfg.output_dir) / f"response_cache_{cfg.backend}.jsonl"
    raw = sample_responses(items, clients, cfg, cache)        # raises LLMUnavailable

    log.info("[STATE] NORMALIZING")
    responses = normalize_responses(raw, items)

    log.info("[STATE] EVALUATING  -> skipped (LLM-as-Judge belongs to the remaining 50%)")

    log.info("[STATE] DETECTING")
    metrics = analyze_all(items, responses, cfg)
    summary = summarize(metrics, cfg)

    log.info("[STATE] RESOLVING  -> skipped (Debate & Resolution belongs to the remaining 50%)")
    log.info("[STATE] RESULTS_STORED -> GENERATING_REPORT")
    report = generate_report(items, responses, metrics, summary, cfg)
    log.info("[STATE] REPORT_DELIVERED  -> %s", report)
    return {"items": items, "responses": responses, "metrics": metrics,
            "summary": summary, "report": report}
