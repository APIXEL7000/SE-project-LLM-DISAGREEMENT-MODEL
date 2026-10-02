"""Command-line entry point.

    python main.py                         # offline demo with the mock backend
    python main.py --data data/my.csv --backend anthropic --models claude-haiku-4-5-20251001
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.config import Config, ConfigError
from src.data_collection import DatasetError
from src.llm_clients import LLMUnavailable
from src.pipeline import run_pipeline


def parse_args(argv=None) -> Config:
    d = Config()
    p = argparse.ArgumentParser(description="LLM disagreement & uncertainty analysis (first 50%)")
    p.add_argument("--data", type=Path, default=d.data_path, help="dataset CSV")
    p.add_argument("--out", type=Path, default=d.output_dir, help="output directory")
    p.add_argument("--backend", choices=["mock", "anthropic", "ollama"], default=d.backend)
    p.add_argument("--models", nargs="+", default=d.models, help="model names / ids")
    p.add_argument("--samples", type=int, default=d.n_samples, help="responses per (prompt, model)")
    p.add_argument("--temperature", type=float, default=d.temperature)
    p.add_argument("--threshold", type=float, default=d.agreement_threshold,
                   help="flag items whose agreement is below this")
    p.add_argument("--limit", type=int, default=None, help="only first N items")
    p.add_argument("--no-cache", action="store_true", help="ignore/skip the response cache")
    p.add_argument("--seed", type=int, default=d.seed, help="mock backend seed")
    a = p.parse_args(argv)
    return Config(data_path=a.data, output_dir=a.out, backend=a.backend, models=a.models,
                  n_samples=a.samples, temperature=a.temperature, agreement_threshold=a.threshold,
                  limit=a.limit, use_cache=not a.no_cache, seed=a.seed)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    cfg = parse_args(argv)
    try:
        result = run_pipeline(cfg)
    except (DatasetError, ConfigError) as exc:          # STD: configDone [invalid]
        logging.error("Invalid input: %s", exc)
        return 2
    except LLMUnavailable as exc:                       # STD: queryFailed
        logging.error("LLM Model Pool unavailable: %s", exc)
        return 3
    print("\n=== SUMMARY ===")
    print(json.dumps(result["summary"], indent=2))
    print(f"\nFull report: {result['report']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
