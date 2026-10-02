"""Data Collection Module  (DFD Process P1, STD states CONFIGURING -> DATASET_READY).

Loads the evaluation dataset from CSV, validates it, and prepares one prompt
per item.  Items may carry an optional distribution of human votes; this is what
the literature (Lee et al. 2023; Ni et al. 2025) compares LLM output against.

CSV columns
-----------
id           unique item id                                   (required)
task_type    'classification' or 'open'                       (required)
instruction  task instruction shown to the LLM                (required)
text         the content to judge / the question              (required)
options      'a|b|c' label set (classification only)
human_votes  'a:7|b:3' annotator vote counts (optional)
reference    reference answer (open items; used by the mock backend now and by
             the LLM-as-Judge module in the remaining 50%)
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CLASSIFICATION = "classification"
OPEN = "open"
REQUIRED_COLUMNS = ["id", "task_type", "instruction", "text"]


class DatasetError(ValueError):
    """Raised when the dataset is invalid (STD guard: [dataset invalid])."""


@dataclass(frozen=True)
class Item:
    item_id: str
    task_type: str
    instruction: str
    text: str
    options: tuple[str, ...] = ()
    human_votes: dict[str, int] | None = None
    reference: str | None = None

    def human_distribution(self) -> np.ndarray | None:
        """Human vote shares in `options` order, or None if no votes."""
        if not self.human_votes:
            return None
        counts = np.array([self.human_votes.get(o, 0) for o in self.options], dtype=float)
        return counts / counts.sum()


def parse_options(raw: str) -> tuple[str, ...]:
    return tuple(o.strip() for o in raw.split("|") if o.strip())


def parse_votes(raw: str) -> dict[str, int]:
    """'entailment:7|neutral:3' -> {'entailment': 7, 'neutral': 3}."""
    votes: dict[str, int] = {}
    for part in raw.split("|"):
        part = part.strip()
        if not part:
            continue
        label, sep, count = part.rpartition(":")
        if not sep or not label.strip():
            raise ValueError(f"bad vote entry '{part}' (expected label:count)")
        votes[label.strip()] = int(count)
    return votes


def load_dataset(path: Path, limit: int | None = None) -> list[Item]:
    """Load and validate the dataset. Collects *all* problems before raising."""
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"dataset file not found: {path}")
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise DatasetError(f"missing required column(s): {missing}")
        rows = list(reader)
    if not rows:
        raise DatasetError("dataset is empty")

    items: list[Item] = []
    errors: list[str] = []
    seen: set[str] = set()
    for line_no, row in enumerate(rows, start=2):   # line 1 is the header
        rid = (row.get("id") or "").strip()
        where = f"line {line_no} (id='{rid}')"
        if not rid:
            errors.append(f"line {line_no}: empty id"); continue
        if rid in seen:
            errors.append(f"{where}: duplicate id")
        seen.add(rid)

        ttype = (row.get("task_type") or "").strip().lower()
        if ttype not in (CLASSIFICATION, OPEN):
            errors.append(f"{where}: task_type must be 'classification' or 'open'"); continue
        if not (row.get("text") or "").strip():
            errors.append(f"{where}: empty text")
        if not (row.get("instruction") or "").strip():
            errors.append(f"{where}: empty instruction")

        options = parse_options(row.get("options") or "")
        if ttype == CLASSIFICATION and len(options) < 2:
            errors.append(f"{where}: classification items need >= 2 options")

        votes = None
        if (row.get("human_votes") or "").strip():
            try:
                votes = parse_votes(row["human_votes"])
            except ValueError as exc:
                errors.append(f"{where}: {exc}")
            else:
                if ttype != CLASSIFICATION:
                    errors.append(f"{where}: human_votes only supported for classification")
                elif not set(votes) <= set(options):
                    errors.append(f"{where}: human_votes labels {sorted(set(votes) - set(options))} not in options")
                elif sum(votes.values()) <= 0 or min(votes.values()) < 0:
                    errors.append(f"{where}: human_votes counts must be >= 0 and sum > 0")

        items.append(Item(
            item_id=rid, task_type=ttype,
            instruction=(row.get("instruction") or "").strip(),
            text=(row.get("text") or "").strip(),
            options=options, human_votes=votes,
            reference=(row.get("reference") or "").strip() or None,
        ))
    if errors:
        raise DatasetError("; ".join(errors))
    return items[:limit] if limit else items


def build_prompt(item: Item) -> str:
    """Prepare the LLM prompt for one item (paper: 'prepares prompts')."""
    if item.task_type == CLASSIFICATION:
        return (f"{item.instruction}\n\n{item.text}\n\n"
                f"Answer with exactly one of: {', '.join(item.options)}. "
                f"Reply with only that label.")
    return f"{item.instruction}\n\n{item.text}\n\nAnswer concisely in a few words."
