"""Response normalisation  (DFD sub-process 3.1, STD state NORMALIZING).

Different phrasings of the same answer ('Paris', 'The answer is paris.') must
count as one answer, otherwise disagreement would be overestimated.

Known limitation: for classification the *earliest* option mentioned in the text
is taken, so 'not entailment' would be read as 'entailment'.  A stricter parser
or a judge model is part of the remaining work.
"""
from __future__ import annotations

import re
import string

import pandas as pd

from .data_collection import CLASSIFICATION, Item

INVALID = "__invalid__"
_PREFIXES = ("the answer is", "my answer is", "answer:", "i think", "it is", "it's")
_ARTICLES = ("the ", "a ", "an ")


def normalize_choice(raw: str, options: tuple[str, ...]) -> str:
    """Map free text to the option mentioned first; INVALID if none is found."""
    text = raw.lower()
    best_pos, best_opt = None, INVALID
    for opt in options:
        # 'not_sarcastic' also matches 'not sarcastic' / 'not-sarcastic'
        pattern = r"\b" + re.escape(opt.lower()).replace("_", "[ _-]") + r"\b"
        m = re.search(pattern, text)
        if m and (best_pos is None or m.start() < best_pos):
            best_pos, best_opt = m.start(), opt
    return best_opt


def normalize_open(raw: str) -> str:
    """Lower-case, drop answer prefixes, punctuation and leading articles."""
    text = raw.lower().strip()
    for p in _PREFIXES:
        if text.startswith(p):
            text = text[len(p):].strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    for a in _ARTICLES:
        if text.startswith(a):
            text = text[len(a):]
    return text or INVALID


def normalize_responses(df: pd.DataFrame, items: list[Item]) -> pd.DataFrame:
    """Add `answer` (normalised) and `valid` (bool) columns to the responses."""
    by_id = {i.item_id: i for i in items}
    out = df.copy()

    def _norm(row) -> str:
        item = by_id[row["item_id"]]
        if item.task_type == CLASSIFICATION:
            return normalize_choice(str(row["raw_text"]), item.options)
        return normalize_open(str(row["raw_text"]))

    out["answer"] = out.apply(_norm, axis=1) if len(out) else pd.Series(dtype=str)
    out["valid"] = out["answer"] != INVALID
    return out
