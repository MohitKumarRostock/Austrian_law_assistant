#!/usr/bin/env python3
"""
Loads Austrian-law TRAIN/TEST query sets from JSON/JSONL and exports legacy constants.

Exports (legacy API):
  - BASE_QUERY_SET
  - QUERY_SET
  - TRAIN_QUERY_SET
  - TEST_QUERY_SET

Expected per-query fields:
  - query_id (str)
  - query_text (str)
  - consensus_law (str)

Recommended: place train.jsonl and test.jsonl next to this file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

Query = Dict[str, Any]


def _read_jsonl(path: Path) -> List[Query]:
    out: List[Query] = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL in {path} at line {ln}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Each JSONL line must be an object/dict: {path} line {ln}")
            out.append(obj)
    return out


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_queries_file(path: Path) -> List[Query]:
    if not path.exists():
        raise FileNotFoundError(str(path))

    suf = path.suffix.lower()
    if suf == ".jsonl":
        qs = _read_jsonl(path)
    elif suf == ".json":
        obj = _read_json(path)
        if not isinstance(obj, list):
            raise ValueError(f"Expected JSON array in {path} (train/test file). Got: {type(obj).__name__}")
        qs = obj
    else:
        # Try JSONL as a default
        qs = _read_jsonl(path)

    if not isinstance(qs, list):
        raise ValueError(f"Expected list of queries in {path}. Got: {type(qs).__name__}")
    return qs


def _validate_queries(qs: List[Query], split_name: str) -> None:
    if not qs:
        raise ValueError(f"{split_name} query set is empty.")

    required = ("query_id", "query_text", "consensus_law")
    ids: List[str] = []

    for i, q in enumerate(qs):
        if not isinstance(q, dict):
            raise ValueError(f"{split_name}[{i}] must be an object/dict.")
        for k in required:
            if k not in q or not str(q[k]).strip():
                raise ValueError(f"{split_name}[{i}] missing/empty required field '{k}'.")
        ids.append(str(q["query_id"]).strip())

    if len(set(ids)) != len(ids):
        # Show first few duplicates
        seen = set()
        dups = []
        for x in ids:
            if x in seen:
                dups.append(x)
                if len(dups) >= 10:
                    break
            seen.add(x)
        raise ValueError(f"{split_name} contains duplicate query_id values (examples: {dups}).")


def _resolve_paths() -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """
    Returns (combined_json, train_path, test_path) where only one strategy is selected.
    """
    # Strategy 1: single combined JSON file with {"train": [...], "test": [...]}
    combined = os.environ.get("AUSTLAW_QUERYSET_JSON")
    if combined:
        return Path(combined), None, None

    # Strategy 2: explicit train/test files
    train_p = os.environ.get("AUSTLAW_TRAIN_QUERIES")
    test_p = os.environ.get("AUSTLAW_TEST_QUERIES")
    if train_p and test_p:
        return None, Path(train_p), Path(test_p)

    # Strategy 3: directory override
    qs_dir = os.environ.get("AUSTLAW_QUERYSET_DIR")
    here = Path(__file__).resolve().parent
    base = Path(qs_dir) if qs_dir else here

    # Preferred defaults
    candidates = [
        (base / "train.jsonl", base / "test.jsonl"),
        (base / "train.json", base / "test.json"),
    ]
    for tr, te in candidates:
        if tr.exists() and te.exists():
            return None, tr, te

    return None, None, None


def load_query_sets() -> Tuple[List[Query], List[Query], str, Dict[str, str]]:
    """
    Returns (train, test, source, paths)
    """
    combined, train_path, test_path = _resolve_paths()

    if combined:
        obj = _read_json(combined)
        if not isinstance(obj, dict) or "train" not in obj or "test" not in obj:
            raise ValueError(f"{combined} must be an object with keys 'train' and 'test'.")
        train = obj["train"]
        test = obj["test"]
        if not isinstance(train, list) or not isinstance(test, list):
            raise ValueError(f"{combined} keys 'train'/'test' must be arrays.")
        _validate_queries(train, "TRAIN_QUERY_SET")
        _validate_queries(test, "TEST_QUERY_SET")
        return train, test, f"combined_json:{combined}", {"combined": str(combined)}

    if train_path and test_path:
        train = _load_queries_file(train_path)
        test = _load_queries_file(test_path)
        _validate_queries(train, "TRAIN_QUERY_SET")
        _validate_queries(test, "TEST_QUERY_SET")
        return (
            train,
            test,
            "separate_files",
            {"train": str(train_path), "test": str(test_path)},
        )

    # Strategy 4: fallback to legacy embedded module, if present
    try:
        from query_set_legacy_paraphrase import (  # type: ignore
            BASE_QUERY_SET as LEGACY_BASE,
            QUERY_SET as LEGACY_ALL,
            TRAIN_QUERY_SET as LEGACY_TRAIN,
            TEST_QUERY_SET as LEGACY_TEST,
        )

        # Light validation (legacy assumed correct)
        return (
            list(LEGACY_TRAIN),
            list(LEGACY_TEST),
            "legacy_module:query_set_legacy_paraphrase",
            {"legacy": "query_set_legacy_paraphrase"},
        )
    except Exception as e:
        raise RuntimeError(
            "No external query files found and legacy fallback module not available.\n"
            "Provide one of:\n"
            "  - AUSTLAW_QUERYSET_JSON=/path/to/querysets.json (with keys train/test)\n"
            "  - AUSTLAW_TRAIN_QUERIES=/path/to/train.jsonl and AUSTLAW_TEST_QUERIES=/path/to/test.jsonl\n"
            "  - Place train_10k.jsonl and test_2k.jsonl next to query_set.py\n"
        ) from e


# --------- Module-level exports (legacy API) ---------
TRAIN_QUERY_SET, TEST_QUERY_SET, DATASET_SOURCE, DATASET_PATHS = load_query_sets()
QUERY_SET: List[Query] = list(TRAIN_QUERY_SET) + list(TEST_QUERY_SET)
BASE_QUERY_SET: List[Query] = QUERY_SET[: min(100, len(QUERY_SET))]

__all__ = [
    "BASE_QUERY_SET",
    "QUERY_SET",
    "TRAIN_QUERY_SET",
    "TEST_QUERY_SET",
    "DATASET_SOURCE",
    "DATASET_PATHS",
    "load_query_sets",
]
