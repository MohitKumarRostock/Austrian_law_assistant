#!/usr/bin/env python3
"""
precompute_kahm_query_embeddings.py

Exact precomputation of KAHM query embeddings so evaluation/inference can load them
from disk instead of recomputing.

IMPORTANT
---------
This script must produce EXACTLY the same KAHM query embeddings as
`evaluate_three_embeddings.py` when run on the same query set and models.

Key correctness rules:
- Only the query text is embedded (no other fields are passed into the IDF–SVD pipeline).
  This prevents leakage from fields like `consensus_law`.
- Uses the same IDF–SVD pipeline and the same KAHM distance-gated multi-model selection.
- Deterministic tie-breaking is forced to "first" to match evaluation behavior.

Output
------
Compressed .npz with:
- embeddings: (N, D_out) float32, L2-normalized (row-major)
- query_ids (optional): (N,) object array if extractable
- chosen_model: (N,) int16 (index into model_names)
- min_distance: (N,) float32 (best distance score from gating)
- model_names: (M,) object array (names corresponding to chosen_model indices)
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import joblib


def _str2bool(v: object) -> bool:
    if isinstance(v, bool):
        return bool(v)
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v!r}")


def l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, eps).astype(np.float32)
    return X / n


# ----------------------------- Query extraction -----------------------------
def _pick_from_mapping(obj: Any, keys: Sequence[str]) -> str:
    if not isinstance(obj, dict):
        return ""
    for k in keys:
        if k in obj:
            v = obj.get(k, "")
            if isinstance(v, (list, tuple)) and v:
                v = v[0]
            if v is None:
                continue
            s = str(v).strip()
            if s:
                return s
    return ""


def _pick_from_object_attrs(obj: Any, keys: Sequence[str]) -> str:
    for k in keys:
        if hasattr(obj, k):
            v = getattr(obj, k)
            if isinstance(v, (list, tuple)) and v:
                v = v[0]
            if v is None:
                continue
            s = str(v).strip()
            if s:
                return s
    return ""


def extract_query_texts(qs: Sequence[Any]) -> List[str]:
    """
    Match evaluate_three_embeddings.py behavior: extract ONLY query text.

    Supported keys: query_text/query/question/text/prompt/q/input
    Supports dicts, tuples/lists, and objects with attributes.
    """
    keys = ["query_text", "query", "question", "text", "prompt", "q", "input"]
    texts: List[str] = []
    for q in qs:
        t = _pick_from_mapping(q, keys)
        if not t and isinstance(q, (list, tuple)):
            if len(q) >= 2 and isinstance(q[1], str) and q[1].strip():
                t = str(q[1]).strip()
            elif len(q) >= 1 and isinstance(q[0], str) and q[0].strip():
                t = str(q[0]).strip()
        if not t:
            t = _pick_from_object_attrs(q, keys)
        texts.append(t)
    return texts


def extract_query_ids(qs: Sequence[Any]) -> List[str]:
    """
    Extract query_id for alignment/debugging. Mirrors evaluate_three_embeddings.py
    (robust but best-effort).
    """
    keys = ["query_id", "id", "qid", "uid"]
    out: List[str] = []
    for q in qs:
        qid = _pick_from_mapping(q, keys)
        if not qid and isinstance(q, (list, tuple)):
            if len(q) >= 1 and isinstance(q[0], str) and q[0].strip():
                if re.match(r"^[A-Za-z0-9_\-\.]+$", q[0]) and len(q[0]) <= 80:
                    qid = str(q[0]).strip()
            if not qid and len(q) >= 2 and isinstance(q[1], str) and q[1].strip():
                if re.match(r"^[A-Za-z0-9_\-\.]+$", q[1]) and len(q[1]) <= 80:
                    qid = str(q[1]).strip()
        if not qid:
            qid = _pick_from_object_attrs(q, keys)
        out.append(str(qid).strip())
    return out


def load_query_set(module_attr: str) -> List[Any]:
    """Load query set from module.attr (e.g., query_set.TEST_QUERY_SET)."""
    if "." not in module_attr:
        raise ValueError("--query_set must be module.attr (e.g., query_set.TEST_QUERY_SET)")
    mod_name, attr = module_attr.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    qs = getattr(mod, attr, None)
    if qs is None:
        raise AttributeError(f"Query set attribute not found: {module_attr}")
    out = list(qs)
    if not out:
        raise ValueError(f"Loaded empty query set from {module_attr}")
    return out


# ----------------------------- KAHM loading -----------------------------
def load_kahm_model(path: str) -> dict:
    from kahm_regression import load_kahm_regressor
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return load_kahm_regressor(path)


def load_kahm_models_from_dir(dir_path: str) -> Dict[str, dict]:
    d = Path(str(dir_path)).expanduser()
    if not d.exists() or not d.is_dir():
        raise FileNotFoundError(dir_path)
    paths = sorted(d.glob("*.joblib"))
    if not paths:
        raise FileNotFoundError(f"No *.joblib models found in {dir_path}")
    models: Dict[str, dict] = {}
    for fp in paths:
        models[fp.stem] = load_kahm_model(str(fp))
    return models


def _kahm_model_path_exists(path: str) -> bool:
    p = str(path or "").strip()
    if not p:
        return False
    if os.path.isdir(p):
        try:
            return any(Path(p).glob("*.joblib"))
        except Exception:
            return False
    return os.path.exists(p)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Precompute KAHM query embeddings (exact, leakage-safe).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--idf_svd_model", default="idf_svd_model.joblib", help="Path to idf_svd_model.joblib")

    # Keep the legacy flags for compatibility, but prefer --query_set.
    ap.add_argument("--query_set", default="query_set.TEST_QUERY_SET", help="Query set as module.attr")
    ap.add_argument("--query_set_module", default="", help="(legacy) Python module containing the query list")
    ap.add_argument("--query_set_name", default="", help="(legacy) Variable name containing the query list")

    ap.add_argument(
        "--kahm_query_model",
        default="kahm_query_regressors_by_law",
        help="KAHM query regressor path: directory of *.joblib or a single *.joblib file.",
    )
    ap.add_argument("--kahm_mode", default="soft", choices=["soft", "hard"])
    ap.add_argument("--kahm_batch", type=int, default=1024)
    ap.add_argument("--output_npz", default="test_queries_kahm_embeddings.npz", help="Output .npz file path")

    # Speed-only warm caching; must not change arithmetic
    if hasattr(argparse, "BooleanOptionalAction"):
        ap.add_argument(
            "--materialize_classifier",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Preload disk-backed classifiers into RAM (speed only).",
        )
        ap.add_argument(
            "--cache_cluster_centers",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Cache cluster centers for faster gating (speed only).",
        )
    else:
        ap.add_argument("--materialize_classifier", type=_str2bool, default=True)
        ap.add_argument("--cache_cluster_centers", type=_str2bool, default=True)

    ap.add_argument("--show_progress", type=_str2bool, default=False, help="Show progress bars if supported.")

    args = ap.parse_args()

    # ----------------------------- Load queries -----------------------------
    if str(args.query_set_module).strip() and str(args.query_set_name).strip():
        mod = importlib.import_module(str(args.query_set_module).strip())
        qs = list(getattr(mod, str(args.query_set_name).strip()))
        if not qs:
            raise ValueError("Query set is empty (legacy module/name).")
    else:
        qs = load_query_set(str(args.query_set).strip())

    texts = extract_query_texts(qs)
    if not texts:
        raise ValueError("No queries loaded.")
    n_empty = sum(1 for t in texts if not t)
    if n_empty:
        # Keep running; evaluation does too, but warn loudly.
        print(f"WARNING: {n_empty}/{len(texts)} queries have empty text.")

    # Optional ids for alignment/debug
    qids = extract_query_ids(qs)
    qids_ok = all(str(x).strip() for x in qids) and (len(set(qids)) == len(qids))
    if not qids_ok:
        qids = []

    # ----------------------------- IDF–SVD -----------------------------
    pipe = joblib.load(str(args.idf_svd_model))
    # CRITICAL: pass ONLY the extracted query texts (prevents leakage).
    X = pipe.transform(texts)
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"IDF–SVD transform output must be 2D; got {X.shape}")
    X = l2_normalize_rows(X)

    # ----------------------------- Load KAHM model(s) -----------------------------
    q_path = str(args.kahm_query_model).strip()
    if not q_path:
        raise ValueError("--kahm_query_model is required.")
    if not _kahm_model_path_exists(q_path):
        raise FileNotFoundError(f"--kahm_query_model not found: {q_path}")

    if os.path.isdir(q_path):
        models = load_kahm_models_from_dir(q_path)

        # Optional warm caches (speed-only)
        if bool(getattr(args, "materialize_classifier", True)) or bool(getattr(args, "cache_cluster_centers", True)):
            try:
                from combine_kahm_regressors_generalized_fast import prepare_kahm_model_for_inference
                for m in models.values():
                    prepare_kahm_model_for_inference(
                        m,
                        materialize_classifier=bool(getattr(args, "materialize_classifier", True)),
                        cache_cluster_centers=bool(getattr(args, "cache_cluster_centers", True)),
                        show_progress=bool(getattr(args, "show_progress", False)),
                    )
            except Exception:
                pass

        # Combine via min-distance gating (must match evaluation tie-break)
        try:
            from combine_kahm_regressors_generalized_fast import combine_kahm_regressors_distance_gated_multi
        except Exception:
            from combine_kahm_regressors_generalized import combine_kahm_regressors_distance_gated_multi

        Y, chosen, best_score, _all_scores, names = combine_kahm_regressors_distance_gated_multi(
            X,
            models=models,
            input_layout="row",
            output_layout="row",
            mode=str(args.kahm_mode),
            batch_size=int(args.kahm_batch),
            tie_break="first",
            show_progress=bool(getattr(args, "show_progress", False)),
            return_all_scores=False,
        )
        model_names = list(names)
    else:
        # Single model file
        model = load_kahm_model(q_path)
        from kahm_regression import kahm_regress

        Xt = np.ascontiguousarray(X.T)  # (D_in, N)
        Yt = kahm_regress(
            model,
            Xt,
            mode=str(args.kahm_mode),
            batch_size=int(args.kahm_batch),
            show_progress=bool(getattr(args, "show_progress", False)),
        )
        Y = np.asarray(Yt.T, dtype=np.float32)
        chosen = np.zeros((Y.shape[0],), dtype=np.int16)
        best_score = np.zeros((Y.shape[0],), dtype=np.float32)
        model_names = [Path(q_path).stem]

    Y = l2_normalize_rows(np.asarray(Y, dtype=np.float32))

    out = Path(str(args.output_npz))
    out.parent.mkdir(parents=True, exist_ok=True)

    save_kwargs = dict(
        embeddings=Y,
        chosen_model=np.asarray(chosen, dtype=np.int16),
        min_distance=np.asarray(best_score, dtype=np.float32),
        model_names=np.asarray(list(model_names), dtype=object),
    )
    if qids:
        save_kwargs["query_ids"] = np.asarray(qids, dtype=object)

    np.savez_compressed(out, **save_kwargs)
    print(f"Saved: {out}  embeddings={Y.shape}  query_ids={'yes' if qids else 'no'}")


if __name__ == "__main__":
    main()
