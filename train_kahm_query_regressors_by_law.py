#!/usr/bin/env python3
"""train_kahm_query_regressors_by_law.py

Train *law-specific* KAHM query regressors (IDF–SVD -> Mixedbread) per
`consensus_law`, then evaluate a distance-gated multi-model combination
(using combine_kahm_regressors_generalized.py) on the full TEST_QUERY_SET.

Design goals
------------
- Keep the same training pipeline + hyperparameters as the standalone
  query regressor trainer, without importing that script as a module.
- Only law-specific change: if n_clusters > N_train_for_law, clamp to
  N_train_for_law.
  (Practically, we clamp to the *core* training count after any validation
   split, to avoid KMeans errors.)

Data
----
Reads TRAIN_QUERY_SET and TEST_QUERY_SET from query_set.py. Each query must have:
  - query_id (str)
  - query_text (str)
  - consensus_law (str)

Outputs
-------
- Saves one regressor per law to an output directory.
- Prints per-law metrics (if test samples exist) and overall combined metrics.

Example
-------
python train_kahm_query_regressors_by_law.py   --idf_svd_model idf_svd_model.joblib   --queries_npz_train queries_embedding_index_train.npz   --queries_npz_test  queries_embedding_index_test.npz   --out kahm_query_regressors_by_law/

Notes
-----
- This script intentionally does *not* create a single serialized "combined" model.
  Combination is evaluated via distance-gating at inference time.
"""

from __future__ import annotations

import os

# Keep consistent with standalone query-regressor training defaults
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import gc
import hashlib
import re
import sys
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, ContextManager, Dict, Iterator, List, Optional, Protocol, Sequence, Tuple, cast, runtime_checkable

import numpy as np

from kahm_regression import (
    kahm_regress,
    save_kahm_regressor,
    train_kahm_regressor,
    tune_cluster_centers_nlms,
    tune_soft_params,
)

from query_set import TRAIN_QUERY_SET, TEST_QUERY_SET  # type: ignore

from combine_kahm_regressors_generalized import combine_kahm_regressors_distance_gated_multi


# ----------------------------- BlockSafe (optional) -----------------------------
try:
    from otfl_blocksafe import enable_otfl_blocksafe, _BLOCKSAFE_STATS  # type: ignore
except Exception:
    enable_otfl_blocksafe = None
    _BLOCKSAFE_STATS = None


@runtime_checkable
class _ContextManagerLike(Protocol):
    """Runtime-checkable protocol for context managers."""

    def __enter__(self) -> Any: ...

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any: ...


def _as_blocksafe_context(obj: Any) -> ContextManager[None]:
    """Normalize BlockSafe return values to a real context manager."""
    if isinstance(obj, _ContextManagerLike):
        return cast(ContextManager[None], obj)

    if callable(obj):
        teardown = cast(Callable[[], Any], obj)

        @contextmanager
        def _cm() -> Iterator[None]:
            try:
                yield
            finally:
                try:
                    teardown()
                except Exception as exc:
                    print(
                        f"WARNING: BlockSafe teardown failed: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )

        return cast(ContextManager[None], _cm())

    return nullcontext()


# ----------------------------- Defaults -----------------------------
DEFAULT_IDF_SVD_MODEL = "idf_svd_model.joblib"
DEFAULT_QUERIES_NPZ = "queries_embedding_index.npz"  # optional combined file (back-compat)
DEFAULT_QUERIES_NPZ_TRAIN = "queries_embedding_index_train.npz"
DEFAULT_QUERIES_NPZ_TEST = "queries_embedding_index_test.npz"
DEFAULT_OUT = "kahm_query_regressors_by_law"

DEFAULT_N_CLUSTERS = 300
DEFAULT_SUBSPACE_DIM = 20
DEFAULT_NB = 100
DEFAULT_RANDOM_STATE = 0
DEFAULT_INPUT_SCALE = 1.0

DEFAULT_KMEANS_KIND = "full"  # {'auto','full','minibatch'}
DEFAULT_KMEANS_BATCH_SIZE = 4096
DEFAULT_MAX_TRAIN_PER_CLUSTER = None
DEFAULT_MODEL_DTYPE = "float32"
DEFAULT_CLUSTER_CENTER_NORMALIZATION = "none"  # none|l2|auto_l2

DEFAULT_AE_CACHE_ROOT = "kahm_ae_cache"
DEFAULT_OVERWRITE_AE_DIR = False

DEFAULT_EVAL_SOFT = True
DEFAULT_TUNE_SOFT = True
DEFAULT_TUNE_NLMS = True
DEFAULT_VAL_FRACTION = 0.05
DEFAULT_VAL_MAX_SAMPLES = 5000
DEFAULT_SOFT_ALPHAS = (5.0, 8.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0)
DEFAULT_SOFT_TOPKS = (2, 5, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 50, 75, 100, 125, 150, 175, 200)

DEFAULT_BLOCKSAFE_ENABLED = True
DEFAULT_BLOCKSAFE_BACKEND = "threading"
DEFAULT_BLOCKSAFE_JITTER_STD = 1e-5
DEFAULT_BLOCKSAFE_JITTER_TRIES = 6
DEFAULT_BLOCKSAFE_JITTER_GROWTH = 2.0
DEFAULT_BLOCKSAFE_EPS_FACTOR = 10.0
DEFAULT_BLOCKSAFE_LOG_FIRST = 100
DEFAULT_BLOCKSAFE_L2_NORMALIZED = True


# ----------------------------- Utilities -----------------------------
def as_float_ndarray(x: Any, *, min_dtype: np.dtype = np.dtype(np.float32)) -> np.ndarray:
    """Convert input to a floating ndarray without downcasting precision."""
    if hasattr(x, "toarray"):
        x = x.toarray()
    x = np.asarray(x)
    if x.dtype.kind in ("i", "u", "b"):
        return x.astype(np.float64, copy=False)
    if x.dtype.kind != "f":
        return x.astype(min_dtype, copy=False)
    if x.dtype.itemsize < min_dtype.itemsize:
        return x.astype(min_dtype, copy=False)
    return x


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = as_float_ndarray(x)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array; got shape={x.shape}")
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def parse_float_list(arg: str) -> List[float]:
    return [float(x.strip()) for x in arg.split(",") if x.strip()]


def parse_topk_list(arg: str) -> List[Optional[int]]:
    out: List[Optional[int]] = []
    for tok in arg.split(","):
        t = tok.strip().lower()
        if not t:
            continue
        if t in ("none", "null"):
            out.append(None)
        else:
            out.append(int(t))
    return out


def compute_embedding_metrics(Y_pred: np.ndarray, Y_true: np.ndarray) -> Dict[str, float]:
    """Compute MSE, overall R^2, and cosine similarity stats for (D, N) embeddings."""
    if Y_pred.ndim != 2 or Y_true.ndim != 2:
        raise ValueError(f"Expected 2D arrays; got {Y_pred.shape}, {Y_true.shape}")
    if Y_pred.shape != Y_true.shape:
        raise ValueError(f"Shape mismatch: pred={Y_pred.shape} true={Y_true.shape}")

    D, N = Y_true.shape
    diff = Y_pred - Y_true
    mse = float(np.mean(diff * diff))

    y = Y_true.reshape(-1)
    y_mean = float(np.mean(y))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    ss_res = float(np.sum(diff.reshape(-1) ** 2))
    r2_overall = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")

    num = np.einsum("dn,dn->n", Y_pred, Y_true)
    den = np.linalg.norm(Y_pred, axis=0) * np.linalg.norm(Y_true, axis=0)
    cos = num / np.maximum(den, 1e-12)

    return dict(
        mse=mse,
        r2_overall=r2_overall,
        cos_mean=float(np.mean(cos)),
        cos_p10=float(np.percentile(cos, 10)),
        cos_p50=float(np.percentile(cos, 50)),
        cos_p90=float(np.percentile(cos, 90)),
        n=N,
        d=D,
    )


def embed_idf_svd_queries(idf_svd_model_path: str, texts: Sequence[str]) -> np.ndarray:
    import joblib

    pipe = joblib.load(idf_svd_model_path)
    X = pipe.transform(list(texts))
    X = as_float_ndarray(X)
    X = l2_normalize_rows(X)
    return X


def load_precomputed_mb_queries_npz(path: str, query_ids: Sequence[str]) -> np.ndarray:
    """Load precomputed Mixedbread embeddings from NPZ and align by query_id."""
    d = np.load(path, allow_pickle=False)
    if "query_id" not in d or "embeddings" not in d:
        raise ValueError(
            f"Queries NPZ '{path}' must contain keys 'query_id' and 'embeddings'. Keys: {list(d.keys())}"
        )

    qid_npz = np.asarray(d["query_id"])
    Y_npz = as_float_ndarray(d["embeddings"])

    if qid_npz.ndim != 1 or Y_npz.ndim != 2:
        raise ValueError(
            f"Queries NPZ '{path}': expected query_id (Q,), embeddings (Q,D); got {qid_npz.shape}, {Y_npz.shape}"
        )

    map_npz = {str(qid_npz[i]): i for i in range(qid_npz.shape[0])}
    missing = [qid for qid in query_ids if qid not in map_npz]
    if missing:
        raise ValueError(f"Queries NPZ '{path}' missing {len(missing)} query_ids. Example: {missing[:10]}")

    Y = np.vstack([Y_npz[map_npz[qid]] for qid in query_ids]).astype(Y_npz.dtype, copy=False)
    Y = l2_normalize_rows(Y)
    return Y


def embed_mb_queries_on_the_fly(model_name: str, device: str, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
    """Optional fallback (torch required): compute Mixedbread embeddings for texts."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    Y = model.encode(
        list(texts),
        batch_size=int(batch_size),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    Y = as_float_ndarray(Y)
    Y = l2_normalize_rows(Y)
    del model
    gc.collect()
    return Y


def _sanitize_for_path(s: str, *, max_len: int = 64) -> str:
    s0 = str(s).strip()
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", s0)
    cleaned = cleaned.strip("._-")
    if not cleaned:
        cleaned = "unknown"
    if len(cleaned) > max_len:
        h = hashlib.sha1(s0.encode("utf-8")).hexdigest()[:10]
        cleaned = cleaned[: max(1, max_len - 11)] + "_" + h
    return cleaned


def _extract_ids_texts_laws(qs: Sequence[Dict[str, Any]], name: str) -> Tuple[List[str], List[str], List[str]]:
    ids: List[str] = []
    texts: List[str] = []
    laws: List[str] = []
    for i, q in enumerate(qs):
        qid = str(q.get("query_id", "")).strip()
        txt = str(q.get("query_text", "")).strip()
        law = str(q.get("consensus_law", "")).strip()
        if not qid:
            raise ValueError(f"{name}[{i}] has empty query_id")
        if not txt:
            raise ValueError(f"{name}[{i}] has empty query_text")
        if not law:
            raise ValueError(f"{name}[{i}] has empty consensus_law")
        ids.append(qid)
        texts.append(txt)
        laws.append(law)

    if len(set(ids)) != len(ids):
        raise ValueError(f"{name} has duplicate query_id values")

    return ids, texts, laws


def _resolve_out_dir_and_prefix(out_arg: str) -> Tuple[Path, str]:
    p = Path(str(out_arg)).expanduser()
    if p.suffix.lower() == ".joblib":
        out_dir = p.parent if str(p.parent) else Path(".")
        prefix = p.stem
        return out_dir, prefix
    return p, "kahm_query_regressor"


def _print_metrics(prefix: str, m: Dict[str, float]) -> None:
    print(prefix)
    print(f"  MSE:               {m['mse']:.6f}")
    print(f"  Overall R^2:       {m['r2_overall']:.4f}")
    print(f"  Cosine mean:       {m['cos_mean']:.4f}")
    print(f"  Cosine p10/p50/p90:{m['cos_p10']:.4f} / {m['cos_p50']:.4f} / {m['cos_p90']:.4f}")
    print(f"  N:                 {int(m['n'])}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Train a separate KAHM query regressor per consensus_law "
            "(IDF–SVD -> Mixedbread), then evaluate a distance-gated combination "
            "across all laws on the full test set."
        )
    )

    p.add_argument("--idf_svd_model", default=DEFAULT_IDF_SVD_MODEL, help="Path to idf_svd_model.joblib (required).")
    p.add_argument("--queries_npz", default="", help="Optional path to a combined precomputed Mixedbread query embeddings NPZ (backward compatible).")
    p.add_argument("--queries_npz_train", default=DEFAULT_QUERIES_NPZ_TRAIN, help="Path to precomputed Mixedbread TRAIN query embeddings NPZ.")
    p.add_argument("--queries_npz_test", default=DEFAULT_QUERIES_NPZ_TEST, help="Path to precomputed Mixedbread TEST query embeddings NPZ.")
    p.add_argument("--require_npz", action="store_true", help="If set, require NPZ targets and do not fall back to on-the-fly MB embedding.")
    p.add_argument("--out", default=DEFAULT_OUT, help="Output directory for saved law-specific KAHM query regressor joblibs.")

    p.add_argument("--mb_model", default="mixedbread-ai/deepset-mxbai-embed-de-large-v1", help="Mixedbread model name (fallback only).")
    p.add_argument("--mb_device", default="cpu", help="Device for fallback MB embedding (cpu/cuda/mps).")
    p.add_argument("--mb_batch", type=int, default=64, help="Batch size for fallback MB embedding.")
    p.add_argument("--force_mb_on_the_fly", action="store_true", help="Ignore NPZ targets and compute MB embeddings with sentence_transformers (torch required).")

    p.add_argument("--model_id", default=None, help="Identifier used to create a unique autoencoder directory under --ae_cache_root (defaults to stem of --out path).")
    p.add_argument("--ae_cache_root", default=DEFAULT_AE_CACHE_ROOT, help=f"Root directory for saved per-cluster autoencoders (default: {DEFAULT_AE_CACHE_ROOT})")
    p.add_argument("--ae_dir", default=None, help="Explicit directory to save per-cluster autoencoders. Overrides --ae_cache_root/--model_id.")
    p.add_argument("--overwrite_ae_dir", action="store_true", default=DEFAULT_OVERWRITE_AE_DIR, help="Allow overwriting an existing AE directory.")

    p.add_argument("--n_clusters", type=int, default=DEFAULT_N_CLUSTERS, help=f"Number of output clusters (default: {DEFAULT_N_CLUSTERS})")
    p.add_argument("--subspace_dim", type=int, default=DEFAULT_SUBSPACE_DIM, help=f"Subspace dimension (default: {DEFAULT_SUBSPACE_DIM})")
    p.add_argument("--nb", type=int, default=DEFAULT_NB, help=f"Nb (default: {DEFAULT_NB})")
    p.add_argument("--random_state", type=int, default=DEFAULT_RANDOM_STATE, help=f"Random seed (default: {DEFAULT_RANDOM_STATE})")
    p.add_argument("--input_scale", type=float, default=DEFAULT_INPUT_SCALE, help=f"Input scaling (default: {DEFAULT_INPUT_SCALE})")

    p.add_argument("--kmeans_kind", default=DEFAULT_KMEANS_KIND, choices=["auto", "full", "minibatch"], help="KMeans implementation choice.")
    p.add_argument("--kmeans_batch_size", type=int, default=DEFAULT_KMEANS_BATCH_SIZE, help="MiniBatchKMeans batch size (if used).")
    p.add_argument("--max_train_per_cluster", type=int, default=DEFAULT_MAX_TRAIN_PER_CLUSTER, help="Cap training samples per cluster (optional).")
    p.add_argument("--model_dtype", default=DEFAULT_MODEL_DTYPE, choices=["float32", "float64"], help="Storage dtype inside the model.")
    p.add_argument("--cluster_center_normalization", default=DEFAULT_CLUSTER_CENTER_NORMALIZATION, choices=["none", "l2", "auto_l2"], help="Normalization for output cluster centers.")

    p.add_argument("--val_fraction", type=float, default=DEFAULT_VAL_FRACTION, help="Fraction of TRAIN queries used for validation/tuning.")
    p.add_argument("--val_max_samples", type=int, default=DEFAULT_VAL_MAX_SAMPLES, help="Max validation samples.")
    p.add_argument("--eval_soft", action="store_true", default=DEFAULT_EVAL_SOFT, help="Evaluate soft-mode regression.")
    p.add_argument("--tune_soft", action="store_true", default=DEFAULT_TUNE_SOFT, help="Tune soft-mode parameters (alpha/topk) on validation set.")
    p.add_argument("--tune_nlms", action="store_true", default=DEFAULT_TUNE_NLMS, help="Refine cluster centers with NLMS (optional).")
    p.add_argument("--soft_alphas", default=",".join(str(x) for x in DEFAULT_SOFT_ALPHAS), help="Comma-separated alphas for soft tuning.")
    p.add_argument("--soft_topks", default=",".join("none" if x is None else str(x) for x in DEFAULT_SOFT_TOPKS), help="Comma-separated topk values for soft tuning (use 'none').")

    p.add_argument("--preload_eval_classifier", action="store_true", help="Preload per-cluster autoencoders into RAM for evaluation (requires RAM).")

    p.add_argument("--blocksafe", action="store_true", default=DEFAULT_BLOCKSAFE_ENABLED, help="Enable OTFL BlockSafe (if available).")
    p.add_argument("--blocksafe_backend", default=DEFAULT_BLOCKSAFE_BACKEND, choices=["threading", "multiprocessing"], help="BlockSafe backend.")
    p.add_argument("--blocksafe_jitter_std", type=float, default=DEFAULT_BLOCKSAFE_JITTER_STD)
    p.add_argument("--blocksafe_jitter_tries", type=int, default=DEFAULT_BLOCKSAFE_JITTER_TRIES)
    p.add_argument("--blocksafe_jitter_growth", type=float, default=DEFAULT_BLOCKSAFE_JITTER_GROWTH)
    p.add_argument("--blocksafe_eps_factor", type=float, default=DEFAULT_BLOCKSAFE_EPS_FACTOR)
    p.add_argument("--blocksafe_log_first", type=int, default=DEFAULT_BLOCKSAFE_LOG_FIRST)
    p.add_argument("--blocksafe_l2_normalized", action="store_true", default=DEFAULT_BLOCKSAFE_L2_NORMALIZED)

    p.add_argument("--combined_mode", default="soft", choices=["soft", "hard"], help="Combination inference mode (default: soft).")
    p.add_argument("--combined_batch_size", type=int, default=2048, help="Batch size used by distance-gated combiner (default: 2048).")
    p.add_argument("--no_combined_progress", action="store_true", help="Disable progress bars during combined evaluation.")

    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    idf_svd_model_path = str(args.idf_svd_model)
    if not Path(idf_svd_model_path).exists():
        raise FileNotFoundError(f"idf_svd_model not found: {idf_svd_model_path}")

    train_qs = list(TRAIN_QUERY_SET)
    test_qs = list(TEST_QUERY_SET)

    train_ids, train_texts, train_laws = _extract_ids_texts_laws(train_qs, "TRAIN_QUERY_SET")
    test_ids, test_texts, test_laws = _extract_ids_texts_laws(test_qs, "TEST_QUERY_SET")

    out_dir, out_prefix = _resolve_out_dir_and_prefix(str(args.out))
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = nullcontext()
    if args.blocksafe and enable_otfl_blocksafe is not None:
        ctx = enable_otfl_blocksafe(
            backend=str(args.blocksafe_backend),
            jitter_std=float(args.blocksafe_jitter_std),
            jitter_tries=int(args.blocksafe_jitter_tries),
            jitter_growth=float(args.blocksafe_jitter_growth),
            eps_factor=float(args.blocksafe_eps_factor),
            log_first=int(args.blocksafe_log_first),
            l2_normalized=bool(args.blocksafe_l2_normalized),
        )

    print(f"Embedding IDF–SVD queries using: {idf_svd_model_path}")
    X_train_all = embed_idf_svd_queries(idf_svd_model_path, train_texts)
    X_test_all = embed_idf_svd_queries(idf_svd_model_path, test_texts)


    def _resolve_npz(which: str) -> Optional[str]:
        combined = str(args.queries_npz).strip()
        split_path = str(getattr(args, f"queries_npz_{which}", "")).strip()
        if split_path and Path(split_path).exists():
            return split_path
        if combined and Path(combined).exists():
            return combined
        return None

    npz_train = _resolve_npz("train")
    npz_test = _resolve_npz("test")

    if bool(args.force_mb_on_the_fly):
        npz_train = None
        npz_test = None

    use_npz = (npz_train is not None) and (npz_test is not None)

    if use_npz:
        print(f"Loading precomputed Mixedbread TRAIN query embeddings: {npz_train}")
        Y_train_all = load_precomputed_mb_queries_npz(str(npz_train), train_ids)
        print(f"Loading precomputed Mixedbread TEST  query embeddings: {npz_test}")
        Y_test_all = load_precomputed_mb_queries_npz(str(npz_test), test_ids)
    else:
        if bool(getattr(args, "require_npz", False)):
            missing = []
            if npz_train is None:
                missing.append("train")
            if npz_test is None:
                missing.append("test")
            raise FileNotFoundError(
                "Missing required NPZ targets for: "
                + ", ".join(missing)
                + ". Provide --queries_npz_train/--queries_npz_test (or a combined --queries_npz) or unset --require_npz."
            )
        print("Computing Mixedbread query embeddings on-the-fly (torch required).")
        Y_train_all = embed_mb_queries_on_the_fly(
            str(args.mb_model), str(args.mb_device), train_texts, batch_size=int(args.mb_batch)
        )
        Y_test_all = embed_mb_queries_on_the_fly(
            str(args.mb_model), str(args.mb_device), test_texts, batch_size=int(args.mb_batch)
        )

    train_idx_by_law: Dict[str, List[int]] = {}
    for i, law in enumerate(train_laws):
        train_idx_by_law.setdefault(law, []).append(i)

    test_idx_by_law: Dict[str, List[int]] = {}
    for i, law in enumerate(test_laws):
        test_idx_by_law.setdefault(law, []).append(i)

    laws = sorted(train_idx_by_law.keys())

    unseen_test_laws = sorted(set(test_idx_by_law.keys()) - set(train_idx_by_law.keys()))
    if unseen_test_laws:
        print(
            "WARNING: test set contains consensus_law values not present in training set.\n"
            f"  Unseen laws (count={len(unseen_test_laws)}): {unseen_test_laws[:20]}"
            + (" ..." if len(unseen_test_laws) > 20 else "")
        )

    print(f"\nTraining {len(laws)} law-specific regressors ...")

    models_by_law: Dict[str, dict] = {}
    saved_paths: Dict[str, str] = {}

    for law in laws:
        idx_tr = np.asarray(train_idx_by_law[law], dtype=np.int64)
        idx_te = np.asarray(test_idx_by_law.get(law, []), dtype=np.int64)

        X_tr = X_train_all[idx_tr]
        Y_tr = Y_train_all[idx_tr]
        X_te = X_test_all[idx_te] if idx_te.size else None
        Y_te = Y_test_all[idx_te] if idx_te.size else None

        N_train = int(X_tr.shape[0])
        N_test = int(0 if X_te is None else X_te.shape[0])

        print("\n" + "=" * 90)
        print(f"Law: {law} | N_train={N_train} | N_test={N_test}")

        rng = np.random.RandomState(int(args.random_state))
        idx = np.arange(N_train, dtype=np.int64)
        rng.shuffle(idx)

        n_val = 0
        if float(args.val_fraction) > 0:
            n_val = int(round(float(args.val_fraction) * N_train))
            n_val = min(n_val, int(args.val_max_samples))
        n_val = max(0, min(n_val, N_train - 2))

        val_idx = idx[:n_val]
        core_idx = idx[n_val:]

        X_val = X_tr[val_idx] if n_val > 0 else None
        Y_val = Y_tr[val_idx] if n_val > 0 else None
        X_core = X_tr[core_idx]
        Y_core = Y_tr[core_idx]

        X_train_col = X_core.T
        Y_train_col = Y_core.T

        n_clusters_eff = int(min(int(args.n_clusters), int(X_core.shape[0])))
        n_clusters_eff = max(1, n_clusters_eff)

        if n_clusters_eff != int(args.n_clusters):
            print(
                f"Clamping n_clusters: requested={int(args.n_clusters)} -> effective={n_clusters_eff} "
                f"(law train_core N={int(X_core.shape[0])})"
            )

        safe_law = _sanitize_for_path(law)
        out_path = out_dir / f"{out_prefix}__law={safe_law}.joblib"

        base_model_id = str(args.model_id) if args.model_id else out_path.stem
        run_model_id = f"{base_model_id}__law={safe_law}"

        ae_dir = None
        if args.ae_dir is not None:
            ae_dir = str(Path(str(args.ae_dir)) / f"law={safe_law}")

        t0 = time.time()
        try:
            with _as_blocksafe_context(ctx):
                model = train_kahm_regressor(
                    X=X_train_col,
                    Y=Y_train_col,
                    n_clusters=n_clusters_eff,
                    subspace_dim=int(args.subspace_dim),
                    Nb=int(args.nb),
                    random_state=int(args.random_state),
                    verbose=True,
                    input_scale=float(args.input_scale) if args.input_scale is not None else 1.0,
                    kmeans_kind=str(args.kmeans_kind),
                    kmeans_batch_size=int(args.kmeans_batch_size),
                    max_train_per_cluster=(None if args.max_train_per_cluster is None else int(args.max_train_per_cluster)),
                    model_dtype=str(args.model_dtype),
                    cluster_center_normalization=str(args.cluster_center_normalization),
                    save_ae_to_disk=False,
                    ae_cache_root=str(args.ae_cache_root),
                    ae_dir=ae_dir,
                    overwrite_ae_dir=bool(args.overwrite_ae_dir),
                    model_id=run_model_id,
                    singleton_strategy="augment",
                    singleton_aux_mix=0.1,
                )
        except Exception as exc:
            print(f"ERROR: training failed for law={law}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue

        t_train = time.time() - t0
        try:
            print(f"Trained model_id={model.get('model_id')} | classifier_dir={model.get('classifier_dir')} | time={t_train:.1f}s")
        except Exception:
            print(f"Training time={t_train:.1f}s")

        tuning_result = None
        if bool(args.tune_soft):
            alphas = tuple(parse_float_list(str(args.soft_alphas)))
            topks = tuple(parse_topk_list(str(args.soft_topks)))
            topk_candidates_eff = [k for k in topks if (k is not None and k <= n_clusters_eff)]
            if not topk_candidates_eff:
                topk_candidates_eff = [1]
                print(f"[{law}] WARNING: no valid topk candidates <= n_clusters_eff={n_clusters_eff}; using topk=1 only.")
            if X_val is None or Y_val is None:
                print("WARNING: tune_soft requested, but validation split is empty. Skipping tuning.")
            else:
                print("Tuning soft parameters on validation set...")
                tuning_result = tune_soft_params(
                    model,
                    X_val.T,
                    Y_val.T,
                    alphas=alphas,
                    topks=topk_candidates_eff,
                    n_jobs=1,
                    verbose=True,
                )

        nlms_results = None
        if bool(args.tune_nlms):
            if X_val is None or Y_val is None:
                print("WARNING: tune_nlms requested, but validation split is empty. Skipping NLMS.")
            else:
                print("Refining cluster centers with NLMS...")
                nlms_results = tune_cluster_centers_nlms(
                    model,
                    np.hstack([X_val.T, X_train_col]),
                    np.hstack([Y_val.T, Y_train_col]),
                    mu=0.1,
                    epsilon=1,
                    epochs=20,
                    batch_size=1024,
                    shuffle=True,
                    random_state=int(args.random_state),
                    anchor_lambda=0.0,
                    n_jobs=1,
                    preload_classifier=True,
                    verbose=True,
                    alpha=(tuning_result.best_alpha if tuning_result is not None else None),
                    topk=(tuning_result.best_topk if tuning_result is not None else None),
                )

        metrics_soft = None
        if N_test > 0 and X_te is not None and Y_te is not None:
            X_eval_col = X_te.T
            Y_eval_col = Y_te.T
            if bool(args.eval_soft) or bool(args.tune_soft):
                Y_pred_soft = kahm_regress(
                    model,
                    X_eval_col,
                    mode="soft",
                    return_probabilities=False,
                    batch_size=1024,
                )
                metrics_soft = compute_embedding_metrics(Y_pred_soft, Y_eval_col)
                _print_metrics("Soft-mode metrics (law test subset):", metrics_soft)
        else:
            print("No test samples for this law; skipping per-law test evaluation.")

        created_at = datetime.now(timezone.utc).isoformat()
        try:
            tuning_payload = asdict(tuning_result) if tuning_result is not None else None
        except Exception:
            tuning_payload = tuning_result

        meta = {
            "created_at_utc": created_at,
            "script": os.path.basename(__file__),
            "consensus_law": law,
            "paths": {
                "idf_svd_model": args.idf_svd_model,
                "queries_npz": (str(args.queries_npz).strip() if (use_npz and str(args.queries_npz).strip()) else None),
                "queries_npz_train": (npz_train if use_npz else None),
                "queries_npz_test": (npz_test if use_npz else None),
                "out": str(out_path),
            },
            "data": {
                "n_train_queries": int(N_train),
                "n_test_queries": int(N_test),
                "n_train_core": int(X_core.shape[0]),
                "n_val": int(0 if X_val is None else X_val.shape[0]),
                "d_in": int(X_tr.shape[1]),
                "d_out": int(Y_tr.shape[1]),
            },
            "hyperparams": {
                "n_clusters_requested": int(args.n_clusters),
                "n_clusters_effective": int(n_clusters_eff),
                "subspace_dim": int(args.subspace_dim),
                "Nb": int(args.nb),
                "random_state": int(args.random_state),
                "input_scale": float(args.input_scale),
                "kmeans_kind": str(args.kmeans_kind),
                "kmeans_batch_size": int(args.kmeans_batch_size),
                "max_train_per_cluster": (None if args.max_train_per_cluster is None else int(args.max_train_per_cluster)),
                "model_dtype": str(args.model_dtype),
                "cluster_center_normalization": str(args.cluster_center_normalization),
                "val_fraction": float(args.val_fraction),
                "val_max_samples": int(args.val_max_samples),
                "eval_soft": bool(args.eval_soft),
                "tune_soft": bool(args.tune_soft),
                "tune_nlms": bool(args.tune_nlms),
                "soft_alphas": list(parse_float_list(str(args.soft_alphas))),
                "soft_topks": list(parse_topk_list(str(args.soft_topks))),
            },
            "tuning": tuning_payload,
            "nlms": (None if nlms_results is None else str(nlms_results)),
            "metrics": {
                "soft": metrics_soft,
                "eval_split": "law_test_subset",
            },
        }

        try:
            model["meta"] = meta
        except Exception:
            pass

        save_kahm_regressor(model, str(out_path))
        print(f"Saved law regressor to: {out_path}")

        models_by_law[law] = model
        saved_paths[law] = str(out_path)

    if not models_by_law:
        raise RuntimeError("No law-specific models were trained successfully.")

    print("\n" + "=" * 90)
    print(f"Evaluating combined regressor on full test set (N={int(X_test_all.shape[0])}) ...")

    if len(models_by_law) == 1:
        only_law = next(iter(models_by_law.keys()))
        print(f"Only one trained model (law={only_law}); skipping gating and using that model for all test queries.")
        Y_pred_col = kahm_regress(
            models_by_law[only_law],
            X_test_all.T,
            mode=str(args.combined_mode),
            return_probabilities=False,
            batch_size=1024,
        )
        chosen_idx = np.zeros((X_test_all.shape[0],), dtype=np.int16)
        names = [only_law]
    else:
        Y_pred_row, chosen_idx, best_score, all_scores, names = combine_kahm_regressors_distance_gated_multi(
            X_test_all,
            models=models_by_law,
            input_layout="row",
            output_layout="row",
            mode=str(args.combined_mode),
            alpha=None,
            topk=None,
            batch_size=int(args.combined_batch_size),
            tie_break="first",
            show_progress=(not bool(args.no_combined_progress)),
            return_all_scores=True,
        )
        Y_pred_col = Y_pred_row.T

    metrics_combined = compute_embedding_metrics(Y_pred_col, Y_test_all.T)
    _print_metrics("Combined metrics (full test set):", metrics_combined)

    print("\nChosen-model distribution (by gating index):")
    chosen_idx = np.asarray(chosen_idx, dtype=np.int64).reshape(-1)
    idx_to_name = {i: names[i] for i in range(len(names))}
    for i in range(len(names)):
        cnt = int(np.sum(chosen_idx == i))
        print(f"  {i:>3d} | {idx_to_name[i]} | {cnt}")

    print("\nCombined metrics per-law (restricted to each law's test queries):")
    any_per_law = False
    for law, idxs in sorted(test_idx_by_law.items(), key=lambda kv: kv[0]):
        if not idxs:
            continue
        any_per_law = True
        idxs_np = np.asarray(idxs, dtype=np.int64)
        Y_pred_law = Y_pred_col[:, idxs_np]
        Y_true_law = Y_test_all.T[:, idxs_np]
        m = compute_embedding_metrics(Y_pred_law, Y_true_law)
        print(f"- {law} (N={len(idxs)})")
        print(f"    cos_mean={m['cos_mean']:.4f} | mse={m['mse']:.6f} | r2={m['r2_overall']:.4f}")
    if not any_per_law:
        print("  (no per-law test subsets found)")

    print("\nSaved models:")
    for law in sorted(saved_paths.keys()):
        print(f"  {law}: {saved_paths[law]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
