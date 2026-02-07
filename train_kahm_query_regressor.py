#!/usr/bin/env python3
"""
train_kahm_query_regressor.py

Train a *query-specific* KAHM regression model mapping L2-normalized IDF–SVD query
embeddings (inputs) -> L2-normalized Mixedbread query embeddings (targets).

Data source
----------
Reads TRAIN_QUERY_SET and TEST_QUERY_SET from query_set.py.
Queries must contain at least:
  - query_id (str)
  - query_text (str)

Targets (recommended)
---------------------
Provide precomputed Mixedbread query embeddings via --queries_npz
(created by build_query_embedding_index_npz.py). This keeps training torch-free.

Output
------
Saves a KAHM regressor joblib via kahm_regression.save_kahm_regressor().

Example (recommended 2-step)
----------------------------
1) Precompute Mixedbread query embeddings once (separate process):
   python build_query_embedding_index_npz.py 

2) Train the query regressor (torch-free):
   python train_kahm_query_regressor.py 

Notes
-----
- KAHM internally clusters outputs (Mixedbread space). If n_clusters is too large
  relative to N_train, you will create many tiny clusters which can destabilize
  regression. For ~10k queries, start with n_clusters ~ 500–2000.

"""

from __future__ import annotations

import os

# Safe defaults (keep after future import)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import gc
import hashlib
import sys
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, ContextManager, Dict, Iterable, Iterator, List, Optional, Protocol, Sequence, Tuple, runtime_checkable, cast

import numpy as np

try:
    from kahm_regression import (
        kahm_regress,
        preload_kahm_classifier,
        save_kahm_regressor,
        train_kahm_regressor,
        tune_soft_params,
        tune_cluster_centers_nlms,
    )
except ImportError:  # pragma: no cover
    from kahm_regression import kahm_regress, save_kahm_regressor, train_kahm_regressor, tune_soft_params, tune_cluster_centers_nlms
    preload_kahm_classifier = None  # type: ignore


# ----------------------------- BlockSafe (optional) -----------------------------
try:
    from otfl_blocksafe import enable_otfl_blocksafe, _BLOCKSAFE_STATS  # type: ignore
except Exception:
    enable_otfl_blocksafe = None
    _BLOCKSAFE_STATS = None


@runtime_checkable
class _ContextManagerLike(Protocol):
    """Runtime-checkable protocol for context managers (for Pylance/mypy friendliness)."""

    def __enter__(self) -> Any: ...

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any: ...


def _as_blocksafe_context(obj: Any) -> ContextManager[None]:
    """Normalize BlockSafe return values to a real context manager.

    Some environments return a context manager; others return a teardown callable.
    This wrapper makes both usable with a single `with` statement, and avoids
    static type checker warnings.
    """
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


# ----------------------------- Query set import -----------------------------
def _load_query_sets() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    from query_set import TRAIN_QUERY_SET, TEST_QUERY_SET  # type: ignore
    train_qs = list(TRAIN_QUERY_SET)
    test_qs = list(TEST_QUERY_SET)
    if not train_qs:
        raise RuntimeError("Imported TRAIN_QUERY_SET is empty.")
    if not test_qs:
        raise RuntimeError("Imported TEST_QUERY_SET is empty.")
    return train_qs, test_qs


TRAIN_QS, TEST_QS = _load_query_sets()


# ----------------------------- Defaults -----------------------------
DEFAULT_IDF_SVD_MODEL = "idf_svd_model.joblib"
DEFAULT_QUERIES_NPZ = "queries_embedding_index.npz"  # optional combined file (back-compat)
DEFAULT_QUERIES_NPZ_TRAIN = "queries_embedding_index_train.npz"
DEFAULT_QUERIES_NPZ_TEST  = "queries_embedding_index_test.npz"
DEFAULT_OUT = "kahm_query_regressor_idf_to_mixedbread.joblib"

DEFAULT_N_CLUSTERS = 20000
DEFAULT_SUBSPACE_DIM = 20
DEFAULT_NB = 100
DEFAULT_RANDOM_STATE = 0
DEFAULT_INPUT_SCALE = 1.0

# Classifier scaling controls
DEFAULT_KMEANS_KIND = "full"  # {'auto','full','minibatch'}
DEFAULT_KMEANS_BATCH_SIZE = 4096
DEFAULT_MAX_TRAIN_PER_CLUSTER = None
DEFAULT_MODEL_DTYPE = "float32"
DEFAULT_CLUSTER_CENTER_NORMALIZATION = "none"  # none|l2|auto_l2

# AE persistence / collision-avoidance defaults (disk-backed classifier dir)
DEFAULT_AE_CACHE_ROOT = "kahm_ae_cache"
DEFAULT_OVERWRITE_AE_DIR = False

# Soft-mode tuning defaults
DEFAULT_EVAL_SOFT = True
DEFAULT_TUNE_SOFT = True
DEFAULT_TUNE_NLMS = True
DEFAULT_VAL_FRACTION = 0.05
DEFAULT_VAL_MAX_SAMPLES = 5000
DEFAULT_SOFT_ALPHAS = (5.0, 8.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0)
DEFAULT_SOFT_TOPKS = (2, 5, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 50, 75, 100, 125, 150, 175, 200)

# BlockSafe defaults
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


def sha256_fingerprint(*parts: bytes) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.hexdigest()


def compute_embedding_metrics(Y_pred: np.ndarray, Y_true: np.ndarray) -> Dict[str, float]:
    """Compute MSE, overall R^2, and cosine similarity stats for (D, N) embeddings."""
    if Y_pred.ndim != 2 or Y_true.ndim != 2:
        raise ValueError(f"Expected 2D arrays; got {Y_pred.shape}, {Y_true.shape}")
    if Y_pred.shape != Y_true.shape:
        raise ValueError(f"Shape mismatch: pred={Y_pred.shape} true={Y_true.shape}")

    D, N = Y_true.shape
    diff = Y_pred - Y_true
    mse = float(np.mean(diff * diff))

    # R^2 overall (flattened) using sum-of-squares identity
    y = Y_true.reshape(-1)
    y_mean = float(np.mean(y))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    ss_res = float(np.sum(diff.reshape(-1) ** 2))
    r2_overall = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")

    # Cosine similarities per column (assumes L2-normalized columns, but be safe)
    # cos = <a,b> / (||a|| ||b||)
    num = np.einsum("dn,dn->n", Y_pred, Y_true)
    den = np.linalg.norm(Y_pred, axis=0) * np.linalg.norm(Y_true, axis=0)
    cos = num / np.maximum(den, 1e-12)

    cos_mean = float(np.mean(cos))
    cos_p10 = float(np.percentile(cos, 10))
    cos_p50 = float(np.percentile(cos, 50))
    cos_p90 = float(np.percentile(cos, 90))

    return dict(
        mse=mse,
        r2_overall=r2_overall,
        cos_mean=cos_mean,
        cos_p10=cos_p10,
        cos_p50=cos_p50,
        cos_p90=cos_p90,
        n=N,
        d=D,
    )


def _validate_and_extract(qs: Sequence[Dict[str, Any]], name: str) -> Tuple[List[str], List[str]]:
    ids: List[str] = []
    texts: List[str] = []
    for i, q in enumerate(qs):
        qid = str(q.get("query_id", "")).strip()
        txt = str(q.get("query_text", "")).strip()
        if not qid:
            raise ValueError(f"{name}[{i}] has empty query_id.")
        if not txt:
            raise ValueError(f"{name}[{i}] has empty query_text.")
        ids.append(qid)
        texts.append(txt)
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name} has duplicate query_id values.")
    return ids, texts


def embed_idf_svd_queries(idf_svd_model_path: str, texts: Sequence[str]) -> np.ndarray:
    import joblib
    pipe = joblib.load(idf_svd_model_path)
    X = pipe.transform(list(texts))
    X = as_float_ndarray(X)
    X = l2_normalize_rows(X)
    return X


def load_precomputed_mb_queries_npz(path: str, query_ids: Sequence[str]) -> np.ndarray:
    """
    Load precomputed Mixedbread query embeddings from NPZ and align them by query_id.
    Expects keys: query_id, embeddings.
    """
    d = np.load(path, allow_pickle=False)
    if "query_id" not in d or "embeddings" not in d:
        raise ValueError(f"Queries NPZ '{path}' must contain keys 'query_id' and 'embeddings'. Keys: {list(d.keys())}")

    qid_npz = np.asarray(d["query_id"])
    Y_npz = as_float_ndarray(d["embeddings"])

    if qid_npz.ndim != 1 or Y_npz.ndim != 2:
        raise ValueError(f"Queries NPZ '{path}': expected query_id (Q,), embeddings (Q,D); got {qid_npz.shape}, {Y_npz.shape}")

    # Map NPZ row index by query_id
    map_npz = {str(qid_npz[i]): i for i in range(qid_npz.shape[0])}
    missing = [qid for qid in query_ids if qid not in map_npz]
    if missing:
        raise ValueError(f"Queries NPZ '{path}' missing {len(missing)} query_ids. Example: {missing[:10]}")

    Y = np.vstack([Y_npz[map_npz[qid]] for qid in query_ids]).astype(Y_npz.dtype, copy=False)
    Y = l2_normalize_rows(Y)
    return Y


def embed_mb_queries_on_the_fly(model_name: str, device: str, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
    """
    Optional fallback (torch required): compute Mixedbread embeddings for texts.
    """
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
    # Reduce peak RAM
    del model
    gc.collect()
    return Y


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a query-specific KAHM regressor (IDF–SVD -> Mixedbread) using TRAIN/TEST_QUERY_SET from query_set.py.")

    p.add_argument("--idf_svd_model", default=DEFAULT_IDF_SVD_MODEL, help="Path to idf_svd_model.joblib (required).")
    p.add_argument("--queries_npz", default="", help="Optional path to a combined precomputed Mixedbread query embeddings NPZ (backward compatible).")
    p.add_argument("--queries_npz_train", default=DEFAULT_QUERIES_NPZ_TRAIN, help="Path to precomputed Mixedbread TRAIN query embeddings NPZ.")
    p.add_argument("--queries_npz_test",  default=DEFAULT_QUERIES_NPZ_TEST,  help="Path to precomputed Mixedbread TEST query embeddings NPZ.")
    p.add_argument("--require_npz", action="store_true", help="If set, require NPZ targets and do not fall back to on-the-fly MB embedding.")
    p.add_argument("--out", default=DEFAULT_OUT, help="Output path for saved KAHM query regressor joblib")

    # Optional on-the-fly MB embedding (only used if --queries_npz is missing or --force_mb_on_the_fly is set)
    p.add_argument("--mb_model", default="mixedbread-ai/deepset-mxbai-embed-de-large-v1", help="Mixedbread model name (fallback only).")
    p.add_argument("--mb_device", default="cpu", help="Device for fallback MB embedding (cpu/cuda/mps).")
    p.add_argument("--mb_batch", type=int, default=64, help="Batch size for fallback MB embedding.")

    p.add_argument("--force_mb_on_the_fly", action="store_true", help="Ignore --queries_npz and compute MB embeddings with sentence_transformers (torch required).")

    # AE persistence / collision avoidance
    p.add_argument("--model_id", default=None, help="Identifier used to create a unique autoencoder directory under --ae_cache_root (defaults to stem of --out).")
    p.add_argument("--ae_cache_root", default=DEFAULT_AE_CACHE_ROOT, help=f"Root directory for saved per-cluster autoencoders (default: {DEFAULT_AE_CACHE_ROOT})")
    p.add_argument("--ae_dir", default=None, help="Explicit directory to save per-cluster autoencoders. Overrides --ae_cache_root/--model_id.")
    p.add_argument("--overwrite_ae_dir", action="store_true", default=DEFAULT_OVERWRITE_AE_DIR, help="Allow overwriting an existing AE directory.")

    # KAHM hyperparameters
    p.add_argument("--n_clusters", type=int, default=DEFAULT_N_CLUSTERS, help=f"Number of output clusters (default: {DEFAULT_N_CLUSTERS})")
    p.add_argument("--subspace_dim", type=int, default=DEFAULT_SUBSPACE_DIM, help=f"Subspace dimension (default: {DEFAULT_SUBSPACE_DIM})")
    p.add_argument("--nb", type=int, default=DEFAULT_NB, help=f"Nb (default: {DEFAULT_NB})")
    p.add_argument("--random_state", type=int, default=DEFAULT_RANDOM_STATE, help=f"Random seed (default: {DEFAULT_RANDOM_STATE})")
    p.add_argument("--input_scale", type=float, default=DEFAULT_INPUT_SCALE, help=f"Input scaling (default: {DEFAULT_INPUT_SCALE})")

    # Scalability controls
    p.add_argument("--kmeans_kind", default=DEFAULT_KMEANS_KIND, choices=["auto", "full", "minibatch"], help="KMeans implementation choice.")
    p.add_argument("--kmeans_batch_size", type=int, default=DEFAULT_KMEANS_BATCH_SIZE, help="MiniBatchKMeans batch size (if used).")
    p.add_argument("--max_train_per_cluster", type=int, default=DEFAULT_MAX_TRAIN_PER_CLUSTER, help="Cap training samples per cluster (optional).")
    p.add_argument("--model_dtype", default=DEFAULT_MODEL_DTYPE, choices=["float32", "float64"], help="Storage dtype inside the model.")
    p.add_argument("--cluster_center_normalization", default=DEFAULT_CLUSTER_CENTER_NORMALIZATION, choices=["none", "l2", "auto_l2"], help="Normalization for output cluster centers.")

    # Validation / tuning
    p.add_argument("--val_fraction", type=float, default=DEFAULT_VAL_FRACTION, help="Fraction of TRAIN queries used for validation/tuning.")
    p.add_argument("--val_max_samples", type=int, default=DEFAULT_VAL_MAX_SAMPLES, help="Max validation samples.")
    p.add_argument("--eval_soft", action="store_true", default=DEFAULT_EVAL_SOFT, help="Evaluate soft-mode regression.")
    p.add_argument("--tune_soft", action="store_true", default=DEFAULT_TUNE_SOFT, help="Tune soft-mode parameters (alpha/topk) on validation set.")
    p.add_argument("--tune_nlms", action="store_true", default=DEFAULT_TUNE_NLMS, help="Refine cluster centers with NLMS (optional).")
    p.add_argument("--soft_alphas", default=",".join(str(x) for x in DEFAULT_SOFT_ALPHAS), help="Comma-separated alphas for soft tuning.")
    p.add_argument("--soft_topks", default=",".join("none" if x is None else str(x) for x in DEFAULT_SOFT_TOPKS), help="Comma-separated topk values for soft tuning (use 'none').")

    p.add_argument("--preload_eval_classifier", action="store_true", help="Preload per-cluster autoencoders into RAM for evaluation (requires RAM).")

    # BlockSafe
    p.add_argument("--blocksafe", action="store_true", default=DEFAULT_BLOCKSAFE_ENABLED, help="Enable OTFL BlockSafe (if available).")
    p.add_argument("--blocksafe_backend", default=DEFAULT_BLOCKSAFE_BACKEND, choices=["threading", "multiprocessing"], help="BlockSafe backend.")
    p.add_argument("--blocksafe_jitter_std", type=float, default=DEFAULT_BLOCKSAFE_JITTER_STD)
    p.add_argument("--blocksafe_jitter_tries", type=int, default=DEFAULT_BLOCKSAFE_JITTER_TRIES)
    p.add_argument("--blocksafe_jitter_growth", type=float, default=DEFAULT_BLOCKSAFE_JITTER_GROWTH)
    p.add_argument("--blocksafe_eps_factor", type=float, default=DEFAULT_BLOCKSAFE_EPS_FACTOR)
    p.add_argument("--blocksafe_log_first", type=int, default=DEFAULT_BLOCKSAFE_LOG_FIRST)
    p.add_argument("--blocksafe_l2_normalized", action="store_true", default=DEFAULT_BLOCKSAFE_L2_NORMALIZED)

    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    idf_svd_model_path = str(args.idf_svd_model)
    if not Path(idf_svd_model_path).exists():
        raise FileNotFoundError(f"idf_svd_model not found: {idf_svd_model_path}")

    train_ids, train_texts = _validate_and_extract(TRAIN_QS, "TRAIN_QUERY_SET")
    test_ids, test_texts = _validate_and_extract(TEST_QS, "TEST_QUERY_SET")

    # Optional BlockSafe enablement
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

    # Compute inputs (IDF–SVD)
    print(f"Embedding IDF–SVD queries using: {idf_svd_model_path}")
    X_train_all = embed_idf_svd_queries(idf_svd_model_path, train_texts)
    X_test_all = embed_idf_svd_queries(idf_svd_model_path, test_texts)

    # Apply optional scaling
    if float(args.input_scale) != 1.0:
        X_train_all = X_train_all * float(args.input_scale)
        X_test_all = X_test_all * float(args.input_scale)

    
    # Compute targets (Mixedbread)
    # Preferred: load from NPZ (transformer-free). Supports split NPZ (train/test) or a combined NPZ (back-compat).
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
                "Missing required NPZ targets for: " + ", ".join(missing) +
                ". Provide --queries_npz_train/--queries_npz_test (or a combined --queries_npz) or unset --require_npz."
            )
        print("Computing Mixedbread query embeddings on-the-fly (torch required).")
        Y_train_all = embed_mb_queries_on_the_fly(str(args.mb_model), str(args.mb_device), train_texts, batch_size=int(args.mb_batch))
        Y_test_all = embed_mb_queries_on_the_fly(str(args.mb_model), str(args.mb_device), test_texts, batch_size=int(args.mb_batch))

    N_train, D_in = X_train_all.shape
    N_test, D_out = Y_test_all.shape[0], Y_train_all.shape[1]

    print(f"\nTrain queries: N={N_train}, D_in={D_in}, D_out={D_out}")
    print(f"Test  queries: N={X_test_all.shape[0]}")

    if int(args.n_clusters) >= int(max(10, N_train // 2)):
        print(f"WARNING: n_clusters={int(args.n_clusters)} is very large relative to N_train={N_train}. "
              "Expect many tiny clusters; consider 500–2000 for ~10k queries.")

    # Build validation split from train if requested / needed
    rng = np.random.RandomState(int(args.random_state))
    idx = np.arange(N_train, dtype=np.int64)
    rng.shuffle(idx)

    n_val = 0
    if float(args.val_fraction) > 0:
        n_val = int(round(float(args.val_fraction) * N_train))
        n_val = min(n_val, int(args.val_max_samples))
    n_val = max(0, min(n_val, N_train - 2))  # keep at least 2 for training

    val_idx = idx[:n_val]
    core_idx = idx[n_val:]

    X_val_all = X_train_all[val_idx] if n_val > 0 else None
    Y_val_all = Y_train_all[val_idx] if n_val > 0 else None
    X_train_core = X_train_all[core_idx]
    Y_train_core = Y_train_all[core_idx]

    # KAHM expects (D, N)
    X_train = X_train_core.T
    Y_train = Y_train_core.T
    # Evaluation is always on TEST set (never validation), regardless of --val_fraction.
    # Validation split (if any) is used only for tuning.
    X_eval = X_test_all.T
    Y_eval = Y_test_all.T
    eval_name = "test"

    # Choose model_id and AE directory naming
    run_model_id = str(args.model_id) if args.model_id else Path(str(args.out)).stem

    t0 = time.time()

    # BlockSafe compatibility: normalize return to a true context manager.
    with _as_blocksafe_context(ctx):
        model = train_kahm_regressor(
            X=X_train,
            Y=Y_train,
            n_clusters=int(args.n_clusters),
            subspace_dim=int(args.subspace_dim),
            Nb=int(args.nb),
            random_state=int(args.random_state),
            verbose=True,
            input_scale=1.0,  # input_scale already applied above
            kmeans_kind=str(args.kmeans_kind),
            kmeans_batch_size=int(args.kmeans_batch_size),
            max_train_per_cluster=(None if args.max_train_per_cluster is None else int(args.max_train_per_cluster)),
            model_dtype=str(args.model_dtype),
            cluster_center_normalization=str(args.cluster_center_normalization),
            save_ae_to_disk=False,
            ae_cache_root=str(args.ae_cache_root),
            ae_dir=(None if args.ae_dir is None else str(args.ae_dir)),
            overwrite_ae_dir=bool(args.overwrite_ae_dir),
            model_id=run_model_id,
            singleton_strategy="augment",  # {'augment','merge'}
            singleton_aux_mix = 0.1,
        )

    t_train = time.time() - t0
    try:
        print(f"\nTrained classifier_dir: {model.get('classifier_dir')} | model_id={model.get('model_id')} | time={t_train:.1f}s")
    except Exception:
        print(f"\nTraining time={t_train:.1f}s")

    if args.blocksafe and _BLOCKSAFE_STATS is not None:
        try:
            print("[blocksafe] stats:", asdict(_BLOCKSAFE_STATS))
        except Exception:
            pass

    # Optional soft tuning
    tuning_result = None
    if bool(args.tune_soft):
        alphas = tuple(parse_float_list(str(args.soft_alphas)))
        topks = tuple(parse_topk_list(str(args.soft_topks)))
        if X_val_all is None or Y_val_all is None:
            print("WARNING: tuning soft parameters requested, but validation set is empty. Skipping tuning.")
        else:
            print("\nTuning soft parameters on validation set...")
            tuning_result = tune_soft_params(
                model,
                X_val_all.T,
                Y_val_all.T,
                alphas=alphas,
                topks=topks,
                n_jobs=1,
                verbose=True,
            )

    # Optional NLMS refinement (requires a validation split)
    nlms_results = None
    if bool(args.tune_nlms):
        if X_val_all is None or Y_val_all is None:
            print("WARNING: NLMS refinement requested, but validation set is empty. Skipping NLMS refinement.")
        else:
            print("\nRefining cluster centers with NLMS...")
            nlms_results = tune_cluster_centers_nlms(
                model,
                np.hstack([X_val_all.T, X_train]),
                np.hstack([Y_val_all.T, Y_train]),
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

    # Evaluate
    print(f"\nEvaluating on {eval_name} set ...")

    if bool(args.preload_eval_classifier) and preload_kahm_classifier is not None:
        try:
            print("Preloading per-cluster autoencoders into RAM for evaluation ...")
            preload_kahm_classifier(model, n_jobs=min(8, (os.cpu_count() or 1)))
        except Exception as exc:
            print("WARNING: preload_kahm_classifier failed (continuing).")
            print(f"  Reason: {type(exc).__name__}: {exc}")

    metrics_soft = None
    if bool(args.eval_soft) or bool(args.tune_soft):
        Y_pred_soft = kahm_regress(model, X_eval, mode="soft", return_probabilities=False, batch_size=1024)
        metrics_soft = compute_embedding_metrics(Y_pred_soft, Y_eval)
        print("Soft-mode metrics:")
        print(f"  MSE:               {metrics_soft['mse']:.6f}")
        print(f"  Overall R^2:       {metrics_soft['r2_overall']:.4f}")
        print(f"  Cosine mean:       {metrics_soft['cos_mean']:.4f}")
        print(f"  Cosine p10/p50/p90:{metrics_soft['cos_p10']:.4f} / {metrics_soft['cos_p50']:.4f} / {metrics_soft['cos_p90']:.4f}")
        if model.get("soft_alpha") is not None or model.get("soft_topk") is not None:
            print(f"  Using soft params from model: alpha={model.get('soft_alpha')}, topk={model.get('soft_topk')}")

    # Save model with metadata
    created_at = datetime.now(timezone.utc).isoformat()
    try:
        tuning_payload = asdict(tuning_result) if tuning_result is not None else None
    except Exception:
        tuning_payload = tuning_result

    qid_fp = sha256_fingerprint(("\n".join(train_ids)).encode("utf-8"), ("\n".join(test_ids)).encode("utf-8"))

    model_meta = {
        "created_at_utc": created_at,
        "script": os.path.basename(__file__),
        "paths": {
            "idf_svd_model": args.idf_svd_model,
            "queries_npz": (str(args.queries_npz).strip() if (use_npz and str(args.queries_npz).strip()) else None),
            "queries_npz_train": (npz_train if use_npz else None),
            "queries_npz_test": (npz_test if use_npz else None),
            "out": args.out,
        },
        "data": {
            "n_train_queries": int(N_train),
            "n_test_queries": int(X_test_all.shape[0]),
            "n_train_core": int(X_train_core.shape[0]),
            "n_val": int(0 if X_val_all is None else X_val_all.shape[0]),
            "d_in": int(D_in),
            "d_out": int(D_out),
            "train_query_id_fingerprint_sha256": qid_fp,
        },
        "hyperparams": {
            "n_clusters": int(args.n_clusters),
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
            "eval_split": eval_name,
        },
    }

    # Attach metadata inside model dict (kahm_regression strips runtime-only keys on save)
    try:
        model["meta"] = model_meta
    except Exception:
        pass

    out_path = str(args.out)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    save_kahm_regressor(model, out_path)
    print(f"\nSaved query regressor to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
