#!/usr/bin/env python3
"""
train_kahm_embedding_regressor.py

Train a KAHM-based regression model predicting L2-normalized Mixedbread
embeddings (targets) from L2-normalized IDF–SVD embeddings (inputs), where both
are stored as NPZ “index bundles” keyed by sentence_id.

New (important):
  - If --include_queries is enabled, you can provide --queries_npz containing
    precomputed Mixedbread query embeddings for QUERY_SET.
  - This keeps the training process torch-free (no SentenceTransformer import)
    and avoids mixed OpenMP runtime issues.

Example (recommended 2-step):
  1) Precompute query embeddings (separate process):
     python build_query_embedding_index_npz.py \
        --out queries_embedding_index.npz \
        --model mixedbread-ai/deepset-mxbai-embed-de-large-v1 \
        --device cpu

  2) Train with query augmentation without torch:
     python train_kahm_embedding_regressor.py \
        --idf_svd_npz embedding_index_idf_svd.npz \
        --semantic_npz embedding_index.npz \
        --idf_svd_model idf_svd_model.joblib \
        --include_queries \
        --queries_npz queries_embedding_index.npz \
        --out kahm_regressor_idf_to_mixedbread.joblib
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
import time
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    # Prefer the memory-optimized implementation if present.
    from kahm_regression import (
        kahm_regress,
        save_kahm_regressor,
        train_kahm_regressor,
        tune_soft_params,
    )
except ImportError:  # pragma: no cover
    # Fallback to the baseline implementation.
    from kahm_regression import kahm_regress, save_kahm_regressor, train_kahm_regressor, tune_soft_params



# ----------------------------- QUERY_SET import -----------------------------
def _load_query_set() -> List[Dict[str, Any]]:
    from query_set import TRAIN_QUERY_SET  # type: ignore
    qs = list(TRAIN_QUERY_SET)
    if not qs:
        raise RuntimeError("Imported QUERY_SET is empty.")
    return qs





QUERY_SET = _load_query_set()


# ----------------------------- BlockSafe (optional) -----------------------------
try:
    from otfl_blocksafe import enable_otfl_blocksafe, _BLOCKSAFE_STATS  # type: ignore
except Exception:
    enable_otfl_blocksafe = None
    _BLOCKSAFE_STATS = None


# ----------------------------- Defaults -----------------------------
DEFAULT_IDF_SVD_NPZ = "embedding_index_idf_svd.npz"
DEFAULT_SEMANTIC_NPZ = "embedding_index.npz"
DEFAULT_IDF_SVD_MODEL = "idf_svd_model.joblib"
DEFAULT_OUT = "kahm_regressor_idf_to_mixedbread.joblib"

DEFAULT_INCLUDE_QUERIES = True
DEFAULT_QUERIES_NPZ = "queries_embedding_index.npz"

DEFAULT_N_CLUSTERS = 20000
DEFAULT_SUBSPACE_DIM = 20
DEFAULT_NB = 100  # IMPORTANT: as requested
DEFAULT_TRAIN_FRACTION = 1.0
DEFAULT_RANDOM_STATE = 0
DEFAULT_INPUT_SCALE = 1.0

# Memory / scaling controls (classifier side)
DEFAULT_KMEANS_KIND = "full"  # {'auto','full','minibatch'}
DEFAULT_KMEANS_BATCH_SIZE = 4096
DEFAULT_MAX_TRAIN_PER_CLUSTER = None
DEFAULT_MODEL_DTYPE = "float32"
DEFAULT_CLUSTER_CENTER_NORMALIZATION = "none"  # none|l2|auto_l2

# Soft-mode memory controls
DEFAULT_SOFT_BATCH_SIZE = 2048

# Soft-mode tuning defaults
DEFAULT_EVAL_SOFT = True
DEFAULT_TUNE_SOFT = True
DEFAULT_VAL_FRACTION = 0.01
DEFAULT_VAL_MAX_SAMPLES = 5000
DEFAULT_SOFT_ALPHAS = (2.0, 5.0, 8.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 18.0, 20.0, 25.0, 50.0)
DEFAULT_SOFT_TOPKS = (2, 5, 10, 11, 12, 13, 14, 15, 20, 25, 50)

# BlockSafe defaults
DEFAULT_BLOCKSAFE_ENABLED = True
DEFAULT_BLOCKSAFE_BACKEND = "threading"
DEFAULT_BLOCKSAFE_JITTER_STD = 1e-5
DEFAULT_BLOCKSAFE_JITTER_TRIES = 6
DEFAULT_BLOCKSAFE_JITTER_GROWTH = 2.0
DEFAULT_BLOCKSAFE_EPS_FACTOR = 10.0
DEFAULT_BLOCKSAFE_LOG_FIRST = 100
DEFAULT_BLOCKSAFE_L2_NORMALIZED = True

# Degenerecy check
DEFAULT_NO_DEGENERACY_CHECK = False


# ----------------------------- Utilities -----------------------------

def as_float_ndarray(x: Any, *, min_dtype: np.dtype = np.dtype(np.float32)) -> np.ndarray:
    """Convert input to a floating ndarray without downcasting precision.

    - Converts scipy sparse matrices to dense via .toarray() when available.
    - Integer/bool inputs are promoted to float64.
    - Floating inputs keep their existing precision unless below `min_dtype`,
      in which case they are promoted to `min_dtype`.
    """
    if hasattr(x, "toarray"):
        x = x.toarray()
    arr = np.asarray(x)
    if arr.dtype.kind not in "fc":
        arr = arr.astype(np.float64, copy=False)
    dtype = np.result_type(arr.dtype, min_dtype)
    return arr.astype(dtype, copy=False)

def l2_normalize_rows(mat: np.ndarray, eps: float = 1e-12, *, inplace: bool = False) -> np.ndarray:
    """L2-normalize rows of a 2D array.

    Memory behavior:
      - inplace=False (default): returns a new normalized array (does not mutate the input).
      - inplace=True: normalizes the provided array in-place (or a writable copy).
    """
    mat = as_float_ndarray(mat)
    dtype = mat.dtype
    eps_t = dtype.type(eps)

    if inplace:
        out = mat if mat.flags.writeable else mat.copy()
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        np.maximum(norms, eps_t, out=norms)
        out /= norms
        return out

    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, eps_t).astype(dtype, copy=False)
    return mat / norms



def _npz_scalar_to_str(x: Any) -> Optional[str]:
    try:
        if x is None:
            return None
        if isinstance(x, np.ndarray) and x.shape == ():
            return str(x.item())
        return str(x)
    except Exception:
        return None


def load_npz_bundle(path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    d = np.load(path, allow_pickle=False)
    if "sentence_id" not in d or "embeddings" not in d:
        raise ValueError(f"NPZ '{path}' must contain keys 'sentence_id' and 'embeddings'. Keys: {list(d.keys())}")

    ids = np.asarray(d["sentence_id"], dtype=np.int64)
    emb = as_float_ndarray(d["embeddings"])

    if ids.ndim != 1:
        raise ValueError(f"NPZ '{path}': sentence_id must be 1D; got {ids.shape}.")
    if emb.ndim != 2:
        raise ValueError(f"NPZ '{path}': embeddings must be 2D; got {emb.shape}.")
    if emb.shape[0] != ids.shape[0]:
        raise ValueError(
            f"NPZ '{path}': embeddings rows must match sentence_id length; got {emb.shape[0]} vs {ids.shape[0]}."
        )

    meta: Dict[str, Any] = {}
    for k in d.files:
        if k in ("sentence_id", "embeddings"):
            continue
        v = d[k]
        if isinstance(v, np.ndarray) and v.shape == ():
            meta[k] = v.item()
        else:
            meta[k] = f"<array shape={getattr(v, 'shape', None)} dtype={getattr(v, 'dtype', None)}>"
    return ids, emb, meta


def align_by_sentence_id(
    ids_x: np.ndarray,
    X: np.ndarray,
    ids_y: np.ndarray,
    Y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids_x = np.asarray(ids_x, dtype=np.int64)
    ids_y = np.asarray(ids_y, dtype=np.int64)

    sx = np.argsort(ids_x)
    sy = np.argsort(ids_y)

    ids_xs = ids_x[sx]
    ids_ys = ids_y[sy]

    common = np.intersect1d(ids_xs, ids_ys, assume_unique=False)
    if common.size == 0:
        raise ValueError("No overlapping sentence_id values between the two NPZ bundles.")

    px = np.searchsorted(ids_xs, common)
    py = np.searchsorted(ids_ys, common)

    X_aligned = X[sx[px]]
    Y_aligned = Y[sy[py]]

    return common, X_aligned, Y_aligned


def train_test_split_indices(N: int, train_fraction: float, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, test_idx) for N samples.

    Memory notes:
      - Special-cases train_fraction >= 1.0 to avoid allocating a full permutation.
      - Preserves the original behavior of always returning at least one training sample.
    """
    N = int(N)
    if N <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    tf = float(train_fraction)

    # Fast path: all-train, no-test.
    if tf >= 1.0:
        return np.arange(N, dtype=np.int64), np.array([], dtype=np.int64)

    rng = np.random.RandomState(int(random_state))

    # Fast path: effectively 0% train still returns 1 sample, without a full permutation.
    if tf <= 0.0:
        i = int(rng.randint(0, N))
        train_idx = np.array([i], dtype=np.int64)
        if N == 1:
            return train_idx, np.array([], dtype=np.int64)
        test_idx = np.concatenate(
            [np.arange(0, i, dtype=np.int64), np.arange(i + 1, N, dtype=np.int64)],
            axis=0,
        )
        return train_idx, test_idx

    perm = rng.permutation(N)
    n_train = int(tf * N)
    n_train = max(1, min(N, n_train))
    return perm[:n_train], perm[n_train:]


def split_train_val_indices(train_idx: np.ndarray, val_fraction: float, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
    train_idx = np.asarray(train_idx, dtype=np.int64)
    if train_idx.size < 10 or val_fraction <= 0.0:
        return train_idx, np.array([], dtype=np.int64)

    rng = np.random.RandomState(int(random_state) + 1337)
    perm = rng.permutation(train_idx.size)
    n_val = int(val_fraction * train_idx.size)
    n_val = max(1, min(train_idx.size - 1, n_val))
    val_pos = perm[:n_val]
    train_pos = perm[n_val:]
    return train_idx[train_pos], train_idx[val_pos]


def compute_embedding_metrics(
    Y_pred: np.ndarray | Tuple[np.ndarray, np.ndarray],
    Y_true: np.ndarray,
) -> Dict[str, float]:
    """Compute MSE, overall R^2, and cosine similarity stats for (D, N) embeddings.

    Memory notes:
      - avoids computing (Y_pred - Y_true) twice
      - avoids forming full centered matrices for R^2 (uses sum-of-squares identity)
      - avoids allocating a full (D, N) product for dot-products (uses einsum)
    """
    if isinstance(Y_pred, tuple):
        Y_pred = Y_pred[0]

    Y_pred = as_float_ndarray(Y_pred)
    Y_true = as_float_ndarray(Y_true)

    # Use a shared working dtype (no precision downcast).
    work_dtype = np.result_type(Y_pred.dtype, Y_true.dtype)
    Y_pred = Y_pred.astype(work_dtype, copy=False)
    Y_true = Y_true.astype(work_dtype, copy=False)

    # Squared error (allocate once; reuse for both mse and residual_ss)
    diff = Y_pred - Y_true
    np.square(diff, out=diff)
    residual_ss = float(np.sum(diff))
    mse = float(np.mean(diff))

    # Total sum of squares without forming a centered (D, N) matrix:
    #   sum_j (y_ij - mean_i)^2 = sum_j y_ij^2 - N * mean_i^2
    N = int(Y_true.shape[1])
    mean_true = Y_true.mean(axis=1)
    sum_sq_true = float(np.einsum("ij,ij->", Y_true, Y_true))
    total_ss = float(sum_sq_true - N * float(np.sum(mean_true * mean_true)))
    r2_overall = float(1.0 - residual_ss / total_ss) if total_ss > 0 else float("nan")

    # Cosine similarity: dot(Y_pred, Y_true) / ||Y_pred||, assuming Y_true approx unit-norm
    dot = np.einsum("ij,ij->j", Y_pred, Y_true)
    pred_norm = np.maximum(np.linalg.norm(Y_pred, axis=0), 1e-12)
    cos = dot / pred_norm

    return {
        "mse": mse,
        "r2_overall": r2_overall,
        "cos_mean": float(np.mean(cos)),
        "cos_p10": float(np.percentile(cos, 10)),
        "cos_p50": float(np.percentile(cos, 50)),
        "cos_p90": float(np.percentile(cos, 90)),
    }

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


# ----------------------------- Query helpers -----------------------------
def embed_queries_idf_svd(idf_svd_model_path: str, D_in_expected: int) -> np.ndarray:
    import joblib

    texts = [str(q.get("query_text", "")).strip() for q in QUERY_SET]
    print(f"\nEmbedding {len(texts)} queries with IDF–SVD pipeline: {idf_svd_model_path}")

    pipe = joblib.load(idf_svd_model_path)
    X_q = pipe.transform(texts)

    X_q = as_float_ndarray(X_q)
    if X_q.ndim != 2:
        raise ValueError(f"IDF–SVD pipeline output must be 2D; got {X_q.shape}.")
    if X_q.shape[1] != int(D_in_expected):
        raise ValueError(f"IDF–SVD query dim mismatch: got {X_q.shape[1]}, expected {D_in_expected}")

    X_q = l2_normalize_rows(X_q)
    print(f"Query IDF–SVD embeddings shape (L2-normalized): {X_q.shape}")
    return X_q


def load_precomputed_mixedbread_queries_npz(path: str, D_out_expected: int) -> np.ndarray:
    """
    Load precomputed Mixedbread query embeddings from NPZ and align them to QUERY_SET by query_id.
    Expects keys: query_id, embeddings.
    """
    d = np.load(path, allow_pickle=False)
    if "query_id" not in d or "embeddings" not in d:
        raise ValueError(f"Queries NPZ '{path}' must contain keys 'query_id' and 'embeddings'. Keys: {list(d.keys())}")

    qid_npz = np.asarray(d["query_id"])
    Y_npz = as_float_ndarray(d["embeddings"])

    if qid_npz.ndim != 1 or Y_npz.ndim != 2:
        raise ValueError(f"Queries NPZ '{path}': expected query_id (Q,), embeddings (Q,D); got {qid_npz.shape}, {Y_npz.shape}")

    if Y_npz.shape[1] != int(D_out_expected):
        raise ValueError(f"Queries NPZ '{path}': embedding dim {Y_npz.shape[1]} != expected {D_out_expected}")

    # Build mapping and reorder to match QUERY_SET
    query_ids = [str(q.get("query_id", "")).strip() for q in QUERY_SET]
    if any(not qid for qid in query_ids):
        raise ValueError("QUERY_SET contains empty query_id entries.")
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("QUERY_SET contains duplicate query_id values (must be unique).")

    map_npz = {str(qid_npz[i]): i for i in range(qid_npz.shape[0])}
    missing = [qid for qid in query_ids if qid not in map_npz]
    if missing:
        raise ValueError(f"Queries NPZ '{path}' missing {len(missing)} query_ids from QUERY_SET. Example: {missing[:10]}")

    Y = np.vstack([Y_npz[map_npz[qid]] for qid in query_ids]).astype(Y_npz.dtype, copy=False)
    Y = l2_normalize_rows(Y)  # enforce normalization for cosine geometry
    print(f"Loaded precomputed Mixedbread query embeddings: {path} | shape={Y.shape} (L2-normalized)")
    return Y


def embed_queries_mixedbread_on_the_fly(model_name: str, device: str, D_out_expected: int) -> np.ndarray:
    """
    Fallback: compute Mixedbread query embeddings in-process.
    This imports SentenceTransformer/torch and may reintroduce OpenMP conflicts on some setups.
    """
    from sentence_transformers import SentenceTransformer

    texts = [str(q.get("query_text", "")).strip() for q in QUERY_SET]
    texts_prefixed = ["query: " + t for t in texts]

    print(f"Embedding {len(texts_prefixed)} queries with Mixedbread model: {model_name} (device={device})")
    model = SentenceTransformer(model_name, device=device, truncate_dim=int(D_out_expected))

    Y_q = model.encode(
        texts_prefixed,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    if Y_q.ndim != 2:
        raise ValueError(f"Mixedbread encode output must be 2D; got {Y_q.shape}.")
    if Y_q.shape[1] != int(D_out_expected):
        if Y_q.shape[1] > int(D_out_expected):
            Y_q = Y_q[:, : int(D_out_expected)]
        else:
            raise ValueError(f"Mixedbread query dim mismatch: got {Y_q.shape[1]}, expected {D_out_expected}")

    Y_q = l2_normalize_rows(Y_q)
    print(f"Query Mixedbread embeddings shape (L2-normalized): {Y_q.shape}")
    return Y_q


def choose_device(requested: str) -> str:
    if requested and requested.lower() != "auto":
        return requested
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


# ----------------------------- CLI -----------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train KAHM regressor mapping IDF–SVD embeddings -> Mixedbread embeddings from NPZ bundles.")

    p.add_argument("--idf_svd_npz", default=DEFAULT_IDF_SVD_NPZ, help="Path to embedding_index_idf_svd.npz")
    p.add_argument("--semantic_npz", default=DEFAULT_SEMANTIC_NPZ, help="Path to embedding_index.npz")
    p.add_argument("--idf_svd_model", default=DEFAULT_IDF_SVD_MODEL, help="Path to idf_svd_model.joblib (required if --include_queries)")
    p.add_argument("--out", default=DEFAULT_OUT, help="Output path for saved KAHM regressor joblib")

    p.add_argument("--include_queries", action="store_true", default=DEFAULT_INCLUDE_QUERIES, help="Include QUERY_SET as additional training samples")
    p.add_argument(
        "--queries_npz",
        default=DEFAULT_QUERIES_NPZ,
        help="Precomputed Mixedbread QUERY_SET embeddings NPZ (recommended). Used only if --include_queries.",
    )
    p.add_argument(
        "--mixedbread_model",
        default=None,
        help="Only used if queries_npz is missing and on-the-fly Mixedbread query embeddings are computed.",
    )
    p.add_argument(
        "--mixedbread_device",
        default="cpu",
        help="Only used if queries_npz is missing and on-the-fly Mixedbread query embeddings are computed (cpu/cuda/mps/auto).",
    )

    p.add_argument("--n_clusters", type=int, default=DEFAULT_N_CLUSTERS)
    p.add_argument("--subspace_dim", type=int, default=DEFAULT_SUBSPACE_DIM)
    p.add_argument("--nb", type=int, default=DEFAULT_NB)
    p.add_argument("--train_fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    p.add_argument("--random_state", type=int, default=DEFAULT_RANDOM_STATE)
    p.add_argument("--input_scale", type=float, default=DEFAULT_INPUT_SCALE)

    # Scaling / memory controls
    p.add_argument(
        "--kmeans_kind",
        default=DEFAULT_KMEANS_KIND,
        help="KMeans backend: auto/full/minibatch (auto switches to MiniBatchKMeans for large n_clusters)",
    )
    p.add_argument("--kmeans_batch_size", type=int, default=DEFAULT_KMEANS_BATCH_SIZE, help="MiniBatchKMeans batch_size")
    p.add_argument(
        "--max_train_per_cluster",
        type=int,
        default=DEFAULT_MAX_TRAIN_PER_CLUSTER,
        help="Cap classifier training samples per output cluster (largest impact on RAM when n_clusters is large).",
    )
    p.add_argument(
        "--model_dtype",
        default=DEFAULT_MODEL_DTYPE,
        help="Downcast numeric arrays inside the trained model (float32 recommended).",
    )

    p.add_argument(
        "--cluster_center_normalization",
        default=DEFAULT_CLUSTER_CENTER_NORMALIZATION,
        choices=["none", "l2", "auto_l2"],
        help="Normalize output cluster centroids: none (general), l2 (directional/unit embeddings), auto_l2 (detect unit-norm Y).",
    )

    p.add_argument(
        "--soft_batch_size",
        type=int,
        default=DEFAULT_SOFT_BATCH_SIZE,
        help="Batch size for soft-mode inference/eval to avoid allocating C×N distance/probability matrices.",
    )

    p.add_argument("--no_degeneracy_check", action="store_true", default=DEFAULT_NO_DEGENERACY_CHECK, help="Disable OTFL degeneracy probing of clusters")

    # Soft-mode tuning / evaluation
    p.add_argument("--eval_soft", action="store_true", default=DEFAULT_EVAL_SOFT, help="Also evaluate 'soft' prediction mode on the test set")
    p.add_argument("--tune_soft", action="store_true", default=DEFAULT_TUNE_SOFT, help="Tune (alpha, topk) for soft mode on a validation split")
    p.add_argument("--val_fraction", type=float, default=DEFAULT_VAL_FRACTION, help="Validation fraction from training sentences (only if --tune_soft)")
    p.add_argument("--val_max_samples", type=int, default=DEFAULT_VAL_MAX_SAMPLES, help="Max validation samples used for tuning (subsample if larger)")
    p.add_argument("--soft_alphas", default=",".join(str(x) for x in DEFAULT_SOFT_ALPHAS), help="Comma-separated alphas for tuning grid")
    p.add_argument("--soft_topks", default=",".join(str(x) for x in DEFAULT_SOFT_TOPKS), help="Comma-separated topk values for tuning grid (use 'None')")

    # BlockSafe controls
    p.add_argument("--blocksafe", action="store_true", default=DEFAULT_BLOCKSAFE_ENABLED, help="Enable OTFL BlockSafe monkey patch (recommended).")
    p.add_argument("--no_blocksafe", action="store_false", dest="blocksafe", help="Disable OTFL BlockSafe.")
    p.add_argument("--blocksafe_backend", default=DEFAULT_BLOCKSAFE_BACKEND, help="joblib backend for OTFL when BlockSafe is enabled (default: threading)")
    p.add_argument("--blocksafe_jitter_std", type=float, default=DEFAULT_BLOCKSAFE_JITTER_STD)
    p.add_argument("--blocksafe_jitter_tries", type=int, default=DEFAULT_BLOCKSAFE_JITTER_TRIES)
    p.add_argument("--blocksafe_jitter_growth", type=float, default=DEFAULT_BLOCKSAFE_JITTER_GROWTH)
    p.add_argument("--blocksafe_eps_factor", type=float, default=DEFAULT_BLOCKSAFE_EPS_FACTOR)
    p.add_argument("--blocksafe_log_first", type=int, default=DEFAULT_BLOCKSAFE_LOG_FIRST)
    p.add_argument("--blocksafe_l2_normalized", action="store_true", default=DEFAULT_BLOCKSAFE_L2_NORMALIZED)
    p.add_argument("--blocksafe_seed", type=int, default=0, help="Random seed used inside BlockSafe jitter logic")

    return p


# ----------------------------- Main -----------------------------
def main() -> int:
    args = build_arg_parser().parse_args()
    t_start = time.perf_counter()

    # Load NPZ bundles
    print(f"Loading IDF–SVD NPZ:   {args.idf_svd_npz}")
    ids_x, X_all, meta_x = load_npz_bundle(args.idf_svd_npz)

    print(f"Loading Semantic NPZ:  {args.semantic_npz}")
    ids_y, Y_all, meta_y = load_npz_bundle(args.semantic_npz)

    print(f"IDF–SVD:   ids={ids_x.shape}, X={X_all.shape}")
    print(f"Semantic:  ids={ids_y.shape}, Y={Y_all.shape}")

    # Align by sentence_id
    common_ids, X, Y = align_by_sentence_id(ids_x, X_all, ids_y, Y_all)
    # Drop large unaligned arrays early to reduce peak RAM.
    try:
        del X_all, Y_all, ids_x, ids_y
    except Exception:
        pass

    N, D_in = X.shape
    _, D_out = Y.shape
    print(f"Aligned by sentence_id: N_common={N}, D_in={D_in}, D_out={D_out}")

    fp_x = _npz_scalar_to_str(meta_x.get("dataset_fingerprint_sha256"))
    fp_y = _npz_scalar_to_str(meta_y.get("dataset_fingerprint_sha256"))
    if fp_x and fp_y:
        print("Dataset fingerprint:", "MATCH" if fp_x == fp_y else "MISMATCH (continuing).")

        # Train/test split
    train_idx_sent, test_idx_sent = train_test_split_indices(N, args.train_fraction, args.random_state)
    print(f"Train sentence samples: {train_idx_sent.size}")
    print(f"Test  sentence samples: {test_idx_sent.size}")

    # Optional validation split for soft tuning (sentences only)
    train_core_idx_sent = train_idx_sent
    val_idx_sent = np.array([], dtype=np.int64)
    if args.tune_soft:
        train_core_idx_sent, val_idx_sent = split_train_val_indices(train_idx_sent, args.val_fraction, args.random_state)
        print(f"Validation sentence samples (for soft tuning): {val_idx_sent.size}")
        print(f"Core train sentence samples (after val split): {train_core_idx_sent.size}")

    # L2-normalize only the required splits (avoids keeping full normalized copies)
    X_train_sent = l2_normalize_rows(X[train_core_idx_sent], inplace=True)
    Y_train_sent = l2_normalize_rows(Y[train_core_idx_sent], inplace=True)
    X_test_sent = l2_normalize_rows(X[test_idx_sent], inplace=True)
    Y_test_sent = l2_normalize_rows(Y[test_idx_sent], inplace=True)

    # Validation splits (sentences only). Keep as (N_val, D) until transpose later.
    X_val_sent: Optional[np.ndarray] = None
    Y_val_sent: Optional[np.ndarray] = None
    if val_idx_sent.size > 0:
        X_val_sent = l2_normalize_rows(X[val_idx_sent], inplace=True)
        Y_val_sent = l2_normalize_rows(Y[val_idx_sent], inplace=True)

    # Free large aligned matrices early (helps peak RAM for large corpora)
    try:
        del X, Y, common_ids
    except Exception:
        pass


    # Optional query augmentation
    if args.include_queries:
        if not args.idf_svd_model or not os.path.exists(args.idf_svd_model):
            raise ValueError("--include_queries requires a valid --idf_svd_model path (idf_svd_model.joblib).")

        # Always compute IDF–SVD query embeddings (no torch)
        X_q = embed_queries_idf_svd(args.idf_svd_model, D_in_expected=D_in)

        # Prefer precomputed Mixedbread queries NPZ to keep training torch-free
        Y_q: np.ndarray
        if args.queries_npz and os.path.exists(args.queries_npz):
            Y_q = load_precomputed_mixedbread_queries_npz(args.queries_npz, D_out_expected=D_out)
        else:
            # Fallback to on-the-fly Mixedbread query embedding (may reintroduce OpenMP warnings)
            mb_model = args.mixedbread_model or _npz_scalar_to_str(meta_y.get("model")) or "mixedbread-ai/deepset-mxbai-embed-de-large-v1"
            device = choose_device(args.mixedbread_device)
            Y_q = embed_queries_mixedbread_on_the_fly(mb_model, device=device, D_out_expected=D_out)

        X_train_all = np.vstack([X_train_sent, X_q])
        Y_train_all = np.vstack([Y_train_sent, Y_q])
        print(f"Total training samples (sentences + queries): {X_train_all.shape[0]}")
    else:
        X_train_all = X_train_sent
        Y_train_all = Y_train_sent
        print("Training only on sentence embeddings (no query augmentation).")
        print(f"Total training samples (sentences only): {X_train_all.shape[0]}")
        print(f"Query samples available (not used): {len(QUERY_SET)}")

    N_train_total = X_train_all.shape[0]
    if N_train_total < 2 * int(args.n_clusters):
        raise ValueError(
            f"Not enough training samples ({N_train_total}) for n_clusters={args.n_clusters}. "
            f"Need N_train_total >= 2 * n_clusters."
        )

    # KAHM expects (D, N)
    X_train = X_train_all.T
    Y_train = Y_train_all.T
    X_test = X_test_sent.T
    Y_test = Y_test_sent.T


        # Validation tensors (sentences only). Used for soft tuning when available and as evaluation split when present.
    X_val: Optional[np.ndarray] = None
    Y_val: Optional[np.ndarray] = None
    if X_val_sent is not None and Y_val_sent is not None and X_val_sent.shape[0] > 0:
        X_val = X_val_sent.T
        Y_val = Y_val_sent.T

        val_max = int(args.val_max_samples)
        if val_max > 0 and X_val.shape[1] > val_max:
            rng = np.random.RandomState(int(args.random_state) + 4242)
            sel = rng.choice(X_val.shape[1], size=val_max, replace=False)
            X_val = X_val[:, sel]
            Y_val = Y_val[:, sel]
            print(f"Subsampled validation set to {val_max} samples for validation/evaluation.")

        # Free the (N_val, D) copies once transposed.
        try:
            del X_val_sent, Y_val_sent
        except Exception:
            pass

# Choose evaluation split: validation if present, else the held-out test set.
    X_eval = X_test
    Y_eval = Y_test
    eval_name = "test"
    if X_val is not None and Y_val is not None and X_val.shape[1] > 0:
        X_eval = X_val
        Y_eval = Y_val
        eval_name = "validation"
    
    # Enable BlockSafe and force threading backend
    ctx = nullcontext()
    blocksafe_info = {"enabled": False}
    if args.blocksafe:
        if enable_otfl_blocksafe is None:
            raise ImportError("BlockSafe requested but otfl_blocksafe.py could not be imported.")
        ctx_factory = enable_otfl_blocksafe(
            jitter_std=float(args.blocksafe_jitter_std),
            jitter_tries=int(args.blocksafe_jitter_tries),
            jitter_growth=float(args.blocksafe_jitter_growth),
            eps_factor=float(args.blocksafe_eps_factor),
            backend=str(args.blocksafe_backend),
            l2_normalized=bool(args.blocksafe_l2_normalized),
            log_first=int(args.blocksafe_log_first),
            random_seed=int(args.blocksafe_seed),
        )
        ctx = ctx_factory()
        blocksafe_info = {
            "enabled": True,
            "backend": str(args.blocksafe_backend),
            "jitter_std": float(args.blocksafe_jitter_std),
            "jitter_tries": int(args.blocksafe_jitter_tries),
            "jitter_growth": float(args.blocksafe_jitter_growth),
            "eps_factor": float(args.blocksafe_eps_factor),
            "log_first": int(args.blocksafe_log_first),
            "l2_normalized": bool(args.blocksafe_l2_normalized),
            "seed": int(args.blocksafe_seed),
        }

    # Train KAHM regressor
    print("\nTraining KAHM regressor (IDF–SVD → Mixedbread, L2-normalized spaces) ...")
    with ctx:
        model = train_kahm_regressor(
            X=X_train,
            Y=Y_train,
            n_clusters=int(args.n_clusters),
            subspace_dim=int(args.subspace_dim),
            Nb=int(args.nb),
            random_state=int(args.random_state),
            verbose=True,
            input_scale=float(args.input_scale),
            kmeans_kind=str(args.kmeans_kind),
            kmeans_batch_size=int(args.kmeans_batch_size),
            max_train_per_cluster=(None if args.max_train_per_cluster is None else int(args.max_train_per_cluster)),
            model_dtype=str(args.model_dtype),
            cluster_center_normalization=str(args.cluster_center_normalization)
        )

    if args.blocksafe and _BLOCKSAFE_STATS is not None:
        try:
            print("[blocksafe] stats:", asdict(_BLOCKSAFE_STATS))
        except Exception:
            pass

    # Optional soft tuning
    tuning_result = None
    if args.tune_soft:
        if X_val is None or Y_val is None or X_val.shape[1] == 0:
            print("WARNING: --tune_soft requested, but validation set is empty. Skipping tuning.")
        else:
            alphas = tuple(parse_float_list(args.soft_alphas))
            topks = tuple(parse_topk_list(args.soft_topks))
            print("\nTuning soft parameters on validation set (sentences only) ...")
            assert X_val is not None and Y_val is not None
            tuning_result = tune_soft_params(
                model,
                X_val,
                Y_val,
                alphas=alphas,
                topks=topks,
                n_jobs=1,
                verbose=True
            )

    # Evaluate on validation (if present) or test (fallback)
    print(f"\nEvaluating on {eval_name} set (sentences only) ...")
    Y_pred_hard = kahm_regress(model, X_eval, mode="hard")
    metrics_hard = compute_embedding_metrics(Y_pred_hard, Y_eval)

    print("Hard-mode metrics:")
    print(f"  Test MSE:           {metrics_hard['mse']:.6f}")
    print(f"  Overall R^2:        {metrics_hard['r2_overall']:.4f}")
    print(f"  Cosine mean:        {metrics_hard['cos_mean']:.4f}")
    print(f"  Cosine p10/p50/p90: {metrics_hard['cos_p10']:.4f} / {metrics_hard['cos_p50']:.4f} / {metrics_hard['cos_p90']:.4f}")

    metrics_soft = None
    if args.eval_soft or args.tune_soft:
        try:
            bs = int(args.soft_batch_size)
            Y_pred_soft = kahm_regress(model, X_eval, mode="soft", batch_size=(bs if bs > 0 else None))
            metrics_soft = compute_embedding_metrics(Y_pred_soft, Y_eval)
            print("Soft-mode metrics:")
            print(f"  Test MSE:           {metrics_soft['mse']:.6f}")
            print(f"  Overall R^2:        {metrics_soft['r2_overall']:.4f}")
            print(f"  Cosine mean:        {metrics_soft['cos_mean']:.4f}")
            print(f"  Cosine p10/p50/p90: {metrics_soft['cos_p10']:.4f} / {metrics_soft['cos_p50']:.4f} / {metrics_soft['cos_p90']:.4f}")
            if model.get("soft_alpha") is not None or model.get("soft_topk") is not None:
                print(f"  Using soft params from model: alpha={model.get('soft_alpha')}, topk={model.get('soft_topk')}")
        except Exception as exc:
            print("WARNING: Soft-mode evaluation failed (continuing).")
            print(f"  Reason: {type(exc).__name__}: {exc}")

    # Save model with metadata
    created_at = datetime.now(timezone.utc).isoformat()
    try:
        tuning_payload = asdict(tuning_result) if tuning_result is not None else None
    except Exception:
        tuning_payload = tuning_result

    model_meta = {
        "created_at_utc": created_at,
        "script": os.path.basename(__file__),
        "paths": {
            "idf_svd_npz": args.idf_svd_npz,
            "semantic_npz": args.semantic_npz,
            "idf_svd_model": args.idf_svd_model if args.include_queries else None,
            "queries_npz": args.queries_npz if (args.include_queries and args.queries_npz and os.path.exists(args.queries_npz)) else None,
            "out": args.out,
        },
        "data": {
            "n_common_sentences": int(N),
            "d_in": int(D_in),
            "d_out": int(D_out),
            "n_train_sentences_core": int(train_core_idx_sent.size),
            "n_test_sentences": int(X_test_sent.shape[0]),
            "n_val_sentences_used": int(0 if X_val is None else X_val.shape[1]),
            "include_queries": bool(args.include_queries),
            "n_queries": int(len(QUERY_SET)) if args.include_queries else 0,
            "train_fraction": float(args.train_fraction),
        },
        "npz_meta_idf_svd": meta_x,
        "npz_meta_semantic": meta_y,
        "fingerprints": {
            "idf_svd_npz_dataset_fingerprint_sha256": fp_x,
            "semantic_npz_dataset_fingerprint_sha256": fp_y,
        },
        "hyperparams": {
            "n_clusters": int(args.n_clusters),
            "subspace_dim": int(args.subspace_dim),
            "Nb": int(args.nb),
            "random_state": int(args.random_state),
            "input_scale": float(args.input_scale),
            "enable_degeneracy_check": bool(not args.no_degeneracy_check),
        },
        "blocksafe": blocksafe_info,
        "soft": {
            "tune_soft": bool(args.tune_soft),
            "eval_soft": bool(args.eval_soft),
            "val_fraction": float(args.val_fraction),
            "val_max_samples": int(args.val_max_samples),
            "soft_alphas": parse_float_list(args.soft_alphas),
            "soft_topks": [("None" if x is None else int(x)) for x in parse_topk_list(args.soft_topks)],
            "tuning_result": tuning_payload,
            "model_soft_alpha": model.get("soft_alpha"),
            "model_soft_topk": model.get("soft_topk"),
        },
        "metrics": {"hard": metrics_hard, "soft": metrics_soft},
    }

    if args.blocksafe and _BLOCKSAFE_STATS is not None:
        try:
            model_meta["blocksafe"]["stats"] = asdict(_BLOCKSAFE_STATS)
        except Exception:
            pass

    model["meta"] = model_meta

    print(f"\nSaving model to: {args.out}")
    save_kahm_regressor(model, args.out)

    t_end = time.perf_counter()
    print(f"Done. Total time: {t_end - t_start:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
