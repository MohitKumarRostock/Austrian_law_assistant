#!/usr/bin/env python3
"""
train_kahm_embedding_regressor_by_law.py

Train *law-specific* KAHM regressors mapping IDF–SVD embeddings (inputs) to
Mixedbread embeddings (targets) for Austrian law sentences, then evaluate a
distance-gated multi-model combination on the full (held-out) test split.

Inputs
------
- --idf_svd_npz: NPZ bundle with keys: sentence_id (int64), embeddings (N,D_in)
- --semantic_npz: NPZ bundle with keys: sentence_id (int64), embeddings (N,D_out)
- --sentences_parquet: parquet containing sentence metadata with at least:
    * sentence_id column (default: sentence_id)
    * law label column (auto-detected or via --law-col)

Combination
-----------
This script does not serialize a single "combined" model. It saves one model per
law to --out_dir and evaluates a combined predictor via min-distance gating
(using combine_kahm_regressors_generalized.py).

Example
-------
python train_kahm_embedding_regressor_by_law.py \
  --sentences_parquet ris_sentences.parquet \
  --idf_svd_npz embedding_index_idf_svd.npz \
  --semantic_npz embedding_index.npz \
  --idf_svd_model idf_svd_model.joblib \
  --out_dir kahm_embedding_regressors_by_law/ \
  --train_fraction 0.9
"""

from __future__ import annotations

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import re
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

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

from combine_kahm_regressors_generalized import combine_kahm_regressors_distance_gated_multi


# ----------------------------- BlockSafe (optional) -----------------------------
try:
    from otfl_blocksafe import enable_otfl_blocksafe, _BLOCKSAFE_STATS  # type: ignore
except Exception:
    enable_otfl_blocksafe = None
    _BLOCKSAFE_STATS = None


# ----------------------------- Defaults (match train_kahm_embedding_regressor.py) -----------------------------
DEFAULT_SENTENCES_PARQUET = "ris_sentences.parquet"
DEFAULT_SENTENCE_ID_COL = "sentence_id"
DEFAULT_LAW_COL = ""  # empty => auto-detect

DEFAULT_IDF_SVD_NPZ = "embedding_index_idf_svd.npz"
DEFAULT_SEMANTIC_NPZ = "embedding_index.npz"
DEFAULT_IDF_SVD_MODEL = "idf_svd_model.joblib"
DEFAULT_OUT_DIR = "kahm_embedding_regressors_by_law"

DEFAULT_N_CLUSTERS = 20000
DEFAULT_SUBSPACE_DIM = 20
DEFAULT_NB = 100
DEFAULT_TRAIN_FRACTION = 1.0  # match train_kahm_embedding_regressor.py (set <1.0 to create a test split)
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
DEFAULT_SOFT_ALPHAS = "5,8,10,11,12,13,14,15,16,17,18,19,20"
DEFAULT_SOFT_TOPKS = "2,5,8,10,11,12,13,14,15,16,17,18,19,20,25,50,75,100,125,150,175,200"

DEFAULT_BLOCKSAFE_ENABLED = True
DEFAULT_BLOCKSAFE_BACKEND = "threading"
DEFAULT_BLOCKSAFE_JITTER_STD = 1e-5
DEFAULT_BLOCKSAFE_JITTER_TRIES = 6
DEFAULT_BLOCKSAFE_JITTER_GROWTH = 2.0
DEFAULT_BLOCKSAFE_EPS_FACTOR = 10.0
DEFAULT_BLOCKSAFE_LOG_FIRST = 100
DEFAULT_BLOCKSAFE_L2_NORMALIZED = True

DEFAULT_NO_DEGENERACY_CHECK = False


# ----------------------------- Utilities -----------------------------

def as_float_ndarray(x: Any, *, min_dtype: np.dtype = np.dtype(np.float32)) -> np.ndarray:
    if hasattr(x, "toarray"):
        x = x.toarray()
    arr = np.asarray(x)
    if arr.dtype.kind not in "fc":
        arr = arr.astype(np.float64, copy=False)
    dtype = np.result_type(arr.dtype, min_dtype)
    return arr.astype(dtype, copy=False)


def l2_normalize_rows(mat: np.ndarray, eps: float = 1e-12, *, inplace: bool = False) -> np.ndarray:
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


def load_npz_bundle(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as d:
        if "sentence_id" not in d or "embeddings" not in d:
            raise ValueError(f"NPZ '{path}' must contain keys 'sentence_id' and 'embeddings'. Keys: {list(d.keys())}")
        ids = np.asarray(d["sentence_id"], dtype=np.int64)
        emb = as_float_ndarray(d["embeddings"])
    if ids.ndim != 1:
        raise ValueError(f"NPZ '{path}': sentence_id must be 1D; got {ids.shape}")
    if emb.ndim != 2:
        raise ValueError(f"NPZ '{path}': embeddings must be 2D; got {emb.shape}")
    if emb.shape[0] != ids.shape[0]:
        raise ValueError(f"NPZ '{path}': embeddings rows {emb.shape[0]} != sentence_id rows {ids.shape[0]}")
    order = np.argsort(ids)
    return ids[order], emb[order]


def align_by_sentence_id(
    ids_x: np.ndarray, X: np.ndarray,
    ids_y: np.ndarray, Y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    common = np.intersect1d(ids_x, ids_y, assume_unique=False)
    if common.size == 0:
        raise RuntimeError("No overlapping sentence_id values between the two NPZ bundles.")
    ix = np.searchsorted(ids_x, common)
    iy = np.searchsorted(ids_y, common)
    return common, X[ix], Y[iy]


def infer_law_col(df_cols: Iterable[str]) -> Optional[str]:
    cols = list(df_cols)
    candidates = ["consensus_law", "law", "law_id", "law_type", "norm", "gesetz", "act", "source_law"]
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def sanitize_token(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_.+=-]+", "_", s)
    s = s.strip("._")
    return s or "UNKNOWN"


def train_test_split_indices(N: int, train_fraction: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    N = int(N)
    if N <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    tf = float(train_fraction)
    if tf >= 1.0:
        return np.arange(N, dtype=np.int64), np.array([], dtype=np.int64)

    n_train = int(np.floor(tf * N))
    n_train = max(1, min(N, n_train))
    perm = rng.permutation(N)
    train_idx = perm[:n_train].astype(np.int64, copy=False)
    test_idx = perm[n_train:].astype(np.int64, copy=False)
    return train_idx, test_idx


def split_train_val_indices(train_idx: np.ndarray, val_fraction: float, rng: np.random.Generator, val_max_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    train_idx = np.asarray(train_idx, dtype=np.int64)
    if train_idx.size == 0:
        return train_idx, np.array([], dtype=np.int64)

    vf = float(val_fraction)
    if vf <= 0.0:
        return train_idx, np.array([], dtype=np.int64)

    n_val = int(np.floor(vf * train_idx.size))
    n_val = max(0, min(train_idx.size - 1, n_val))  # keep at least 1 core train
    if val_max_samples is not None:
        n_val = min(n_val, int(val_max_samples))
    if n_val <= 0:
        return train_idx, np.array([], dtype=np.int64)

    perm = rng.permutation(train_idx.size)
    val_sel = perm[:n_val]
    core_sel = perm[n_val:]
    return train_idx[core_sel], train_idx[val_sel]


def compute_embedding_metrics(
    Y_pred: np.ndarray | Tuple[np.ndarray, np.ndarray],
    Y_true: np.ndarray,
) -> Dict[str, float]:
    if isinstance(Y_pred, tuple):
        Y_pred = Y_pred[0]

    Y_pred = as_float_ndarray(Y_pred)
    Y_true = as_float_ndarray(Y_true)

    work_dtype = np.result_type(Y_pred.dtype, Y_true.dtype)
    Y_pred = Y_pred.astype(work_dtype, copy=False)
    Y_true = Y_true.astype(work_dtype, copy=False)

    diff = Y_pred - Y_true
    np.square(diff, out=diff)
    residual_ss = float(np.sum(diff))
    mse = float(np.mean(diff))

    N = int(Y_true.shape[1])
    mean_true = Y_true.mean(axis=1)
    sum_sq_true = float(np.einsum("ij,ij->", Y_true, Y_true))
    total_ss = float(sum_sq_true - N * float(np.sum(mean_true * mean_true)))
    r2_overall = float(1.0 - residual_ss / total_ss) if total_ss > 0 else float("nan")

    dot = np.einsum("ij,ij->j", Y_pred, Y_true)
    pred_norm = np.sqrt(np.einsum("ij,ij->j", Y_pred, Y_pred))
    pred_norm = np.maximum(pred_norm, np.asarray(1e-12, dtype=pred_norm.dtype))
    cos = dot / pred_norm

    return {
        "mse": mse,
        "r2_overall": r2_overall,
        "cos_mean": float(np.mean(cos)),
        "cos_std": float(np.std(cos)),
        "cos_p05": float(np.quantile(cos, 0.05)),
        "cos_p50": float(np.quantile(cos, 0.50)),
        "cos_p95": float(np.quantile(cos, 0.95)),
    }


def parse_float_list(s: str) -> List[float]:
    out: List[float] = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if tok:
            out.append(float(tok))
    return out


def parse_topk_list(s: str) -> List[int | None]:
    out: List[int | None] = []
    for tok in str(s).split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        if tok in {"none", "null"}:
            out.append(None)
        else:
            out.append(int(tok))
    return out


def maybe_enable_blocksafe(args: argparse.Namespace) -> None:
    if not args.blocksafe:
        return
    if enable_otfl_blocksafe is None:
        print("WARNING: blocksafe requested but otfl_blocksafe not available; continuing without it.")
        return
    enable_otfl_blocksafe(
        backend=str(args.blocksafe_backend),
        jitter_std=float(args.blocksafe_jitter_std),
        jitter_tries=int(args.blocksafe_jitter_tries),
        jitter_growth=float(args.blocksafe_jitter_growth),
        eps_factor=float(args.blocksafe_eps_factor),
        log_first=int(args.blocksafe_log_first),
        l2_normalized=bool(args.blocksafe_l2_normalized),
    )


# ----------------------------- CLI -----------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train per-law KAHM embedding regressors and evaluate distance-gated combination.")
    p.add_argument("--sentences_parquet", type=str, default=DEFAULT_SENTENCES_PARQUET, help="Parquet containing sentence metadata (incl. law label).")
    p.add_argument("--sentence_id_col", type=str, default=DEFAULT_SENTENCE_ID_COL, help="Sentence id column name in parquet.")
    p.add_argument("--law_col", type=str, default=DEFAULT_LAW_COL, help="Law label column name in parquet. Empty => auto-detect.")
    p.add_argument("--idf_svd_npz", type=str, default=DEFAULT_IDF_SVD_NPZ, help="Input IDF–SVD embeddings NPZ (X).")
    p.add_argument("--semantic_npz", type=str, default=DEFAULT_SEMANTIC_NPZ, help="Target Mixedbread embeddings NPZ (Y).")
    p.add_argument("--idf_svd_model", type=str, default=DEFAULT_IDF_SVD_MODEL, help="Path to IDF–SVD model joblib (stored in manifest only).")

    p.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR, help="Directory to write per-law models + manifest.")
    p.add_argument("--train_fraction", type=float, default=DEFAULT_TRAIN_FRACTION, help="Per-law train fraction (rest is test).")
    p.add_argument("--random_state", type=int, default=DEFAULT_RANDOM_STATE, help="Random seed for per-law splits.")

    # KAHM hyperparameters (match baseline defaults)
    p.add_argument("--n_clusters", type=int, default=DEFAULT_N_CLUSTERS)
    p.add_argument("--subspace_dim", type=int, default=DEFAULT_SUBSPACE_DIM)
    p.add_argument("--nb", type=int, default=DEFAULT_NB)
    p.add_argument("--input_scale", type=float, default=DEFAULT_INPUT_SCALE)
    p.add_argument("--kmeans_kind", type=str, default=DEFAULT_KMEANS_KIND, choices=("auto", "full", "minibatch"))
    p.add_argument("--kmeans_batch_size", type=int, default=DEFAULT_KMEANS_BATCH_SIZE)
    p.add_argument("--max_train_per_cluster", type=int, default=None)
    p.add_argument("--model_dtype", type=str, default=DEFAULT_MODEL_DTYPE)
    p.add_argument("--cluster_center_normalization", type=str, default=DEFAULT_CLUSTER_CENTER_NORMALIZATION)

    # Autoencoder cache
    p.add_argument("--ae_cache_root", type=str, default=DEFAULT_AE_CACHE_ROOT)
    p.add_argument("--overwrite_ae_dir", action="store_true", default=DEFAULT_OVERWRITE_AE_DIR)

    # Soft tuning / NLMS
    p.add_argument("--eval_soft", action=argparse.BooleanOptionalAction, default=DEFAULT_EVAL_SOFT)
    p.add_argument("--tune_soft", action=argparse.BooleanOptionalAction, default=DEFAULT_TUNE_SOFT)
    p.add_argument("--tune_nlms", action=argparse.BooleanOptionalAction, default=DEFAULT_TUNE_NLMS)
    p.add_argument("--val_fraction", type=float, default=DEFAULT_VAL_FRACTION)
    p.add_argument("--val_max_samples", type=int, default=DEFAULT_VAL_MAX_SAMPLES)
    p.add_argument("--soft_alphas", type=str, default=DEFAULT_SOFT_ALPHAS)
    p.add_argument("--soft_topks", type=str, default=DEFAULT_SOFT_TOPKS)

    # Evaluation speed
    p.add_argument("--preload_eval_classifier", action="store_true", default=False, help="Preload per-cluster AEs into RAM for evaluation (RAM heavy).")
    p.add_argument("--eval_batch_size", type=int, default=2048)

    # BlockSafe
    p.add_argument("--blocksafe", action=argparse.BooleanOptionalAction, default=DEFAULT_BLOCKSAFE_ENABLED)
    p.add_argument("--blocksafe_backend", type=str, default=DEFAULT_BLOCKSAFE_BACKEND)
    p.add_argument("--blocksafe_jitter_std", type=float, default=DEFAULT_BLOCKSAFE_JITTER_STD)
    p.add_argument("--blocksafe_jitter_tries", type=int, default=DEFAULT_BLOCKSAFE_JITTER_TRIES)
    p.add_argument("--blocksafe_jitter_growth", type=float, default=DEFAULT_BLOCKSAFE_JITTER_GROWTH)
    p.add_argument("--blocksafe_eps_factor", type=float, default=DEFAULT_BLOCKSAFE_EPS_FACTOR)
    p.add_argument("--blocksafe_log_first", type=int, default=DEFAULT_BLOCKSAFE_LOG_FIRST)
    p.add_argument("--blocksafe_l2_normalized", action=argparse.BooleanOptionalAction, default=DEFAULT_BLOCKSAFE_L2_NORMALIZED)

    p.add_argument("--no_degeneracy_check", action="store_true", default=DEFAULT_NO_DEGENERACY_CHECK)

    return p.parse_args(list(argv) if argv is not None else None)


# ----------------------------- Main -----------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest_by_law.json"

    maybe_enable_blocksafe(args)

    # Load embeddings
    print("Loading embedding bundles ...")
    ids_x, X_all = load_npz_bundle(str(args.idf_svd_npz))
    ids_y, Y_all = load_npz_bundle(str(args.semantic_npz))
    common_ids, X_aligned, Y_aligned = align_by_sentence_id(ids_x, X_all, ids_y, Y_all)
    # Reduce peak memory: keep only aligned float32 embeddings
    X_aligned = X_aligned.astype(np.float32, copy=False)
    Y_aligned = Y_aligned.astype(np.float32, copy=False)
    try:
        del X_all, Y_all, ids_x, ids_y
    except Exception:
        pass

    N, D_in = X_aligned.shape
    _, D_out = Y_aligned.shape
    print(f"Aligned embeddings: N={N} | D_in={D_in} | D_out={D_out}")

    # Load parquet metadata
    print(f"Loading sentence metadata: {args.sentences_parquet}")
    df_meta = pd.read_parquet(args.sentences_parquet)

    sid_col = str(args.sentence_id_col)
    if sid_col not in df_meta.columns:
        raise ValueError(f"Parquet is missing sentence id col '{sid_col}'. Columns: {list(df_meta.columns)}")

    law_col = str(args.law_col).strip()
    if not law_col:
        law_col = infer_law_col(df_meta.columns) or ""
        if not law_col:
            raise ValueError(
                "Could not auto-detect law column in parquet. Provide --law_col explicitly.\n"
                f"Columns: {list(df_meta.columns)}"
            )
        print(f"Auto-detected law column: {law_col}")

    if law_col not in df_meta.columns:
        raise ValueError(f"Parquet is missing law col '{law_col}'. Columns: {list(df_meta.columns)}")

    df_meta = df_meta[[sid_col, law_col]].copy()
    df_meta[sid_col] = pd.to_numeric(df_meta[sid_col], errors="coerce").astype("Int64")
    df_meta = df_meta.dropna(subset=[sid_col])
    df_meta[sid_col] = df_meta[sid_col].astype(np.int64)
    df_meta = df_meta.drop_duplicates(subset=[sid_col], keep="first")

    # Join law labels
    df_ids = pd.DataFrame({sid_col: common_ids})
    df_join = df_ids.merge(df_meta, on=sid_col, how="left")
    mask = df_join[law_col].notna().to_numpy()

    if not np.any(mask):
        raise RuntimeError("After joining parquet metadata, no aligned samples have a law label.")
    if np.any(~mask):
        print(f"WARNING: dropping {int(np.sum(~mask))} aligned samples with no law label in parquet.")

    common_ids = common_ids[mask]
    X_aligned = X_aligned[mask]
    Y_aligned = Y_aligned[mask]
    laws = df_join.loc[mask, law_col].astype(str).to_numpy()

    unique_laws = sorted(set(laws))
    print(f"Usable samples after law join: N={X_aligned.shape[0]} | unique laws={len(unique_laws)}")

    rng_global = np.random.default_rng(int(args.random_state))

    # Group indices by law
    law_to_indices: Dict[str, np.ndarray] = {}
    for law in unique_laws:
        law_to_indices[law] = np.where(laws == law)[0].astype(np.int64, copy=False)

    # Storage
    models_by_law: Dict[str, dict] = {}
    split_info: Dict[str, Dict[str, Any]] = {}
    test_chunks: List[Tuple[str, np.ndarray, np.ndarray]] = []  # (law, X_row, Y_col)

    # Train per law
    for law in unique_laws:
        idx_global = law_to_indices[law]
        N_law = int(idx_global.size)
        if N_law <= 0:
            continue

        # deterministic per-law RNG
        rng = np.random.default_rng(rng_global.integers(0, 2**32 - 1, dtype=np.uint32).item())

        train_idx_local, test_idx_local = train_test_split_indices(N_law, float(args.train_fraction), rng)
        train_idx_global = idx_global[train_idx_local]
        test_idx_global = idx_global[test_idx_local]

        train_core_global = train_idx_global
        val_global = np.array([], dtype=np.int64)
        if bool(args.tune_soft) or bool(args.tune_nlms):
            train_core_global, val_global = split_train_val_indices(
                train_idx_global,
                float(args.val_fraction),
                rng,
                int(args.val_max_samples),
            )

        n_train_core = int(train_core_global.size)
        if n_train_core <= 0:
            print(f"[{law}] skipping: no training samples after val split.")
            continue

        # Prepare training rows
        X_train_rows = X_aligned[train_core_global]
        Y_train_rows = Y_aligned[train_core_global]

        n_clusters_eff = int(min(int(args.n_clusters), n_train_core))
        n_clusters_eff = max(1, n_clusters_eff)

        if n_clusters_eff == 1:
            # KAHM's singleton augmentation strategy requires at least 2 total training points.
            # For rare laws with only 1 training sample after the split, synthesize a nearby input
            # sample while keeping the output identical.
            x0 = X_train_rows[0]
            y0 = Y_train_rows[0]
            noise = rng.normal(size=x0.shape).astype(x0.dtype, copy=False)
            noise_norm = float(np.linalg.norm(noise) + 1e-12)
            x0_norm = float(np.linalg.norm(x0) + 1e-12)
            jitter_mag = 1e-3 * x0_norm
            x1 = x0 + (jitter_mag * noise / noise_norm)
            X_train_rows = np.vstack([X_train_rows, x1[None, :]])
            Y_train_rows = np.vstack([Y_train_rows, y0[None, :]])
            n_train_core = 2



        # Train matrices as (D,N) float32
        X_train = l2_normalize_rows(X_train_rows, inplace=False).astype(np.float32, copy=False).T
        Y_train = l2_normalize_rows(Y_train_rows, inplace=False).astype(np.float32, copy=False).T

        X_val: Optional[np.ndarray] = None
        Y_val: Optional[np.ndarray] = None
        if val_global.size > 0:
            X_val = l2_normalize_rows(X_aligned[val_global], inplace=False).astype(np.float32, copy=False).T
            Y_val = l2_normalize_rows(Y_aligned[val_global], inplace=False).astype(np.float32, copy=False).T

        if X_val is None:
            X_val = X_train
            Y_val = Y_train
            print(f"[{law}] WARNING: no validation set; using training data for tuning (not recommended).")

        print(f"\n=== Training law={law} (N={N_law}) ===")
        print(f"train_core={n_train_core} | val={val_global.size} | test={test_idx_global.size} | n_clusters={n_clusters_eff}")

        run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        model_id = f"emb_law_{sanitize_token(law)}_{run_stamp}"

        t0 = time.perf_counter()
        model = train_kahm_regressor(
            X=X_train,
            Y=Y_train,
            n_clusters=n_clusters_eff,
            subspace_dim=int(args.subspace_dim),
            Nb=int(args.nb),
            random_state=int(args.random_state),
            verbose=False,
            input_scale=float(args.input_scale),
            kmeans_kind=str(args.kmeans_kind),
            kmeans_batch_size=int(args.kmeans_batch_size),
            max_train_per_cluster=(None if args.max_train_per_cluster is None else int(args.max_train_per_cluster)),
            model_dtype=str(args.model_dtype),
            cluster_center_normalization=str(args.cluster_center_normalization),
            save_ae_to_disk=False,
            ae_cache_root=str(args.ae_cache_root),
            ae_dir=None,
            overwrite_ae_dir=bool(args.overwrite_ae_dir),
            model_id=model_id,
            singleton_strategy="augment",
            singleton_aux_mix=0.1,
        )
        t1 = time.perf_counter()
        print(f"[{law}] training done in {t1 - t0:.1f}s")

        # Optional tuning
        tuning_result = None
        if bool(args.tune_soft):
            if X_val is None or Y_val is None or X_val.shape[1] == 0:
                print(f"[{law}] tune_soft requested but validation set empty; skipping.")
            else:
                alphas = tuple(parse_float_list(args.soft_alphas))
                topks = tuple(parse_topk_list(args.soft_topks))
                # filter out None values from parsed topk list before numeric comparison
                topk_candidates_eff = [k for k in topks if (k is not None and k <= n_clusters_eff)]
                if not topk_candidates_eff:
                    topk_candidates_eff = [1]
                    print(f"[{law}] WARNING: no valid topk candidates <= n_clusters_eff={n_clusters_eff}; using topk=1 only.")

                print(f"[{law}] tuning soft params on val ...")
                tuning_result = tune_soft_params(
                    model,
                    X_val,
                    Y_val,
                    alphas=alphas,
                    topks=topk_candidates_eff,
                    n_jobs=1,
                    verbose=True,
                )

        if bool(args.tune_nlms):
            if X_val is None or Y_val is None or X_val.shape[1] == 0:
                print(f"[{law}] tune_nlms requested but validation set empty; skipping.")
            else:
                print(f"[{law}] NLMS refinement on val ...")
                _ = tune_cluster_centers_nlms(
                    model, 
                    np.hstack([X_val, X_train]),
                    np.hstack([Y_val, Y_train]),
                    mu=0.1,
                    epsilon=1,
                    epochs=20,
                    batch_size=1024,
                    shuffle=True,
                    random_state=0,
                    anchor_lambda=0.0,   # set e.g. 1e-3 to pull gently toward initial KMeans centers
                    n_jobs=1,
                    preload_classifier=True,
                    verbose=False,
                    alpha=tuning_result.best_alpha if tuning_result is not None else None,
                    topk=tuning_result.best_topk if tuning_result is not None else None
                    )
        # Save model
        law_tok = sanitize_token(law)
        out_model_path = out_dir / f"kahm_embedding_regressor__law={law_tok}.joblib"
        save_kahm_regressor(model, str(out_model_path))
        models_by_law[law] = model

        if bool(args.preload_eval_classifier) and preload_kahm_classifier is not None:
            try:
                preload_kahm_classifier(model)
            except Exception as e:
                print(f"[{law}] WARNING: preload_eval_classifier failed: {e}")

        # Per-law eval + collect combined test chunks
        law_metrics_hard = None
        law_metrics_soft = None

        if test_idx_global.size > 0:
            X_test_row = l2_normalize_rows(X_aligned[test_idx_global], inplace=False).astype(np.float32, copy=False)
            Y_test_col = l2_normalize_rows(Y_aligned[test_idx_global], inplace=False).astype(np.float32, copy=False).T
        else: 
            print(f"[{law}] no test samples (train_fraction may be 1.0); using training data for testing (not recommended).")
            X_test_row = X_train.T
            Y_test_col = Y_train

        Y_hat_hard = kahm_regress(model, X_test_row.T, mode="hard")
        law_metrics_hard = compute_embedding_metrics(Y_hat_hard, Y_test_col)

        if bool(args.eval_soft):
            Y_hat_soft = kahm_regress(model, X_test_row.T, mode="soft")  # uses tuned params if present
            law_metrics_soft = compute_embedding_metrics(Y_hat_soft, Y_test_col)

        test_chunks.append((law, X_test_row, Y_test_col))
    
            

        split_info[law] = {
            "law_token": law_tok,
            "n_total": N_law,
            "n_train_core": int(train_core_global.size),
            "n_val": int(val_global.size),
            "n_test": int(test_idx_global.size),
            "n_clusters_eff": int(n_clusters_eff),
            "model_path": str(out_model_path),
            "metrics_hard": law_metrics_hard,
            "metrics_soft": law_metrics_soft,
            "model_id": str(model.get("model_id", model_id)),
            "classifier_dir": str(model.get("classifier_dir", "")),
            "soft_alpha": model.get("soft_alpha", None),
            "soft_topk": model.get("soft_topk", None),
        }

    if not models_by_law:
        raise RuntimeError("No per-law models were trained (check law column / data).")

    # Optional blocksafe stats
    if bool(args.blocksafe) and _BLOCKSAFE_STATS is not None:
        try:
            print("[blocksafe] stats:", asdict(_BLOCKSAFE_STATS))
        except Exception:
            pass

    # Combined evaluation
    if not test_chunks:
        print("\nNo per-law test samples collected (train_fraction may be 1.0 everywhere). Skipping combined eval.")
        combined = {"n_test_total": 0, "metrics_hard": None, "metrics_soft": None, "chosen_counts": None}
    else:
        X_test_all_row = np.concatenate([c[1] for c in test_chunks], axis=0).astype(np.float32, copy=False)
        Y_test_all_col = np.concatenate([c[2] for c in test_chunks], axis=1).astype(np.float32, copy=False)

        print(f"\n=== Combined evaluation on union test split: N_test={X_test_all_row.shape[0]} ===")

        # Combine models via min-distance gating
        # Use alpha/topk=None so each model's tuned soft params (if any) are used.
        Y_comb_hard, chosen_hard, best_score_hard, all_scores_hard, names = combine_kahm_regressors_distance_gated_multi(
            X_test_all_row,
            models=models_by_law,
            input_layout="row",
            output_layout="col",
            mode="hard",
            alpha=None,
            topk=None,
            batch_size=int(args.eval_batch_size),
            tie_break="first",
            show_progress=True,
            return_all_scores=False,
        )
        metrics_hard = compute_embedding_metrics(Y_comb_hard, Y_test_all_col)
        print("[combined][hard] ", json.dumps(metrics_hard, ensure_ascii=False))

        metrics_soft = None
        chosen_soft = None
        if bool(args.eval_soft):
            Y_comb_soft, chosen_soft, best_score_soft, all_scores_soft, names2 = combine_kahm_regressors_distance_gated_multi(
                X_test_all_row,
                models=models_by_law,
                input_layout="row",
                output_layout="col",
                mode="soft",
                alpha=None,
                topk=None,
                batch_size=int(args.eval_batch_size),
                tie_break="first",
                show_progress=True,
                return_all_scores=False,
            )
            metrics_soft = compute_embedding_metrics(Y_comb_soft, Y_test_all_col)
            print("[combined][soft] ", json.dumps(metrics_soft, ensure_ascii=False))

        # Count routing decisions
        chosen_counts: Dict[str, int] = {}
        for name in names:
            chosen_counts[name] = 0
        for idx in chosen_hard.tolist():
            chosen_counts[names[int(idx)]] += 1

        print("Chosen model counts (hard):")
        for k, v in sorted(chosen_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {k}: {v}")

        combined = {
            "n_test_total": int(X_test_all_row.shape[0]),
            "metrics_hard": metrics_hard,
            "metrics_soft": metrics_soft,
            "chosen_counts_hard": chosen_counts,
        }

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "idf_svd_npz": str(args.idf_svd_npz),
        "semantic_npz": str(args.semantic_npz),
        "sentences_parquet": str(args.sentences_parquet),
        "sentence_id_col": str(args.sentence_id_col),
        "law_col": str(law_col),
        "idf_svd_model": str(args.idf_svd_model),
        "kahm_params": {
            "n_clusters_requested": int(args.n_clusters),
            "subspace_dim": int(args.subspace_dim),
            "nb": int(args.nb),
            "input_scale": float(args.input_scale),
            "kmeans_kind": str(args.kmeans_kind),
            "kmeans_batch_size": int(args.kmeans_batch_size),
            "max_train_per_cluster": (None if args.max_train_per_cluster is None else int(args.max_train_per_cluster)),
            "model_dtype": str(args.model_dtype),
            "cluster_center_normalization": str(args.cluster_center_normalization),
            "ae_cache_root": str(args.ae_cache_root),
            "overwrite_ae_dir": bool(args.overwrite_ae_dir),
            "tune_soft": bool(args.tune_soft),
            "tune_nlms": bool(args.tune_nlms),
            "eval_soft": bool(args.eval_soft),
            "val_fraction": float(args.val_fraction),
            "val_max_samples": int(args.val_max_samples),
        },
        "split": {
            "train_fraction": float(args.train_fraction),
            "random_state": int(args.random_state),
        },
        "per_law": split_info,
        "combined": combined,
    }

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote manifest: {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
