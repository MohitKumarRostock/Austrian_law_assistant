#!/usr/bin/env python3
"""evaluate_three_embeddings.py

Evaluation script for law retrieval using FAISS cosine (inner product on L2-normalized vectors).

This version supports *two distinct KAHM use-cases* in addition to the standard baselines:

  (A) Mixedbread baseline:        q_MB  -> corpus_MB
  (B) KAHM query translation:     q_IDF -> KAHM(q) -> corpus_MB
      - This is the intended production architecture when MB can be precomputed on the corpus,
        but MB inference is not allowed/desired at query-time.
  (C) Full KAHM approximation:    q_IDF -> KAHM(q) -> corpus_KAHM
      - Intended for candidate generation and semantic routing when MB is not available.
  (D) IDF–SVD baseline:           q_IDF -> corpus_IDF

It loads a query set module (default: query_set.TEST_QUERY_SET), builds FAISS indices for the
corpus embeddings, embeds queries for each method, and prints evaluation metrics.

Expected NPZ format:
  - keys: "sentence_id" (1D int64), "embeddings" (2D float32)
  - optional scalar metadata fields (e.g., from precompute_kahm_corpus_npz.py)

Expected corpus parquet columns:
  - sentence_id (unique), law_type, page, sentence
"""

from __future__ import annotations

# --- Safety guards for native libraries (avoid OpenMP/MKL/thread runtime clashes) ---
# These segfaults are most commonly triggered on macOS when importing FAISS (OpenMP) before PyTorch.
import os as _os
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Import PyTorch early (before FAISS) and cap its threads.
try:
    import torch as _torch  # type: ignore
    _torch.set_num_threads(1)
    if hasattr(_torch, "set_num_interop_threads"):
        _torch.set_num_interop_threads(1)
except Exception:
    _torch = None  # type: ignore


import argparse
import importlib
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _cap_faiss_threads(n: int = 1) -> None:
    """Best-effort cap for FAISS OpenMP threads."""
    try:
        import faiss  # type: ignore
        if int(n) > 0:
            faiss.omp_set_num_threads(int(n))
    except Exception:
        pass



# ----------------------------- Defaults -----------------------------
DEFAULT_CORPUS_PARQUET = "ris_sentences.parquet"
DEFAULT_SEMANTIC_NPZ = "embedding_index.npz"
DEFAULT_IDF_SVD_NPZ = "embedding_index_idf_svd.npz"
DEFAULT_KAHM_CORPUS_NPZ = "embedding_index_kahm_mixedbread_approx.npz"
DEFAULT_IDF_SVD_MODEL = "idf_svd_model.joblib"
DEFAULT_KAHM_MODEL = "kahm_regressor_idf_to_mixedbread.joblib"

DEFAULT_MIXEDBREAD_FALLBACK_MODEL = "mixedbread-ai/deepset-mxbai-embed-de-large-v1"
DEFAULT_QUERY_PREFIX = "query: "
DEFAULT_MB_QUERY_NPZ = ""

DEFAULT_EVAL_K = 10
DEFAULT_K_GRID = [1, 3, 5, 10, 20]
DEFAULT_PRIMARY_K_FOR_HYPOTHESIS = 10
DEFAULT_PREDOMINANCE_FRACTION = 0.5
DEFAULT_BOOTSTRAP_SAMPLES = 5000
DEFAULT_BOOTSTRAP_SEED = 0
DEFAULT_BOOTSTRAP_ENABLED = True

DEFAULT_KAHM_MODE = "soft"
DEFAULT_REGRESS_BATCH = 4096
DEFAULT_QUERY_BATCH = 64

DEFAULT_KAHM_N_JOBS = 1
DEFAULT_KAHM_BACKEND = "threading"

# How to handle a precomputed KAHM corpus NPZ whose metadata does not match
# the current run:
#  - auto: recompute on mismatch
#  - force: always use precomputed if present
#  - strict: error on mismatch
DEFAULT_KAHM_PRECOMPUTED_POLICY = "auto"

DEFAULT_FAISS_THREADS = 0

# ----------------------------- Hypothesis thresholds (defaults) -----------------------------
# These defaults are conservative, aligned with typical retrieval non-inferiority framing.
# All margins are *absolute* deltas for metrics in [0,1], except lift which is on the raw scale.

# H1 (production retrieval): KAHM best-practice retrieval (routing + law-aware post-processing)
# should be non-inferior to the Mixedbread baseline.
#
# H1b (candidate generation): KAHM query translation should be non-inferior to Mixedbread on
# recall-oriented metrics (hit@k, MRR@k), without necessarily matching Mixedbread top-1.
DEFAULT_H1_MARGIN_HIT = 0.06
DEFAULT_H1_MARGIN_MRR = 0.085
DEFAULT_H1_MARGIN_TOP1 = 0.07
DEFAULT_H1_MARGIN_CONS = 0.05
DEFAULT_H1_MARGIN_LIFT = 10.0
DEFAULT_H1_MARGIN_MAJ = 0.10

# H2 (routing/candidate generation): Full KAHM corpus should be non-inferior to MB on routing
# signals, and materially better than IDF–SVD.
DEFAULT_H2_MARGIN_HIT = 0.08
DEFAULT_H2_MARGIN_CONS = 0.08
DEFAULT_H2_MARGIN_MAJ = 0.10
DEFAULT_H2_MARGIN_LIFT = 10.0

# Router confidence grid (for selective routing / abstention analysis)
DEFAULT_ROUTER_CONS_THRESHOLDS = "0.4,0.45,0.5,0.55,0.6,0.65,0.7"
DEFAULT_ROUTER_LIFT_THRESHOLDS = "5,10,20,30,40"
DEFAULT_ROUTER_MIN_COVERAGE = 0.50

# ----------------------------- KAHM best-practice evaluation defaults -----------------------------
# "Best possible" use of KAHM in this repo is typically:
#   1) Use the full-KAHM index (cheap) to estimate a law posterior + confidence signals.
#   2) Use KAHM(query→MB corpus) for high-recall candidate generation.
#   3) Apply law-aware filtering and/or reranking of those candidates based on the router posterior
#      to recover top-1 quality without running Mixedbread at query-time.
DEFAULT_KAHM_BEST_ENABLED = True
DEFAULT_KAHM_BEST_ROUTER_K = 50
DEFAULT_KAHM_BEST_CAND_K = 200
DEFAULT_KAHM_BEST_MODE = "hybrid"  # filter | rerank | hybrid
DEFAULT_KAHM_BEST_TOP_LAWS_GRID = "1,2,3"
DEFAULT_KAHM_BEST_LAMBDA_GRID = "0,0.05,0.1,0.2,0.4"
# If filtering is enabled (mode=filter|hybrid), limit how many of the final
# top-k slots can be occupied by router-selected laws. This prevents overly
# aggressive filtering from harming hit@k.
DEFAULT_KAHM_BEST_LAW_SLOTS_GRID = "1,2,3,5,10"
DEFAULT_KAHM_BEST_OPTIMIZE = "top1"  # top1 | mrr | hit
DEFAULT_KAHM_BEST_TUNE = True
DEFAULT_KAHM_BEST_EPS = 1e-12


# Script version banner (helps confirm you are running the updated file)
SCRIPT_VERSION = "2025-12-30-kahm-best-v5.1"



# ----------------------------- Numeric helpers -----------------------------
def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps).astype(np.float32)
    return x / n


def assert_finite(name: str, x: np.ndarray) -> None:
    x = np.asarray(x)
    if not np.isfinite(x).all():
        if x.ndim == 2:
            bad_rows = (~np.isfinite(x)).any(axis=1).nonzero()[0]
            example = bad_rows[:10].tolist()
            raise ValueError(
                f"{name}: found non-finite values (NaN/Inf) in {bad_rows.size} rows. "
                f"Example bad row indices: {example}"
            )
        raise ValueError(f"{name}: found non-finite values (NaN/Inf).")


def summarize_vectors(name: str, x: np.ndarray) -> None:
    x = np.asarray(x)
    if x.ndim != 2:
        print(f"{name}: shape={getattr(x, 'shape', None)}", flush=True)
        return
    n = np.linalg.norm(x.astype(np.float32, copy=False), axis=1)
    zero_rows = int((n == 0).sum())
    print(
        f"{name}: shape={x.shape} | norms: mean={float(n.mean()):.4f} "
        f"p50={float(np.median(n)):.4f} min={float(n.min()):.4f} max={float(n.max()):.4f} "
        f"| zero-rows={zero_rows} ({(zero_rows / max(1, x.shape[0])):.2%})",
        flush=True,
    )


# ----------------------------- IO helpers -----------------------------
def load_npz_bundle(path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"NPZ not found: {path}")

    d = np.load(path, allow_pickle=False)
    if "sentence_id" not in d or "embeddings" not in d:
        raise ValueError(f"NPZ '{path}' must contain keys 'sentence_id' and 'embeddings'. Keys: {list(d.keys())}")

    ids = np.asarray(d["sentence_id"], dtype=np.int64)
    emb = np.asarray(d["embeddings"], dtype=np.float32)

    if ids.ndim != 1:
        raise ValueError(f"NPZ '{path}': sentence_id must be 1D; got {ids.shape}.")
    if emb.ndim != 2:
        raise ValueError(f"NPZ '{path}': embeddings must be 2D; got {emb.shape}.")
    if emb.shape[0] != ids.shape[0]:
        raise ValueError(
            f"NPZ '{path}': embeddings rows must match sentence_id length; got {emb.shape[0]} vs {ids.shape[0]}."
        )
    if np.unique(ids).size != ids.size:
        raise ValueError(f"NPZ '{path}': duplicate sentence_id values detected (must be unique).")

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


def load_query_embeddings_npz(path: str, query_set: List[Dict[str, Any]], expected_dim: int) -> np.ndarray:
    """Load precomputed query embeddings from an NPZ.

    Supported formats:
      - keys: {"embeddings", "query_id"} (preferred). The loader aligns by query_id.
      - keys: {"q"} or {"emb"} or {"X"}: embeddings only. If length matches query_set, uses order.

    Returns: float32 array (n_queries, expected_dim), L2-normalized.
    """
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"MB query NPZ not found: {path}")
    with np.load(path, allow_pickle=True) as z:
        keys = set(z.files)
        emb_key = None
        for k in ("embeddings", "q_embeddings", "q", "emb", "X"):
            if k in keys:
                emb_key = k
                break
        if emb_key is None:
            raise ValueError(f"MB query NPZ {path} has no recognized embedding key. Found keys: {sorted(keys)}")
        emb = np.asarray(z[emb_key], dtype=np.float32)

        if emb.ndim != 2:
            raise ValueError(f"MB query NPZ {path}: embeddings must be 2D, got shape {emb.shape}")
        if int(emb.shape[1]) != int(expected_dim):
            raise ValueError(
                f"MB query NPZ {path}: dim mismatch. expected_dim={expected_dim} but got {emb.shape[1]}"
            )

        n_q = len(query_set)
        if "query_id" in keys:
            qids_npz = [str(x) for x in np.asarray(z["query_id"]).tolist()]
            pos = {qid: i for i, qid in enumerate(qids_npz)}
            out = np.zeros((n_q, expected_dim), dtype=np.float32)
            missing = 0
            for i, q in enumerate(query_set):
                qid = str(q.get("query_id", f"q{i}"))
                j = pos.get(qid, None)
                if j is None:
                    missing += 1
                    continue
                out[i] = emb[int(j)]
            if missing > 0:
                raise ValueError(
                    f"MB query NPZ {path}: missing {missing}/{n_q} query_ids from the current query_set."
                )
            return l2_normalize_rows(out)

        # Fallback: order-based
        if int(emb.shape[0]) != int(n_q):
            raise ValueError(
                f"MB query NPZ {path}: row mismatch. expected {n_q} queries but got {emb.shape[0]}"
            )
        return l2_normalize_rows(emb)


def subset_by_ids(ids: np.ndarray, emb: np.ndarray, keep_ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.int64)
    keep_ids = np.asarray(keep_ids, dtype=np.int64)

    pos = {int(sid): i for i, sid in enumerate(ids.tolist())}
    idx = np.array([pos[int(sid)] for sid in keep_ids.tolist()], dtype=np.int64)
    return emb[idx]


def align_by_sentence_id(
    ids_a: np.ndarray, emb_a: np.ndarray, ids_b: np.ndarray, emb_b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(ids_a, dtype=np.int64)
    b = np.asarray(ids_b, dtype=np.int64)

    common = np.intersect1d(a, b, assume_unique=False)
    common = np.asarray(common, dtype=np.int64)
    # keep the common ids sorted (np.intersect1d returns sorted)
    emb_a_aligned = subset_by_ids(a, emb_a, common)
    emb_b_aligned = subset_by_ids(b, emb_b, common)
    return common, emb_a_aligned, emb_b_aligned


def load_corpus_info(parquet_path: str, sentence_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Corpus parquet not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    required = {"sentence_id", "law_type", "page", "sentence"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Corpus parquet missing columns {sorted(missing)}. Found: {list(df.columns)}")

    ids = df["sentence_id"].astype(np.int64).to_numpy()
    if np.unique(ids).size != ids.size:
        raise ValueError("Corpus parquet: duplicate sentence_id values detected (must be unique).")

    df = df.set_index("sentence_id", drop=False)
    sid = np.asarray(sentence_ids, dtype=np.int64)

    missing_ids = np.setdiff1d(sid, df.index.to_numpy(dtype=np.int64), assume_unique=False)
    if missing_ids.size:
        raise ValueError(
            f"Corpus parquet missing {missing_ids.size} ids referenced by embeddings. Example: {missing_ids[:10].tolist()}"
        )

    sub = df.loc[sid]
    law = sub["law_type"].astype(str).to_numpy()
    page = sub["page"].astype(int).to_numpy()
    sent = sub["sentence"].astype(str).to_numpy()
    return law, page, sent


# ----------------------------- Query set loading -----------------------------
def load_query_set(module_name: str, attr_name: str) -> List[Dict[str, Any]]:
    mod = importlib.import_module(module_name)
    qs = getattr(mod, attr_name, None)
    if qs is None:
        raise AttributeError(f"Query set attribute not found: {module_name}.{attr_name}")
    out = list(qs)
    if not out:
        raise ValueError(f"Loaded empty query set from {module_name}.{attr_name}")
    return out


# ----------------------------- FAISS helpers -----------------------------
def build_faiss_index(emb: np.ndarray, *, index_type: str, hnsw_m: int, ef_search: int):
    import faiss  # type: ignore

    X = np.ascontiguousarray(emb.astype(np.float32, copy=False))
    d = int(X.shape[1])

    if index_type == "flat":
        index: Any = faiss.IndexFlatIP(d)
        index.add(X)
        return index

    if index_type == "hnsw":
        index: Any = faiss.IndexHNSWFlat(d, int(hnsw_m), faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efSearch = int(ef_search)
        index.add(X)
        return index

    raise ValueError(f"Unknown FAISS index_type: {index_type}")


def faiss_search(index, q_emb: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    Q = np.ascontiguousarray(q_emb.astype(np.float32, copy=False))
    scores, idx = index.search(Q, int(k))
    return scores, idx


# ----------------------------- Embedding models -----------------------------
def load_idf_svd_model(path: str):
    import joblib

    if not os.path.exists(path):
        raise FileNotFoundError(f"IDF–SVD model not found: {path}")
    return joblib.load(path)


def embed_queries_idf_svd(pipe, texts: List[str]) -> np.ndarray:
    X = pipe.transform(texts)
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"IDF–SVD transform output must be 2D; got {X.shape}")
    return l2_normalize_rows(X)


def build_mixedbread_embedder(model_name: str, device: str, dim: int):
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(model_name, device=device, truncate_dim=int(dim))

    def _embed(texts: List[str], *, batch_size: int) -> np.ndarray:
        q_texts = [DEFAULT_QUERY_PREFIX + t for t in texts]
        Y = m.encode(
            q_texts,
            batch_size=int(batch_size),
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype(np.float32)
        if Y.ndim != 2:
            raise ValueError(f"Mixedbread encode output must be 2D; got {Y.shape}")
        if Y.shape[1] != int(dim):
            if Y.shape[1] > int(dim):
                Y = Y[:, : int(dim)]
            else:
                raise ValueError(f"Mixedbread embedding dim mismatch: got {Y.shape[1]}, expected {dim}")
        return l2_normalize_rows(Y)

    return _embed


def load_kahm_model(path: str) -> dict:
    from kahm_regression import load_kahm_regressor

    if not os.path.exists(path):
        raise FileNotFoundError(f"KAHM model not found: {path}")
    return load_kahm_regressor(path)


def kahm_regress_batched(
    model: dict,
    X: np.ndarray,
    *,
    mode: str,
    batch_size: int,
    n_jobs: int,
    backend: str,
) -> np.ndarray:
    """
    X: (N,D_in) row-major, assumed already L2-normalized for cosine semantics.
    Returns (N,D_out) L2-normalized.

    Uses joblib parallel_backend(backend) to avoid loky forking by default.
    """
    from kahm_regression import kahm_regress
    from joblib import parallel_backend

    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"KAHM regression input must be 2D; got {X.shape}")

    n, _ = X.shape
    # Determine output dim by running a tiny batch
    with parallel_backend(str(backend)):
        Yt0 = kahm_regress(model, X[:1].T, n_jobs=int(max(1, n_jobs)), mode=str(mode))
    d_out = int(np.asarray(Yt0).shape[0])

    Y = np.zeros((n, d_out), dtype=np.float32)
    Xt = X.T  # kahm_regress expects (D, N)
    bs = int(max(1, batch_size))
    n_jobs_eff = int(max(1, n_jobs))

    for j0 in range(0, n, bs):
        j1 = min(n, j0 + bs)
        with parallel_backend(str(backend)):
            Yt = kahm_regress(model, Xt[:, j0:j1], n_jobs=n_jobs_eff, mode=str(mode))
        Y[j0:j1, :] = np.asarray(Yt.T, dtype=np.float32)

    return l2_normalize_rows(Y)


# ----------------------------- KAHM NPZ compatibility -----------------------------
def _npz_scalar_to_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return str(x)
    try:
        if isinstance(x, np.ndarray) and x.shape == ():
            return str(x.item())
    except Exception:
        pass
    return None


def kahm_npz_compatible(
    *,
    meta_k: Dict[str, Any],
    meta_idf: Dict[str, Any],
    kahm_model_path: str,
    idf_svd_npz_path: str,
    kahm_mode: str,
) -> Tuple[bool, List[str]]:
    """
    Verify that a precomputed KAHM corpus NPZ appears compatible with the current run.
    If it was built with a different regressor or mode, rankings can collapse to near-random.
    """
    reasons: List[str] = []
    want_kahm = os.path.basename(str(kahm_model_path))
    got_kahm = str(meta_k.get("source_kahm_model", "")).strip()
    if got_kahm and (got_kahm != want_kahm):
        reasons.append(f"source_kahm_model={got_kahm!r} != current kahm_model={want_kahm!r}")

    want_mode = str(kahm_mode)
    got_mode = str(meta_k.get("kahm_mode", "")).strip()
    if got_mode and (got_mode != want_mode):
        reasons.append(f"kahm_mode={got_mode!r} != current kahm_mode={want_mode!r}")

    want_idf_npz = os.path.basename(str(idf_svd_npz_path))
    got_idf_npz = str(meta_k.get("source_idf_svd_npz", "")).strip()
    if got_idf_npz and (got_idf_npz != want_idf_npz):
        reasons.append(f"source_idf_svd_npz={got_idf_npz!r} != current idf_svd_npz={want_idf_npz!r}")

    got_fp = str(meta_k.get("source_idf_svd_fingerprint_sha256", "")).strip()
    want_fp = _npz_scalar_to_str(meta_idf.get("dataset_fingerprint_sha256")) or ""
    if got_fp and want_fp and (got_fp != want_fp):
        reasons.append("IDF–SVD dataset_fingerprint_sha256 mismatch between corpus NPZ and current IDF–SVD NPZ")

    ok = (len(reasons) == 0)
    return ok, reasons


# ----------------------------- Bootstrap helpers -----------------------------
def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    ci: Tuple[float, float] = (2.5, 97.5),
) -> Tuple[float, Tuple[float, float]]:
    # Bootstrap CI for the mean over queries (sampling queries with replacement).
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return float("nan"), (float("nan"), float("nan"))
    point = float(np.mean(v))
    rng = np.random.default_rng(int(seed))
    n = int(v.size)
    B = int(max(1, n_boot))
    boot = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = rng.integers(0, n, size=n, dtype=np.int64)
        boot[b] = float(np.mean(v[idx]))
    lo, hi = np.percentile(boot, [ci[0], ci[1]]).tolist()
    return point, (float(lo), float(hi))


def _fmt_ci(point: float, ci: Tuple[float, float], *, digits: int = 3) -> str:
    if not np.isfinite(point) or not np.isfinite(ci[0]) or not np.isfinite(ci[1]):
        return "nan"
    return f"{point:.{digits}f} [{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"


# ----------------------------- Evaluation -----------------------------

def evaluate_on_query_set(
    query_set: List[Dict[str, Any]],
    *,
    k: int,
    predominance_fraction: float,
    q_true_mb: Optional[np.ndarray],
    q_idf: np.ndarray,
    q_kahm: np.ndarray,
    index_mb_corpus,
    index_kahm_corpus,
    index_idf,
    law_arr: np.ndarray,
    bootstrap: bool = DEFAULT_BOOTSTRAP_ENABLED,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,

    # --- KAHM best-practice routed retrieval (optional) ---
    kahm_best: bool = DEFAULT_KAHM_BEST_ENABLED,
    kahm_best_router_k: int = DEFAULT_KAHM_BEST_ROUTER_K,
    kahm_best_cand_k: int = DEFAULT_KAHM_BEST_CAND_K,
    kahm_best_mode: str = DEFAULT_KAHM_BEST_MODE,
    kahm_best_top_laws_grid: Optional[List[int]] = None,
    kahm_best_law_slots_grid: Optional[List[int]] = None,
    kahm_best_lambda_grid: Optional[List[float]] = None,
    kahm_best_min_coverage: float = DEFAULT_ROUTER_MIN_COVERAGE,
    kahm_best_optimize: str = DEFAULT_KAHM_BEST_OPTIMIZE,
    kahm_best_tune: bool = DEFAULT_KAHM_BEST_TUNE,
    router_cons_thresholds: Optional[List[float]] = None,
    router_lift_thresholds: Optional[List[float]] = None,
) -> Dict[str, Any]:
    k = int(k)
    pred_frac = float(predominance_fraction)

    consensuses = [str(q.get("consensus_law", "")).strip() for q in query_set]
    n_total = len(consensuses)

    # Corpus law priors (sentence-level) for lift normalization
    law_counts = Counter([str(x) for x in law_arr.tolist()])
    total_sent = float(max(1, int(law_arr.size)))
    law_prior = {lw: (float(cnt) / total_sent) for lw, cnt in law_counts.items()}

    # Run FAISS searches once (enables overlap diagnostics)
    kahm_best_enabled = bool(kahm_best) and (index_kahm_corpus is not None)
    cand_k = int(max(k, int(kahm_best_cand_k))) if kahm_best_enabled else int(k)
    router_k = int(max(k, int(kahm_best_router_k))) if kahm_best_enabled else int(k)

    idx_mb: Optional[np.ndarray] = None
    if q_true_mb is not None:
        _, idx_mb = faiss_search(index_mb_corpus, q_true_mb, k)

    # (B) KAHM query translation into MB space, retrieved against MB corpus index.
    # If kahm_best is enabled we also fetch a larger candidate pool for reranking/filtering.
    scores_kahm_qmb_all, idx_kahm_qmb_all = faiss_search(index_mb_corpus, q_kahm, cand_k)
    idx_kahm_qmb = idx_kahm_qmb_all[:, :k]

    # (C) Full KAHM approximation (optional). If kahm_best is enabled we fetch router_k neighbours
    # to estimate a law posterior.
    idx_kahm_full: Optional[np.ndarray] = None
    scores_kahm_full_all: Optional[np.ndarray] = None
    idx_kahm_full_all: Optional[np.ndarray] = None
    if index_kahm_corpus is not None:
        scores_kahm_full_all, idx_kahm_full_all = faiss_search(index_kahm_corpus, q_kahm, router_k)
        idx_kahm_full = idx_kahm_full_all[:, :k]

    # (D) IDF–SVD baseline
    _, idx_idf = faiss_search(index_idf, q_idf, k)

    def _score_from_idx(name: str, idx: np.ndarray, *, ref_idx: np.ndarray | None = None) -> Dict[str, Any]:
        idx = np.asarray(idx, dtype=np.int64)
        n = int(idx.shape[0])

        # Per-query arrays (micro metrics)
        top1_v: List[float] = []
        majority_v: List[float] = []
        hit_v: List[float] = []
        cons_frac_v: List[float] = []
        lift_v: List[float] = []
        mrr_ul_v: List[float] = []

        # Router-oriented per-query signals (available at inference-time)
        #   - maj_law: predicted law (mode of retrieved laws)
        #   - maj_frac: mode fraction in top-k
        #   - maj_lift: prior-normalized maj_frac
        #   - maj_correct: maj_law == consensus law (ground truth, for evaluation only)
        maj_law_v: List[str] = []
        maj_frac_v: List[float] = []
        maj_lift_v: List[float] = []
        maj_correct_v: List[float] = []

        # Optional overlap arrays (vs reference retrieval)
        sent_jacc_v: List[float] = []
        sent_ovl_v: List[float] = []
        law_jacc_v: List[float] = []

        # Macro aggregates by consensus law
        per_law: Dict[str, Dict[str, float]] = {}

        valid = 0
        for i in range(n):
            cons = consensuses[i]
            if not cons:
                continue

            row_idx = idx[i]
            row_idx = [int(j) for j in row_idx.tolist() if int(j) >= 0]
            if not row_idx:
                continue

            retrieved_laws = [str(law_arr[j]) for j in row_idx]
            if not retrieved_laws:
                continue

            valid += 1

            # Majority/predominance accuracy
            c = Counter(retrieved_laws)
            maj_law, maj_count = c.most_common(1)[0]
            is_majority = (maj_law == cons) and (float(maj_count) / float(max(1, len(retrieved_laws))) >= pred_frac)

            # Router confidence signals (do NOT depend on ground truth)
            maj_frac = float(maj_count) / float(max(1, len(retrieved_laws)))
            maj_prior = float(law_prior.get(str(maj_law), 0.0))
            maj_lift = maj_frac / maj_prior if maj_prior > 0.0 else 0.0
            maj_correct = 1.0 if (str(maj_law) == cons) else 0.0

            # Top-1
            is_top1 = (retrieved_laws[0] == cons)

            # Hit@k
            is_hit = (cons in retrieved_laws)

            # Consensus fraction
            cons_frac = float(c.get(cons, 0)) / float(max(1, len(retrieved_laws)))

            # Lift (prior-normalized cons frac)
            prior = float(law_prior.get(cons, 0.0))
            lift = cons_frac / prior if prior > 0.0 else 0.0

            # Unique-law MRR@k
            seen = set()
            unique_laws: List[str] = []
            for lw in retrieved_laws:
                if lw not in seen:
                    unique_laws.append(lw)
                    seen.add(lw)
            try:
                rank = unique_laws.index(cons) + 1
                mrr_ul = 1.0 / float(rank)
            except ValueError:
                mrr_ul = 0.0

            top1_v.append(1.0 if is_top1 else 0.0)
            majority_v.append(1.0 if is_majority else 0.0)
            hit_v.append(1.0 if is_hit else 0.0)
            cons_frac_v.append(float(cons_frac))
            lift_v.append(float(lift))
            mrr_ul_v.append(float(mrr_ul))

            maj_law_v.append(str(maj_law))
            maj_frac_v.append(float(maj_frac))
            maj_lift_v.append(float(maj_lift))
            maj_correct_v.append(float(maj_correct))

            # Overlap diagnostics vs reference retrieval (typically vs true MB)
            if ref_idx is not None:
                ref_row = ref_idx[i]
                ref_row = [int(j) for j in ref_row.tolist() if int(j) >= 0]
                # sentence-id overlap in row-index space (same aligned corpus ordering)
                A = set(row_idx)
                B = set(ref_row)
                inter = len(A.intersection(B))
                union = len(A.union(B))
                sent_jacc = float(inter) / float(union) if union > 0 else 0.0
                sent_ovl = float(inter) / float(max(1, k))

                # law-set overlap
                laws_a = set(str(law_arr[j]) for j in A)
                laws_b = set(str(law_arr[j]) for j in B)
                inter_l = len(laws_a.intersection(laws_b))
                union_l = len(laws_a.union(laws_b))
                law_jacc = float(inter_l) / float(union_l) if union_l > 0 else 0.0

                sent_jacc_v.append(sent_jacc)
                sent_ovl_v.append(sent_ovl)
                law_jacc_v.append(law_jacc)

            # Macro buckets
            b = per_law.get(cons)
            if b is None:
                b = {"count": 0.0, "top1": 0.0, "hit": 0.0, "cons_frac": 0.0, "lift": 0.0, "mrr_ul": 0.0, "majority": 0.0}
                per_law[cons] = b
            b["count"] += 1.0
            b["top1"] += 1.0 if is_top1 else 0.0
            b["hit"] += 1.0 if is_hit else 0.0
            b["cons_frac"] += float(cons_frac)
            b["lift"] += float(lift)
            b["mrr_ul"] += float(mrr_ul)
            b["majority"] += 1.0 if is_majority else 0.0

        # Convert to numpy arrays
        top1_a = np.asarray(top1_v, dtype=np.float64)
        majority_a = np.asarray(majority_v, dtype=np.float64)
        hit_a = np.asarray(hit_v, dtype=np.float64)
        cons_frac_a = np.asarray(cons_frac_v, dtype=np.float64)
        lift_a = np.asarray(lift_v, dtype=np.float64)
        mrr_ul_a = np.asarray(mrr_ul_v, dtype=np.float64)

        maj_frac_a = np.asarray(maj_frac_v, dtype=np.float64)
        maj_lift_a = np.asarray(maj_lift_v, dtype=np.float64)
        maj_correct_a = np.asarray(maj_correct_v, dtype=np.float64)

        # Point estimates + bootstrap CIs (micro)
        if bootstrap:
            top1_pt, top1_ci = _bootstrap_mean_ci(top1_a, n_boot=int(bootstrap_samples), seed=int(bootstrap_seed))
            maj_pt, maj_ci = _bootstrap_mean_ci(majority_a, n_boot=int(bootstrap_samples), seed=int(bootstrap_seed + 1))
            hit_pt, hit_ci = _bootstrap_mean_ci(hit_a, n_boot=int(bootstrap_samples), seed=int(bootstrap_seed + 2))
            mrr_pt, mrr_ci = _bootstrap_mean_ci(mrr_ul_a, n_boot=int(bootstrap_samples), seed=int(bootstrap_seed + 3))
            cf_pt, cf_ci = _bootstrap_mean_ci(cons_frac_a, n_boot=int(bootstrap_samples), seed=int(bootstrap_seed + 4))
            lift_pt, lift_ci = _bootstrap_mean_ci(lift_a, n_boot=int(bootstrap_samples), seed=int(bootstrap_seed + 5))
        else:
            top1_pt, top1_ci = float(np.mean(top1_a)) if top1_a.size else float("nan"), (float("nan"), float("nan"))
            maj_pt, maj_ci = float(np.mean(majority_a)) if majority_a.size else float("nan"), (float("nan"), float("nan"))
            hit_pt, hit_ci = float(np.mean(hit_a)) if hit_a.size else float("nan"), (float("nan"), float("nan"))
            mrr_pt, mrr_ci = float(np.mean(mrr_ul_a)) if mrr_ul_a.size else float("nan"), (float("nan"), float("nan"))
            cf_pt, cf_ci = float(np.mean(cons_frac_a)) if cons_frac_a.size else float("nan"), (float("nan"), float("nan"))
            lift_pt, lift_ci = float(np.mean(lift_a)) if lift_a.size else float("nan"), (float("nan"), float("nan"))

        # Macro averages (law-balanced)
        if per_law:
            n_laws = float(len(per_law))
            macro_top1 = sum((b["top1"] / b["count"]) for b in per_law.values()) / n_laws
            macro_hit = sum((b["hit"] / b["count"]) for b in per_law.values()) / n_laws
            macro_cf = sum((b["cons_frac"] / b["count"]) for b in per_law.values()) / n_laws
            macro_lift = sum((b["lift"] / b["count"]) for b in per_law.values()) / n_laws
            macro_mrr = sum((b["mrr_ul"] / b["count"]) for b in per_law.values()) / n_laws
            macro_maj = sum((b["majority"] / b["count"]) for b in per_law.values()) / n_laws
        else:
            macro_top1 = macro_hit = macro_cf = macro_lift = macro_mrr = macro_maj = float("nan")

        out: Dict[str, Any] = {
            "method": name,
            "n_queries": int(valid),
            "n_queries_total": int(n_total),
            "k": int(k),
            "predominance_fraction": float(pred_frac),

            # Robust micro metrics + CIs
            "hit_at_k": float(hit_pt),
            "hit_at_k_ci": [float(hit_ci[0]), float(hit_ci[1])],
            "mrr_unique_law_at_k": float(mrr_pt),
            "mrr_unique_law_at_k_ci": [float(mrr_ci[0]), float(mrr_ci[1])],
            "mean_consensus_fraction_in_topk": float(cf_pt),
            "mean_consensus_fraction_in_topk_ci": [float(cf_ci[0]), float(cf_ci[1])],
            "mean_prior_normalized_consensus_fraction_lift": float(lift_pt),
            "mean_prior_normalized_consensus_fraction_lift_ci": [float(lift_ci[0]), float(lift_ci[1])],

            # Macro (law-balanced) versions (point estimates)
            "macro_hit_at_k": float(macro_hit),
            "macro_mrr_unique_law_at_k": float(macro_mrr),
            "macro_mean_consensus_fraction_in_topk": float(macro_cf),
            "macro_mean_lift": float(macro_lift),

            # Legacy/secondary + CI
            "majority_accuracy": float(maj_pt),
            "majority_accuracy_ci": [float(maj_ci[0]), float(maj_ci[1])],
            "top1_accuracy": float(top1_pt),
            "top1_accuracy_ci": [float(top1_ci[0]), float(top1_ci[1])],
        }

        # Overlap aggregates if computed
        if ref_idx is not None and sent_jacc_v:
            sj = np.asarray(sent_jacc_v, dtype=np.float64)
            so = np.asarray(sent_ovl_v, dtype=np.float64)
            lj = np.asarray(law_jacc_v, dtype=np.float64)
            if bootstrap:
                sj_pt, sj_ci = _bootstrap_mean_ci(sj, n_boot=int(bootstrap_samples), seed=int(bootstrap_seed + 10))
                so_pt, so_ci = _bootstrap_mean_ci(so, n_boot=int(bootstrap_samples), seed=int(bootstrap_seed + 11))
                lj_pt, lj_ci = _bootstrap_mean_ci(lj, n_boot=int(bootstrap_samples), seed=int(bootstrap_seed + 12))
            else:
                sj_pt, sj_ci = float(np.mean(sj)), (float("nan"), float("nan"))
                so_pt, so_ci = float(np.mean(so)), (float("nan"), float("nan"))
                lj_pt, lj_ci = float(np.mean(lj)), (float("nan"), float("nan"))

            out.update(
                {
                    "overlap_sentence_jaccard_vs_ref": float(sj_pt),
                    "overlap_sentence_jaccard_vs_ref_ci": [float(sj_ci[0]), float(sj_ci[1])],
                    "overlap_sentence_fraction_vs_ref": float(so_pt),
                    "overlap_sentence_fraction_vs_ref_ci": [float(so_ci[0]), float(so_ci[1])],
                    "overlap_law_jaccard_vs_ref": float(lj_pt),
                    "overlap_law_jaccard_vs_ref_ci": [float(lj_ci[0]), float(lj_ci[1])],
                }
            )

        # Keep per-query arrays for paired bootstrap comparisons (not dumped unless out_json used)
        out["_per_query"] = {
            "top1": top1_a.tolist(),
            "majority": majority_a.tolist(),
            "hit": hit_a.tolist(),
            "cons_frac": cons_frac_a.tolist(),
            "lift": lift_a.tolist(),
            "mrr_ul": mrr_ul_a.tolist(),

            # Router signals
            "maj_law": maj_law_v,
            "maj_frac": maj_frac_a.tolist(),
            "maj_lift": maj_lift_a.tolist(),
            "maj_correct": maj_correct_a.tolist(),
        }
        if ref_idx is not None and sent_jacc_v:
            out["_per_query"].update(
                {
                    "sent_jacc": np.asarray(sent_jacc_v, dtype=np.float64).tolist(),
                    "sent_ovl": np.asarray(sent_ovl_v, dtype=np.float64).tolist(),
                    "law_jacc": np.asarray(law_jacc_v, dtype=np.float64).tolist(),
                }
            )

        return out

    # Score methods
    # Reference retrieval for overlap diagnostics is the MB baseline if available.
    ref_idx = idx_mb

    mb: Optional[Dict[str, Any]] = None
    if idx_mb is not None:
        mb = _score_from_idx("true_mixedbread", idx_mb, ref_idx=None)

    kahm_qmb = _score_from_idx(
        "kahm_query_on_mb_corpus",
        idx_kahm_qmb,
        ref_idx=ref_idx,
    )

    kahm_full: Optional[Dict[str, Any]] = None
    if idx_kahm_full is not None:
        kahm_full = _score_from_idx(
            "kahm_full_approx",
            idx_kahm_full,
            ref_idx=ref_idx,
        )


    kahm_best_out: Optional[Dict[str, Any]] = None
    if kahm_best_enabled and (idx_kahm_full_all is not None) and (scores_kahm_full_all is not None):
        # Precompute router law posteriors + LLRs per query
        n_q = int(idx_kahm_qmb_all.shape[0])
        cand_laws_all = law_arr[idx_kahm_qmb_all]  # shape (n_q, cand_k)
        router_laws_all = law_arr[idx_kahm_full_all]  # shape (n_q, router_k)

        # Provide default grids if not passed.
        cons_grid = router_cons_thresholds if router_cons_thresholds is not None else _parse_float_list(
            DEFAULT_ROUTER_CONS_THRESHOLDS, name="router_cons_thresholds"
        )
        lift_grid = router_lift_thresholds if router_lift_thresholds is not None else _parse_float_list(
            DEFAULT_ROUTER_LIFT_THRESHOLDS, name="router_lift_thresholds"
        )
        top_laws_grid = kahm_best_top_laws_grid if kahm_best_top_laws_grid is not None else _parse_int_list(
            DEFAULT_KAHM_BEST_TOP_LAWS_GRID, name="kahm_best_top_laws_grid"
        )
        law_slots_grid = kahm_best_law_slots_grid if kahm_best_law_slots_grid is not None else _parse_int_list(
            DEFAULT_KAHM_BEST_LAW_SLOTS_GRID, name="kahm_best_law_slots_grid"
        )
        lambda_grid = kahm_best_lambda_grid if kahm_best_lambda_grid is not None else _parse_float_list(
            DEFAULT_KAHM_BEST_LAMBDA_GRID, name="kahm_best_lambda_grid"
        )

        if not cons_grid:
            cons_grid = [0.55]
        if not lift_grid:
            lift_grid = [10.0]
        if not top_laws_grid:
            top_laws_grid = [1]
        if not law_slots_grid:
            law_slots_grid = [1]
        if not lambda_grid:
            lambda_grid = [0.1]

        # Router stats per query
        router_info: List[Dict[str, Any]] = []
        for i in range(n_q):
            r_laws = [str(x) for x in router_laws_all[i].tolist() if str(x)]
            if not r_laws:
                router_info.append({"counts": Counter(), "llr": {}, "maj_law": "", "maj_frac": 0.0, "maj_lift": 0.0})
                continue
            c = Counter(r_laws)
            maj_law, maj_count = c.most_common(1)[0]
            maj_frac = float(maj_count) / float(len(r_laws))
            prior = float(law_prior.get(str(maj_law), 0.0))
            maj_lift = maj_frac / max(float(prior), float(DEFAULT_KAHM_BEST_EPS))
            denom = float(len(r_laws))
            p = {lw: float(cnt) / denom for lw, cnt in c.items()}
            llr = {
                lw: float(
                    np.log((p[lw] + DEFAULT_KAHM_BEST_EPS) / (float(law_prior.get(lw, 0.0)) + DEFAULT_KAHM_BEST_EPS))
                )
                for lw in p.keys()
            }
            router_info.append(
                {"counts": c, "llr": llr, "maj_law": str(maj_law), "maj_frac": float(maj_frac), "maj_lift": float(maj_lift)}
            )

        def _quick_metrics(idx_topk: np.ndarray) -> Dict[str, float]:
            idx_topk = np.asarray(idx_topk, dtype=np.int64)
            n = int(idx_topk.shape[0])
            top1 = 0.0
            hit = 0.0
            mrr_ul_sum = 0.0
            valid = 0
            for i in range(n):
                cons = consensuses[i]
                if not cons:
                    continue
                row = [int(x) for x in idx_topk[i].tolist() if int(x) >= 0]
                if not row:
                    continue
                laws = [str(law_arr[j]) for j in row]
                if not laws:
                    continue
                valid += 1
                top1 += 1.0 if str(laws[0]) == str(cons) else 0.0
                hit += 1.0 if str(cons) in set(laws) else 0.0
                seen: List[str] = []
                for lw in laws:
                    if lw not in seen:
                        seen.append(lw)
                try:
                    rank = 1 + seen.index(str(cons))
                    mrr_ul_sum += 1.0 / float(rank)
                except ValueError:
                    mrr_ul_sum += 0.0
            denom = float(max(1, valid))
            return {"top1": float(top1 / denom), "hit": float(hit / denom), "mrr_ul": float(mrr_ul_sum / denom), "valid": float(valid)}

        def _build_kahm_best_idx(
            *,
            maj_frac_thr: float,
            maj_lift_thr: float,
            top_laws: int,
            law_slots: int,
            lam: float,
            mode: str,
        ) -> Tuple[np.ndarray, float]:
            out_idx = np.full((n_q, k), -1, dtype=np.int64)
            accepted = 0
            for i in range(n_q):
                info = router_info[i]
                ok = (info["maj_frac"] >= float(maj_frac_thr)) and (info["maj_lift"] >= float(maj_lift_thr)) and bool(info["counts"])
                cand_idx = idx_kahm_qmb_all[i].astype(np.int64, copy=False)
                cand_scores = scores_kahm_qmb_all[i].astype(np.float32, copy=False)
                if not ok:
                    out_idx[i, :] = cand_idx[:k]
                    continue

                accepted += 1
                law_ranked = [lw for (lw, _) in info["counts"].most_common(max(1, int(top_laws)))]
                law_set = set(law_ranked)

                if mode in ("rerank", "hybrid") and float(lam) != 0.0:
                    bonuses = np.zeros((cand_idx.shape[0],), dtype=np.float32)
                    llr = info["llr"]
                    for j in range(cand_idx.shape[0]):
                        lw = str(cand_laws_all[i, j])
                        bonuses[j] = float(llr.get(lw, 0.0))
                    rerank_scores = cand_scores + float(lam) * bonuses
                    order = np.argsort(-rerank_scores, kind="mergesort")
                else:
                    order = np.arange(cand_idx.shape[0], dtype=np.int64)

                ordered_idx = cand_idx[order]
                ordered_laws = cand_laws_all[i, order]

                picked: List[int] = []
                if mode in ("filter", "hybrid"):
                    cap = int(max(1, min(int(law_slots), int(k))))
                    for j, sid in enumerate(ordered_idx.tolist()):
                        if len(picked) >= cap:
                            break
                        if str(ordered_laws[j]) in law_set:
                            picked.append(int(sid))

                if len(picked) < k:
                    seen = set(picked)
                    for sid in ordered_idx.tolist():
                        if sid < 0:
                            continue
                        if sid in seen:
                            continue
                        picked.append(int(sid))
                        seen.add(int(sid))
                        if len(picked) >= k:
                            break

                out_idx[i, :] = np.asarray(picked[:k], dtype=np.int64)
            coverage = float(accepted) / float(max(1, n_q))
            return out_idx, coverage

        mode = str(kahm_best_mode).strip().lower()
        if mode not in ("filter", "rerank", "hybrid"):
            mode = "hybrid"

        default_cfg = {
            "maj_frac_thr": 0.55,
            "maj_lift_thr": 10.0,
            "top_laws": 1,
            "law_slots": 3,
            "lam": 0.10,
            "mode": mode,
        }
        best_cfg = dict(default_cfg)
        best_val = -1.0
        best_cov = 0.0

        if bool(kahm_best_tune):
            obj = str(kahm_best_optimize).strip().lower()
            if obj not in ("top1", "mrr", "hit"):
                obj = "top1"

            for maj_frac_thr in cons_grid:
                for maj_lift_thr in lift_grid:
                    for top_laws in top_laws_grid:
                        for law_slots in law_slots_grid:
                            for lam in lambda_grid:
                                idx_try, cov = _build_kahm_best_idx(
                                    maj_frac_thr=float(maj_frac_thr),
                                    maj_lift_thr=float(maj_lift_thr),
                                    top_laws=int(top_laws),
                                    law_slots=int(law_slots),
                                    lam=float(lam),
                                    mode=mode,
                                )
                                if cov < float(kahm_best_min_coverage):
                                    continue
                                qm = _quick_metrics(idx_try)
                                val = float(
                                    qm["top1"] if obj == "top1" else (qm["mrr_ul"] if obj == "mrr" else qm["hit"])
                                )
                                if (val > best_val) or (val == best_val and cov > best_cov) or (
                                    val == best_val
                                    and cov == best_cov
                                    and float(lam) < float(best_cfg.get("lam", 1e9))
                                ):
                                    best_val = val
                                    best_cov = cov
                                    best_cfg = {
                                        "maj_frac_thr": float(maj_frac_thr),
                                        "maj_lift_thr": float(maj_lift_thr),
                                        "top_laws": int(top_laws),
                                        "law_slots": int(law_slots),
                                        "lam": float(lam),
                                        "mode": mode,
                                    }

        idx_best, cov = _build_kahm_best_idx(
            maj_frac_thr=float(best_cfg["maj_frac_thr"]),
            maj_lift_thr=float(best_cfg["maj_lift_thr"]),
            top_laws=int(best_cfg["top_laws"]),
            law_slots=int(best_cfg.get("law_slots", min(1, k))),
            lam=float(best_cfg["lam"]),
            mode=str(best_cfg["mode"]),
        )

        kahm_best_out = _score_from_idx("kahm_best_routed", idx_best, ref_idx=ref_idx)
        kahm_best_out["kahm_best_config"] = {
            "mode": str(best_cfg["mode"]),
            "router_k": int(router_k),
            "cand_k": int(cand_k),
            "maj_frac_thr": float(best_cfg["maj_frac_thr"]),
            "maj_lift_thr": float(best_cfg["maj_lift_thr"]),
            "top_laws": int(best_cfg["top_laws"]),
            "law_slots": int(best_cfg.get("law_slots", min(1, k))),
            "lambda": float(best_cfg["lam"]),
            "coverage": float(cov),
            "optimize": str(kahm_best_optimize),
            "tuned": bool(kahm_best_tune),
        }

    idf = _score_from_idx("pure_idf_svd", idx_idf, ref_idx=ref_idx)

    # Paired comparisons vs true Mixedbread (if available) to validate hypotheses.
    comparisons: Dict[str, Any] = {}
    if bootstrap and mb is not None:
        # Paired bootstrap over query indices in the per-query arrays
        def _paired_delta_ci(a_list: List[float], b_list: List[float], seed_off: int = 0) -> Dict[str, Any]:
            a = np.asarray(a_list, dtype=np.float64)
            b = np.asarray(b_list, dtype=np.float64)
            n = min(int(a.size), int(b.size))
            a = a[:n]
            b = b[:n]
            if n == 0:
                return {"delta": float("nan"), "ci": [float("nan"), float("nan")]}
            d = a - b
            pt, ci = _bootstrap_mean_ci(d, n_boot=int(bootstrap_samples), seed=int(bootstrap_seed + seed_off))
            return {"delta": float(pt), "ci": [float(ci[0]), float(ci[1])]}

        comparisons["kahm_query_minus_mb"] = {
            "hit_at_k": _paired_delta_ci(kahm_qmb["_per_query"]["hit"], mb["_per_query"]["hit"], 100),
            "mrr_unique_law_at_k": _paired_delta_ci(kahm_qmb["_per_query"]["mrr_ul"], mb["_per_query"]["mrr_ul"], 101),
            "mean_consensus_fraction_in_topk": _paired_delta_ci(kahm_qmb["_per_query"]["cons_frac"], mb["_per_query"]["cons_frac"], 102),
            "mean_lift": _paired_delta_ci(kahm_qmb["_per_query"]["lift"], mb["_per_query"]["lift"], 103),
            "majority_accuracy": _paired_delta_ci(kahm_qmb["_per_query"]["majority"], mb["_per_query"]["majority"], 104),
            "top1_accuracy": _paired_delta_ci(kahm_qmb["_per_query"]["top1"], mb["_per_query"]["top1"], 105),
        }

        if kahm_best_out is not None:
            comparisons["kahm_best_minus_mb"] = {
                "hit_at_k": _paired_delta_ci(kahm_best_out["_per_query"]["hit"], mb["_per_query"]["hit"], 150),
                "mrr_unique_law_at_k": _paired_delta_ci(kahm_best_out["_per_query"]["mrr_ul"], mb["_per_query"]["mrr_ul"], 151),
                "mean_consensus_fraction_in_topk": _paired_delta_ci(kahm_best_out["_per_query"]["cons_frac"], mb["_per_query"]["cons_frac"], 152),
                "mean_lift": _paired_delta_ci(kahm_best_out["_per_query"]["lift"], mb["_per_query"]["lift"], 153),
                "majority_accuracy": _paired_delta_ci(kahm_best_out["_per_query"]["majority"], mb["_per_query"]["majority"], 154),
                "top1_accuracy": _paired_delta_ci(kahm_best_out["_per_query"]["top1"], mb["_per_query"]["top1"], 155),
            }
            comparisons["kahm_best_minus_kahm_query"] = {
                "hit_at_k": _paired_delta_ci(kahm_best_out["_per_query"]["hit"], kahm_qmb["_per_query"]["hit"], 156),
                "mrr_unique_law_at_k": _paired_delta_ci(kahm_best_out["_per_query"]["mrr_ul"], kahm_qmb["_per_query"]["mrr_ul"], 157),
                "mean_consensus_fraction_in_topk": _paired_delta_ci(kahm_best_out["_per_query"]["cons_frac"], kahm_qmb["_per_query"]["cons_frac"], 158),
                "mean_lift": _paired_delta_ci(kahm_best_out["_per_query"]["lift"], kahm_qmb["_per_query"]["lift"], 159),
                "majority_accuracy": _paired_delta_ci(kahm_best_out["_per_query"]["majority"], kahm_qmb["_per_query"]["majority"], 160),
                "top1_accuracy": _paired_delta_ci(kahm_best_out["_per_query"]["top1"], kahm_qmb["_per_query"]["top1"], 161),
            }


        if kahm_full is not None:
            comparisons["kahm_full_minus_mb"] = {
                "hit_at_k": _paired_delta_ci(kahm_full["_per_query"]["hit"], mb["_per_query"]["hit"], 106),
                "mrr_unique_law_at_k": _paired_delta_ci(kahm_full["_per_query"]["mrr_ul"], mb["_per_query"]["mrr_ul"], 107),
                "mean_consensus_fraction_in_topk": _paired_delta_ci(kahm_full["_per_query"]["cons_frac"], mb["_per_query"]["cons_frac"], 108),
                "mean_lift": _paired_delta_ci(kahm_full["_per_query"]["lift"], mb["_per_query"]["lift"], 109),
                "majority_accuracy": _paired_delta_ci(kahm_full["_per_query"]["majority"], mb["_per_query"]["majority"], 110),
                "top1_accuracy": _paired_delta_ci(kahm_full["_per_query"]["top1"], mb["_per_query"]["top1"], 111),
            }

        comparisons["idf_minus_mb"] = {
            "hit_at_k": _paired_delta_ci(idf["_per_query"]["hit"], mb["_per_query"]["hit"], 200),
            "mrr_unique_law_at_k": _paired_delta_ci(idf["_per_query"]["mrr_ul"], mb["_per_query"]["mrr_ul"], 201),
            "mean_consensus_fraction_in_topk": _paired_delta_ci(idf["_per_query"]["cons_frac"], mb["_per_query"]["cons_frac"], 202),
            "mean_lift": _paired_delta_ci(idf["_per_query"]["lift"], mb["_per_query"]["lift"], 203),
            "majority_accuracy": _paired_delta_ci(idf["_per_query"]["majority"], mb["_per_query"]["majority"], 204),
            "top1_accuracy": _paired_delta_ci(idf["_per_query"]["top1"], mb["_per_query"]["top1"], 205),
        }

        comparisons["kahm_query_minus_idf"] = {
            "hit_at_k": _paired_delta_ci(kahm_qmb["_per_query"]["hit"], idf["_per_query"]["hit"], 120),
            "mrr_unique_law_at_k": _paired_delta_ci(kahm_qmb["_per_query"]["mrr_ul"], idf["_per_query"]["mrr_ul"], 121),
            "mean_consensus_fraction_in_topk": _paired_delta_ci(kahm_qmb["_per_query"]["cons_frac"], idf["_per_query"]["cons_frac"], 122),
            "mean_lift": _paired_delta_ci(kahm_qmb["_per_query"]["lift"], idf["_per_query"]["lift"], 123),
            "majority_accuracy": _paired_delta_ci(kahm_qmb["_per_query"]["majority"], idf["_per_query"]["majority"], 124),
            "top1_accuracy": _paired_delta_ci(kahm_qmb["_per_query"]["top1"], idf["_per_query"]["top1"], 125),
        }

        if kahm_full is not None:
            comparisons["kahm_full_minus_idf"] = {
                "hit_at_k": _paired_delta_ci(kahm_full["_per_query"]["hit"], idf["_per_query"]["hit"], 126),
                "mrr_unique_law_at_k": _paired_delta_ci(kahm_full["_per_query"]["mrr_ul"], idf["_per_query"]["mrr_ul"], 127),
                "mean_consensus_fraction_in_topk": _paired_delta_ci(kahm_full["_per_query"]["cons_frac"], idf["_per_query"]["cons_frac"], 128),
                "mean_lift": _paired_delta_ci(kahm_full["_per_query"]["lift"], idf["_per_query"]["lift"], 129),
                "majority_accuracy": _paired_delta_ci(kahm_full["_per_query"]["majority"], idf["_per_query"]["majority"], 130),
                "top1_accuracy": _paired_delta_ci(kahm_full["_per_query"]["top1"], idf["_per_query"]["top1"], 131),
            }

    out: Dict[str, Any] = {
        "kahm_query_on_mb_corpus": kahm_qmb,
        "pure_idf_svd": idf,
        "comparisons": comparisons,
        "bootstrap": {"enabled": bool(bootstrap), "samples": int(bootstrap_samples), "seed": int(bootstrap_seed)},
    }
    if mb is not None:
        out["true_mixedbread"] = mb
    if kahm_best_out is not None:
        out["kahm_best_routed"] = kahm_best_out
    if kahm_full is not None:
        out["kahm_full_approx"] = kahm_full
    return out


def print_eval_summary(eval_out: Dict[str, Any]) -> None:
    boot = eval_out.get("bootstrap", {})
    if bool(boot.get("enabled", False)):
        print(
            f"\nBootstrap CIs: enabled | samples={boot.get('samples')} | seed={boot.get('seed')}",
            flush=True,
        )

    ordered: List[str] = []
    if "true_mixedbread" in eval_out:
        ordered.append("true_mixedbread")
    if "kahm_query_on_mb_corpus" in eval_out:
        ordered.append("kahm_query_on_mb_corpus")
    if "kahm_best_routed" in eval_out:
        ordered.append("kahm_best_routed")
    if "kahm_full_approx" in eval_out:
        ordered.append("kahm_full_approx")
    if "pure_idf_svd" in eval_out:
        ordered.append("pure_idf_svd")

    for key in ordered:
        m = eval_out[key]
        print(f"\n[{m['method']}]", flush=True)
        print(
            f"  queries: {m['n_queries']} (of {m.get('n_queries_total', m['n_queries'])}) | "
            f"k={m['k']} | predominance_fraction={m['predominance_fraction']}",
            flush=True,
        )

        # Robust (micro) with CIs
        print(f"  hit@k:               {_fmt_ci(m['hit_at_k'], tuple(m['hit_at_k_ci']))}", flush=True)
        print(f"  MRR@k (unique laws): {_fmt_ci(m['mrr_unique_law_at_k'], tuple(m['mrr_unique_law_at_k_ci']))}", flush=True)
        print(f"  mean cons frac:      {_fmt_ci(m['mean_consensus_fraction_in_topk'], tuple(m['mean_consensus_fraction_in_topk_ci']))}", flush=True)
        print(f"  mean lift (prior):   {_fmt_ci(m['mean_prior_normalized_consensus_fraction_lift'], tuple(m['mean_prior_normalized_consensus_fraction_lift_ci']), digits=3)}", flush=True)

        # Macro (law-balanced) point estimates
        print(f"  macro hit@k:         {m['macro_hit_at_k']:.3f}", flush=True)
        print(f"  macro MRR@k:         {m['macro_mrr_unique_law_at_k']:.3f}", flush=True)
        print(f"  macro cons frac:     {m['macro_mean_consensus_fraction_in_topk']:.3f}", flush=True)
        print(f"  macro lift:          {m['macro_mean_lift']:.3f}", flush=True)

        # Overlap diagnostics (vs true mixedbread retrieval), if present
        if "overlap_sentence_jaccard_vs_ref" in m:
            print(f"  overlap sent Jaccard vs MB@k: {_fmt_ci(m['overlap_sentence_jaccard_vs_ref'], tuple(m['overlap_sentence_jaccard_vs_ref_ci']))}", flush=True)
            print(f"  overlap sent frac vs MB@k:    {_fmt_ci(m['overlap_sentence_fraction_vs_ref'], tuple(m['overlap_sentence_fraction_vs_ref_ci']))}", flush=True)
            print(f"  overlap law Jaccard vs MB@k:  {_fmt_ci(m['overlap_law_jaccard_vs_ref'], tuple(m['overlap_law_jaccard_vs_ref_ci']))}", flush=True)

        # Legacy/secondary with CIs
        print(f"  majority-accuracy:   {_fmt_ci(m['majority_accuracy'], tuple(m['majority_accuracy_ci']))}", flush=True)
        print(f"  top1-accuracy:       {_fmt_ci(m['top1_accuracy'], tuple(m['top1_accuracy_ci']))}", flush=True)
        if "kahm_best_config" in m:
            cfg = m.get("kahm_best_config", {})
            print(
                f"  kahm_best config:    mode={cfg.get('mode')} | router_k={cfg.get('router_k')} | cand_k={cfg.get('cand_k')} | "
                f"maj_frac_thr={cfg.get('maj_frac_thr')} | maj_lift_thr={cfg.get('maj_lift_thr')} | top_laws={cfg.get('top_laws')} | lambda={cfg.get('lambda')}",
                flush=True,
            )
            if "coverage" in cfg:
                print(f"  kahm_best coverage:  {float(cfg.get('coverage')):.3f}", flush=True)

    # Hypothesis-focused paired deltas (bootstrap) where available.
    comp = eval_out.get("comparisons", {})
    if comp:
        print("\n[paired deltas (mean difference; 95% bootstrap CI)]", flush=True)

        def _pdelta(d: Dict[str, Any]) -> str:
            return _fmt_ci(float(d.get("delta", float("nan"))), tuple(d.get("ci", [float("nan"), float("nan")])), digits=3)

        def _print_block(title: str, d: Dict[str, Any]) -> None:
            print(f"  {title}:", flush=True)
            for met, label in [
                ("hit_at_k", "hit@k"),
                ("mrr_unique_law_at_k", "MRR@k(unique laws)"),
                ("mean_consensus_fraction_in_topk", "mean cons frac"),
                ("mean_lift", "mean lift"),
                ("majority_accuracy", "majority-accuracy"),
                ("top1_accuracy", "top1-accuracy"),
            ]:
                if met in d:
                    print(f"    {label:<20} {_pdelta(d[met])}", flush=True)

        if "kahm_query_minus_mb" in comp:
            _print_block("KAHM(query→MB corpus) - MB", comp["kahm_query_minus_mb"])
        if "kahm_best_minus_mb" in comp:
            _print_block("KAHM(best routed) - MB", comp["kahm_best_minus_mb"])
        if "kahm_best_minus_kahm_query" in comp:
            _print_block("KAHM(best routed) - KAHM(query→MB)", comp["kahm_best_minus_kahm_query"])
        if "kahm_full_minus_mb" in comp:
            _print_block("KAHM(full approx) - MB", comp["kahm_full_minus_mb"])
        if "idf_minus_mb" in comp:
            _print_block("IDF–SVD - MB", comp["idf_minus_mb"])

        if "kahm_query_minus_idf" in comp:
            _print_block("KAHM(query→MB corpus) - IDF–SVD", comp["kahm_query_minus_idf"])
        if "kahm_full_minus_idf" in comp:
            _print_block("KAHM(full approx) - IDF–SVD", comp["kahm_full_minus_idf"])



def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Evaluation script for FAISS retrieval with Mixedbread, KAHM query-translation (IDF–SVD→MB space), "
            "full KAHM approximation, and IDF–SVD baselines."
        )
    )

    p.add_argument("--corpus", default=DEFAULT_CORPUS_PARQUET, help="Corpus parquet with sentence_id/law_type/page/sentence")
    p.add_argument("--semantic_npz", default=DEFAULT_SEMANTIC_NPZ, help="NPZ with true Mixedbread corpus embeddings")
    p.add_argument("--idf_svd_npz", default=DEFAULT_IDF_SVD_NPZ, help="NPZ with IDF–SVD corpus embeddings")
    p.add_argument("--idf_svd_model", default=DEFAULT_IDF_SVD_MODEL, help="joblib IDF–SVD pipeline used for queries")

    p.add_argument("--kahm_model", default=DEFAULT_KAHM_MODEL, help="KAHM regressor joblib (IDF–SVD -> approx Mixedbread)")
    p.add_argument("--kahm_mode", default=DEFAULT_KAHM_MODE, help="KAHM regression mode (must match training)")
    p.add_argument("--kahm_corpus_npz", default=DEFAULT_KAHM_CORPUS_NPZ, help="Optional precomputed KAHM corpus NPZ")
    p.add_argument("--kahm_precomputed", choices=["auto", "force", "strict"], default=DEFAULT_KAHM_PRECOMPUTED_POLICY,
                   help="How to handle precomputed KAHM corpus NPZ metadata mismatches")

    p.add_argument("--kahm_n_jobs", type=int, default=DEFAULT_KAHM_N_JOBS, help="KAHM regression n_jobs")
    p.add_argument("--kahm_backend", choices=["threading", "loky"], default=DEFAULT_KAHM_BACKEND, help="joblib backend for KAHM inference")

    p.add_argument("--mixedbread_model", default=DEFAULT_MIXEDBREAD_FALLBACK_MODEL, help="Mixedbread model name (SentenceTransformer) (evaluation baseline)")
    p.add_argument(
        "--mb_query_npz",
        default=DEFAULT_MB_QUERY_NPZ,
        help=(
            "Optional NPZ containing precomputed Mixedbread query embeddings (avoids loading the MB model). "
            "If provided, the script will load MB query embeddings from this NPZ even when --skip_mb_queries is set. "
            "Expected keys: embeddings + query_id (preferred) or embeddings-only with matching row count."
        ),
    )
    p.add_argument("--device", default="cpu", help="Device for SentenceTransformer (e.g., cpu, cuda)")

    p.add_argument(
        "--skip_mb_queries",
        action="store_true",
        help="Skip Mixedbread query embeddings/baseline (useful for inference-only runs without MB model).",
    )
    p.add_argument(
        "--skip_kahm_full",
        action="store_true",
        help="Skip full KAHM corpus retrieval evaluation (only evaluate KAHM(query→MB corpus) and IDF–SVD).",
    )

    p.add_argument("--eval_k", type=int, default=DEFAULT_EVAL_K, help="Evaluate on top-k retrieved results")
    p.add_argument("--k_grid", default=None, help="Comma-separated k values for k-grid evaluation (default: 1,3,5,10,20).")
    p.add_argument("--no_kgrid", action="store_true", help="Disable k-grid evaluation output.")
    p.add_argument("--predominance_fraction", type=float, default=DEFAULT_PREDOMINANCE_FRACTION, help="Threshold for majority/predominance metric")
    p.add_argument("--bootstrap_samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES, help="Number of bootstrap resamples for 95% CIs (micro metrics).")
    p.add_argument("--bootstrap_seed", type=int, default=DEFAULT_BOOTSTRAP_SEED, help="Random seed for bootstrap.")
    p.add_argument("--no_bootstrap", action="store_true", help="Disable bootstrap confidence intervals.")

    # Hypothesis testing configuration (non-inferiority by default)
    p.add_argument(
        "--equivalence_two_sided",
        action="store_true",
        help=(
            "Use two-sided equivalence (CI must lie within ±margin). Default is one-sided non-inferiority "
            "(CI lower bound must be ≥ -margin) for 'as good as' claims."
        ),
    )

    # H1 margins (KAHM(best routed) vs Mixedbread baseline; also used for H1b candidate checks)
    p.add_argument("--h1_margin_hit", type=float, default=DEFAULT_H1_MARGIN_HIT, help="Non-inferiority margin for hit@k (H1: best routed vs MB; H1b: candidate generation checks)")
    p.add_argument("--h1_margin_mrr", type=float, default=DEFAULT_H1_MARGIN_MRR, help="Non-inferiority margin for MRR(unique laws)@k")
    p.add_argument("--h1_margin_top1", type=float, default=DEFAULT_H1_MARGIN_TOP1, help="Non-inferiority margin for top1 accuracy")
    p.add_argument("--h1_margin_cons", type=float, default=DEFAULT_H1_MARGIN_CONS, help="Non-inferiority margin for mean consensus fraction@k")
    p.add_argument("--h1_margin_lift", type=float, default=DEFAULT_H1_MARGIN_LIFT, help="Non-inferiority margin for mean lift@k")
    p.add_argument("--h1_margin_majority", type=float, default=DEFAULT_H1_MARGIN_MAJ, help="Non-inferiority margin for majority-accuracy")

    # H2 margins (Full KAHM approx vs MB) + router superiority vs IDF
    p.add_argument("--h2_margin_hit", type=float, default=DEFAULT_H2_MARGIN_HIT, help="Non-inferiority margin for hit@k (H1: best routed vs MB; H1b: candidate generation checks)")
    p.add_argument("--h2_margin_cons", type=float, default=DEFAULT_H2_MARGIN_CONS, help="Non-inferiority margin for mean consensus fraction@k")
    p.add_argument("--h2_margin_lift", type=float, default=DEFAULT_H2_MARGIN_LIFT, help="Non-inferiority margin for mean lift@k")
    p.add_argument("--h2_margin_majority", type=float, default=DEFAULT_H2_MARGIN_MAJ, help="Non-inferiority margin for majority-accuracy")

    # Router confidence sweep (uses inference-time signals maj_frac and maj_lift)
    p.add_argument(
        "--router_cons_thresholds",
        default=DEFAULT_ROUTER_CONS_THRESHOLDS,
        help="Comma-separated thresholds for maj_frac (e.g., '0.5,0.6,0.7')",
    )
    p.add_argument(
        "--router_lift_thresholds",
        default=DEFAULT_ROUTER_LIFT_THRESHOLDS,
        help="Comma-separated thresholds for maj_lift (e.g., '10,20,30,40')",
    )
    p.add_argument(
        "--router_min_coverage",
        type=float,
        default=DEFAULT_ROUTER_MIN_COVERAGE,
        help="When recommending a router threshold pair from the grid, require at least this coverage.",
    )

    # KAHM best-practice evaluation (routing + law-aware candidate post-processing)
    p.add_argument("--no_kahm_best", action="store_true", help="Disable the KAHM best-practice routed evaluation.")
    p.add_argument("--kahm_best_router_k", type=int, default=DEFAULT_KAHM_BEST_ROUTER_K, help="K for router posterior estimation on KAHM(full) index")
    p.add_argument("--kahm_best_cand_k", type=int, default=DEFAULT_KAHM_BEST_CAND_K, help="Candidate pool size for KAHM(q→MB) retrieval before post-processing")
    p.add_argument("--kahm_best_mode", choices=["filter", "rerank", "hybrid"], default=DEFAULT_KAHM_BEST_MODE, help="How to apply router posterior to candidates")
    p.add_argument("--kahm_best_top_laws_grid", default=DEFAULT_KAHM_BEST_TOP_LAWS_GRID, help="Grid for number of top router laws (comma-separated ints)")
    p.add_argument(
        "--kahm_best_law_slots_grid",
        default=DEFAULT_KAHM_BEST_LAW_SLOTS_GRID,
        help="Grid for max number of top-k slots reserved for router laws when filtering is enabled (comma-separated ints)",
    )
    p.add_argument("--kahm_best_lambda_grid", default=DEFAULT_KAHM_BEST_LAMBDA_GRID, help="Grid for rerank lambda (comma-separated floats)")
    p.add_argument("--kahm_best_optimize", choices=["top1", "mrr", "hit"], default=DEFAULT_KAHM_BEST_OPTIMIZE, help="Objective used when selecting the best kahm_best config")
    p.add_argument("--kahm_best_min_coverage", type=float, default=DEFAULT_ROUTER_MIN_COVERAGE, help="Minimum acceptance coverage when tuning kahm_best thresholds")
    p.add_argument("--no_kahm_best_tune", action="store_true", help="Disable tuning; use a reasonable default kahm_best config")


    p.add_argument("--query_batch", type=int, default=DEFAULT_QUERY_BATCH, help="Batch size for Mixedbread query embeddings")
    p.add_argument("--regress_batch", type=int, default=DEFAULT_REGRESS_BATCH, help="Batch size for KAHM regression")

    p.add_argument("--query_set_module", default="query_set", help="Python module to import query set from")
    p.add_argument("--query_set_name", default="TEST_QUERY_SET", help="Attribute in module containing list of query dicts")

    # FAISS index configuration
    p.add_argument("--faiss_index", choices=["flat", "hnsw"], default="flat", help="FAISS index type")
    p.add_argument("--hnsw_m", type=int, default=32, help="HNSW M parameter")
    p.add_argument("--hnsw_ef_search", type=int, default=128, help="HNSW efSearch parameter")
    p.add_argument("--faiss_threads", type=int, default=DEFAULT_FAISS_THREADS, help="FAISS OpenMP threads (0=default)")

    p.add_argument("--out_json", default="", help="Optional path to write evaluation summary JSON")
    p.add_argument("--per_query_csv", default="", help="Optional path to write per-query top-1 predictions CSV")

    return p

def _parse_k_grid(arg: str | None) -> List[int]:
    if arg is None or not str(arg).strip():
        ks = list(DEFAULT_K_GRID)
    else:
        raw = [x.strip() for x in str(arg).split(",") if x.strip()]
        ks: List[int] = []
        for x in raw:
            ks.append(int(x))
        if not ks:
            ks = list(DEFAULT_K_GRID)
    ks = sorted({int(k) for k in ks if int(k) > 0})
    return ks or list(DEFAULT_K_GRID)


def _parse_float_list(arg: str | None, *, name: str) -> List[float]:
    if arg is None:
        return []
    raw = [x.strip() for x in str(arg).split(",") if x.strip()]
    out: List[float] = []
    for x in raw:
        try:
            out.append(float(x))
        except Exception as e:
            raise ValueError(f"Failed to parse {name} entry {x!r} as float: {e}")
    return out



def _parse_int_list(arg: str | None, *, name: str) -> List[int]:
    if arg is None:
        return []
    raw = [x.strip() for x in str(arg).split(",") if x.strip()]
    out: List[int] = []
    for x in raw:
        try:
            out.append(int(x))
        except Exception as e:
            raise ValueError(f"Failed to parse {name} entry {x!r} as int: {e}")
    return out

def _kgrid_compact_line(label: str, values: List[Tuple[int, float, Tuple[float, float]]], digits: int = 3) -> str:
    parts = []
    for k, pt, ci in values:
        parts.append(f"{k}:{_fmt_ci(pt, ci, digits=digits)}")
    return f"  {label:<24} " + " | ".join(parts)


def print_k_grid_summary(k_to_eval: Dict[int, Dict[str, Any]], *, ks: List[int]) -> None:
    print("\nK-grid evaluation (robust law-level metrics; 95% bootstrap CIs):", flush=True)
    print(f"  ks = {ks}", flush=True)

    # Determine which methods are available in the stored evaluations.
    any_ev = k_to_eval[ks[0]]
    methods: List[Tuple[str, str]] = []
    if "true_mixedbread" in any_ev:
        methods.append(("true_mixedbread", "Mixedbread (true)"))
    if "kahm_query_on_mb_corpus" in any_ev:
        methods.append(("kahm_query_on_mb_corpus", "KAHM(query→MB corpus)"))
    if "kahm_best_routed" in any_ev:
        methods.append(("kahm_best_routed", "KAHM(best routed)"))
    if "kahm_full_approx" in any_ev:
        methods.append(("kahm_full_approx", "KAHM(full approx)"))
    if "pure_idf_svd" in any_ev:
        methods.append(("pure_idf_svd", "IDF–SVD"))
    for key, title in methods:
        print(f"\n[{title}]", flush=True)

        hit = []
        mrr = []
        cf = []
        lift = []
        top1 = []

        def _pt_ci(m: Dict[str, Any], pt_key: str, ci_key: str) -> Tuple[float, Tuple[float, float]]:
            pt = float(m.get(pt_key, float("nan")))
            ci_v = m.get(ci_key, [float("nan"), float("nan")])
            try:
                ci_t = (float(ci_v[0]), float(ci_v[1]))
            except Exception:
                ci_t = (float("nan"), float("nan"))
            return pt, ci_t

        for k in ks:
            m = k_to_eval[k][key]
            pt, ci = _pt_ci(m, "hit_at_k", "hit_at_k_ci")
            hit.append((k, pt, ci))
            pt, ci = _pt_ci(m, "mrr_unique_law_at_k", "mrr_unique_law_at_k_ci")
            mrr.append((k, pt, ci))
            pt, ci = _pt_ci(m, "mean_consensus_fraction_in_topk", "mean_consensus_fraction_in_topk_ci")
            cf.append((k, pt, ci))
            pt, ci = _pt_ci(m, "mean_prior_normalized_consensus_fraction_lift", "mean_prior_normalized_consensus_fraction_lift_ci")
            lift.append((k, pt, ci))
            pt, ci = _pt_ci(m, "top1_accuracy", "top1_accuracy_ci")
            top1.append((k, pt, ci))

        print(_kgrid_compact_line("hit@k", hit), flush=True)
        print(_kgrid_compact_line("MRR@k unique laws", mrr), flush=True)
        print(_kgrid_compact_line("consensus frac@k", cf), flush=True)
        print(_kgrid_compact_line("lift@k (prior)", lift), flush=True)
        print(_kgrid_compact_line("top1 accuracy", top1), flush=True)

        if key != "true_mixedbread":
            law_j = []
            sent_j = []
            for k in ks:
                m = k_to_eval[k][key]
                if "overlap_law_jaccard_vs_ref" in m:
                    law_j.append((k, float(m.get("overlap_law_jaccard_vs_ref", float("nan"))), tuple(m.get("overlap_law_jaccard_vs_ref_ci", [float("nan"), float("nan")]))))
                if "overlap_sentence_jaccard_vs_ref" in m:
                    sent_j.append((k, float(m.get("overlap_sentence_jaccard_vs_ref", float("nan"))), tuple(m.get("overlap_sentence_jaccard_vs_ref_ci", [float("nan"), float("nan")]))))
            if law_j:
                print(_kgrid_compact_line("law Jaccard vs MB", law_j), flush=True)
            if sent_j:
                print(_kgrid_compact_line("sent Jaccard vs MB", sent_j), flush=True)



def print_hypothesis_verdict(
    k_to_eval: Dict[int, Dict[str, Any]],
    *,
    primary_ks: List[int],
    equivalence_two_sided: bool,
    h1_margins: Dict[str, float],
    h2_margins: Dict[str, float],
    router_cons_thresholds: List[float],
    router_lift_thresholds: List[float],
    router_min_coverage: float,
) -> None:
    """Print hypothesis checks using paired deltas + bootstrap CIs.

    Default framing is *non-inferiority* (one-sided): method is "as good as" baseline
    if the 95% CI lower bound for (method - baseline) is >= -margin.

    If --equivalence_two_sided is set, it uses two-sided equivalence: CI must lie within ±margin.

    H1: KAHM best-practice retrieval (routing + law-aware post-processing) ~= MB baseline.
    H1b: KAHM query translation (q_IDF -> KAHM(q) against MB corpus) is adequate for candidate generation.
    H2: Full KAHM approx (q_IDF -> KAHM(q) retrieved against KAHM corpus) is good for routing/candidates
        and materially better than IDF–SVD.
    """
    available = sorted(int(k) for k in k_to_eval.keys())
    if not available:
        print("\nHypothesis check: no k values available.", flush=True)
        return

    def _nearest_k(target: int) -> int:
        if target in k_to_eval:
            return target
        return min(available, key=lambda x: abs(int(x) - int(target)))

    def _get_delta(comp: Dict[str, Any], comp_key: str, met: str) -> Tuple[float, Tuple[float, float]]:
        d = comp.get(comp_key, {}).get(met, None)
        if not isinstance(d, dict):
            return float("nan"), (float("nan"), float("nan"))
        ci = d.get("ci", [float("nan"), float("nan")])
        return float(d.get("delta", float("nan"))), (float(ci[0]), float(ci[1]))

    def _noninferior(delta_ci: Tuple[float, Tuple[float, float]], margin: float) -> bool:
        _, ci = delta_ci
        if not np.isfinite(ci[0]) or not np.isfinite(ci[1]):
            return False
        if equivalence_two_sided:
            return (ci[0] >= -float(margin)) and (ci[1] <= float(margin))
        # one-sided non-inferiority (higher is better): lower CI bound >= -margin
        return ci[0] >= -float(margin)

    def _superior(delta_ci: Tuple[float, Tuple[float, float]]) -> bool:
        _, ci = delta_ci
        return np.isfinite(ci[0]) and (ci[0] > 0.0)

    def _fmt_delta(d: Tuple[float, Tuple[float, float]]) -> str:
        return f"{d[0]:.3f} [{d[1][0]:.3f}, {d[1][1]:.3f}]"

    def _required_margin(d: Tuple[float, Tuple[float, float]]) -> float:
        """Return the smallest NI/equivalence margin that would pass given the CI."""
        _, ci = d
        if not np.isfinite(ci[0]) or not np.isfinite(ci[1]):
            return float("nan")
        if equivalence_two_sided:
            # Need CI within [-m, +m]
            return float(max(0.0, -ci[0], ci[1]))
        # One-sided NI: need lower bound >= -m  -> m >= -lower_bound
        return float(max(0.0, -ci[0]))

    def _status(name: str, d: Tuple[float, Tuple[float, float]], margin: float, ok: bool) -> str:
        req = _required_margin(d)
        if ok:
            return f"    PASS {name} (margin={margin:g})"
        return f"    FAIL {name} (margin={margin:g}; min_required≈{req:.3f})"

    def _router_confidence_report(method: Dict[str, Any], *, title: str) -> None:
        pq = method.get("_per_query", {})
        maj_frac = np.asarray(pq.get("maj_frac", []), dtype=np.float64)
        maj_lift = np.asarray(pq.get("maj_lift", []), dtype=np.float64)
        maj_correct = np.asarray(pq.get("maj_correct", []), dtype=np.float64)
        if maj_frac.size == 0 or maj_lift.size == 0 or maj_correct.size == 0:
            print(f"\n{title}: router confidence report unavailable (missing per-query router signals).", flush=True)
            return

        cons_grid = [float(x) for x in router_cons_thresholds]
        lift_grid = [float(x) for x in router_lift_thresholds]

        print(f"\n{title}: selective routing (abstention) sweep", flush=True)
        print(f"  thresholds: maj_frac in {cons_grid}, maj_lift in {lift_grid}", flush=True)
        print(f"  recommendation constraint: min_coverage >= {router_min_coverage:.2f}", flush=True)
        print("  cols: maj_frac_thr | maj_lift_thr | coverage | majority-acc(accepted)", flush=True)

        best = None  # (acc, coverage, cons_thr, lift_thr)
        for cthr in cons_grid:
            for lthr in lift_grid:
                mask = (maj_frac >= float(cthr)) & (maj_lift >= float(lthr))
                cov = float(mask.mean()) if mask.size else 0.0
                acc = float(np.mean(maj_correct[mask])) if mask.sum() > 0 else float("nan")

                print(f"    {cthr:>10.3f} | {lthr:>11.3f} | {cov:>8.3f} | {acc:>20.3f}", flush=True)

                if np.isfinite(acc) and (cov >= float(router_min_coverage)):
                    cand = (acc, cov, float(cthr), float(lthr))
                    if best is None or (cand[0] > best[0]) or (cand[0] == best[0] and cand[1] > best[1]):
                        best = cand

        if best is not None:
            print(
                f"  Recommended thresholds (maximize accepted accuracy with coverage >= {router_min_coverage:.2f}): "
                f"maj_frac >= {best[2]:.3f}, maj_lift >= {best[3]:.3f} (coverage={best[1]:.3f}, acc={best[0]:.3f})",
                flush=True,
            )
        else:
            print(
                "  No threshold pair met the minimum coverage constraint; consider lowering --router_min_coverage "
                "or widening the threshold grids.",
                flush=True,
            )

    for pk_req in primary_ks:
        pk = _nearest_k(int(pk_req))
        ev = k_to_eval[pk]
        comp = ev.get("comparisons", {})

        # --------------------------- H1 ---------------------------
        # H1 (production retrieval): KAHM best-practice routed retrieval should be non-inferior to
        # the Mixedbread baseline. If kahm_best is not available, we fall back to the plain
        # KAHM query translation baseline (q_IDF -> KAHM(q) retrieved against the MB corpus).
        h1_comp_key = "kahm_best_minus_mb" if "kahm_best_minus_mb" in comp else "kahm_query_minus_mb"
        if h1_comp_key in comp:
            h1_label = "KAHM(best routed)" if h1_comp_key == "kahm_best_minus_mb" else "KAHM(query→MB corpus)"

            d_hit = _get_delta(comp, h1_comp_key, "hit_at_k")
            d_mrr = _get_delta(comp, h1_comp_key, "mrr_unique_law_at_k")
            d_top1 = _get_delta(comp, h1_comp_key, "top1_accuracy")
            d_cons = _get_delta(comp, h1_comp_key, "mean_consensus_fraction_in_topk")
            d_lift = _get_delta(comp, h1_comp_key, "mean_lift")
            d_maj = _get_delta(comp, h1_comp_key, "majority_accuracy")

            ok_hit = _noninferior(d_hit, h1_margins["hit"])
            ok_mrr = _noninferior(d_mrr, h1_margins["mrr"])
            ok_top1 = _noninferior(d_top1, h1_margins["top1"])
            ok_cons = _noninferior(d_cons, h1_margins["cons"])
            ok_lift = _noninferior(d_lift, h1_margins["lift"])
            ok_maj = _noninferior(d_maj, h1_margins["majority"])

            # H1 is meant to be a "production quality" claim: include top-1.
            supported_h1 = ok_hit and ok_mrr and ok_top1 and ok_cons and ok_lift

            print(f"\nHypothesis H1: {h1_label} is as good as MB for production retrieval", flush=True)
            print(f"  k = {pk}", flush=True)
            print(f"  test: {'two-sided equivalence' if equivalence_two_sided else 'one-sided non-inferiority'}", flush=True)
            print(
                "  margins: "
                f"hit={h1_margins['hit']}, mrr={h1_margins['mrr']}, top1={h1_margins['top1']}, "
                f"cons={h1_margins['cons']}, lift={h1_margins['lift']}, maj={h1_margins['majority']}",
                flush=True,
            )
            print(f"  hit@{pk}:                 {h1_label}−MB = {_fmt_delta(d_hit)}", flush=True)
            print(f"  MRR unique@{pk}:           {h1_label}−MB = {_fmt_delta(d_mrr)}", flush=True)
            print(f"  top1@{pk}:                 {h1_label}−MB = {_fmt_delta(d_top1)}", flush=True)
            print(f"  cons frac@{pk}:            {h1_label}−MB = {_fmt_delta(d_cons)}", flush=True)
            print(f"  lift@{pk} (prior):         {h1_label}−MB = {_fmt_delta(d_lift)}", flush=True)
            print(f"  majority@{pk}:             {h1_label}−MB = {_fmt_delta(d_maj)}", flush=True)

            print("  checks:", flush=True)
            print(_status("hit", d_hit, h1_margins["hit"], ok_hit), flush=True)
            print(_status("mrr_unique", d_mrr, h1_margins["mrr"], ok_mrr), flush=True)
            print(_status("top1", d_top1, h1_margins["top1"], ok_top1), flush=True)
            print(_status("cons_frac", d_cons, h1_margins["cons"], ok_cons), flush=True)
            print(_status("lift", d_lift, h1_margins["lift"], ok_lift), flush=True)
            print(_status("majority", d_maj, h1_margins["majority"], ok_maj), flush=True)
            print("  Verdict: Supported." if supported_h1 else "  Verdict: Not fully supported (see deltas above).", flush=True)
        else:
            print("\nHypothesis H1: skipped (MB query baseline not available; run without --skip_mb_queries).", flush=True)

        # --------------------------- H1b ---------------------------
        # H1b (candidate generation): even if KAHM(query→MB corpus) is not perfect at top-1,
        # it should be strong enough on recall-oriented metrics to act as a high-recall candidate
        # pool that is subsequently improved by the routed best-practice step.
        if "kahm_query_minus_mb" in comp:
            d_hit = _get_delta(comp, "kahm_query_minus_mb", "hit_at_k")
            d_mrr = _get_delta(comp, "kahm_query_minus_mb", "mrr_unique_law_at_k")
            d_top1 = _get_delta(comp, "kahm_query_minus_mb", "top1_accuracy")
            d_cons = _get_delta(comp, "kahm_query_minus_mb", "mean_consensus_fraction_in_topk")
            d_lift = _get_delta(comp, "kahm_query_minus_mb", "mean_lift")
            d_maj = _get_delta(comp, "kahm_query_minus_mb", "majority_accuracy")

            ok_hit = _noninferior(d_hit, h1_margins["hit"])
            ok_mrr = _noninferior(d_mrr, h1_margins["mrr"])
            ok_cons = _noninferior(d_cons, h1_margins["cons"])
            ok_lift = _noninferior(d_lift, h1_margins["lift"])
            ok_maj = _noninferior(d_maj, h1_margins["majority"])
            ok_top1 = _noninferior(d_top1, h1_margins["top1"])

            # Candidate generation does not require matching top-1; prioritize recall + law-level breadth.
            supported_h1b = ok_hit and ok_mrr and ok_cons and ok_lift

            print("\nHypothesis H1b: KAHM(query→MB corpus) is adequate for candidate generation", flush=True)
            print(f"  k = {pk}", flush=True)
            print(f"  test: {'two-sided equivalence' if equivalence_two_sided else 'one-sided non-inferiority'}", flush=True)
            print(
                "  margins (applied to recall-oriented metrics): "
                f"hit={h1_margins['hit']}, mrr={h1_margins['mrr']}, cons={h1_margins['cons']}, lift={h1_margins['lift']}",
                flush=True,
            )
            print(f"  hit@{pk}:                 KAHM(q→MB)−MB = {_fmt_delta(d_hit)}", flush=True)
            print(f"  MRR unique@{pk}:           KAHM(q→MB)−MB = {_fmt_delta(d_mrr)}", flush=True)
            print(f"  cons frac@{pk}:            KAHM(q→MB)−MB = {_fmt_delta(d_cons)}", flush=True)
            print(f"  lift@{pk} (prior):         KAHM(q→MB)−MB = {_fmt_delta(d_lift)}", flush=True)
            print(f"  (info) top1@{pk}:           KAHM(q→MB)−MB = {_fmt_delta(d_top1)}", flush=True)
            print(f"  (info) majority@{pk}:       KAHM(q→MB)−MB = {_fmt_delta(d_maj)}", flush=True)

            print("  checks:", flush=True)
            print(_status("hit", d_hit, h1_margins["hit"], ok_hit), flush=True)
            print(_status("mrr_unique", d_mrr, h1_margins["mrr"], ok_mrr), flush=True)
            print(_status("cons_frac", d_cons, h1_margins["cons"], ok_cons), flush=True)
            print(_status("lift", d_lift, h1_margins["lift"], ok_lift), flush=True)
            print("  Verdict: Supported." if supported_h1b else "  Verdict: Not fully supported (see deltas above).", flush=True)
            if supported_h1b and (not ok_top1):
                print("  Note: top1 did not meet the configured margin; this is expected for a candidate-generation role.", flush=True)
        else:
            print("\nHypothesis H1b: skipped (requires MB query baseline).", flush=True)

        # --------------------------- H1c ---------------------------
        # H1c (value-add): KAHM(best routed) should *improve* top-1 quality over plain KAHM query translation.
        if "kahm_best_minus_kahm_query" in comp:
            d_top1 = _get_delta(comp, "kahm_best_minus_kahm_query", "top1_accuracy")
            d_mrr = _get_delta(comp, "kahm_best_minus_kahm_query", "mrr_unique_law_at_k")

            ok_top1_sup = _superior(d_top1)
            ok_mrr_sup = _superior(d_mrr)
            supported_h1c = ok_top1_sup or ok_mrr_sup

            print("\nHypothesis H1c: KAHM(best routed) improves over KAHM(query→MB corpus)", flush=True)
            print(f"  k = {pk}", flush=True)
            print("  test: one-sided superiority (CI lower bound > 0)", flush=True)
            print(f"  top1@{pk}:                 KAHM(best)−KAHM(q→MB) = {_fmt_delta(d_top1)}", flush=True)
            print(f"  MRR unique@{pk}:           KAHM(best)−KAHM(q→MB) = {_fmt_delta(d_mrr)}", flush=True)
            print("  checks:", flush=True)
            print("    PASS top1_superiority" if ok_top1_sup else "    FAIL top1_superiority", flush=True)
            print("    PASS mrr_superiority" if ok_mrr_sup else "    FAIL mrr_superiority", flush=True)
            print("  Verdict: Supported." if supported_h1c else "  Verdict: Not fully supported (no superiority signal).", flush=True)
        else:
            print("\nHypothesis H1c: skipped (requires kahm_best_routed evaluation).", flush=True)

# --------------------------- H2 ---------------------------
        if "kahm_full_minus_mb" in comp and "kahm_full_minus_idf" in comp:
            d_hit_mb = _get_delta(comp, "kahm_full_minus_mb", "hit_at_k")
            d_cons_mb = _get_delta(comp, "kahm_full_minus_mb", "mean_consensus_fraction_in_topk")
            d_lift_mb = _get_delta(comp, "kahm_full_minus_mb", "mean_lift")
            d_maj_mb = _get_delta(comp, "kahm_full_minus_mb", "majority_accuracy")

            ok_hit2 = _noninferior(d_hit_mb, h2_margins["hit"])
            ok_cons2 = _noninferior(d_cons_mb, h2_margins["cons"])
            ok_lift2 = _noninferior(d_lift_mb, h2_margins["lift"])
            ok_maj2 = _noninferior(d_maj_mb, h2_margins["majority"])

            # Superiority vs IDF–SVD (at least one core signal)
            d_hit_idf = _get_delta(comp, "kahm_full_minus_idf", "hit_at_k")
            d_cons_idf = _get_delta(comp, "kahm_full_minus_idf", "mean_consensus_fraction_in_topk")
            d_maj_idf = _get_delta(comp, "kahm_full_minus_idf", "majority_accuracy")
            better_than_idf = _superior(d_hit_idf) or _superior(d_cons_idf) or _superior(d_maj_idf)

            supported_h2 = ok_hit2 and ok_cons2 and ok_lift2 and ok_maj2 and better_than_idf

            print("\nHypothesis H2: Full KAHM approx is good for candidate generation / semantic routing", flush=True)
            print(f"  k = {pk}", flush=True)
            print(f"  test: {'two-sided equivalence' if equivalence_two_sided else 'one-sided non-inferiority'}", flush=True)
            print(
                "  margins: "
                f"hit={h2_margins['hit']}, cons={h2_margins['cons']}, maj={h2_margins['majority']}, lift={h2_margins['lift']}",
                flush=True,
            )
            print(f"  hit@{pk}:                 KAHM(full)−MB = {_fmt_delta(d_hit_mb)}", flush=True)
            print(f"  cons frac@{pk}:            KAHM(full)−MB = {_fmt_delta(d_cons_mb)}", flush=True)
            print(f"  lift@{pk} (prior):         KAHM(full)−MB = {_fmt_delta(d_lift_mb)}", flush=True)
            print(f"  majority@{pk}:             KAHM(full)−MB = {_fmt_delta(d_maj_mb)}", flush=True)

            print("  checks:", flush=True)
            print(_status("hit", d_hit_mb, h2_margins["hit"], ok_hit2), flush=True)
            print(_status("cons_frac", d_cons_mb, h2_margins["cons"], ok_cons2), flush=True)
            print(_status("lift", d_lift_mb, h2_margins["lift"], ok_lift2), flush=True)
            print(_status("majority", d_maj_mb, h2_margins["majority"], ok_maj2), flush=True)
            print(
                "    PASS superiority_vs_idf (at least one routing signal has CI lower bound > 0)"
                if better_than_idf
                else "    FAIL superiority_vs_idf (no routing signal had CI lower bound > 0)",
                flush=True,
            )
            print("  Verdict: Supported." if supported_h2 else "  Verdict: Not fully supported (see deltas above).", flush=True)
            if better_than_idf:
                print(f"  Note: KAHM(full) is materially better than IDF–SVD at k={pk} on at least one routing signal.", flush=True)

            if "kahm_full_approx" in ev:
                _router_confidence_report(ev["kahm_full_approx"], title=f"KAHM(full) @k={pk}")
        else:
            print("\nHypothesis H2: skipped (full KAHM corpus evaluation not available; run without --skip_kahm_full).", flush=True)

def main() -> int:
    args = build_arg_parser().parse_args()

    print(f"Script: {os.path.basename(__file__)} | version={SCRIPT_VERSION} | path={os.path.abspath(__file__)}", flush=True)

    # Optional: cap FAISS threads
    try:
        import faiss  # type: ignore

        if int(args.faiss_threads) > 0:
            faiss.omp_set_num_threads(int(args.faiss_threads))
    except Exception as e:
        print(f"WARNING: could not set FAISS threads: {e}", file=sys.stderr, flush=True)

    # Load query set
    query_set = load_query_set(str(args.query_set_module), str(args.query_set_name))
    q_texts = [str(q.get("query_text", "")).strip() for q in query_set]

    print(f"Loaded query set: {args.query_set_module}.{args.query_set_name} (n={len(query_set)})", flush=True)

    # Load corpus embeddings
    print("\nLoading corpus embeddings ...", flush=True)
    ids_mb, emb_mb, meta_mb = load_npz_bundle(str(args.semantic_npz))
    ids_idf, emb_idf, meta_idf = load_npz_bundle(str(args.idf_svd_npz))

    # Align by sentence_id
    common_ids, emb_mb_a, emb_idf_a = align_by_sentence_id(ids_mb, emb_mb, ids_idf, emb_idf)
    print(f"Aligned corpora: common sentence_ids={common_ids.size}", flush=True)

    # Normalize corpus vectors
    emb_mb_a = l2_normalize_rows(emb_mb_a)
    emb_idf_a = l2_normalize_rows(emb_idf_a)
    assert_finite("emb_mb_a", emb_mb_a)
    assert_finite("emb_idf_a", emb_idf_a)
    summarize_vectors("emb_mb_a", emb_mb_a)
    summarize_vectors("emb_idf_a", emb_idf_a)

    # Load corpus info for scoring
    law_arr, page_arr, sent_arr = load_corpus_info(str(args.corpus), common_ids)

    # Build indices
    print("\nBuilding FAISS indices (true MB + IDF) ...", flush=True)
    index_true_mb = build_faiss_index(emb_mb_a, index_type=str(args.faiss_index), hnsw_m=int(args.hnsw_m), ef_search=int(args.hnsw_ef_search))
    index_idf = build_faiss_index(emb_idf_a, index_type=str(args.faiss_index), hnsw_m=int(args.hnsw_m), ef_search=int(args.hnsw_ef_search))

    # Load models
    print("\nLoading query embedding models ...", flush=True)
    idf_pipe = load_idf_svd_model(str(args.idf_svd_model))
    kahm_model = load_kahm_model(str(args.kahm_model))

    # Mixedbread query baseline: optional.
    # For strict "no MB model" setups, you can provide --mb_query_npz with precomputed MB query embeddings.
    embed_mixedbread = None
    q_true_mb_precomputed: Optional[np.ndarray] = None
    mb_query_npz_path = str(getattr(args, "mb_query_npz", "") or "").strip()
    if mb_query_npz_path:
        print(f"Loading precomputed Mixedbread query embeddings NPZ: {mb_query_npz_path}", flush=True)
        q_true_mb_precomputed = load_query_embeddings_npz(
            mb_query_npz_path,
            query_set=query_set,
            expected_dim=int(emb_mb_a.shape[1]),
        )
        assert_finite("q_true_mb(precomputed)", q_true_mb_precomputed)
        summarize_vectors("q_true_mb(precomputed)", q_true_mb_precomputed)
    elif not bool(getattr(args, "skip_mb_queries", False)):
        embed_mixedbread = build_mixedbread_embedder(
            str(args.mixedbread_model),
            device=str(args.device),
            dim=int(emb_mb_a.shape[1]),
        )
    else:
        print("NOTE: --skip_mb_queries enabled and --mb_query_npz not provided; Mixedbread query baseline will NOT be computed.", flush=True)

    # KAHM corpus embeddings (precomputed or on-the-fly) for full approximation evaluation.
    use_precomputed_kahm = bool(
        (not bool(getattr(args, "skip_kahm_full", False)))
        and args.kahm_corpus_npz
        and os.path.exists(str(args.kahm_corpus_npz))
    )
    # Ensure these are always defined for static type checkers.
    ids_k: Optional[np.ndarray] = None
    emb_k: Optional[np.ndarray] = None
    meta_k: Optional[Dict[str, Any]] = None


    if use_precomputed_kahm:
        print(f"\nLoading precomputed KAHM corpus NPZ: {args.kahm_corpus_npz}", flush=True)
        ids_k, emb_k, meta_k = load_npz_bundle(str(args.kahm_corpus_npz))

        policy = str(args.kahm_precomputed).lower().strip()
        if policy != "force":
            ok, reasons = kahm_npz_compatible(
                meta_k=meta_k,
                meta_idf=meta_idf,
                kahm_model_path=str(args.kahm_model),
                idf_svd_npz_path=str(args.idf_svd_npz),
                kahm_mode=str(args.kahm_mode),
            )
            if not ok:
                msg = " | ".join(reasons)
                if policy == "strict":
                    raise ValueError(
                        "Precomputed KAHM corpus NPZ appears incompatible with this run. "
                        f"Details: {msg}. "
                        "Regenerate the NPZ with precompute_kahm_corpus_npz.py or use --kahm_precomputed force."
                    )
                print(
                    "WARNING: Precomputed KAHM corpus NPZ appears incompatible; recomputing on the fly. "
                    f"Details: {msg}",
                    flush=True,
                )
                use_precomputed_kahm = False

    index_kahm_corpus = None
    emb_kahm_a = None
    if not bool(getattr(args, "skip_kahm_full", False)):
        if use_precomputed_kahm:
            assert ids_k is not None and emb_k is not None
            common2, emb_k_a, _ = align_by_sentence_id(ids_k, emb_k, common_ids, emb_mb_a)
            if common2.size != common_ids.size:
                print(
                    f"WARNING: KAHM corpus NPZ covers only {common2.size}/{common_ids.size} sentences. Using intersection.",
                    flush=True,
                )
                # Reduce all to intersection
                common_ids = common2
                emb_mb_a = subset_by_ids(ids_mb, emb_mb, common_ids)
                emb_idf_a = subset_by_ids(ids_idf, emb_idf, common_ids)
                law_arr, page_arr, sent_arr = load_corpus_info(str(args.corpus), common_ids)

                emb_mb_a = l2_normalize_rows(emb_mb_a)
                emb_idf_a = l2_normalize_rows(emb_idf_a)

                index_true_mb = build_faiss_index(emb_mb_a, index_type=str(args.faiss_index), hnsw_m=int(args.hnsw_m), ef_search=int(args.hnsw_ef_search))
                index_idf = build_faiss_index(emb_idf_a, index_type=str(args.faiss_index), hnsw_m=int(args.hnsw_m), ef_search=int(args.hnsw_ef_search))

            emb_kahm_a = l2_normalize_rows(emb_k_a)
        else:
            print("\nComputing KAHM corpus embeddings on the fly (IDF–SVD → approx Mixedbread) ...", flush=True)
            emb_kahm_a = kahm_regress_batched(
                kahm_model,
                emb_idf_a,
                mode=str(args.kahm_mode),
                batch_size=int(args.regress_batch),
                n_jobs=int(args.kahm_n_jobs),
                backend=str(args.kahm_backend),
            )

        assert emb_kahm_a is not None
        assert_finite("emb_kahm_a", emb_kahm_a)
        summarize_vectors("emb_kahm_a", emb_kahm_a)

        print("\nBuilding FAISS index (KAHM corpus approx) ...", flush=True)
        index_kahm_corpus = build_faiss_index(
            emb_kahm_a,
            index_type=str(args.faiss_index),
            hnsw_m=int(args.hnsw_m),
            ef_search=int(args.hnsw_ef_search),
        )
    else:
        print("NOTE: --skip_kahm_full enabled; full KAHM corpus index will NOT be built.", flush=True)

    # Encode queries
    q_true_mb: Optional[np.ndarray] = None
    if q_true_mb_precomputed is not None:
        q_true_mb = q_true_mb_precomputed
        print("\nUsing precomputed Mixedbread query embeddings (NPZ).", flush=True)
    elif embed_mixedbread is not None:
        print("\nEncoding queries with Mixedbread ...", flush=True)
        q_true_mb = embed_mixedbread(q_texts, batch_size=int(args.query_batch))
        assert_finite("q_true_mb", q_true_mb)
        summarize_vectors("q_true_mb", q_true_mb)
    else:
        print("\nSkipping Mixedbread query encoding (no MB query model loaded).", flush=True)

    print("\nEncoding queries with IDF–SVD ...", flush=True)
    q_idf = embed_queries_idf_svd(idf_pipe, q_texts)
    assert_finite("q_idf", q_idf)
    summarize_vectors("q_idf", q_idf)

    print("\nRegressing queries with KAHM ...", flush=True)
    q_kahm = kahm_regress_batched(
        kahm_model,
        q_idf,
        mode=str(args.kahm_mode),
        batch_size=max(1, int(args.regress_batch)),
        n_jobs=int(args.kahm_n_jobs),
        backend=str(args.kahm_backend),
    )
    assert_finite("q_kahm", q_kahm)
    summarize_vectors("q_kahm", q_kahm)

    # Evaluate
    print(f"\nEvaluating (top-{int(args.eval_k)}) ...", flush=True)
    eval_out = evaluate_on_query_set(
        query_set,
        k=int(args.eval_k),
        predominance_fraction=float(args.predominance_fraction),
        q_true_mb=q_true_mb,
        q_idf=q_idf,
        q_kahm=q_kahm,
        index_mb_corpus=index_true_mb,
        index_kahm_corpus=index_kahm_corpus,
        index_idf=index_idf,
        law_arr=law_arr,
        bootstrap=(not bool(getattr(args, 'no_bootstrap', False))),
        bootstrap_samples=int(getattr(args, 'bootstrap_samples', DEFAULT_BOOTSTRAP_SAMPLES)),
        bootstrap_seed=int(getattr(args, 'bootstrap_seed', DEFAULT_BOOTSTRAP_SEED)),
        kahm_best=(not bool(getattr(args, "no_kahm_best", False))),
        kahm_best_router_k=int(getattr(args, "kahm_best_router_k", DEFAULT_KAHM_BEST_ROUTER_K)),
        kahm_best_cand_k=int(getattr(args, "kahm_best_cand_k", DEFAULT_KAHM_BEST_CAND_K)),
        kahm_best_mode=str(getattr(args, "kahm_best_mode", DEFAULT_KAHM_BEST_MODE)),
        kahm_best_top_laws_grid=_parse_int_list(getattr(args, "kahm_best_top_laws_grid", DEFAULT_KAHM_BEST_TOP_LAWS_GRID), name="kahm_best_top_laws_grid"),
        kahm_best_law_slots_grid=_parse_int_list(getattr(args, "kahm_best_law_slots_grid", DEFAULT_KAHM_BEST_LAW_SLOTS_GRID), name="kahm_best_law_slots_grid"),
        kahm_best_lambda_grid=_parse_float_list(getattr(args, "kahm_best_lambda_grid", DEFAULT_KAHM_BEST_LAMBDA_GRID), name="kahm_best_lambda_grid"),
        kahm_best_min_coverage=float(getattr(args, "kahm_best_min_coverage", DEFAULT_ROUTER_MIN_COVERAGE)),
        kahm_best_optimize=str(getattr(args, "kahm_best_optimize", DEFAULT_KAHM_BEST_OPTIMIZE)),
        kahm_best_tune=(not bool(getattr(args, "no_kahm_best_tune", False))),
        router_cons_thresholds=_parse_float_list(getattr(args, "router_cons_thresholds", DEFAULT_ROUTER_CONS_THRESHOLDS), name="router_cons_thresholds"),
        router_lift_thresholds=_parse_float_list(getattr(args, "router_lift_thresholds", DEFAULT_ROUTER_LIFT_THRESHOLDS), name="router_lift_thresholds"),
    )
    print_eval_summary(eval_out)

    # Hypothesis testing configuration (margins + router grid) – parsed once
    h1_margins = {
        "hit": float(getattr(args, "h1_margin_hit", DEFAULT_H1_MARGIN_HIT)),
        "mrr": float(getattr(args, "h1_margin_mrr", DEFAULT_H1_MARGIN_MRR)),
        "top1": float(getattr(args, "h1_margin_top1", DEFAULT_H1_MARGIN_TOP1)),
        "cons": float(getattr(args, "h1_margin_cons", DEFAULT_H1_MARGIN_CONS)),
        "lift": float(getattr(args, "h1_margin_lift", DEFAULT_H1_MARGIN_LIFT)),
        "majority": float(getattr(args, "h1_margin_majority", DEFAULT_H1_MARGIN_MAJ)),
    }
    h2_margins = {
        "hit": float(getattr(args, "h2_margin_hit", DEFAULT_H2_MARGIN_HIT)),
        "cons": float(getattr(args, "h2_margin_cons", DEFAULT_H2_MARGIN_CONS)),
        "lift": float(getattr(args, "h2_margin_lift", DEFAULT_H2_MARGIN_LIFT)),
        "majority": float(getattr(args, "h2_margin_majority", DEFAULT_H2_MARGIN_MAJ)),
    }

    router_cons = _parse_float_list(
        getattr(args, "router_cons_thresholds", DEFAULT_ROUTER_CONS_THRESHOLDS),
        name="router_cons_thresholds",
    )
    router_lift = _parse_float_list(
        getattr(args, "router_lift_thresholds", DEFAULT_ROUTER_LIFT_THRESHOLDS),
        name="router_lift_thresholds",
    )
    if not router_cons:
        router_cons = _parse_float_list(DEFAULT_ROUTER_CONS_THRESHOLDS, name="router_cons_thresholds")
    if not router_lift:
        router_lift = _parse_float_list(DEFAULT_ROUTER_LIFT_THRESHOLDS, name="router_lift_thresholds")
    router_cons = sorted({float(x) for x in router_cons})
    router_lift = sorted({float(x) for x in router_lift})
    router_min_cov = float(getattr(args, "router_min_coverage", DEFAULT_ROUTER_MIN_COVERAGE))

    # k-grid evaluation + hypothesis blocks
    k_to_eval: Dict[int, Dict[str, Any]] = {int(args.eval_k): eval_out}

    if not bool(getattr(args, "no_kgrid", False)):
        ks = _parse_k_grid(getattr(args, "k_grid", None))
        primary_k = int(DEFAULT_PRIMARY_K_FOR_HYPOTHESIS)
        if primary_k not in ks:
            ks = sorted(set(ks + [primary_k]))
        k_to_eval = {int(args.eval_k): eval_out}
        for kk in ks:
            kk = int(kk)
            if kk in k_to_eval:
                continue
            k_to_eval[kk] = evaluate_on_query_set(
                query_set,
                k=kk,
                predominance_fraction=float(args.predominance_fraction),
                q_true_mb=q_true_mb,
                q_idf=q_idf,
                q_kahm=q_kahm,
                index_mb_corpus=index_true_mb,
                index_kahm_corpus=index_kahm_corpus,
                index_idf=index_idf,
                law_arr=law_arr,
                bootstrap=(not bool(getattr(args, "no_bootstrap", False))),
                bootstrap_samples=int(getattr(args, "bootstrap_samples", DEFAULT_BOOTSTRAP_SAMPLES)),
                bootstrap_seed=int(getattr(args, "bootstrap_seed", DEFAULT_BOOTSTRAP_SEED)),
                kahm_best=(not bool(getattr(args, "no_kahm_best", False))),
                kahm_best_router_k=int(getattr(args, "kahm_best_router_k", DEFAULT_KAHM_BEST_ROUTER_K)),
                kahm_best_cand_k=int(getattr(args, "kahm_best_cand_k", DEFAULT_KAHM_BEST_CAND_K)),
                kahm_best_mode=str(getattr(args, "kahm_best_mode", DEFAULT_KAHM_BEST_MODE)),
                kahm_best_top_laws_grid=_parse_int_list(getattr(args, "kahm_best_top_laws_grid", DEFAULT_KAHM_BEST_TOP_LAWS_GRID), name="kahm_best_top_laws_grid"),
                kahm_best_law_slots_grid=_parse_int_list(getattr(args, "kahm_best_law_slots_grid", DEFAULT_KAHM_BEST_LAW_SLOTS_GRID), name="kahm_best_law_slots_grid"),
                kahm_best_lambda_grid=_parse_float_list(getattr(args, "kahm_best_lambda_grid", DEFAULT_KAHM_BEST_LAMBDA_GRID), name="kahm_best_lambda_grid"),
                kahm_best_min_coverage=float(getattr(args, "kahm_best_min_coverage", DEFAULT_ROUTER_MIN_COVERAGE)),
                kahm_best_optimize=str(getattr(args, "kahm_best_optimize", DEFAULT_KAHM_BEST_OPTIMIZE)),
                kahm_best_tune=(not bool(getattr(args, "no_kahm_best_tune", False))),
                router_cons_thresholds=_parse_float_list(getattr(args, "router_cons_thresholds", DEFAULT_ROUTER_CONS_THRESHOLDS), name="router_cons_thresholds"),
                router_lift_thresholds=_parse_float_list(getattr(args, "router_lift_thresholds", DEFAULT_ROUTER_LIFT_THRESHOLDS), name="router_lift_thresholds"),
            )
        print_k_grid_summary(k_to_eval, ks=ks)
    else:
        print(
            "\nK-grid evaluation: skipped (--no_kgrid). Hypothesis blocks will use the nearest available k.",
            flush=True,
        )

    # Always print H1/H2 + routing threshold recommendation.
    # If k=20 was not evaluated, the nearest available k will be used.
    print_hypothesis_verdict(
        k_to_eval,
        primary_ks=[int(DEFAULT_PRIMARY_K_FOR_HYPOTHESIS), 20],
        equivalence_two_sided=bool(getattr(args, "equivalence_two_sided", False)),
        h1_margins=h1_margins,
        h2_margins=h2_margins,
        router_cons_thresholds=router_cons,
        router_lift_thresholds=router_lift,
        router_min_coverage=router_min_cov,
    )

    # Optional outputs
    if str(args.out_json).strip():
        import json

        out_path = str(args.out_json).strip()
        # Avoid dumping large internal per-query arrays by default
        out_clean = dict(eval_out)
        for _k in ("true_mixedbread", "kahm_query_on_mb_corpus", "kahm_full_approx", "pure_idf_svd"):
            if _k in out_clean and isinstance(out_clean[_k], dict):
                out_clean[_k] = dict(out_clean[_k])
                out_clean[_k].pop("_per_query", None)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_clean, f, ensure_ascii=False, indent=2)
        print(f"\nWrote summary JSON: {out_path}", flush=True)

    if str(args.per_query_csv).strip():
        # Per-query diagnostics at k=args.eval_k (law-level)
        out_csv = str(args.per_query_csv).strip()
        k_csv = int(args.eval_k)

        # MB baseline (optional)
        s_mb = i_mb = None
        if q_true_mb is not None:
            s_mb, i_mb = faiss_search(index_true_mb, q_true_mb, k_csv)

        # KAHM query translation (always available): q_KAHM against MB corpus index
        s_kq, i_kq = faiss_search(index_true_mb, q_kahm, k_csv)

        # Full KAHM approx (optional): q_KAHM against KAHM corpus index
        s_kf = i_kf = None
        if index_kahm_corpus is not None:
            s_kf, i_kf = faiss_search(index_kahm_corpus, q_kahm, k_csv)

        # IDF baseline
        s_i, i_i = faiss_search(index_idf, q_idf, k_csv)

        def _per_query_stats(row_idx: Optional[np.ndarray], cons: str) -> Tuple[int, float, int]:
            if row_idx is None:
                return 0, 0.0, 0
            laws = [str(law_arr[j]) for j in row_idx.tolist() if int(j) >= 0]
            if not laws or not cons:
                return 0, 0.0, 0
            c = Counter(laws)
            consfrac = float(c.get(cons, 0)) / float(max(1, len(laws)))
            hit = 1 if cons in c else 0
            # unique law rank (1-based); 0 if absent
            seen = set()
            uniq = []
            for lw in laws:
                if lw not in seen:
                    uniq.append(lw)
                    seen.add(lw)
            rank = (uniq.index(cons) + 1) if cons in seen else 0
            return hit, consfrac, rank

        rows: List[Dict[str, Any]] = []
        for qi, q in enumerate(query_set):
            qid = str(q.get("query_id", f"q{qi}"))
            qtext = str(q.get("query_text", "")).strip()
            cons = str(q.get("consensus_law", "")).strip()

            pred_mb = str(law_arr[int(i_mb[qi, 0])]) if (i_mb is not None and int(i_mb[qi, 0]) >= 0) else ""
            pred_kq = str(law_arr[int(i_kq[qi, 0])]) if int(i_kq[qi, 0]) >= 0 else ""
            pred_kf = str(law_arr[int(i_kf[qi, 0])]) if (i_kf is not None and int(i_kf[qi, 0]) >= 0) else ""
            pred_i = str(law_arr[int(i_i[qi, 0])]) if int(i_i[qi, 0]) >= 0 else ""

            hit_mb, consfrac_mb, rank_mb = _per_query_stats(i_mb[qi] if i_mb is not None else None, cons)
            hit_kq, consfrac_kq, rank_kq = _per_query_stats(i_kq[qi], cons)
            hit_kf, consfrac_kf, rank_kf = _per_query_stats(i_kf[qi] if i_kf is not None else None, cons)
            hit_i, consfrac_i, rank_i = _per_query_stats(i_i[qi], cons)

            rows.append(
                {
                    "query_id": qid,
                    "query_text": qtext,
                    "consensus_law": cons,
                    "k": k_csv,

                    "pred_law_true_mixedbread": pred_mb,
                    "pred_law_kahm_query_on_mb_corpus": pred_kq,
                    "pred_law_kahm_full_approx": pred_kf,
                    "pred_law_pure_idf_svd": pred_i,

                    "hit_at_k_true_mixedbread": hit_mb,
                    "hit_at_k_kahm_query_on_mb_corpus": hit_kq,
                    "hit_at_k_kahm_full_approx": hit_kf,
                    "hit_at_k_pure_idf_svd": hit_i,

                    "cons_frac_at_k_true_mixedbread": consfrac_mb,
                    "cons_frac_at_k_kahm_query_on_mb_corpus": consfrac_kq,
                    "cons_frac_at_k_kahm_full_approx": consfrac_kf,
                    "cons_frac_at_k_pure_idf_svd": consfrac_i,

                    "unique_law_rank_true_mixedbread": rank_mb,
                    "unique_law_rank_kahm_query_on_mb_corpus": rank_kq,
                    "unique_law_rank_kahm_full_approx": rank_kf,
                    "unique_law_rank_pure_idf_svd": rank_i,

                    "score_true_mixedbread": float(s_mb[qi, 0]) if s_mb is not None else float("nan"),
                    "score_kahm_query_on_mb_corpus": float(s_kq[qi, 0]),
                    "score_kahm_full_approx": float(s_kf[qi, 0]) if s_kf is not None else float("nan"),
                    "score_pure_idf_svd": float(s_i[qi, 0]),
                }
            )

        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"Wrote per-query CSV: {out_csv}", flush=True)


    return 0


if __name__ == "__main__":
    raise SystemExit(main())