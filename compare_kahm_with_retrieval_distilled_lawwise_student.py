#!/usr/bin/env python3
"""
compare_kahm_with_retrieval_distilled_lawwise_student.py

Purpose
-------
Compare the KAHM encoder against a more serious compact neural retrieval
alternative while preserving the same distributed-law training / serving
contract used in the existing neural-lawwise comparison script:

    cheap lexical query features (IDF–SVD) -> predicted Mixedbread query embedding

Added neural alternative
------------------------
This script adds one retrieval-aware compact neural student:

1) retrieval_distilled_student_lawwise
   A compact law-wise residual MLP student that maps IDF–SVD query features to
   the frozen Mixedbread embedding space, but is trained with a *retrieval*
   distillation objective rather than only pointwise embedding regression.

   For each law q, the student is trained so that its similarity distribution
   over that law's frozen Mixedbread corpus embeddings matches the teacher
   query embedding's similarity distribution over the same law-local corpus.
   Besides the full-corpus distillation term, the student can optionally place
   extra weight on the teacher's top-ranked law-local corpus items. Auxiliary
   cosine / MSE alignment to the teacher query embedding is included for
   stability, together with gradient clipping and cosine learning-rate decay.

Why this is a stronger reviewer-facing comparator
-------------------------------------------------
The existing compact neural baselines in compare_kahm_with_neural_lawwise_baselines.py
are direct regressors or prototype-mixture controls. This script adds a more
retrieval-aware compact student without changing the deployment contract, the
law-wise split, or the distance-gated multi-law serving setup.

The comparison therefore addresses the narrower question:

    "Does KAHM still outperform a compact law-wise neural student that is
     explicitly trained to mimic teacher-side retrieval behavior over the
     frozen law-local corpus?"

The direct Mixedbread(true) reference remains the oracle teacher-space query
representation and is not replaced by this student.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Keep thread behavior predictable and broadly aligned with the existing scripts.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def as_float_ndarray(x: Any) -> np.ndarray:
    if hasattr(x, "toarray"):
        x = x.toarray()
    x = np.asarray(x)
    if x.dtype.kind in ("i", "u", "b"):
        return x.astype(np.float64, copy=False)
    if x.dtype.kind != "f":
        return x.astype(np.float32, copy=False)
    if x.dtype.itemsize < np.dtype(np.float32).itemsize:
        return x.astype(np.float32, copy=False)
    return x


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = as_float_ndarray(x)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array; got shape={x.shape}")
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def pairwise_min_sqdist(a: np.ndarray, b: np.ndarray, *, block: int = 2048) -> np.ndarray:
    """Return min_j ||a_i - b_j||^2 for each row i in a."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"Expected 2D arrays; got {a.shape} and {b.shape}")
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"Dimension mismatch: {a.shape} vs {b.shape}")
    if b.shape[0] == 0:
        return np.full((a.shape[0],), np.inf, dtype=np.float32)

    b_norm = np.sum(b * b, axis=1)[None, :]
    out = np.empty((a.shape[0],), dtype=np.float32)
    for start in range(0, a.shape[0], block):
        end = min(a.shape[0], start + block)
        ai = a[start:end]
        ai_norm = np.sum(ai * ai, axis=1, keepdims=True)
        d2 = ai_norm + b_norm - 2.0 * (ai @ b.T)
        np.maximum(d2, 0.0, out=d2)
        out[start:end] = np.min(d2, axis=1).astype(np.float32, copy=False)
    return out


def compute_embedding_metrics(y_pred_col: np.ndarray, y_true_col: np.ndarray) -> Dict[str, float]:
    if y_pred_col.ndim != 2 or y_true_col.ndim != 2:
        raise ValueError(f"Expected 2D arrays; got {y_pred_col.shape}, {y_true_col.shape}")
    if y_pred_col.shape != y_true_col.shape:
        raise ValueError(f"Shape mismatch: {y_pred_col.shape} vs {y_true_col.shape}")

    diff = y_pred_col - y_true_col
    mse = float(np.mean(diff * diff))

    y = y_true_col.reshape(-1)
    y_mean = float(np.mean(y))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    ss_res = float(np.sum(diff.reshape(-1) ** 2))
    r2_overall = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")

    num = np.einsum("dn,dn->n", y_pred_col, y_true_col)
    den = np.linalg.norm(y_pred_col, axis=0) * np.linalg.norm(y_true_col, axis=0)
    cos = num / np.maximum(den, 1e-12)

    return {
        "mse": mse,
        "r2_overall": r2_overall,
        "cos_mean": float(np.mean(cos)),
        "cos_p10": float(np.percentile(cos, 10)),
        "cos_p50": float(np.percentile(cos, 50)),
        "cos_p90": float(np.percentile(cos, 90)),
        "n": int(y_true_col.shape[1]),
        "d": int(y_true_col.shape[0]),
    }


def _bootstrap_mean_ci(x: np.ndarray, *, n_boot: int, seed: int) -> Tuple[float, Tuple[float, float]]:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan"), (float("nan"), float("nan"))
    rng = np.random.default_rng(int(seed))
    n = int(x.size)
    pt = float(np.mean(x))
    bs = np.empty(int(n_boot), dtype=np.float64)
    for b in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        bs[b] = float(np.mean(x[idx]))
    lo, hi = np.quantile(bs, [0.025, 0.975])
    return pt, (float(lo), float(hi))


def _bootstrap_paired_delta_ci(a: np.ndarray, b: np.ndarray, *, n_boot: int, seed: int) -> Tuple[float, Tuple[float, float]]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Paired arrays must have same shape; got {a.shape} vs {b.shape}")
    if a.size == 0:
        return float("nan"), (float("nan"), float("nan"))
    rng = np.random.default_rng(int(seed))
    n = int(a.size)
    d = a - b
    pt = float(np.mean(d))
    bs = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        bs[i] = float(np.mean(d[idx]))
    lo, hi = np.quantile(bs, [0.025, 0.975])
    return pt, (float(lo), float("nan") if math.isnan(hi) else float(hi))


def _fmt_ci(pt: float, ci: Tuple[float, float], digits: int = 3) -> str:
    return f"{pt:.{digits}f} [{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"


def _fmt_delta(pt: float, ci: Tuple[float, float], digits: int = 3) -> str:
    return f"{pt:+.{digits}f} [{ci[0]:+.{digits}f}, {ci[1]:+.{digits}f}]"


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([head, sep] + body)


def parse_hidden_sizes(spec: str) -> Tuple[int, ...]:
    vals = [int(x.strip()) for x in str(spec).split(",") if x.strip()]
    if not vals:
        raise ValueError(f"Empty hidden-size spec: {spec!r}")
    if any(v <= 0 for v in vals):
        raise ValueError(f"Hidden sizes must be positive: {spec!r}")
    return tuple(vals)


def safe_validation_fraction(n_samples: int, requested: float) -> Optional[float]:
    """Return a safe validation fraction for early stopping or None to disable it."""
    if n_samples < 20:
        return None
    vf = float(requested)
    vf = min(max(vf, 0.05), 0.3)
    n_val = int(round(n_samples * vf))
    if n_val < 2 or (n_samples - n_val) < 10:
        return None
    return vf


# -----------------------------------------------------------------------------
# Query-set loading
# -----------------------------------------------------------------------------
def load_query_set(module_attr: str) -> List[Dict[str, Any]]:
    if "." not in module_attr:
        raise ValueError("query-set spec must be module.attr, e.g. query_set.TEST_QUERY_SET")
    mod_name, attr = module_attr.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    qs = getattr(mod, attr, None)
    if qs is None:
        raise AttributeError(f"Query-set attribute not found: {module_attr}")
    out = list(qs)
    if not out:
        raise ValueError(f"Loaded empty query set from {module_attr}")
    return out


def extract_ids_texts_laws(qs: Sequence[Mapping[str, Any]], *, name: str) -> Tuple[List[str], List[str], List[str]]:
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


# -----------------------------------------------------------------------------
# Embedding loading
# -----------------------------------------------------------------------------
def embed_idf_svd_queries(idf_svd_model_path: str, texts: Sequence[str]) -> np.ndarray:
    import joblib

    pipe = joblib.load(idf_svd_model_path)
    x = pipe.transform(list(texts))
    x = as_float_ndarray(x)
    x = l2_normalize_rows(x)
    return x.astype(np.float32, copy=False)


def load_query_npz(path: str, query_ids: Sequence[str]) -> np.ndarray:
    d = np.load(path, allow_pickle=False)
    if "query_id" not in d or "embeddings" not in d:
        raise ValueError(f"Query NPZ '{path}' must contain keys 'query_id' and 'embeddings'. Keys: {list(d.keys())}")
    qid_npz = np.asarray(d["query_id"])
    y_npz = as_float_ndarray(d["embeddings"])
    if qid_npz.ndim != 1 or y_npz.ndim != 2:
        raise ValueError(f"Query NPZ '{path}' has invalid shapes: {qid_npz.shape}, {y_npz.shape}")
    pos = {str(qid_npz[i]): i for i in range(qid_npz.shape[0])}
    missing = [qid for qid in query_ids if qid not in pos]
    if missing:
        raise ValueError(f"Query NPZ '{path}' missing {len(missing)} query_ids. Example: {missing[:10]}")
    y = np.vstack([y_npz[pos[qid]] for qid in query_ids])
    return l2_normalize_rows(y).astype(np.float32, copy=False)


def load_npz_bundle(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    keys = set(data.files)

    sid_key = None
    for k in ("sentence_ids", "ids", "sentence_id"):
        if k in keys:
            sid_key = k
            break
    emb_key = None
    for k in ("embeddings", "embedding", "X", "emb"):
        if k in keys:
            emb_key = k
            break

    if sid_key is None or emb_key is None:
        raise ValueError(f"Unsupported NPZ schema in {path}. Keys: {sorted(keys)}")

    sentence_ids = np.asarray(data[sid_key], dtype=np.int64)
    emb = np.asarray(data[emb_key], dtype=np.float32)
    if sentence_ids.ndim != 1 or emb.ndim != 2:
        raise ValueError(f"Invalid corpus NPZ shapes in {path}: ids={sentence_ids.shape}, emb={emb.shape}")
    if emb.shape[0] != sentence_ids.shape[0]:
        raise ValueError(f"Row mismatch in {path}: emb={emb.shape[0]} ids={sentence_ids.shape[0]}")
    return {"sentence_ids": sentence_ids, "emb": l2_normalize_rows(emb).astype(np.float32, copy=False)}


# -----------------------------------------------------------------------------
# Retrieval helpers
# -----------------------------------------------------------------------------
def load_corpus_parquet(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Corpus parquet not found: {path}")
    df = pd.read_parquet(path)
    required = {"sentence_id", "law_type"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Corpus parquet missing columns: {sorted(missing)}")
    ids = df["sentence_id"].astype(np.int64).to_numpy()
    if np.unique(ids).size != ids.size:
        raise ValueError("Corpus parquet has duplicate sentence_id values")
    return df


def align_by_common_sentence_ids(df: pd.DataFrame, mb: Dict[str, np.ndarray], idf: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    s_df = df["sentence_id"].astype(np.int64).to_numpy()
    s_mb = mb["sentence_ids"].astype(np.int64)
    s_idf = idf["sentence_ids"].astype(np.int64)

    common = np.intersect1d(np.intersect1d(s_df, s_mb), s_idf)
    if common.size == 0:
        raise ValueError("No common sentence_ids across df/MB/IDF bundles")

    def _subset(ids: np.ndarray, emb: np.ndarray, common_ids: np.ndarray) -> np.ndarray:
        pos = {int(s): i for i, s in enumerate(ids.tolist())}
        idx = np.asarray([pos[int(s)] for s in common_ids.tolist()], dtype=np.int64)
        return emb[idx]

    emb_mb = _subset(s_mb, mb["emb"], common)
    emb_idf = _subset(s_idf, idf["emb"], common)

    pos_df = {int(s): i for i, s in enumerate(s_df.tolist())}
    df_idx = np.asarray([pos_df[int(s)] for s in common.tolist()], dtype=np.int64)
    law = df.iloc[df_idx]["law_type"].astype(str).to_numpy()

    return {
        "sentence_ids": common,
        "law": law,
        "emb_mb": emb_mb,
        "emb_idf": emb_idf,
    }


def build_faiss_index(emb: np.ndarray, *, n_threads: Optional[int] = None):
    import faiss  # type: ignore

    if n_threads is not None:
        try:
            faiss.omp_set_num_threads(int(n_threads))
        except Exception:
            pass
    x = np.ascontiguousarray(emb.astype(np.float32, copy=False))
    index: Any = faiss.IndexFlatIP(int(x.shape[1]))
    index.add(x)
    return index


def faiss_search(index: Any, q_emb: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    q = np.ascontiguousarray(q_emb.astype(np.float32, copy=False))
    scores, idx = index.search(q, int(k))
    return scores, idx


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
@dataclass
class PerQuery:
    hit: np.ndarray
    top1: np.ndarray
    majority: np.ndarray
    cons_frac: np.ndarray
    lift: np.ndarray
    mrr_ul: np.ndarray


def _majority_law_tiebreak(laws: List[str], counts: Dict[str, int]) -> Tuple[str, int]:
    if not laws or not counts:
        return "", 0
    max_count = max(int(v) for v in counts.values())
    candidates = [lw for lw, cnt in counts.items() if int(cnt) == max_count]
    if len(candidates) == 1:
        return candidates[0], max_count
    first_pos: Dict[str, int] = {}
    for pos, lw in enumerate(laws):
        if lw not in first_pos:
            first_pos[lw] = int(pos)
    chosen = min(candidates, key=lambda lw: first_pos.get(lw, 10**9))
    return chosen, max_count


def compute_per_query_metrics(*, idx: np.ndarray, law_arr: np.ndarray, consensus_laws: List[str], k: int, predominance_fraction: float) -> PerQuery:
    from collections import Counter

    k = int(k)
    pred_frac = float(predominance_fraction)
    c_all = Counter([str(x) for x in law_arr.tolist()])
    total = float(max(1, int(law_arr.size)))
    prior = {lw: float(cnt) / total for lw, cnt in c_all.items()}

    idx = np.asarray(idx, dtype=np.int64)
    if idx.ndim != 2:
        raise ValueError(f"idx must be 2D; got {idx.shape}")
    if idx.shape[1] < k:
        raise ValueError(f"idx has too few columns: {idx.shape[1]} < k={k}")
    if idx.shape[1] > k:
        idx = idx[:, :k]

    n = int(idx.shape[0])
    if len(consensus_laws) != n:
        raise ValueError(f"consensus_laws length {len(consensus_laws)} != n_queries {n}")

    hit_v = np.zeros(n, dtype=np.float64)
    top1_v = np.zeros(n, dtype=np.float64)
    maj_v = np.zeros(n, dtype=np.float64)
    cf_v = np.zeros(n, dtype=np.float64)
    lift_v = np.zeros(n, dtype=np.float64)
    mrr_v = np.zeros(n, dtype=np.float64)

    for i in range(n):
        cons = str(consensus_laws[i]).strip()
        row = [int(j) for j in idx[i].tolist() if int(j) >= 0]
        laws = [str(law_arr[j]) for j in row]

        hit_v[i] = 1.0 if (cons in laws) else 0.0
        top1_v[i] = 1.0 if (laws and laws[0] == cons) else 0.0

        counts = Counter(laws)
        maj_law, maj_count = _majority_law_tiebreak(laws, dict(counts))
        maj_frac = float(maj_count) / float(max(1, len(laws)))
        maj_v[i] = 1.0 if (maj_law == cons and maj_frac >= pred_frac) else 0.0

        cons_frac = float(counts.get(cons, 0)) / float(max(1, len(laws)))
        cf_v[i] = cons_frac
        cons_prior = float(prior.get(cons, 0.0))
        lift_v[i] = (cons_frac / cons_prior) if cons_prior > 0 else 0.0

        seen = set()
        uniq: List[str] = []
        for lw in laws:
            if lw not in seen:
                uniq.append(lw)
                seen.add(lw)
        try:
            rank = uniq.index(cons) + 1
            mrr_v[i] = 1.0 / float(rank)
        except ValueError:
            mrr_v[i] = 0.0

    return PerQuery(hit=hit_v, top1=top1_v, majority=maj_v, cons_frac=cf_v, lift=lift_v, mrr_ul=mrr_v)


# -----------------------------------------------------------------------------
# KAHM loading
# -----------------------------------------------------------------------------
def load_kahm_model(path: str) -> dict:
    import joblib

    obj = joblib.load(path)
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    raise TypeError(f"Unsupported KAHM model object type in {path}: {type(obj)!r}")


def load_kahm_models_from_dir(dir_path: str) -> Dict[str, dict]:
    d = Path(str(dir_path)).expanduser()
    if not d.exists() or not d.is_dir():
        raise FileNotFoundError(f"KAHM model directory not found: {dir_path}")
    paths = sorted(d.glob("*.joblib"))
    if not paths:
        raise FileNotFoundError(f"No *.joblib models found in directory: {dir_path}")
    models: Dict[str, dict] = {}
    for fp in paths:
        models[fp.stem] = load_kahm_model(str(fp))
    return models


def kahm_regress_distance_gated_multi_models(
    x_row: np.ndarray,
    *,
    models: Mapping[str, dict] | Sequence[dict],
    mode: str,
    batch_size: int,
    tie_break: str = "first",
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    try:
        from combine_kahm_regressors_generalized_fast import combine_kahm_regressors_distance_gated_multi  # type: ignore
    except Exception:
        from combine_kahm_regressors_generalized import combine_kahm_regressors_distance_gated_multi  # type: ignore
    models_for_call = dict(models) if isinstance(models, Mapping) else list(models)
    y, chosen, best_score, _all_scores, names = combine_kahm_regressors_distance_gated_multi(
        x_row,
        models=models_for_call,
        input_layout="row",
        output_layout="row",
        mode=str(mode),
        batch_size=int(batch_size),
        tie_break=str(tie_break),
        show_progress=bool(show_progress),
        return_all_scores=False,
    )
    y = l2_normalize_rows(np.asarray(y, dtype=np.float32))
    return y, np.asarray(chosen), np.asarray(best_score, dtype=np.float32), list(names)


# -----------------------------------------------------------------------------
# Retrieval-distilled neural student
# -----------------------------------------------------------------------------
@dataclass
class RetrievalStudentLawModel:
    law: str
    state_dict: Dict[str, Any]
    input_dim: int
    output_dim: int
    hidden_sizes: Tuple[int, ...]
    dropout: float
    prototypes: np.ndarray
    corpus_embeddings: np.ndarray


@dataclass
class PredictionBundle:
    embeddings: np.ndarray
    chosen_model_names: List[str]
    gating_scores: np.ndarray


def make_semantic_prototypes(y_train: np.ndarray, *, n_prototypes: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    y_train = l2_normalize_rows(np.asarray(y_train, dtype=np.float32))
    n = int(y_train.shape[0])
    c = max(1, min(int(n_prototypes), n))
    if c == 1:
        center = l2_normalize_rows(np.mean(y_train, axis=0, keepdims=True)).astype(np.float32, copy=False)
        labels = np.zeros((n,), dtype=np.int32)
        return center, labels
    km = KMeans(n_clusters=c, n_init=10, random_state=int(seed))
    labels = km.fit_predict(y_train)
    centers = l2_normalize_rows(np.asarray(km.cluster_centers_, dtype=np.float32))
    return centers, np.asarray(labels, dtype=np.int32)


def _resolve_device(requested: str) -> str:
    import torch

    req = str(requested).strip().lower()
    if req == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if req == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    if req not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {requested!r}")
    return req


def _numpy_to_torch(x: np.ndarray, device: str):
    import torch

    return torch.as_tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32, device=device)


class CompactResidualStudent:
    """Small residual MLP with a linear skip path into the teacher space."""

    def __init__(self, input_dim: int, hidden_sizes: Tuple[int, ...], output_dim: int, dropout: float):
        import torch
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self, in_dim: int, hidden: Tuple[int, ...], out_dim: int, p_drop: float):
                super().__init__()
                if not hidden:
                    raise ValueError("hidden_sizes must not be empty")
                blocks: List[nn.Module] = []
                prev = int(in_dim)
                for h in hidden:
                    blocks.append(nn.Linear(prev, int(h)))
                    blocks.append(nn.LayerNorm(int(h)))
                    blocks.append(nn.GELU())
                    if float(p_drop) > 0:
                        blocks.append(nn.Dropout(float(p_drop)))
                    prev = int(h)
                self.backbone = nn.Sequential(*blocks)
                self.head = nn.Linear(prev, int(out_dim))
                self.skip = nn.Linear(int(in_dim), int(out_dim), bias=False)

            def forward(self, x):
                import torch.nn.functional as F

                y = self.head(self.backbone(x)) + self.skip(x)
                y = F.normalize(y, p=2, dim=-1)
                return y

        self.module = _Net(input_dim, hidden_sizes, output_dim, dropout)

    def to(self, device: str):
        self.module.to(device)
        return self

    def parameters(self):
        return self.module.parameters()

    def train(self):
        self.module.train()

    def eval(self):
        self.module.eval()

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.module.load_state_dict(state_dict)

    def __call__(self, x):
        return self.module(x)


def _distillation_loss(
    student_q,
    teacher_q,
    corpus_law,
    *,
    temperature: float,
    kl_weight: float,
    cos_weight: float,
    mse_weight: float,
    focus_topk: int = 0,
    focus_weight: float = 0.0,
):
    import torch
    import torch.nn.functional as F

    t = float(temperature)
    student_logits = (student_q @ corpus_law.T) / t
    with torch.no_grad():
        teacher_logits = (teacher_q @ corpus_law.T) / t
        teacher_probs = F.softmax(teacher_logits, dim=1)

    kl = F.kl_div(F.log_softmax(student_logits, dim=1), teacher_probs, reduction="batchmean")

    focus_kl = student_logits.new_zeros(())
    k = int(focus_topk)
    if float(focus_weight) > 0.0 and k > 0 and student_logits.shape[1] > 1:
        k = min(k, int(student_logits.shape[1]))
        with torch.no_grad():
            top_idx = torch.topk(teacher_logits, k=k, dim=1, largest=True, sorted=False).indices
        student_focus = torch.gather(student_logits, 1, top_idx)
        teacher_focus_logits = torch.gather(teacher_logits, 1, top_idx)
        teacher_focus_probs = F.softmax(teacher_focus_logits, dim=1)
        focus_kl = F.kl_div(F.log_softmax(student_focus, dim=1), teacher_focus_probs, reduction="batchmean")

    cos = 1.0 - F.cosine_similarity(student_q, teacher_q, dim=1).mean()
    mse = F.mse_loss(student_q, teacher_q)
    loss = (
        float(kl_weight) * kl
        + float(focus_weight) * focus_kl
        + float(cos_weight) * cos
        + float(mse_weight) * mse
    )
    return loss, {
        "kl": float(kl.detach().cpu().item()),
        "focus_kl": float(focus_kl.detach().cpu().item()),
        "cos": float(cos.detach().cpu().item()),
        "mse": float(mse.detach().cpu().item()),
    }


def _run_student_batches(model: CompactResidualStudent, x_np: np.ndarray, *, batch_size: int, device: str) -> np.ndarray:
    import torch

    x_np = np.asarray(x_np, dtype=np.float32)
    outs: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, x_np.shape[0], int(batch_size)):
            end = min(x_np.shape[0], start + int(batch_size))
            xb = _numpy_to_torch(x_np[start:end], device)
            yb = model(xb).detach().cpu().numpy().astype(np.float32, copy=False)
            outs.append(yb)
    return np.vstack(outs) if outs else np.empty((0, 0), dtype=np.float32)


def train_retrieval_distilled_student_lawwise(
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_laws: Sequence[str],
    corpus_emb_mb: np.ndarray,
    corpus_laws: Sequence[str],
    *,
    n_prototypes: int,
    seed: int,
    hidden_sizes: Tuple[int, ...],
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    temperature: float,
    kl_weight: float,
    cos_weight: float,
    mse_weight: float,
    focus_topk: int,
    focus_weight: float,
    grad_clip: float,
    lr_min_factor: float,
    validation_fraction: float,
    patience: int,
    device: str,
    log_every: int = 10,
) -> Dict[str, RetrievalStudentLawModel]:
    import torch

    rng = np.random.default_rng(int(seed))
    dev = _resolve_device(device)

    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = l2_normalize_rows(np.asarray(y_train, dtype=np.float32))
    train_laws_arr = np.asarray([str(lw) for lw in train_laws], dtype=object)
    corpus_laws_arr = np.asarray([str(lw) for lw in corpus_laws], dtype=object)

    laws = sorted(set(train_laws_arr.tolist()))
    models: Dict[str, RetrievalStudentLawModel] = {}

    for law in laws:
        q_mask = train_laws_arr == law
        c_mask = corpus_laws_arr == law

        x_l = np.asarray(x_train[q_mask], dtype=np.float32)
        y_l = np.asarray(y_train[q_mask], dtype=np.float32)
        corpus_l = l2_normalize_rows(np.asarray(corpus_emb_mb[c_mask], dtype=np.float32))

        if x_l.shape[0] == 0:
            continue
        if corpus_l.shape[0] == 0:
            raise ValueError(f"No law-local corpus embeddings found for law={law!r}")

        n = int(x_l.shape[0])
        val_frac = safe_validation_fraction(n, validation_fraction)
        order = np.arange(n, dtype=np.int64)
        rng.shuffle(order)

        if val_frac is None:
            train_idx = order
            val_idx = np.asarray([], dtype=np.int64)
        else:
            n_val = max(1, int(round(n * float(val_frac))))
            n_val = min(n_val, max(1, n - 10))
            val_idx = order[:n_val]
            train_idx = order[n_val:]
            if train_idx.size == 0:
                train_idx = order
                val_idx = np.asarray([], dtype=np.int64)

        model = CompactResidualStudent(
            input_dim=int(x_l.shape[1]),
            hidden_sizes=hidden_sizes,
            output_dim=int(y_l.shape[1]),
            dropout=float(dropout),
        ).to(dev)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(epochs)),
            eta_min=float(learning_rate) * max(0.0, float(lr_min_factor)),
        )

        x_train_t = _numpy_to_torch(x_l[train_idx], dev)
        y_train_t = _numpy_to_torch(y_l[train_idx], dev)
        corpus_t = _numpy_to_torch(corpus_l, dev)

        if val_idx.size:
            x_val_t = _numpy_to_torch(x_l[val_idx], dev)
            y_val_t = _numpy_to_torch(y_l[val_idx], dev)
        else:
            x_val_t = None
            y_val_t = None

        best_state = copy.deepcopy(model.state_dict())
        best_val = float("inf")
        stale = 0

        for epoch in range(1, int(epochs) + 1):
            model.train()
            perm = torch.randperm(x_train_t.shape[0], device=dev)
            epoch_losses: List[float] = []

            for start in range(0, int(x_train_t.shape[0]), int(batch_size)):
                batch_ids = perm[start : start + int(batch_size)]
                xb = x_train_t[batch_ids]
                yb = y_train_t[batch_ids]

                optimizer.zero_grad(set_to_none=True)
                pred = model(xb)
                loss, _parts = _distillation_loss(
                    pred,
                    yb,
                    corpus_t,
                    temperature=float(temperature),
                    kl_weight=float(kl_weight),
                    cos_weight=float(cos_weight),
                    mse_weight=float(mse_weight),
                    focus_topk=int(focus_topk),
                    focus_weight=float(focus_weight),
                )
                loss.backward()
                if float(grad_clip) > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu().item()))

            model.eval()
            with torch.no_grad():
                if x_val_t is not None and y_val_t is not None and x_val_t.shape[0] > 0:
                    val_pred = model(x_val_t)
                    val_loss, val_parts = _distillation_loss(
                        val_pred,
                        y_val_t,
                        corpus_t,
                        temperature=float(temperature),
                        kl_weight=float(kl_weight),
                        cos_weight=float(cos_weight),
                        mse_weight=float(mse_weight),
                        focus_topk=int(focus_topk),
                        focus_weight=float(focus_weight),
                    )
                    monitor = float(val_loss.detach().cpu().item())
                else:
                    train_pred = model(x_train_t)
                    val_loss, val_parts = _distillation_loss(
                        train_pred,
                        y_train_t,
                        corpus_t,
                        temperature=float(temperature),
                        kl_weight=float(kl_weight),
                        cos_weight=float(cos_weight),
                        mse_weight=float(mse_weight),
                        focus_topk=int(focus_topk),
                        focus_weight=float(focus_weight),
                    )
                    monitor = float(val_loss.detach().cpu().item())

            if epoch == 1 or epoch % max(1, int(log_every)) == 0 or epoch == int(epochs):
                current_lr = float(optimizer.param_groups[0]["lr"])
                print(
                    f"[retrieval_student][{law}] "
                    f"epoch={epoch:03d}/{int(epochs):03d} "
                    f"train={np.mean(epoch_losses):.6f} "
                    f"monitor={monitor:.6f} "
                    f"lr={current_lr:.6e} "
                    f"kl={val_parts['kl']:.6f} focus={val_parts['focus_kl']:.6f} "
                    f"cos={val_parts['cos']:.6f} mse={val_parts['mse']:.6f}"
                )

            if monitor + 1e-8 < best_val:
                best_val = monitor
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
                if int(patience) > 0 and stale >= int(patience):
                    break

            scheduler.step()

        model.load_state_dict(best_state)
        prototypes, _labels = make_semantic_prototypes(y_l, n_prototypes=n_prototypes, seed=seed)

        models[law] = RetrievalStudentLawModel(
            law=law,
            state_dict=copy.deepcopy(model.state_dict()),
            input_dim=int(x_l.shape[1]),
            output_dim=int(y_l.shape[1]),
            hidden_sizes=hidden_sizes,
            dropout=float(dropout),
            prototypes=prototypes.astype(np.float32, copy=False),
            corpus_embeddings=corpus_l.astype(np.float32, copy=False),
        )

    return models


def predict_retrieval_distilled_student_lawwise(
    models: Mapping[str, RetrievalStudentLawModel],
    x_test: np.ndarray,
    *,
    batch_size: int,
    device: str,
) -> PredictionBundle:
    names = sorted(models.keys())
    if not names:
        raise ValueError("No retrieval-distilled student models available.")

    dev = _resolve_device(device)
    pred_by_name: Dict[str, np.ndarray] = {}
    score_by_name: Dict[str, np.ndarray] = {}

    for name in names:
        model_info = models[name]
        net = CompactResidualStudent(
            input_dim=int(model_info.input_dim),
            hidden_sizes=tuple(model_info.hidden_sizes),
            output_dim=int(model_info.output_dim),
            dropout=float(model_info.dropout),
        ).to(dev)
        net.load_state_dict(model_info.state_dict)
        y_hat = _run_student_batches(net, x_test, batch_size=int(batch_size), device=dev)
        y_hat = l2_normalize_rows(y_hat).astype(np.float32, copy=False)
        pred_by_name[name] = y_hat
        score_by_name[name] = pairwise_min_sqdist(y_hat, model_info.prototypes)

    all_scores = np.vstack([score_by_name[name] for name in names])
    chosen = np.argmin(all_scores, axis=0).astype(np.int64)
    best_score = all_scores[chosen, np.arange(x_test.shape[0])]
    y_best = np.vstack([pred_by_name[names[int(j)]][i] for i, j in enumerate(chosen)]).astype(np.float32, copy=False)
    chosen_names = [names[int(j)] for j in chosen.tolist()]
    return PredictionBundle(embeddings=y_best, chosen_model_names=chosen_names, gating_scores=best_score.astype(np.float32, copy=False))


# -----------------------------------------------------------------------------
# Reporting helpers
# -----------------------------------------------------------------------------
def summarize_methods(
    *,
    per_query: Dict[str, PerQuery],
    ks: Sequence[int],
    methods: Sequence[str],
    bootstrap: int,
    seed: int,
    baseline_ref: str,
) -> Dict[str, Any]:
    metric_keys = ["hit", "mrr_ul", "top1", "majority", "cons_frac", "lift"]
    out: Dict[str, Any] = {"summary_by_k": {}, "deltas_vs_baseline": {}}
    for k in ks:
        out["summary_by_k"][int(k)] = {}
        out["deltas_vs_baseline"][int(k)] = {}
        for method in methods:
            pq = per_query[f"{method}@{int(k)}"]
            out["summary_by_k"][int(k)][method] = {}
            for mk in metric_keys:
                pt, ci = _bootstrap_mean_ci(getattr(pq, mk), n_boot=int(bootstrap), seed=int(seed) + int(k))
                out["summary_by_k"][int(k)][method][mk] = {"pt": pt, "ci": ci}
                if method != baseline_ref:
                    dpt, dci = _bootstrap_paired_delta_ci(
                        getattr(pq, mk),
                        getattr(per_query[f"{baseline_ref}@{int(k)}"], mk),
                        n_boot=int(bootstrap),
                        seed=int(seed) + 1000 + int(k),
                    )
                    out["deltas_vs_baseline"][int(k)].setdefault(method, {})[mk] = {"pt": dpt, "ci": dci}
    return out


def build_markdown_report(
    *,
    methods: Sequence[str],
    ks: Sequence[int],
    summary: Dict[str, Any],
    embedding_metrics: Dict[str, Dict[str, float]],
    baseline_ref: str,
) -> str:
    metric_names = {
        "hit": "Hit@k",
        "mrr_ul": "MRR@k (unique laws)",
        "top1": "Top-1",
        "majority": "Majority-Accuracy",
        "cons_frac": "Mean Consensus Fraction",
        "lift": "Mean Lift",
    }

    lines: List[str] = []
    lines.append("# KAHM comparison against a retrieval-distilled compact neural student")
    lines.append("")
    lines.append("This report compares KAHM to an evaluation-matched compact neural student that preserves the same deployment contract: IDF–SVD query features in, retrieval against a frozen Mixedbread corpus index out.")
    lines.append("")
    lines.append("The added neural baseline remains law-wise and uses the same prototype-distance gating across local models, but it is trained with retrieval distillation over the law-local corpus rather than only pointwise embedding regression.")
    lines.append("")
    lines.append("## Embedding reconstruction diagnostics")
    lines.append("")
    headers = ["Method", "MSE", "R^2", "Cos mean", "Cos p50"]
    rows: List[List[str]] = []
    for method in methods:
        m = embedding_metrics.get(method, {})
        if not m:
            rows.append([method, "n/a", "n/a", "n/a", "n/a"])
        else:
            rows.append([
                method,
                f"{m.get('mse', float('nan')):.6f}",
                f"{m.get('r2_overall', float('nan')):.4f}",
                f"{m.get('cos_mean', float('nan')):.4f}",
                f"{m.get('cos_p50', float('nan')):.4f}",
            ])
    lines.append(_md_table(headers, rows))
    lines.append("")

    for mk, title in metric_names.items():
        lines.append(f"## {title}")
        lines.append("")
        headers = ["k"] + list(methods)
        rows = []
        for k in ks:
            row = [str(k)]
            for method in methods:
                entry = summary["summary_by_k"][int(k)][method][mk]
                row.append(_fmt_ci(float(entry["pt"]), tuple(entry["ci"])))
            rows.append(row)
        lines.append(_md_table(headers, rows))
        lines.append("")

        others = [m for m in methods if m != baseline_ref]
        if others:
            lines.append(f"### Paired deltas vs {baseline_ref}")
            lines.append("")
            headers = ["k"] + others
            rows = []
            for k in ks:
                row = [str(k)]
                for method in others:
                    entry = summary["deltas_vs_baseline"][int(k)][method][mk]
                    row.append(_fmt_delta(float(entry["pt"]), tuple(entry["ci"])))
                rows.append(row)
            lines.append(_md_table(headers, rows))
            lines.append("")

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare KAHM against a retrieval-distilled compact neural law-wise student under the same distributed-law serving contract."
    )

    p.add_argument("--train_query_set", default="query_set.TRAIN_QUERY_SET")
    p.add_argument("--test_query_set", default="query_set.TEST_QUERY_SET")
    p.add_argument("--idf_svd_model", default="idf_svd_model.joblib")
    p.add_argument("--queries_npz_train", default="queries_embedding_index_train.npz", help="Train query Mixedbread embeddings NPZ (query_id + embeddings).")
    p.add_argument("--queries_npz_test", default="queries_embedding_index_test.npz", help="Test query Mixedbread embeddings NPZ (query_id + embeddings).")

    p.add_argument("--kahm_model_dir", default="kahm_query_regressors_by_law", help="Optional directory with law-wise KAHM *.joblib regressors.")
    p.add_argument("--kahm_query_embeddings_npz", default="", help="Optional precomputed KAHM query embeddings for the test set.")
    p.add_argument("--kahm_mode", default="soft", choices=["soft", "hard"])
    p.add_argument("--kahm_batch", type=int, default=1024)

    p.add_argument("--corpus_parquet", default="ris_sentences.parquet")
    p.add_argument("--mb_corpus_npz", default="embedding_index.npz")
    p.add_argument("--idf_corpus_npz", default="embedding_index_idf_svd.npz")
    p.add_argument("--threads", type=int, default=1)

    p.add_argument("--ks", default="3,5,10,15,20")
    p.add_argument("--predominance_fraction", type=float, default=0.1)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--baseline_n_prototypes", type=int, default=32)

    p.add_argument("--student_hidden", default="768,384,192")
    p.add_argument("--student_dropout", type=float, default=0.05)
    p.add_argument("--student_epochs", type=int, default=160)
    p.add_argument("--student_batch_size", type=int, default=128)
    p.add_argument("--student_lr", type=float, default=8e-4)
    p.add_argument("--student_weight_decay", type=float, default=5e-5)
    p.add_argument("--student_temperature", type=float, default=0.05)
    p.add_argument("--student_kl_weight", type=float, default=1.0)
    p.add_argument("--student_cos_weight", type=float, default=0.20)
    p.add_argument("--student_mse_weight", type=float, default=0.05)
    p.add_argument("--student_focus_topk", type=int, default=32)
    p.add_argument("--student_focus_weight", type=float, default=0.35)
    p.add_argument("--student_grad_clip", type=float, default=1.0)
    p.add_argument("--student_lr_min_factor", type=float, default=0.05)
    p.add_argument("--student_validation_fraction", type=float, default=0.10)
    p.add_argument("--student_patience", type=int, default=14)
    p.add_argument("--student_device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--student_log_every", type=int, default=10)

    p.add_argument("--out_dir", default="retrieval_student_comparison")
    p.add_argument("--save_student_query_embeddings", action="store_true")
    p.add_argument("--report_path", default="retrieval_student_comparison/report.md")
    p.add_argument("--results_json_path", default="retrieval_student_comparison/results.json")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    ks = [int(x.strip()) for x in str(args.ks).split(",") if x.strip()]
    if not ks:
        raise ValueError("--ks must contain at least one cutoff")
    k_max = max(ks)

    out_dir = Path(str(args.out_dir)).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_qs = load_query_set(str(args.train_query_set))
    test_qs = load_query_set(str(args.test_query_set))
    train_ids, train_texts, train_laws = extract_ids_texts_laws(train_qs, name="TRAIN_QUERY_SET")
    test_ids, test_texts, test_laws = extract_ids_texts_laws(test_qs, name="TEST_QUERY_SET")

    print(f"Loaded TRAIN queries: {len(train_ids)} | TEST queries: {len(test_ids)}")

    t0 = time.time()
    x_train = embed_idf_svd_queries(str(args.idf_svd_model), train_texts)
    x_test = embed_idf_svd_queries(str(args.idf_svd_model), test_texts)
    y_train = load_query_npz(str(args.queries_npz_train), train_ids)
    y_test = load_query_npz(str(args.queries_npz_test), test_ids)
    print(f"Loaded lexical and teacher query embeddings in {time.time() - t0:.1f}s")

    print("Loading corpus bundles and building aligned corpus view ...")
    df = load_corpus_parquet(str(args.corpus_parquet))
    mb_bundle = load_npz_bundle(str(args.mb_corpus_npz))
    idf_bundle = load_npz_bundle(str(args.idf_corpus_npz))
    aligned = align_by_common_sentence_ids(df, mb_bundle, idf_bundle)

    embedding_predictions: Dict[str, np.ndarray] = {
        "IDF–SVD": x_test,
        "Mixedbread (true)": y_test,
    }
    embedding_metrics: Dict[str, Dict[str, float]] = {
        "Mixedbread (true)": compute_embedding_metrics(y_test.T, y_test.T),
    }

    print("Training retrieval_distilled_student_lawwise baseline ...")
    aligned_laws = [str(x) for x in aligned["law"].tolist()]
    student_models = train_retrieval_distilled_student_lawwise(
        x_train,
        y_train,
        train_laws,
        aligned["emb_mb"],
        aligned_laws,
        n_prototypes=int(args.baseline_n_prototypes),
        seed=int(args.seed),
        hidden_sizes=parse_hidden_sizes(str(args.student_hidden)),
        dropout=float(args.student_dropout),
        epochs=int(args.student_epochs),
        batch_size=int(args.student_batch_size),
        learning_rate=float(args.student_lr),
        weight_decay=float(args.student_weight_decay),
        temperature=float(args.student_temperature),
        kl_weight=float(args.student_kl_weight),
        cos_weight=float(args.student_cos_weight),
        mse_weight=float(args.student_mse_weight),
        focus_topk=int(args.student_focus_topk),
        focus_weight=float(args.student_focus_weight),
        grad_clip=float(args.student_grad_clip),
        lr_min_factor=float(args.student_lr_min_factor),
        validation_fraction=float(args.student_validation_fraction),
        patience=int(args.student_patience),
        device=str(args.student_device),
        log_every=int(args.student_log_every),
    )
    student_pred = predict_retrieval_distilled_student_lawwise(
        student_models,
        x_test,
        batch_size=int(args.student_batch_size),
        device=str(args.student_device),
    )
    embedding_predictions["retrieval_distilled_student_lawwise(query→MB corpus)"] = student_pred.embeddings
    embedding_metrics["retrieval_distilled_student_lawwise(query→MB corpus)"] = compute_embedding_metrics(
        student_pred.embeddings.T,
        y_test.T,
    )

    if args.save_student_query_embeddings:
        np.savez_compressed(
            str(out_dir / "retrieval_distilled_student_query_embeddings_test.npz"),
            query_id=np.asarray(test_ids, dtype=np.str_),
            embeddings=np.asarray(student_pred.embeddings, dtype=np.float32),
        )

    if str(args.kahm_query_embeddings_npz).strip():
        print("Loading precomputed KAHM test embeddings ...")
        y_kahm = load_query_npz(str(args.kahm_query_embeddings_npz), test_ids)
        embedding_predictions["KAHM(query→MB corpus)"] = y_kahm
        embedding_metrics["KAHM(query→MB corpus)"] = compute_embedding_metrics(y_kahm.T, y_test.T)
    elif str(args.kahm_model_dir).strip():
        print("Loading and running KAHM models ...")
        kahm_models = load_kahm_models_from_dir(str(args.kahm_model_dir))
        y_kahm, _chosen, _best, _names = kahm_regress_distance_gated_multi_models(
            x_test,
            models=kahm_models,
            mode=str(args.kahm_mode),
            batch_size=int(args.kahm_batch),
            tie_break="first",
            show_progress=True,
        )
        embedding_predictions["KAHM(query→MB corpus)"] = y_kahm
        embedding_metrics["KAHM(query→MB corpus)"] = compute_embedding_metrics(y_kahm.T, y_test.T)
        if args.save_student_query_embeddings:
            np.savez_compressed(
                str(out_dir / "kahm_query_embeddings_test.npz"),
                query_id=np.asarray(test_ids, dtype=np.str_),
                embeddings=np.asarray(y_kahm, dtype=np.float32),
            )

    print("Building FAISS indices ...")
    index_idf = build_faiss_index(aligned["emb_idf"], n_threads=int(args.threads))
    index_mb = build_faiss_index(aligned["emb_mb"], n_threads=int(args.threads))
    law_arr = np.asarray(aligned["law"], dtype=object)

    per_query: Dict[str, PerQuery] = {}
    method_order: List[str] = []

    for method, emb in embedding_predictions.items():
        if method == "IDF–SVD":
            _, idx = faiss_search(index_idf, emb, k_max)
        else:
            _, idx = faiss_search(index_mb, emb, k_max)
        method_order.append(method)
        for k in ks:
            per_query[f"{method}@{int(k)}"] = compute_per_query_metrics(
                idx=idx,
                law_arr=law_arr,
                consensus_laws=test_laws,
                k=int(k),
                predominance_fraction=float(args.predominance_fraction),
            )

    print("\n=== Quick summary at largest k ===")
    k_star = int(max(ks))
    for method in method_order:
        pq = per_query[f"{method}@{k_star}"]
        print(
            f"{method:55s} | "
            f"MRR@{k_star}={np.mean(pq.mrr_ul):.3f} | "
            f"Hit@{k_star}={np.mean(pq.hit):.3f} | "
            f"Top1={np.mean(pq.top1):.3f} | "
            f"Maj={np.mean(pq.majority):.3f}"
        )

    summary = summarize_methods(
        per_query=per_query,
        ks=ks,
        methods=method_order,
        bootstrap=int(args.bootstrap),
        seed=int(args.seed),
        baseline_ref="IDF–SVD",
    )

    report_md = build_markdown_report(
        methods=method_order,
        ks=ks,
        summary=summary,
        embedding_metrics=embedding_metrics,
        baseline_ref="IDF–SVD",
    )

    if str(args.report_path).strip():
        report_path = Path(str(args.report_path)).expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_md, encoding="utf-8")
        print(f"Wrote report: {report_path}")

    if str(args.results_json_path).strip():
        payload = {
            "ks": ks,
            "methods": method_order,
            "embedding_metrics": embedding_metrics,
            "summary": summary,
            "student_hyperparameters": {
                "hidden_sizes": parse_hidden_sizes(str(args.student_hidden)),
                "dropout": float(args.student_dropout),
                "epochs": int(args.student_epochs),
                "batch_size": int(args.student_batch_size),
                "learning_rate": float(args.student_lr),
                "weight_decay": float(args.student_weight_decay),
                "temperature": float(args.student_temperature),
                "kl_weight": float(args.student_kl_weight),
                "cos_weight": float(args.student_cos_weight),
                "mse_weight": float(args.student_mse_weight),
                "focus_topk": int(args.student_focus_topk),
                "focus_weight": float(args.student_focus_weight),
                "grad_clip": float(args.student_grad_clip),
                "lr_min_factor": float(args.student_lr_min_factor),
                "validation_fraction": float(args.student_validation_fraction),
                "patience": int(args.student_patience),
                "device": str(args.student_device),
            },
        }
        out_path = Path(str(args.results_json_path)).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote JSON: {out_path}")

    default_report = out_dir / "comparison_report.md"
    default_report.write_text(report_md, encoding="utf-8")
    print(f"Wrote default report: {default_report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
