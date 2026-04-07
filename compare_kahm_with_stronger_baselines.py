#!/usr/bin/env python3
"""
compare_kahm_with_stronger_baselines.py

Purpose
-------
Add stronger, evaluation-matched baselines to the existing KAHM query-encoding
pipeline without changing the deployment contract:

    cheap lexical query features (IDF–SVD) -> predicted Mixedbread query embedding

This script keeps the same law-wise training split used by the KAHM pipeline and
adds two reviewer-facing baselines:

1) ridge_lawwise:
   A direct multi-output Ridge regressor per law (IDF–SVD -> Mixedbread), with
   law gating based on the distance of the predicted embedding to that law's
   semantic prototype set.

2) logistic_proto_lawwise:
   A matched prototype-mixture baseline. For each law, Mixedbread teacher
   embeddings are clustered into semantic prototypes, and a multinomial logistic
   regression predicts the posterior over those prototypes from IDF–SVD query
   features. The final embedding is the posterior-weighted prototype mixture.
   This is deliberately close to the KAHM estimator structure, but without the
   KAHM geometry.

The script optionally loads an already-trained directory of law-wise KAHM models,
produces KAHM test-query embeddings, and compares the following systems on the
same downstream retrieval task:

- IDF–SVD
- Mixedbread (true query embeddings)
- KAHM(query->MB corpus)             [if --kahm_model_dir or --kahm_query_embeddings_npz is given]
- ridge_lawwise(query->MB corpus)
- logistic_proto_lawwise(query->MB corpus)

Outputs
-------
- Optional query-embedding NPZ files for each trained baseline.
- Optional JSON with retrieval metrics and paired bootstrap deltas.
- Optional Markdown report.

Notes
-----
- This is meant as a *comparison script*, not as a replacement for your current
  KAHM training/evaluation scripts.
- It reuses the same TRAIN_QUERY_SET / TEST_QUERY_SET convention from query_set.py.
- Retrieval metrics and bootstrap deltas are aligned with the logic in
  evaluate_three_embeddings_storylines.py.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, Ridge


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
    """Return min_j ||a_i - b_j||^2 for each row i in a.

    Both a and b must be row-major matrices.
    """
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
    return pt, (float(lo), float(hi))


def _fmt_ci(pt: float, ci: Tuple[float, float], digits: int = 3) -> str:
    return f"{pt:.{digits}f} [{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"


def _fmt_delta(pt: float, ci: Tuple[float, float], digits: int = 3) -> str:
    return f"{pt:+.{digits}f} [{ci[0]:+.{digits}f}, {ci[1]:+.{digits}f}]"


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([head, sep] + body)


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
    index = faiss.IndexFlatIP(int(x.shape[1]))
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
# Baseline models
# -----------------------------------------------------------------------------
@dataclass
class RidgeLawModel:
    law: str
    reg: Ridge
    prototypes: np.ndarray  # (C, D)


@dataclass
class LogisticPrototypeLawModel:
    law: str
    clf: Optional[LogisticRegression]
    prototypes: np.ndarray  # (C, D)
    cluster_labels: np.ndarray  # retained for debugging / reproducibility


@dataclass
class PredictionBundle:
    embeddings: np.ndarray  # (N, D)
    chosen_model_names: List[str]
    gating_scores: np.ndarray  # (N,)



def make_semantic_prototypes(y_train: np.ndarray, *, n_prototypes: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster teacher embeddings for one law and return (centers, labels)."""
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



def train_ridge_lawwise(
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_laws: Sequence[str],
    *,
    ridge_alpha: float,
    n_prototypes: int,
    seed: int,
) -> Dict[str, RidgeLawModel]:
    laws = sorted(set(str(lw) for lw in train_laws))
    models: Dict[str, RidgeLawModel] = {}
    train_laws_arr = np.asarray([str(lw) for lw in train_laws], dtype=object)

    for law in laws:
        mask = train_laws_arr == law
        x_l = np.asarray(x_train[mask], dtype=np.float32)
        y_l = np.asarray(y_train[mask], dtype=np.float32)
        if x_l.shape[0] == 0:
            continue
        reg = Ridge(alpha=float(ridge_alpha), fit_intercept=False)
        reg.fit(x_l, y_l)
        prototypes, _labels = make_semantic_prototypes(y_l, n_prototypes=n_prototypes, seed=seed)
        models[law] = RidgeLawModel(law=law, reg=reg, prototypes=prototypes)
    return models



def predict_ridge_lawwise(models: Mapping[str, RidgeLawModel], x_test: np.ndarray) -> PredictionBundle:
    names = sorted(models.keys())
    if not names:
        raise ValueError("No ridge models available")

    pred_by_name: Dict[str, np.ndarray] = {}
    score_by_name: Dict[str, np.ndarray] = {}
    for name in names:
        model = models[name]
        y_hat = model.reg.predict(x_test).astype(np.float32, copy=False)
        y_hat = l2_normalize_rows(y_hat).astype(np.float32, copy=False)
        pred_by_name[name] = y_hat
        score_by_name[name] = pairwise_min_sqdist(y_hat, model.prototypes)

    all_scores = np.vstack([score_by_name[name] for name in names])  # (M, N)
    chosen = np.argmin(all_scores, axis=0).astype(np.int64)
    best_score = all_scores[chosen, np.arange(x_test.shape[0])]
    y_best = np.vstack([pred_by_name[names[int(j)]][i] for i, j in enumerate(chosen)]).astype(np.float32, copy=False)
    chosen_names = [names[int(j)] for j in chosen.tolist()]
    return PredictionBundle(embeddings=y_best, chosen_model_names=chosen_names, gating_scores=best_score.astype(np.float32, copy=False))



def train_logistic_proto_lawwise(
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_laws: Sequence[str],
    *,
    n_prototypes: int,
    seed: int,
    logistic_c: float,
    logistic_max_iter: int,
) -> Dict[str, LogisticPrototypeLawModel]:
    laws = sorted(set(str(lw) for lw in train_laws))
    models: Dict[str, LogisticPrototypeLawModel] = {}
    train_laws_arr = np.asarray([str(lw) for lw in train_laws], dtype=object)

    for law in laws:
        mask = train_laws_arr == law
        x_l = np.asarray(x_train[mask], dtype=np.float32)
        y_l = np.asarray(y_train[mask], dtype=np.float32)
        if x_l.shape[0] == 0:
            continue

        prototypes, labels = make_semantic_prototypes(y_l, n_prototypes=n_prototypes, seed=seed)
        uniq = np.unique(labels)
        clf: Optional[LogisticRegression]
        if uniq.size <= 1:
            clf = None
        else:
            try:
                clf = LogisticRegression(
                    C=float(logistic_c),
                    fit_intercept=True,
                    solver="lbfgs",
                    multi_class="multinomial",
                    max_iter=int(logistic_max_iter),
                    random_state=int(seed),)
            except TypeError:
                clf = LogisticRegression(
                    C=float(logistic_c),
                    fit_intercept=True,
                    solver="lbfgs",
                    max_iter=int(logistic_max_iter),
                    random_state=int(seed),)
            clf.fit(x_l, labels)
        models[law] = LogisticPrototypeLawModel(law=law, clf=clf, prototypes=prototypes, cluster_labels=labels)
    return models



def predict_logistic_proto_lawwise(models: Mapping[str, LogisticPrototypeLawModel], x_test: np.ndarray) -> PredictionBundle:
    names = sorted(models.keys())
    if not names:
        raise ValueError("No logistic-prototype models available")

    pred_by_name: Dict[str, np.ndarray] = {}
    score_by_name: Dict[str, np.ndarray] = {}
    for name in names:
        model = models[name]
        if model.clf is None:
            y_hat = np.repeat(model.prototypes[:1], x_test.shape[0], axis=0)
        else:
            proba = model.clf.predict_proba(x_test).astype(np.float32, copy=False)
            y_hat = (proba @ model.prototypes).astype(np.float32, copy=False)
        y_hat = l2_normalize_rows(y_hat).astype(np.float32, copy=False)
        pred_by_name[name] = y_hat
        score_by_name[name] = pairwise_min_sqdist(y_hat, model.prototypes)

    all_scores = np.vstack([score_by_name[name] for name in names])
    chosen = np.argmin(all_scores, axis=0).astype(np.int64)
    best_score = all_scores[chosen, np.arange(x_test.shape[0])]
    y_best = np.vstack([pred_by_name[names[int(j)]][i] for i, j in enumerate(chosen)]).astype(np.float32, copy=False)
    chosen_names = [names[int(j)] for j in chosen.tolist()]
    return PredictionBundle(embeddings=y_best, chosen_model_names=chosen_names, gating_scores=best_score.astype(np.float32, copy=False))


# -----------------------------------------------------------------------------
# KAHM loading / inference (reusing the existing model format)
# -----------------------------------------------------------------------------
def load_kahm_model(path: str) -> dict:
    from kahm_regression import load_kahm_regressor

    if not os.path.exists(path):
        raise FileNotFoundError(f"KAHM model not found: {path}")
    return load_kahm_regressor(path)



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
        from combine_kahm_regressors_generalized import combine_kahm_regressors_distance_gated_multi

    y, chosen, best_score, _all_scores, names = combine_kahm_regressors_distance_gated_multi(
        x_row,
        models=models,
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
    lines.append("# KAHM comparison against stronger baselines")
    lines.append("")
    lines.append("This report compares KAHM to evaluation-matched baselines that preserve the same deployment contract: IDF–SVD query features in, retrieval against a frozen Mixedbread corpus index out.")
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
    p = argparse.ArgumentParser(description="Compare KAHM against stronger evaluation-matched baselines.")

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

    p.add_argument("--baseline_methods", default="ridge,logistic_proto", help="Comma-separated subset of: ridge,logistic_proto")
    p.add_argument("--ridge_alpha", type=float, default=1.0)
    p.add_argument("--baseline_n_prototypes", type=int, default=32)
    p.add_argument("--logistic_c", type=float, default=1.0)
    p.add_argument("--logistic_max_iter", type=int, default=1000)

    p.add_argument("--out_dir", default="stronger_baseline_comparison")
    p.add_argument("--save_baseline_query_embeddings", action="store_true")
    p.add_argument("--report_path", default="stronger_baseline_comparison/report.md")
    p.add_argument("--results_json_path", default="stronger_baseline_comparison/results.json")
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
    print(f"Loaded lexical and teacher embeddings in {time.time() - t0:.1f}s")

    embedding_predictions: Dict[str, np.ndarray] = {
        "IDF–SVD": x_test,
        "Mixedbread (true)": y_test,
    }
    embedding_metrics: Dict[str, Dict[str, float]] = {
        "Mixedbread (true)": compute_embedding_metrics(y_test.T, y_test.T),
    }

    baseline_methods = [x.strip().lower() for x in str(args.baseline_methods).split(",") if x.strip()]
    allowed = {"ridge", "logistic_proto"}
    bad = sorted(set(baseline_methods) - allowed)
    if bad:
        raise ValueError(f"Unknown baseline methods: {bad}")

    if "ridge" in baseline_methods:
        print("Training ridge_lawwise baseline ...")
        models_ridge = train_ridge_lawwise(
            x_train,
            y_train,
            train_laws,
            ridge_alpha=float(args.ridge_alpha),
            n_prototypes=int(args.baseline_n_prototypes),
            seed=int(args.seed),
        )
        ridge_pred = predict_ridge_lawwise(models_ridge, x_test)
        embedding_predictions["ridge_lawwise(query→MB corpus)"] = ridge_pred.embeddings
        embedding_metrics["ridge_lawwise(query→MB corpus)"] = compute_embedding_metrics(ridge_pred.embeddings.T, y_test.T)
        if args.save_baseline_query_embeddings:
            np.savez_compressed(
                str(out_dir / "ridge_query_embeddings_test.npz"),
                query_id=np.asarray(test_ids, dtype=object),
                embeddings=np.asarray(ridge_pred.embeddings, dtype=np.float32),
            )

    if "logistic_proto" in baseline_methods:
        print("Training logistic_proto_lawwise baseline ...")
        models_lp = train_logistic_proto_lawwise(
            x_train,
            y_train,
            train_laws,
            n_prototypes=int(args.baseline_n_prototypes),
            seed=int(args.seed),
            logistic_c=float(args.logistic_c),
            logistic_max_iter=int(args.logistic_max_iter),
        )
        lp_pred = predict_logistic_proto_lawwise(models_lp, x_test)
        embedding_predictions["logistic_proto_lawwise(query→MB corpus)"] = lp_pred.embeddings
        embedding_metrics["logistic_proto_lawwise(query→MB corpus)"] = compute_embedding_metrics(lp_pred.embeddings.T, y_test.T)
        if args.save_baseline_query_embeddings:
            np.savez_compressed(
                str(out_dir / "logistic_proto_query_embeddings_test.npz"),
                query_id=np.asarray(test_ids, dtype=object),
                embeddings=np.asarray(lp_pred.embeddings, dtype=np.float32),
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
        if args.save_baseline_query_embeddings:
            np.savez_compressed(
                str(out_dir / "kahm_query_embeddings_test.npz"),
                query_id=np.asarray(test_ids, dtype=object),
                embeddings=np.asarray(y_kahm, dtype=np.float32),
            )

    print("Loading corpus bundles and building FAISS indices ...")
    df = load_corpus_parquet(str(args.corpus_parquet))
    mb_bundle = load_npz_bundle(str(args.mb_corpus_npz))
    idf_bundle = load_npz_bundle(str(args.idf_corpus_npz))
    aligned = align_by_common_sentence_ids(df, mb_bundle, idf_bundle)

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

    # Print quick console summary at largest k.
    print("\n=== Quick summary at largest k ===")
    k_star = int(max(ks))
    for method in method_order:
        pq = per_query[f"{method}@{k_star}"]
        print(
            f"{method:40s} | "
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
        }
        out_path = Path(str(args.results_json_path)).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote JSON: {out_path}")

    # Also store a default report in the output directory for convenience.
    default_report = out_dir / "comparison_report.md"
    default_report.write_text(report_md, encoding="utf-8")
    print(f"Wrote default report: {default_report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
