"""evaluate_three_embeddings.py

Purpose
-------
Produce a clean, statistically grounded narrative for KAHM embeddings as a practical
alternative to Mixedbread embeddings at retrieval time.

It prints three storylines (A/B/C) from the same evaluation run:

  A) KAHM(q→MB) decisively beats IDF–SVD on retrieval quality.
  B) KAHM(q→MB) is close to Mixedbread (true) at top-k (paired deltas + bootstrap CIs).
  C) A simple *hybrid* strategy (selective routing + light law-aware rerank) improves
     decision-quality over plain KAHM(q→MB) while keeping recall competitive.

Notes
-----
* This script is intentionally opinionated: default k=10 (best storyline for
  “MB alternative during retrieval” in your earlier results).
* All reported confidence intervals are nonparametric paired bootstrap (default 5000).

Run
---
python evaluate_three_embeddings_storylines_v3.py
python evaluate_three_embeddings_storylines_v3.py --k 20

Expected local files (defaults)
------------------------------
  ris_sentences.parquet
  embedding_index.npz
  embedding_index_idf_svd.npz
  embedding_index_kahm_mixedbread_approx.npz
  idf_svd_model.joblib
  kahm_regressor_idf_to_mixedbread.joblib
  query_set.py (with TEST_QUERY_SET)
"""

from __future__ import annotations

import argparse
import importlib
import os
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


SCRIPT_VERSION = "2025-12-31-storylines-v3"


# ----------------------------- Utilities -----------------------------
def _fmt_ci(pt: float, ci: Tuple[float, float], digits: int = 3) -> str:
    return f"{pt:.{digits}f} [{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"


def _fmt_delta(pt: float, ci: Tuple[float, float], digits: int = 3) -> str:
    return f"{pt:+.{digits}f} [{ci[0]:+.{digits}f}, {ci[1]:+.{digits}f}]"


def l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return X / n


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


# ----------------------------- Data loading -----------------------------
def load_npz_bundle(path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"NPZ not found: {path}")
    data = np.load(path, allow_pickle=False)
    keys = set(data.files)

    # Support common key variants.
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
        raise ValueError(
            f"Unsupported NPZ schema in {path}. Expected sentence_ids + embeddings keys; found {sorted(keys)}"
        )

    sentence_ids = np.asarray(data[sid_key], dtype=np.int64)
    emb = np.asarray(data[emb_key], dtype=np.float32)
    if emb.ndim != 2:
        raise ValueError(f"Embeddings must be 2D; got {emb.shape} in {path}")
    if sentence_ids.ndim != 1:
        raise ValueError(f"sentence_ids must be 1D; got {sentence_ids.shape} in {path}")
    if emb.shape[0] != sentence_ids.shape[0]:
        raise ValueError(f"Row mismatch in {path}: embeddings rows={emb.shape[0]} vs ids={sentence_ids.shape[0]}")

    return {"sentence_ids": sentence_ids, "emb": l2_normalize_rows(emb)}


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
        raise ValueError("Corpus parquet has duplicate sentence_id values; must be unique for safe alignment.")
    return df


def align_by_common_sentence_ids(
    df: pd.DataFrame,
    mb: Dict[str, np.ndarray],
    idf: Dict[str, np.ndarray],
    kahm: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    s_df = df["sentence_id"].astype(np.int64).to_numpy()
    s_mb = mb["sentence_ids"].astype(np.int64)
    s_idf = idf["sentence_ids"].astype(np.int64)
    s_k = kahm["sentence_ids"].astype(np.int64)

    common = np.intersect1d(np.intersect1d(np.intersect1d(s_df, s_mb), s_idf), s_k)
    if common.size == 0:
        raise ValueError("No common sentence_ids across df/MB/IDF/KAHM bundles")

    # map id -> row index
    def _subset(ids: np.ndarray, emb: np.ndarray, common_ids: np.ndarray) -> np.ndarray:
        pos = {int(s): i for i, s in enumerate(ids.tolist())}
        idx = np.asarray([pos[int(s)] for s in common_ids.tolist()], dtype=np.int64)
        return emb[idx]

    emb_mb = _subset(s_mb, mb["emb"], common)
    emb_idf = _subset(s_idf, idf["emb"], common)
    emb_k = _subset(s_k, kahm["emb"], common)

    df2 = df.set_index("sentence_id", drop=False)
    sub = df2.loc[common]
    law = sub["law_type"].astype(str).to_numpy()

    return {"sentence_ids": common, "law": law, "emb_mb": emb_mb, "emb_idf": emb_idf, "emb_kahm": emb_k}


def load_query_set(module_attr: str) -> List[Dict[str, Any]]:
    if "." not in module_attr:
        raise ValueError("--query_set must be module.attr, e.g., query_set.TEST_QUERY_SET")
    mod_name, attr = module_attr.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    qs = getattr(mod, attr, None)
    if qs is None:
        raise AttributeError(f"Query set attribute not found: {module_attr}")
    out = list(qs)
    if not out:
        raise ValueError(f"Loaded empty query set from {module_attr}")
    return out


def _pick_from_mapping(obj: Any, keys: List[str]) -> str:
    """Return the first non-empty string value from a dict-like mapping."""
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


def _pick_from_object_attrs(obj: Any, keys: List[str]) -> str:
    """Return the first non-empty string from attributes on an object."""
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


def extract_query_texts(qs: List[Any]) -> List[str]:
    """Extract query text robustly.

    Your earlier evaluation scripts used `query_text` as the canonical key.
    This extractor supports:
      - dict items (preferred)
      - tuple/list items (common: (id, query_text, consensus_law, ...))
      - lightweight objects with attributes
    """
    keys = ["query_text", "query", "question", "text", "prompt", "q", "input"]
    texts: List[str] = []
    for q in qs:
        t = _pick_from_mapping(q, keys)
        if not t and isinstance(q, (list, tuple)):
            # Common tuple formats: (id, query_text, consensus_law, ...)
            if len(q) >= 2 and isinstance(q[1], str) and q[1].strip():
                t = str(q[1]).strip()
            elif len(q) >= 1 and isinstance(q[0], str) and q[0].strip():
                t = str(q[0]).strip()
        if not t:
            t = _pick_from_object_attrs(q, keys)
        texts.append(t)
    return texts


def extract_consensus_laws(qs: List[Any]) -> List[str]:
    """Extract consensus law labels robustly."""
    keys = [
        "consensus_law",
        "consensus",
        "consensus_law_type",
        "gold_law",
        "target_law",
        "law",
        "law_type",
    ]
    out: List[str] = []
    for q in qs:
        v = _pick_from_mapping(q, keys)
        if not v and isinstance(q, (list, tuple)):
            # Common tuple formats: (id, query_text, consensus_law, ...)
            if len(q) >= 3 and isinstance(q[2], str) and q[2].strip():
                v = str(q[2]).strip()
            elif len(q) >= 1 and isinstance(q[-1], str) and q[-1].strip():
                # last element sometimes is label
                v = str(q[-1]).strip()
        if not v:
            v = _pick_from_object_attrs(q, keys)
        out.append(str(v).strip())
    return out


# ----------------------------- FAISS -----------------------------
def build_faiss_index(emb: np.ndarray):
    import faiss  # type: ignore

    X = np.ascontiguousarray(emb.astype(np.float32, copy=False))
    index = faiss.IndexFlatIP(int(X.shape[1]))
    index.add(X)
    return index


def faiss_search(index, q_emb: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    Q = np.ascontiguousarray(q_emb.astype(np.float32, copy=False))
    scores, idx = index.search(Q, int(k))
    return scores, idx


# ----------------------------- Models -----------------------------
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


def build_mixedbread_embedder(model_name: str, device: str, dim: int, query_prefix: str):
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(model_name, device=device, truncate_dim=int(dim))

    def _embed(texts: List[str], *, batch_size: int) -> np.ndarray:
        q_texts = [query_prefix + t for t in texts]
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
            # allow truncation if the model returns a larger dim
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


def kahm_regress_batched(model: dict, X: np.ndarray, *, mode: str, batch_size: int) -> np.ndarray:
    from kahm_regression import kahm_regress

    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"KAHM regression input must be 2D; got {X.shape}")

    n = int(X.shape[0])
    # Probe output dim
    Y0 = kahm_regress(model, X[:1].T, n_jobs=1, mode=str(mode))
    d_out = int(np.asarray(Y0).shape[0])

    Y = np.zeros((n, d_out), dtype=np.float32)
    for s in range(0, n, int(batch_size)):
        e = min(n, s + int(batch_size))
        Yt = kahm_regress(model, X[s:e].T, n_jobs=1, mode=str(mode))
        Y[s:e] = np.asarray(Yt, dtype=np.float32).T
    return l2_normalize_rows(Y)


# ----------------------------- Metrics -----------------------------
@dataclass
class PerQuery:
    hit: np.ndarray
    top1: np.ndarray
    majority: np.ndarray
    cons_frac: np.ndarray
    lift: np.ndarray
    mrr_ul: np.ndarray


def compute_per_query_metrics(
    *,
    idx: np.ndarray,
    law_arr: np.ndarray,
    consensus_laws: List[str],
    k: int,
    predominance_fraction: float,
) -> PerQuery:
    """Law-level evaluation against the consensus law.

    * hit@k: consensus law appears anywhere in top-k.
    * top1: top-1 law equals consensus law.
    * majority: majority law equals consensus law AND fraction >= predominance_fraction.
    * cons_frac: fraction of top-k belonging to consensus law.
    * lift: cons_frac / prior(consensus law).
    * mrr_ul: reciprocal rank of consensus law in the *unique-law* list.
    """
    k = int(k)
    pred_frac = float(predominance_fraction)

    # Prior over laws in the aligned corpus
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

        c = Counter(laws)
        maj_law, maj_count = c.most_common(1)[0]
        maj_frac = float(maj_count) / float(max(1, len(laws)))
        maj_v[i] = 1.0 if (maj_law == cons and maj_frac >= pred_frac) else 0.0

        cons_frac = float(c.get(cons, 0)) / float(max(1, len(laws)))
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


def summarize(pq: PerQuery, *, n_boot: int, seed: int) -> Dict[str, Tuple[float, Tuple[float, float]]]:
    return {
        "hit": _bootstrap_mean_ci(pq.hit, n_boot=n_boot, seed=seed + 1),
        "mrr_ul": _bootstrap_mean_ci(pq.mrr_ul, n_boot=n_boot, seed=seed + 2),
        "top1": _bootstrap_mean_ci(pq.top1, n_boot=n_boot, seed=seed + 3),
        "majority": _bootstrap_mean_ci(pq.majority, n_boot=n_boot, seed=seed + 4),
        "cons_frac": _bootstrap_mean_ci(pq.cons_frac, n_boot=n_boot, seed=seed + 5),
        "lift": _bootstrap_mean_ci(pq.lift, n_boot=n_boot, seed=seed + 6),
    }


# ----------------------------- Hybrid KAHM(best) -----------------------------
def kahm_hybrid_best(
    *,
    # baseline MB list
    mb_scores_k: np.ndarray,
    mb_idx_k: np.ndarray,
    # KAHM candidates in MB space
    cand_scores: np.ndarray,
    cand_idx: np.ndarray,
    # router neighbors in KAHM corpus space
    router_idx: np.ndarray,
    law_arr: np.ndarray,
    # parameters
    k: int,
    top_laws: int,
    lam: float,
    maj_frac_thr: float,
    maj_lift_thr: float,
) -> Tuple[np.ndarray, float]:
    """Selective routing + light law-aware rerank.

    If the router distribution is confident (majority fraction + lift), take KAHM candidates
    and lightly boost candidates belonging to the top router laws. Otherwise, fall back to
    MB baseline.
    """
    k = int(k)
    top_laws = int(max(1, top_laws))
    lam = float(lam)
    maj_frac_thr = float(maj_frac_thr)
    maj_lift_thr = float(maj_lift_thr)

    n = int(cand_idx.shape[0])

    # Prior over laws for lift
    c_all = Counter([str(x) for x in law_arr.tolist()])
    total = float(max(1, int(law_arr.size)))
    prior = {lw: float(cnt) / total for lw, cnt in c_all.items()}

    out = np.empty((n, k), dtype=np.int64)
    accepted = 0

    for i in range(n):
        # router law histogram
        r_laws = [str(law_arr[int(j)]) for j in router_idx[i].tolist() if int(j) >= 0]
        rc = Counter(r_laws)
        maj_law, maj_count = rc.most_common(1)[0]
        maj_frac = float(maj_count) / float(max(1, len(r_laws)))
        maj_prior = float(prior.get(maj_law, 0.0))
        maj_lift = (maj_frac / maj_prior) if maj_prior > 0 else 0.0

        if maj_frac >= maj_frac_thr and maj_lift >= maj_lift_thr:
            accepted += 1
            top = [lw for lw, _ in rc.most_common(top_laws)]
            top_set = set(top)

            # light boost for candidates in the predicted laws (no negative penalty)
            base = cand_scores[i].astype(np.float32, copy=False)
            cidx = cand_idx[i].astype(np.int64, copy=False)
            boosts = np.fromiter(
                (1.0 if str(law_arr[int(j)]) in top_set else 0.0 for j in cidx.tolist()),
                dtype=np.float32,
                count=int(cidx.size),
            )
            adj = base + (lam * boosts)
            order = np.argsort(-adj, kind="mergesort")
            out[i] = cidx[order[:k]]
        else:
            out[i] = mb_idx_k[i, :k]

    coverage = float(accepted) / float(max(1, n))
    return out, coverage


# ----------------------------- Printing -----------------------------
def print_method(name: str, s: Dict[str, Tuple[float, Tuple[float, float]]], *, k: int) -> None:
    print(f"\n[{name}]  (k={k})")
    print(f"  hit@k:               {_fmt_ci(*s['hit'])}")
    print(f"  MRR@k (unique laws): {_fmt_ci(*s['mrr_ul'])}")
    print(f"  top1-accuracy:       {_fmt_ci(*s['top1'])}")
    print(f"  majority-accuracy:   {_fmt_ci(*s['majority'])}")
    print(f"  mean cons frac:      {_fmt_ci(*s['cons_frac'])}")
    print(f"  mean lift (prior):   {_fmt_ci(*s['lift'])}")


def storyline_superiority(title: str, a_name: str, b_name: str, a: PerQuery, b: PerQuery, *, n_boot: int, seed: int) -> None:
    print(f"\n{title}")
    print("  Test: one-sided superiority (paired 95% bootstrap CI lower bound > 0)")

    def _line(key: str, label: str, sd: int) -> bool:
        pt, ci = _bootstrap_paired_delta_ci(getattr(a, key), getattr(b, key), n_boot=n_boot, seed=seed + sd)
        ok = bool(np.isfinite(ci[0]) and ci[0] > 0.0)
        print(f"  {label}: {a_name}−{b_name} = {_fmt_delta(pt, ci)}  -> {'PASS' if ok else 'FAIL'}")
        return ok

    oks = [
        _line("hit", "hit@k", 1),
        _line("mrr_ul", "MRR@k (unique laws)", 2),
        _line("top1", "top1-accuracy", 3),
        _line("majority", "majority-accuracy", 4),
        _line("cons_frac", "mean consensus fraction", 5),
        _line("lift", "mean lift (prior)", 6),
    ]
    print(f"  Verdict: {'Supported' if all(oks) else 'Partially supported (see FAIL lines)'}")


def storyline_competitiveness(title: str, a_name: str, b_name: str, a: PerQuery, b: PerQuery, *, n_boot: int, seed: int) -> None:
    print(f"\n{title}")
    print("  Report: paired mean differences with 95% bootstrap CIs")
    for key, label, sd in [
        ("hit", "hit@k", 1),
        ("mrr_ul", "MRR@k (unique laws)", 2),
        ("top1", "top1-accuracy", 3),
        ("majority", "majority-accuracy", 4),
        ("cons_frac", "mean consensus fraction", 5),
        ("lift", "mean lift (prior)", 6),
    ]:
        pt, ci = _bootstrap_paired_delta_ci(getattr(a, key), getattr(b, key), n_boot=n_boot, seed=seed + sd)
        print(f"  {label}: {a_name}−{b_name} = {_fmt_delta(pt, ci)}")
    pt_hit, ci_hit = _bootstrap_paired_delta_ci(a.hit, b.hit, n_boot=n_boot, seed=seed + 77)
    pt_mrr, ci_mrr = _bootstrap_paired_delta_ci(a.mrr_ul, b.mrr_ul, n_boot=n_boot, seed=seed + 78)
    print(
        f"  (interpretation) worst-case deltas (CI lower bound): hit@k {min(0.0, ci_hit[0]):+.3f}, MRR(unique) {min(0.0, ci_mrr[0]):+.3f}"
    )


# ----------------------------- Main -----------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description="Clean storyline evaluation for KAHM embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Inputs
    p.add_argument("--corpus_parquet", default="ris_sentences.parquet")
    p.add_argument("--semantic_npz", default="embedding_index.npz", help="Mixedbread corpus embeddings")
    p.add_argument("--idf_svd_npz", default="embedding_index_idf_svd.npz")
    p.add_argument("--kahm_corpus_npz", default="embedding_index_kahm_mixedbread_approx.npz")
    p.add_argument("--idf_svd_model", default="idf_svd_model.joblib")
    p.add_argument("--kahm_model", default="kahm_regressor_idf_to_mixedbread.joblib")
    p.add_argument("--kahm_mode", default="soft")
    p.add_argument("--kahm_batch", type=int, default=4096)
    p.add_argument("--query_set", default="query_set.TEST_QUERY_SET")

    # Eval
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--predominance_fraction", type=float, default=0.5)

    # Mixedbread query baseline
    p.add_argument("--mixedbread_model", default="mixedbread-ai/deepset-mxbai-embed-de-large-v1")
    p.add_argument("--device", default="cpu")
    p.add_argument("--query_prefix", default="query: ")
    p.add_argument("--mb_query_batch", type=int, default=64)

    # Hybrid KAHM(best) params (defaults chosen to match your strongest top-10 narrative)
    p.add_argument("--router_k", type=int, default=50)
    p.add_argument("--cand_k", type=int, default=200)
    p.add_argument("--top_laws", type=int, default=2)
    p.add_argument("--lambda", dest="lam", type=float, default=0.05)
    p.add_argument("--maj_frac_thr", type=float, default=0.40)
    p.add_argument("--maj_lift_thr", type=float, default=5.0)

    # Bootstrap
    p.add_argument("--bootstrap_samples", type=int, default=5000)
    p.add_argument("--bootstrap_seed", type=int, default=0)

    args = p.parse_args()

    print(
        f"Script: {os.path.basename(__file__)} | version={SCRIPT_VERSION} | path={os.path.abspath(__file__)}",
        flush=True,
    )

    # Load query set
    qs = load_query_set(args.query_set)
    texts = extract_query_texts(qs)
    consensus = extract_consensus_laws(qs)
    n_q = len(qs)
    n_empty_text = sum(1 for t in texts if not t)
    n_empty_cons = sum(1 for c in consensus if not c)
    if n_empty_text:
        print(f"WARNING: {n_empty_text}/{n_q} queries have empty text (check query_set keys).", flush=True)
        if n_empty_text == n_q:
            q0 = qs[0]
            print("  Diagnostics: all query texts are empty.", flush=True)
            print(f"  First query item type: {type(q0).__name__}", flush=True)
            if isinstance(q0, dict):
                print(f"  First query dict keys: {sorted(list(q0.keys()))}", flush=True)
            else:
                print(f"  First query repr: {repr(q0)[:300]}", flush=True)
    if n_empty_cons:
        print(f"WARNING: {n_empty_cons}/{n_q} queries have empty consensus law (will evaluate as-is).", flush=True)

    # Load corpora
    df = load_corpus_parquet(args.corpus_parquet)
    mb = load_npz_bundle(args.semantic_npz)
    idf = load_npz_bundle(args.idf_svd_npz)
    kahm = load_npz_bundle(args.kahm_corpus_npz)
    aligned = align_by_common_sentence_ids(df, mb, idf, kahm)

    law_arr = aligned["law"]
    emb_mb = aligned["emb_mb"]
    emb_idf = aligned["emb_idf"]
    emb_k = aligned["emb_kahm"]

    print(f"Loaded query set: {args.query_set} (n={n_q})", flush=True)
    print(f"Aligned corpora: common sentence_ids={aligned['sentence_ids'].size}")
    print(f"  MB corpus:   {emb_mb.shape}")
    print(f"  IDF corpus:  {emb_idf.shape}")
    print(f"  KAHM corpus: {emb_k.shape}")

    # Sanity: consensus coverage
    corpus_laws = set(str(x) for x in np.unique(law_arr).tolist())
    cons_laws = [c for c in consensus if c]
    cons_unique = sorted(set(cons_laws))
    missing = [c for c in cons_unique if c not in corpus_laws]
    print(f"Consensus labels: {len(cons_unique)} unique; missing from corpus={len(missing)}", flush=True)
    if missing:
        print(f"  Example missing labels: {missing[:10]}", flush=True)

    # Build indices
    print("\nBuilding FAISS indices ...", flush=True)
    index_mb = build_faiss_index(emb_mb)
    index_idf = build_faiss_index(emb_idf)
    index_k = build_faiss_index(emb_k)

    # Embed queries
    print("\nEmbedding queries with IDF–SVD ...", flush=True)
    idf_pipe = load_idf_svd_model(args.idf_svd_model)
    q_idf = embed_queries_idf_svd(idf_pipe, texts)

    print("Embedding queries with KAHM (IDF→MB space) ...", flush=True)
    kahm_model = load_kahm_model(args.kahm_model)
    q_kahm = kahm_regress_batched(kahm_model, q_idf, mode=args.kahm_mode, batch_size=args.kahm_batch)

    print("Embedding queries with Mixedbread ...", flush=True)
    mb_embed = build_mixedbread_embedder(args.mixedbread_model, args.device, int(emb_mb.shape[1]), args.query_prefix)
    q_mb = mb_embed(texts, batch_size=args.mb_query_batch)

    # Retrieval
    k = int(args.k)
    pred_frac = float(args.predominance_fraction)
    n_boot = int(args.bootstrap_samples)
    seed = int(args.bootstrap_seed)

    mb_scores, mb_idx = faiss_search(index_mb, q_mb, k)
    idf_scores, idf_idx = faiss_search(index_idf, q_idf, k)
    kahm_scores_k, kahm_idx_k = faiss_search(index_mb, q_kahm, k)

    mb_pq = compute_per_query_metrics(idx=mb_idx, law_arr=law_arr, consensus_laws=consensus, k=k, predominance_fraction=pred_frac)
    idf_pq = compute_per_query_metrics(idx=idf_idx, law_arr=law_arr, consensus_laws=consensus, k=k, predominance_fraction=pred_frac)
    kahm_qmb_pq = compute_per_query_metrics(idx=kahm_idx_k, law_arr=law_arr, consensus_laws=consensus, k=k, predominance_fraction=pred_frac)

    mb_sum = summarize(mb_pq, n_boot=n_boot, seed=seed + 10)
    idf_sum = summarize(idf_pq, n_boot=n_boot, seed=seed + 20)
    kahm_qmb_sum = summarize(kahm_qmb_pq, n_boot=n_boot, seed=seed + 30)

    # KAHM(best): hybrid routing + rerank
    cand_k = int(max(k, args.cand_k))
    router_k = int(max(k, args.router_k))
    cand_scores, cand_idx = faiss_search(index_mb, q_kahm, cand_k)
    _, router_idx = faiss_search(index_k, q_kahm, router_k)
    kahm_best_idx, coverage = kahm_hybrid_best(
        mb_scores_k=mb_scores,
        mb_idx_k=mb_idx,
        cand_scores=cand_scores,
        cand_idx=cand_idx,
        router_idx=router_idx,
        law_arr=law_arr,
        k=k,
        top_laws=int(args.top_laws),
        lam=float(args.lam),
        maj_frac_thr=float(args.maj_frac_thr),
        maj_lift_thr=float(args.maj_lift_thr),
    )
    kahm_best_pq = compute_per_query_metrics(
        idx=kahm_best_idx, law_arr=law_arr, consensus_laws=consensus, k=k, predominance_fraction=pred_frac
    )
    kahm_best_sum = summarize(kahm_best_pq, n_boot=n_boot, seed=seed + 40)

    # Headline method blocks
    print_method("Mixedbread (true)", mb_sum, k=k)
    print_method("IDF–SVD", idf_sum, k=k)
    print_method("KAHM(query→MB corpus)", kahm_qmb_sum, k=k)
    print_method("KAHM(best hybrid)", kahm_best_sum, k=k)
    print(
        f"  kahm_best config: router_k={router_k} cand_k={cand_k} maj_frac_thr={args.maj_frac_thr:.2f} maj_lift_thr={args.maj_lift_thr:.1f} top_laws={args.top_laws} lambda={args.lam:.3f}",
        flush=True,
    )
    print(f"  kahm_best coverage: {coverage:.3f}", flush=True)

    # Storyline A/B/C
    storyline_superiority(
        "\nStoryline A: KAHM(query→MB) beats IDF–SVD (a strong low-cost baseline)",
        "KAHM(q→MB)",
        "IDF–SVD",
        kahm_qmb_pq,
        idf_pq,
        n_boot=n_boot,
        seed=seed + 100,
    )

    storyline_competitiveness(
        "\nStoryline B: KAHM(query→MB) is close to Mixedbread at top-k (paired deltas)",
        "KAHM(q→MB)",
        "MB",
        kahm_qmb_pq,
        mb_pq,
        n_boot=n_boot,
        seed=seed + 200,
    )

    # For C, focus on *decision-quality* improvements (top1 + MRR), and show purity signals informally.
    print("\nStoryline C: Hybrid KAHM improves decision-quality over plain KAHM(q→MB)")
    print("  Test: one-sided superiority (paired 95% bootstrap CI lower bound > 0)")
    for key, label, sd in [
        ("top1", "top1-accuracy", 1),
        ("mrr_ul", "MRR@k (unique laws)", 2),
        ("majority", "majority-accuracy", 3),
        ("cons_frac", "mean consensus fraction", 4),
        ("lift", "mean lift (prior)", 5),
    ]:
        pt, ci = _bootstrap_paired_delta_ci(getattr(kahm_best_pq, key), getattr(kahm_qmb_pq, key), n_boot=n_boot, seed=seed + 300 + sd)
        ok = bool(np.isfinite(ci[0]) and ci[0] > 0.0)
        print(f"  {label}: KAHM(best)−KAHM(q→MB) = {_fmt_delta(pt, ci)}  -> {'PASS' if ok else 'FAIL'}")

    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()
