"""evaluate_three_embeddings.py

Purpose
-------
Clean, publication-friendly evaluation script that tells a simple, defensible story about
KAHM embeddings as a retrieval-time alternative to Mixedbread (MB).

It prints three storylines (A/B/C) from the *same* run:

  A) Effectiveness vs a strong low-cost baseline:
     KAHM(q→MB) decisively beats IDF–SVD on retrieval quality.

  B) Competitiveness vs MB at top-k:
     KAHM(q→MB) is close to MB on top-k retrieval quality (paired deltas + bootstrap CIs).

  C) Alignment / "right direction" evidence:
     Full-KAHM corpus embeddings are aligned with MB embeddings (cosine alignment), and
     full-KAHM retrieval neighborhoods overlap strongly with MB neighborhoods.

Why this version
----------------
Your earlier "hybrid" storyline can be perceived as complicated and (depending on the
implementation) may rely on MB fallback, which dilutes the claim "KAHM alone works".
This v4 replaces that with an alignment storyline that directly supports the argument:

  "KAHM approximates MB embedding geometry well enough that nearest-neighbor retrieval
   behaves similarly." 

All confidence intervals use nonparametric *paired* bootstrap (default 5000).

Run
---
python evaluate_three_embeddings_storylines_v4.py
python evaluate_three_embeddings_storylines_v4.py --k 20

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
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


SCRIPT_VERSION = "2025-12-31-storylines-v4"


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


def _bootstrap_paired_delta_ci(
    a: np.ndarray, b: np.ndarray, *, n_boot: int, seed: int
) -> Tuple[float, Tuple[float, float]]:
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


def extract_consensus_laws(qs: List[Any]) -> List[str]:
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
            if len(q) >= 3 and isinstance(q[2], str) and q[2].strip():
                v = str(q[2]).strip()
            elif len(q) >= 1 and isinstance(q[-1], str) and q[-1].strip():
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


# ----------------------------- Alignment metrics -----------------------------
def cosine_rowwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity. Assumes rows are L2-normalized."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"cosine_rowwise shape mismatch: {a.shape} vs {b.shape}")
    return np.sum(a * b, axis=1).astype(np.float64)


def jaccard_topk_rows(a_idx: np.ndarray, b_idx: np.ndarray, *, k: int) -> np.ndarray:
    """Jaccard overlap of sentence-id sets in the top-k lists."""
    a_idx = np.asarray(a_idx, dtype=np.int64)[:, : int(k)]
    b_idx = np.asarray(b_idx, dtype=np.int64)[:, : int(k)]
    n = int(a_idx.shape[0])
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        A = set(int(x) for x in a_idx[i].tolist() if int(x) >= 0)
        B = set(int(x) for x in b_idx[i].tolist() if int(x) >= 0)
        u = len(A | B)
        out[i] = (len(A & B) / u) if u else 0.0
    return out


def overlap_frac_topk_rows(a_idx: np.ndarray, b_idx: np.ndarray, *, k: int) -> np.ndarray:
    """Intersection size divided by k (fixed-k overlap fraction)."""
    a_idx = np.asarray(a_idx, dtype=np.int64)[:, : int(k)]
    b_idx = np.asarray(b_idx, dtype=np.int64)[:, : int(k)]
    n = int(a_idx.shape[0])
    out = np.zeros(n, dtype=np.float64)
    kf = float(max(1, int(k)))
    for i in range(n):
        A = set(int(x) for x in a_idx[i].tolist() if int(x) >= 0)
        B = set(int(x) for x in b_idx[i].tolist() if int(x) >= 0)
        out[i] = float(len(A & B)) / kf
    return out


def law_jaccard_topk_rows(a_idx: np.ndarray, b_idx: np.ndarray, law_arr: np.ndarray, *, k: int) -> np.ndarray:
    """Jaccard overlap of *unique laws* present in the top-k lists."""
    a_idx = np.asarray(a_idx, dtype=np.int64)[:, : int(k)]
    b_idx = np.asarray(b_idx, dtype=np.int64)[:, : int(k)]
    n = int(a_idx.shape[0])
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        A = set(str(law_arr[int(x)]) for x in a_idx[i].tolist() if int(x) >= 0)
        B = set(str(law_arr[int(x)]) for x in b_idx[i].tolist() if int(x) >= 0)
        u = len(A | B)
        out[i] = (len(A & B) / u) if u else 0.0
    return out


# ----------------------------- Main -----------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description="Clean storyline evaluation for KAHM embeddings (v4: alignment storyline).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--corpus_parquet", default="ris_sentences.parquet")
    p.add_argument("--semantic_npz", default="embedding_index.npz", help="Mixedbread corpus embeddings")
    p.add_argument("--idf_svd_npz", default="embedding_index_idf_svd.npz")
    p.add_argument("--kahm_corpus_npz", default="embedding_index_kahm_mixedbread_approx.npz")
    p.add_argument("--idf_svd_model", default="idf_svd_model.joblib")
    p.add_argument("--kahm_model", default="kahm_regressor_idf_to_mixedbread.joblib")
    p.add_argument("--kahm_mode", default="soft")
    p.add_argument("--kahm_batch", type=int, default=4096)
    p.add_argument("--query_set", default="query_set.TEST_QUERY_SET")

    p.add_argument("--k", type=int, default=10)
    p.add_argument("--predominance_fraction", type=float, default=0.5)

    p.add_argument("--mixedbread_model", default="mixedbread-ai/deepset-mxbai-embed-de-large-v1")
    p.add_argument("--device", default="cpu")
    p.add_argument("--query_prefix", default="query: ")
    p.add_argument("--mb_query_batch", type=int, default=64)

    p.add_argument("--bootstrap_samples", type=int, default=5000)
    p.add_argument("--bootstrap_seed", type=int, default=0)

    args = p.parse_args()

    print(
        f"Script: {os.path.basename(__file__)} | version={SCRIPT_VERSION} | path={os.path.abspath(__file__)}",
        flush=True,
    )

    qs = load_query_set(args.query_set)
    texts = extract_query_texts(qs)
    consensus = extract_consensus_laws(qs)
    n_q = len(qs)
    n_empty_text = sum(1 for t in texts if not t)
    if n_empty_text:
        print(f"WARNING: {n_empty_text}/{n_q} queries have empty text (check query_set keys).", flush=True)

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

    # Retrieval + quality metrics
    k = int(args.k)
    pred_frac = float(args.predominance_fraction)
    n_boot = int(args.bootstrap_samples)
    seed = int(args.bootstrap_seed)

    mb_scores, mb_idx = faiss_search(index_mb, q_mb, k)
    idf_scores, idf_idx = faiss_search(index_idf, q_idf, k)

    # KAHM as a drop-in replacement for MB at query-time (search MB corpus)
    kahm_qmb_scores, kahm_qmb_idx = faiss_search(index_mb, q_kahm, k)

    # Full-KAHM retrieval (search KAHM corpus)
    kahm_full_scores, kahm_full_idx = faiss_search(index_k, q_kahm, k)

    mb_pq = compute_per_query_metrics(idx=mb_idx, law_arr=law_arr, consensus_laws=consensus, k=k, predominance_fraction=pred_frac)
    idf_pq = compute_per_query_metrics(idx=idf_idx, law_arr=law_arr, consensus_laws=consensus, k=k, predominance_fraction=pred_frac)
    kahm_qmb_pq = compute_per_query_metrics(idx=kahm_qmb_idx, law_arr=law_arr, consensus_laws=consensus, k=k, predominance_fraction=pred_frac)
    kahm_full_pq = compute_per_query_metrics(idx=kahm_full_idx, law_arr=law_arr, consensus_laws=consensus, k=k, predominance_fraction=pred_frac)

    mb_sum = summarize(mb_pq, n_boot=n_boot, seed=seed + 10)
    idf_sum = summarize(idf_pq, n_boot=n_boot, seed=seed + 20)
    kahm_qmb_sum = summarize(kahm_qmb_pq, n_boot=n_boot, seed=seed + 30)
    kahm_full_sum = summarize(kahm_full_pq, n_boot=n_boot, seed=seed + 40)

    # Headline blocks
    print_method("Mixedbread (true)", mb_sum, k=k)
    print_method("IDF–SVD", idf_sum, k=k)
    print_method("KAHM(query→MB corpus)", kahm_qmb_sum, k=k)
    print_method("Full-KAHM (query→KAHM corpus)", kahm_full_sum, k=k)

    # Storyline A/B
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

    # Storyline C: alignment evidence
    print("\nStoryline C: Full-KAHM embeddings are aligned with MB (geometry + neighborhood overlap)")
    print("  Part C1: Embedding-space cosine alignment")

    cos_corpus = cosine_rowwise(emb_k, emb_mb)
    cos_query = cosine_rowwise(q_kahm, q_mb)
    pt_c, ci_c = _bootstrap_mean_ci(cos_corpus, n_boot=n_boot, seed=seed + 300)
    pt_q, ci_q = _bootstrap_mean_ci(cos_query, n_boot=n_boot, seed=seed + 301)
    print(f"    corpus cosine(KAHM, MB): {_fmt_ci(pt_c, ci_c, digits=4)}")
    print(f"    query  cosine(KAHM, MB): {_fmt_ci(pt_q, ci_q, digits=4)}")

    print("  Part C2: Retrieval-neighborhood overlap vs MB")
    sent_j_full = jaccard_topk_rows(kahm_full_idx, mb_idx, k=k)
    sent_f_full = overlap_frac_topk_rows(kahm_full_idx, mb_idx, k=k)
    law_j_full = law_jaccard_topk_rows(kahm_full_idx, mb_idx, law_arr, k=k)

    pt_sj, ci_sj = _bootstrap_mean_ci(sent_j_full, n_boot=n_boot, seed=seed + 310)
    pt_sf, ci_sf = _bootstrap_mean_ci(sent_f_full, n_boot=n_boot, seed=seed + 311)
    pt_lj, ci_lj = _bootstrap_mean_ci(law_j_full, n_boot=n_boot, seed=seed + 312)

    print(f"    sentence Jaccard@{k} (Full-KAHM vs MB): {_fmt_ci(pt_sj, ci_sj)}")
    print(f"    sentence overlap frac@{k}            : {_fmt_ci(pt_sf, ci_sf)}")
    print(f"    law-set Jaccard@{k} (Full-KAHM vs MB): {_fmt_ci(pt_lj, ci_lj)}")

    # Context: show Full-KAHM is *more* aligned to MB than IDF is.
    sent_j_idf = jaccard_topk_rows(idf_idx, mb_idx, k=k)
    law_j_idf = law_jaccard_topk_rows(idf_idx, mb_idx, law_arr, k=k)
    d_sj_pt, d_sj_ci = _bootstrap_paired_delta_ci(sent_j_full, sent_j_idf, n_boot=n_boot, seed=seed + 320)
    d_lj_pt, d_lj_ci = _bootstrap_paired_delta_ci(law_j_full, law_j_idf, n_boot=n_boot, seed=seed + 321)
    ok_sj = bool(np.isfinite(d_sj_ci[0]) and d_sj_ci[0] > 0)
    ok_lj = bool(np.isfinite(d_lj_ci[0]) and d_lj_ci[0] > 0)
    print("  Part C3: Alignment gain vs IDF–SVD (paired deltas)")
    print(f"    sentence Jaccard delta: (Full-KAHM−IDF) = {_fmt_delta(d_sj_pt, d_sj_ci)}  -> {'PASS' if ok_sj else 'FAIL'}")
    print(f"    law-set Jaccard delta : (Full-KAHM−IDF) = {_fmt_delta(d_lj_pt, d_lj_ci)}  -> {'PASS' if ok_lj else 'FAIL'}")
    print("    Interpretation: PASS means Full-KAHM neighborhoods are *statistically* closer to MB than IDF neighborhoods.")

    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()