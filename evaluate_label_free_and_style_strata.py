#!/usr/bin/env python3
"""
evaluate_label_free_and_style_strata.py

Purpose
-------
Produce diagnostic retrieval results for:

1) label-free queries
2) query-style strata

The script is designed to match the data / evaluation contract already used by
existing comparison scripts in this project:

    cheap lexical query features (IDF–SVD) -> retrieval against a frozen corpus

It reuses the same query-set convention (module.attr, e.g.
`query_set.TEST_QUERY_SET`) and the same query-embedding NPZ schema
(`query_id`, `embeddings`).

What this script evaluates
--------------------------
- Full test set
- Label-free subset (no mention of the gold law abbreviation)
- Strict label-free subset (no mention of any law abbreviation in the label universe)
- Gold-label-mentioned subset
- Per-style subsets (e.g. nl_short, nl_long, scenario, procedural, authority,
  keyword, fragment)

What counts as "label-free"
----------------------------
The generator injects literal law abbreviations into `query_text` only as
optional low-probability hints. This script therefore detects label mentions via
case-insensitive exact abbreviation matching against the query text.

Inputs / methods
----------------
The script automatically supports:
- IDF–SVD                        (recomputed from query text)
- Mixedbread (true)              (loaded from `--queries_npz_test`)
- KAHM(query→MB corpus)          (optional: loaded from NPZ or from model dir)
- Additional MB-space methods    (repeat `--method_npz Name=path/to/file.npz`)

All extra NPZ methods are assumed to live in the same semantic space as the
Mixedbread corpus index.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


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
    d = a - b
    rng = np.random.default_rng(int(seed))
    n = int(d.size)
    pt = float(np.mean(d))
    bs = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        bs[i] = float(np.mean(d[idx]))
    lo, hi = np.quantile(bs, [0.025, 0.975])
    return pt, (float(lo), float(hi))



def _md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    line1 = "| " + " | ".join(headers) + " |"
    line2 = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([line1, line2] + body)



def _fmt_ci(pt: float, ci: Tuple[float, float]) -> str:
    if not np.isfinite(pt):
        return "n/a"
    lo, hi = ci
    return f"{pt:.3f} [{lo:.3f}, {hi:.3f}]"



def _fmt_delta(pt: float, ci: Tuple[float, float]) -> str:
    if not np.isfinite(pt):
        return "n/a"
    lo, hi = ci
    sign = "+" if pt >= 0 else ""
    return f"{sign}{pt:.3f} [{lo:.3f}, {hi:.3f}]"


# -----------------------------------------------------------------------------
# Query-set loading
# -----------------------------------------------------------------------------
def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {e}") from e
    if not rows:
        raise ValueError(f"Loaded empty JSONL query set from {path}")
    return rows



def load_query_set(spec: str) -> List[Dict[str, Any]]:
    spec = str(spec).strip()
    if spec.endswith(".jsonl") or spec.endswith(".json"):
        return _load_jsonl(spec)
    if "." not in spec:
        raise ValueError(
            "query-set spec must be module.attr (e.g. query_set.TEST_QUERY_SET) "
            "or a path to train/test JSONL"
        )
    mod_name, attr = spec.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    qs = getattr(mod, attr, None)
    if qs is None:
        raise AttributeError(f"Query-set attribute not found: {spec}")
    out = list(qs)
    if not out:
        raise ValueError(f"Loaded empty query set from {spec}")
    return out



def extract_query_fields(qs: Sequence[Mapping[str, Any]], *, name: str) -> Dict[str, List[str]]:
    ids: List[str] = []
    texts: List[str] = []
    laws: List[str] = []
    styles: List[str] = []
    topics: List[str] = []
    for i, q in enumerate(qs):
        qid = str(q.get("query_id", "")).strip()
        txt = str(q.get("query_text", "")).strip()
        law = str(q.get("consensus_law", "")).strip()
        sty = str(q.get("style", "")).strip()
        topic = str(q.get("topic_id", "")).strip()
        if not qid:
            raise ValueError(f"{name}[{i}] has empty query_id")
        if not txt:
            raise ValueError(f"{name}[{i}] has empty query_text")
        if not law:
            raise ValueError(f"{name}[{i}] has empty consensus_law")
        ids.append(qid)
        texts.append(txt)
        laws.append(law)
        styles.append(sty)
        topics.append(topic)
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name} has duplicate query_id values")
    return {
        "query_id": ids,
        "query_text": texts,
        "consensus_law": laws,
        "style": styles,
        "topic_id": topics,
    }


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



def build_search_index(emb: np.ndarray, *, n_threads: Optional[int] = None):
    try:
        import faiss  # type: ignore

        if n_threads is not None:
            try:
                faiss.omp_set_num_threads(int(n_threads))
            except Exception:
                pass
        x = np.ascontiguousarray(emb.astype(np.float32, copy=False))
        index: Any = faiss.IndexFlatIP(int(x.shape[1]))
        index.add(x)
        return ("faiss", index)
    except Exception:
        x = np.ascontiguousarray(emb.astype(np.float32, copy=False))
        return ("numpy", x)



def search_index(index_obj: Any, q_emb: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    kind, payload = index_obj
    q = np.ascontiguousarray(q_emb.astype(np.float32, copy=False))
    k = int(k)
    if kind == "faiss":
        scores, idx = payload.search(q, k)
        return scores, idx

    # Numpy fallback: inner-product search.
    x = payload
    sims = q @ x.T
    if sims.shape[1] <= k:
        idx = np.argsort(-sims, axis=1)
    else:
        part = np.argpartition(-sims, kth=np.arange(k), axis=1)[:, :k]
        part_scores = np.take_along_axis(sims, part, axis=1)
        order = np.argsort(-part_scores, axis=1)
        idx = np.take_along_axis(part, order, axis=1)
    scores = np.take_along_axis(sims, idx, axis=1)
    return scores, idx.astype(np.int64, copy=False)


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
# KAHM loading / inference
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
# Label-free / strata helpers
# -----------------------------------------------------------------------------
def compile_law_patterns(laws: Iterable[str]) -> Dict[str, re.Pattern[str]]:
    pats: Dict[str, re.Pattern[str]] = {}
    for law in sorted(set(str(x).strip() for x in laws if str(x).strip())):
        escaped = re.escape(law)
        pats[law] = re.compile(rf"(?<!\w){escaped}(?!\w)", flags=re.IGNORECASE | re.UNICODE)
    return pats



def detect_label_mentions(texts: Sequence[str], gold_laws: Sequence[str], all_laws: Sequence[str]) -> Dict[str, np.ndarray]:
    gold_patterns = compile_law_patterns(gold_laws)
    all_patterns = compile_law_patterns(all_laws)

    gold_mentioned = np.zeros((len(texts),), dtype=bool)
    any_law_mentioned = np.zeros((len(texts),), dtype=bool)
    mentioned_laws: List[List[str]] = []

    for i, (txt, gold) in enumerate(zip(texts, gold_laws)):
        text = str(txt)
        gold_pat = gold_patterns.get(str(gold).strip())
        gold_mentioned[i] = bool(gold_pat.search(text)) if gold_pat is not None else False

        row_hits: List[str] = []
        for law, pat in all_patterns.items():
            if pat.search(text):
                row_hits.append(law)
        any_law_mentioned[i] = bool(row_hits)
        mentioned_laws.append(row_hits)

    return {
        "gold_mentioned": gold_mentioned,
        "gold_free": ~gold_mentioned,
        "any_law_mentioned": any_law_mentioned,
        "any_law_free": ~any_law_mentioned,
        "mentioned_laws": np.asarray(mentioned_laws, dtype=object),
    }



def build_group_masks(styles: Sequence[str], label_info: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    styles_arr = np.asarray([str(x) for x in styles], dtype=object)
    groups: Dict[str, np.ndarray] = {
        "full": np.ones((styles_arr.shape[0],), dtype=bool),
        "label_free_gold": np.asarray(label_info["gold_free"], dtype=bool),
        "label_free_any_law": np.asarray(label_info["any_law_free"], dtype=bool),
        "gold_label_mentioned": np.asarray(label_info["gold_mentioned"], dtype=bool),
        "any_law_mentioned": np.asarray(label_info["any_law_mentioned"], dtype=bool),
    }
    for sty in list(dict.fromkeys(styles_arr.tolist())):
        if not sty:
            continue
        groups[f"style/{sty}"] = styles_arr == sty
    return groups


# -----------------------------------------------------------------------------
# Summaries / reporting
# -----------------------------------------------------------------------------
def summarize_group(
    *,
    per_query_by_method: Dict[str, Dict[int, PerQuery]],
    methods: Sequence[str],
    ks: Sequence[int],
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
            pq = per_query_by_method[method][int(k)]
            out["summary_by_k"][int(k)][method] = {}
            for mk in metric_keys:
                pt, ci = _bootstrap_mean_ci(getattr(pq, mk), n_boot=int(bootstrap), seed=int(seed) + int(k))
                out["summary_by_k"][int(k)][method][mk] = {"pt": pt, "ci": ci}
                if method != baseline_ref:
                    dpt, dci = _bootstrap_paired_delta_ci(
                        getattr(pq, mk),
                        getattr(per_query_by_method[baseline_ref][int(k)], mk),
                        n_boot=int(bootstrap),
                        seed=int(seed) + 1000 + int(k),
                    )
                    out["deltas_vs_baseline"][int(k)].setdefault(method, {})[mk] = {"pt": dpt, "ci": dci}
    return out



def build_markdown_report(
    *,
    methods: Sequence[str],
    ks: Sequence[int],
    group_summaries: Mapping[str, Any],
    group_counts: Mapping[str, int],
    baseline_ref: str,
    label_stats: Mapping[str, Any],
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
    lines.append("# Label-free and query-style diagnostic report")
    lines.append("")
    lines.append(
        "This report evaluates the same retrieval outputs on the full test set, label-free subsets, and per-style strata. "
        "Only `IDF–SVD` is evaluated on the IDF index; all other methods are evaluated against the frozen semantic corpus index."
    )
    lines.append("")
    lines.append("## Query-group counts")
    lines.append("")
    lines.append(f"- Total queries: {int(group_counts.get('full', 0))}")
    lines.append(f"- Gold-label-mentioned: {int(group_counts.get('gold_label_mentioned', 0))}")
    lines.append(f"- Gold-label-free: {int(group_counts.get('label_free_gold', 0))}")
    lines.append(f"- Any-law-mentioned: {int(group_counts.get('any_law_mentioned', 0))}")
    lines.append(f"- Any-law-free: {int(group_counts.get('label_free_any_law', 0))}")
    if label_stats:
        lines.append(f"- Gold-label mention rate: {float(label_stats.get('gold_mention_rate', float('nan'))):.3f}")
        lines.append(f"- Any-law mention rate: {float(label_stats.get('any_law_mention_rate', float('nan'))):.3f}")
    lines.append("")

    group_order = [
        "full",
        "label_free_gold",
        "label_free_any_law",
        "gold_label_mentioned",
        "any_law_mentioned",
    ] + [g for g in group_summaries.keys() if g.startswith("style/")]

    seen = set()
    for group_name in group_order:
        if group_name not in group_summaries or group_name in seen:
            continue
        seen.add(group_name)
        summary = group_summaries[group_name]
        lines.append(f"## {group_name}")
        lines.append("")
        lines.append(f"Queries in group: {int(group_counts.get(group_name, 0))}")
        lines.append("")
        for mk, title in metric_names.items():
            lines.append(f"### {title}")
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
                lines.append(f"#### Paired deltas vs {baseline_ref}")
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
def parse_method_npz(items: Sequence[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for raw in items:
        s = str(raw).strip()
        if not s:
            continue
        if "=" not in s:
            raise ValueError(f"--method_npz expects NAME=PATH, got: {raw}")
        name, path = s.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"Invalid --method_npz value: {raw}")
        out.append((name, path))
    return out



def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate label-free queries and query-style strata under the existing retrieval contract.")

    p.add_argument("--test_query_set", default="query_set.TEST_QUERY_SET")
    p.add_argument("--idf_svd_model", default="idf_svd_model.joblib")
    p.add_argument("--queries_npz_test", default="queries_embedding_index_test.npz", help="Test query Mixedbread embeddings NPZ (query_id + embeddings).")

    p.add_argument("--kahm_model_dir", default="kahm_query_regressors_by_law", help="Optional directory with law-wise KAHM *.joblib regressors.")
    p.add_argument("--kahm_query_embeddings_npz", default="", help="Optional precomputed KAHM query embeddings for the test set.")
    p.add_argument("--kahm_mode", default="soft", choices=["soft", "hard"])
    p.add_argument("--kahm_batch", type=int, default=1024)

    p.add_argument(
        "--method_npz",
        action="append",
        default=[],
        help="Additional MB-space query embeddings in NAME=PATH.npz format. Repeatable.",
    )

    p.add_argument("--corpus_parquet", default="ris_sentences.parquet")
    p.add_argument("--mb_corpus_npz", default="embedding_index.npz")
    p.add_argument("--idf_corpus_npz", default="embedding_index_idf_svd.npz")
    p.add_argument("--threads", type=int, default=1)

    p.add_argument("--ks", default="3,5,10,15,20")
    p.add_argument("--predominance_fraction", type=float, default=0.1)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--baseline_ref", default="IDF–SVD")

    p.add_argument("--report_path", default="label_free_and_style_diagnostics/report.md")
    p.add_argument("--results_json_path", default="label_free_and_style_diagnostics/results.json")
    p.add_argument("--group_membership_csv", default="label_free_and_style_diagnostics/query_groups.csv")
    return p



def main() -> int:
    args = build_arg_parser().parse_args()

    ks = [int(x.strip()) for x in str(args.ks).split(",") if x.strip()]
    if not ks:
        raise ValueError("--ks must contain at least one cutoff")
    k_max = max(ks)

    test_qs = load_query_set(str(args.test_query_set))
    test_fields = extract_query_fields(test_qs, name="TEST_QUERY_SET")
    test_ids = test_fields["query_id"]
    test_texts = test_fields["query_text"]
    test_laws = test_fields["consensus_law"]
    test_styles = test_fields["style"]
    print(f"Loaded TEST queries: {len(test_ids)}")

    t0 = time.time()
    x_test = embed_idf_svd_queries(str(args.idf_svd_model), test_texts)
    y_test = load_query_npz(str(args.queries_npz_test), test_ids)
    print(f"Loaded lexical and teacher query embeddings in {time.time() - t0:.1f}s")

    method_embeddings: Dict[str, np.ndarray] = {
        "IDF–SVD": x_test,
        "Mixedbread (true)": y_test,
    }

    # Optional KAHM.
    if str(args.kahm_query_embeddings_npz).strip():
        print("Loading precomputed KAHM test embeddings ...")
        method_embeddings["KAHM(query→MB corpus)"] = load_query_npz(str(args.kahm_query_embeddings_npz), test_ids)
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
        method_embeddings["KAHM(query→MB corpus)"] = y_kahm

    # Optional additional MB-space NPZ methods.
    for name, path in parse_method_npz(args.method_npz):
        print(f"Loading extra method embeddings: {name} <- {path}")
        if name in method_embeddings:
            raise ValueError(f"Duplicate method name: {name}")
        method_embeddings[name] = load_query_npz(path, test_ids)

    baseline_ref = str(args.baseline_ref)
    if baseline_ref not in method_embeddings:
        raise ValueError(f"--baseline_ref '{baseline_ref}' not among methods: {sorted(method_embeddings.keys())}")

    print("Loading corpus bundles and building retrieval indices ...")
    df = load_corpus_parquet(str(args.corpus_parquet))
    mb_bundle = load_npz_bundle(str(args.mb_corpus_npz))
    idf_bundle = load_npz_bundle(str(args.idf_corpus_npz))
    aligned = align_by_common_sentence_ids(df, mb_bundle, idf_bundle)

    index_idf = build_search_index(aligned["emb_idf"], n_threads=int(args.threads))
    index_mb = build_search_index(aligned["emb_mb"], n_threads=int(args.threads))
    law_arr = np.asarray(aligned["law"], dtype=object)

    print("Running retrieval for all methods ...")
    retrieved_idx_by_method: Dict[str, np.ndarray] = {}
    for method, emb in method_embeddings.items():
        if method == "IDF–SVD":
            _, idx = search_index(index_idf, emb, k_max)
        else:
            _, idx = search_index(index_mb, emb, k_max)
        retrieved_idx_by_method[method] = idx

    label_info = detect_label_mentions(test_texts, test_laws, sorted(set(test_laws)))
    group_masks = build_group_masks(test_styles, label_info)
    group_counts = {name: int(mask.sum()) for name, mask in group_masks.items()}
    label_stats = {
        "gold_mention_rate": float(np.mean(label_info["gold_mentioned"])) if test_ids else float("nan"),
        "any_law_mention_rate": float(np.mean(label_info["any_law_mentioned"])) if test_ids else float("nan"),
    }

    # Save query-level group membership for auditability.
    if str(args.group_membership_csv).strip():
        gm_path = Path(str(args.group_membership_csv)).expanduser()
        gm_path.parent.mkdir(parents=True, exist_ok=True)
        gm_df = pd.DataFrame({
            "query_id": test_ids,
            "consensus_law": test_laws,
            "style": test_styles,
            "query_text": test_texts,
            "gold_label_mentioned": np.asarray(label_info["gold_mentioned"], dtype=bool),
            "any_law_mentioned": np.asarray(label_info["any_law_mentioned"], dtype=bool),
            "mentioned_laws": [";".join(xs) for xs in label_info["mentioned_laws"].tolist()],
        })
        gm_df.to_csv(gm_path, index=False)
        print(f"Wrote group-membership CSV: {gm_path}")

    methods = list(method_embeddings.keys())
    group_summaries: Dict[str, Any] = {}

    print("Computing subset summaries ...")
    for group_name, mask in group_masks.items():
        n_group = int(np.sum(mask))
        if n_group <= 0:
            print(f"Skipping empty group: {group_name}")
            continue
        print(f"  Group {group_name:24s} | n={n_group}")

        group_per_method: Dict[str, Dict[int, PerQuery]] = {}
        mask_idx = np.flatnonzero(mask)
        group_laws = [test_laws[i] for i in mask_idx.tolist()]

        for method in methods:
            idx_full = retrieved_idx_by_method[method]
            idx_sub = idx_full[mask_idx]
            group_per_method[method] = {}
            for k in ks:
                group_per_method[method][int(k)] = compute_per_query_metrics(
                    idx=idx_sub,
                    law_arr=law_arr,
                    consensus_laws=group_laws,
                    k=int(k),
                    predominance_fraction=float(args.predominance_fraction),
                )

        group_summaries[group_name] = summarize_group(
            per_query_by_method=group_per_method,
            methods=methods,
            ks=ks,
            bootstrap=int(args.bootstrap),
            seed=int(args.seed),
            baseline_ref=baseline_ref,
        )

    report_md = build_markdown_report(
        methods=methods,
        ks=ks,
        group_summaries=group_summaries,
        group_counts=group_counts,
        baseline_ref=baseline_ref,
        label_stats=label_stats,
    )

    if str(args.report_path).strip():
        report_path = Path(str(args.report_path)).expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_md, encoding="utf-8")
        print(f"Wrote report: {report_path}")

    if str(args.results_json_path).strip():
        payload = {
            "ks": ks,
            "methods": methods,
            "baseline_ref": baseline_ref,
            "group_counts": group_counts,
            "label_stats": label_stats,
            "group_summaries": group_summaries,
        }
        out_path = Path(str(args.results_json_path)).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote JSON: {out_path}")

    # Console summary at the largest k.
    k_star = int(max(ks))
    print("\n=== Quick summary at largest k ===")
    for group_name in ["full", "label_free_gold", "label_free_any_law"] + [g for g in group_summaries if g.startswith("style/")]:
        if group_name not in group_summaries:
            continue
        print(f"\n[{group_name}] n={group_counts.get(group_name, 0)}")
        for method in methods:
            entry = group_summaries[group_name]["summary_by_k"][k_star][method]
            print(
                f"{method:40s} | "
                f"MRR@{k_star}={entry['mrr_ul']['pt']:.3f} | "
                f"Hit@{k_star}={entry['hit']['pt']:.3f} | "
                f"Top1={entry['top1']['pt']:.3f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
