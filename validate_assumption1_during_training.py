#!/usr/bin/env python3
"""validate_assumption1_during_training.py

Train law-specific KAHM query regressors exactly as in
`train_kahm_query_regressors_by_law.py`, but do not save the trained models.
Instead, compute an in-training diagnostic for Assumption 1 / inequality (54)
of the manuscript:

    ||Phi_c||^2_{L2(P_x)} >= N_c / N

This script fixes the cluster-alignment issue of post-hoc validation from saved
models by computing the cluster counts from the same training partition used to
fit each KAHM and then verifying that the reproduced initial cluster centers
match the centers returned by `train_kahm_regressor`.

What is reported
----------------
For each law and each cluster c, the script computes:
- LHS (empirical, original support): mean_i Phi_c(x_i)^2 over the ORIGINAL core
  training queries used to fit the law-specific KAHM.
- RHS (original support): N_c / N using the ORIGINAL core cluster counts.
- Margin: LHS - RHS.

Additionally, because the training pipeline augments singleton clusters with one
auxiliary sample, the script also reports the same quantities on the EFFECTIVE
augmented training support used internally by the KAHM trainer:
- LHS (augmented support): mean_i Phi_c(x_i)^2 over the AUGMENTED core support.
- RHS (augmented support): N_c^aug / N^aug using the post-augmentation counts.

Outputs
-------
- <out_prefix>_assumption1_report.md : single narrative report suitable for the manuscript text
- Optionally, with --write_details, the script also writes:
  - <out_prefix>_clusters.csv : cluster-level diagnostics
  - <out_prefix>_laws.csv     : law-level summaries
  - <out_prefix>_report.json  : global summary including worst-case margins

Notes
-----
- No trained models are saved.
- The script mirrors the law-wise split, validation holdout, KMeans settings,
  singleton handling, soft-parameter tuning, and optional NLMS refinement from
  the provided training wrapper.
- By default, the *primary* diagnostic is the original-core support because it
  corresponds to real queries rather than auxiliary singleton points. The
  augmented-support diagnostic is included for transparency because the trainer
  fits the KAHM classifier on that effective support.
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
import gc
import json
import re
import sys
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, ContextManager, Dict, Iterator, List, Optional, Protocol, Sequence, Tuple, cast, runtime_checkable

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

from kahm_regression import (
    kahm_regress,
    train_kahm_regressor,
    tune_soft_params,
)
from query_set import TRAIN_QUERY_SET, TEST_QUERY_SET  # type: ignore

try:
    from otfl_blocksafe import enable_otfl_blocksafe, _BLOCKSAFE_STATS  # type: ignore
except Exception:
    enable_otfl_blocksafe = None
    _BLOCKSAFE_STATS = None


# ----------------------------- Defaults -----------------------------
DEFAULT_IDF_SVD_MODEL = "idf_svd_model.joblib"
DEFAULT_QUERIES_NPZ = "queries_embedding_index.npz"
DEFAULT_QUERIES_NPZ_TRAIN = "queries_embedding_index_train.npz"
DEFAULT_QUERIES_NPZ_TEST = "queries_embedding_index_test.npz"
DEFAULT_OUT_PREFIX = "assumption1_margin"
DEFAULT_REPORT_SUPPORT = "core_original"
DEFAULT_REPORT_TOLERANCE = 3e-3

DEFAULT_N_CLUSTERS = 300
DEFAULT_SUBSPACE_DIM = 20
DEFAULT_NB = 100
DEFAULT_RANDOM_STATE = 0
DEFAULT_INPUT_SCALE = 1.0

DEFAULT_KMEANS_KIND = "full"
DEFAULT_KMEANS_BATCH_SIZE = 4096
DEFAULT_MAX_TRAIN_PER_CLUSTER = None
DEFAULT_MODEL_DTYPE = "float32"
DEFAULT_CLUSTER_CENTER_NORMALIZATION = "none"

DEFAULT_AE_CACHE_ROOT = "kahm_ae_cache"
DEFAULT_OVERWRITE_AE_DIR = False

DEFAULT_TUNE_SOFT = True
DEFAULT_TUNE_NLMS = False
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


# ----------------------------- Utility helpers -----------------------------
@runtime_checkable
class _ContextManagerLike(Protocol):
    def __enter__(self) -> Any: ...
    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any: ...


def _as_blocksafe_context(obj: Any) -> ContextManager[None]:
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


def as_float_ndarray(x: Any, *, min_dtype: np.dtype = np.dtype(np.float32)) -> np.ndarray:
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


def l2_normalize_columns(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = as_float_ndarray(x)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array; got shape={x.shape}")
    n = np.linalg.norm(x, axis=0, keepdims=True)
    return x / np.maximum(n, eps)


def should_auto_l2_normalize_targets(y: np.ndarray) -> bool:
    y = as_float_ndarray(y)
    if y.ndim != 2 or y.shape[1] == 0:
        return False
    norms = np.linalg.norm(y, axis=0)
    p10, p50, p90 = np.percentile(norms, [10, 50, 90]).tolist()
    return (0.90 <= p50 <= 1.10) and (p10 >= 0.80) and (p90 <= 1.20)


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


def embed_idf_svd_queries(idf_svd_model_path: str, texts: Sequence[str]) -> np.ndarray:
    import joblib

    pipe = joblib.load(idf_svd_model_path)
    x = pipe.transform(list(texts))
    x = as_float_ndarray(x)
    x = l2_normalize_rows(x)
    return x


def load_precomputed_mb_queries_npz(path: str, query_ids: Sequence[str]) -> np.ndarray:
    d = np.load(path, allow_pickle=False)
    if "query_id" not in d or "embeddings" not in d:
        raise ValueError(
            f"Queries NPZ '{path}' must contain keys 'query_id' and 'embeddings'. Keys: {list(d.keys())}"
        )

    qid_npz = np.asarray(d["query_id"])
    y_npz = as_float_ndarray(d["embeddings"])
    if qid_npz.ndim != 1 or y_npz.ndim != 2:
        raise ValueError(
            f"Queries NPZ '{path}': expected query_id (Q,), embeddings (Q,D); got {qid_npz.shape}, {y_npz.shape}"
        )

    map_npz = {str(qid_npz[i]): i for i in range(qid_npz.shape[0])}
    missing = [qid for qid in query_ids if qid not in map_npz]
    if missing:
        raise ValueError(f"Queries NPZ '{path}' missing {len(missing)} query_ids. Example: {missing[:10]}")

    y = np.vstack([y_npz[map_npz[qid]] for qid in query_ids]).astype(y_npz.dtype, copy=False)
    y = l2_normalize_rows(y)
    return y


def embed_mb_queries_on_the_fly(model_name: str, device: str, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    y = model.encode(
        list(texts),
        batch_size=int(batch_size),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    y = as_float_ndarray(y)
    y = l2_normalize_rows(y)
    del model
    gc.collect()
    return y


def _sanitize_for_path(s: str, *, max_len: int = 64) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s).strip()).strip("._-")
    if not cleaned:
        cleaned = "unknown"
    return cleaned[:max_len]


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


# ----------------------------- Partition reproduction -----------------------------
@dataclass
class PartitionDiagnostics:
    x_aug: np.ndarray               # (D_in, N_aug)
    y_aug: np.ndarray               # (D_out, N_aug)
    labels_orig_mapped: np.ndarray  # (N_orig,)
    labels_aug_mapped: np.ndarray   # (N_aug,)
    counts_orig: np.ndarray         # (C_eff,)
    counts_aug: np.ndarray          # (C_eff,)
    cluster_centers_init: np.ndarray  # (D_out, C_eff)
    n_clusters_eff: int
    n_orig: int
    n_aug: int
    singleton_count_before_aug: int
    cluster_center_normalization_applied: str


def reproduce_training_partition(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_clusters: int,
    random_state: Optional[int],
    kmeans_kind: str,
    kmeans_batch_size: int,
    cluster_center_normalization: str,
    singleton_strategy: str = "augment",
    singleton_aux_mix: float = 0.1,
) -> PartitionDiagnostics:
    """Reproduce the output-clustering part of train_kahm_regressor exactly enough
    to recover cluster counts and row ordering before calling the real trainer.

    Inputs are column-major: x=(D_in,N), y=(D_out,N).
    """
    x = as_float_ndarray(x)
    y = as_float_ndarray(y)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be 2D matrices shaped (D, N).")
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"Sample mismatch: x.shape={x.shape}, y.shape={y.shape}")

    d_in, n = x.shape
    n_orig = int(n)
    if int(n_clusters) > n:
        raise ValueError(f"n_clusters={int(n_clusters)} cannot exceed N={int(n)}")

    y_t: Optional[np.ndarray] = None
    skip_kmeans = int(n_clusters) == int(n)
    if skip_kmeans:
        labels_zero_based = np.arange(int(n), dtype=np.int64)
        kmeans = None
    else:
        y_t = y.T
        kind = str(kmeans_kind).lower().strip()
        if kind not in ("auto", "full", "minibatch"):
            raise ValueError("kmeans_kind must be one of {'auto','full','minibatch'}")
        use_minibatch = (kind == "minibatch") or (kind == "auto" and int(n_clusters) >= 2000)
        if use_minibatch:
            kmeans = MiniBatchKMeans(
                n_clusters=int(n_clusters),
                random_state=random_state,
                batch_size=int(kmeans_batch_size),
                n_init="auto",
                reassignment_ratio=0.01,
            )
        else:
            kmeans = KMeans(
                n_clusters=int(n_clusters),
                random_state=random_state,
                n_init="auto",
            )
        kmeans.fit(y_t)
        labels_zero_based = kmeans.labels_.astype(np.int64, copy=False)

    counts_before = np.bincount(labels_zero_based, minlength=int(n_clusters))
    singletons = np.where(counts_before == 1)[0]

    # Record labels for ORIGINAL samples before augmentation. For augment strategy,
    # the original sample labels do not change; only auxiliary samples are appended.
    labels_zero_based_orig = labels_zero_based.copy()

    if singletons.size > 0:
        strategy = str(singleton_strategy).lower().strip()
        if strategy not in ("augment", "merge"):
            raise ValueError("singleton_strategy must be one of {'augment','merge'}")

        if strategy == "merge":
            if kmeans is None:
                raise ValueError("singleton_strategy='merge' is incompatible with n_clusters == N")
            centers = kmeans.cluster_centers_
            c2 = np.einsum("ij,ij->i", centers, centers)
            counts = counts_before.copy()
            for cl in singletons:
                sample_indices = np.where(labels_zero_based == cl)[0]
                if sample_indices.size != 1:
                    continue
                s_idx = int(sample_indices[0])
                y_sample = y[:, s_idx]
                y2 = float(np.dot(y_sample, y_sample))
                d2 = c2 + y2 - 2.0 * centers.dot(y_sample)
                candidates = np.where(counts >= 2)[0]
                if candidates.size == 0:
                    candidates = np.where(counts >= 1)[0]
                candidates = candidates[candidates != cl]
                if candidates.size == 0:
                    continue
                target = int(candidates[np.argmin(d2[candidates])])
                labels_zero_based[s_idx] = target
                labels_zero_based_orig[s_idx] = target
                counts[target] += 1
                counts[cl] -= 1
        else:
            order = np.argsort(labels_zero_based, kind="mergesort")
            sorted_labels = labels_zero_based[order]
            split_points = np.flatnonzero(np.diff(sorted_labels)) + 1
            groups = np.split(order, split_points)
            cluster_members: List[np.ndarray] = [np.empty(0, dtype=np.int64) for _ in range(int(n_clusters))]
            for g in groups:
                if g.size == 0:
                    continue
                lab = int(labels_zero_based[int(g[0])])
                cluster_members[lab] = g.astype(np.int64, copy=False)

            mix = float(singleton_aux_mix)
            mix = 0.0 if not np.isfinite(mix) else max(0.0, min(1.0, mix))

            x_nn = np.asarray(x.T, dtype=np.float32 if x.dtype == np.float64 else x.dtype)
            n_nn = int(x_nn.shape[0])
            if n_nn < 2:
                raise RuntimeError("Cannot augment singleton clusters with fewer than 2 total points.")
            k_nn = 5 if n_nn >= 5 else n_nn
            nn = NearestNeighbors(n_neighbors=k_nn, metric="euclidean", algorithm="auto")
            nn.fit(x_nn)
            _, nn_idx = nn.kneighbors(x_nn, return_distance=True)
            ar = np.arange(n_nn, dtype=nn_idx.dtype)
            nearest_global = nn_idx[:, 1].astype(np.int64, copy=False)
            self_mask = nearest_global == ar
            if np.any(self_mask):
                for kk in range(2, nn_idx.shape[1]):
                    repl = self_mask & (nn_idx[:, kk] != ar)
                    if np.any(repl):
                        nearest_global[repl] = nn_idx[repl, kk]
                    self_mask = nearest_global == ar
                    if not np.any(self_mask):
                        break
            if np.any(self_mask):
                bad = np.where(self_mask)[0]
                for i in bad:
                    nearest_global[int(i)] = int((int(i) + 1) % n_nn)

            x_new_cols: List[np.ndarray] = []
            y_new_cols: List[np.ndarray] = []
            new_labels: List[int] = []
            for cl in singletons:
                members = cluster_members[int(cl)]
                if members.size != 1:
                    continue
                s_idx = int(members[0])
                x_singleton = x[:, s_idx]
                nn_local = int(nearest_global[s_idx])
                if nn_local == s_idx:
                    nn_local = int((s_idx + 1) % int(x.shape[1]))
                x_neighbor = x[:, nn_local]
                x_aux = (1.0 - mix) * x_singleton + mix * x_neighbor
                x_norm = float(np.linalg.norm(x_singleton)) + 1e-12
                aux_norm = float(np.linalg.norm(x_aux)) + 1e-12
                x_aux = x_aux * (x_norm / aux_norm)
                x_new_cols.append(x_aux.reshape(-1, 1))
                y_new_cols.append(y[:, s_idx].reshape(-1, 1))
                new_labels.append(int(cl))

            if x_new_cols:
                x = np.concatenate([x] + x_new_cols, axis=1)
                y = np.concatenate([y] + y_new_cols, axis=1)
                labels_zero_based = np.concatenate(
                    [labels_zero_based, np.asarray(new_labels, dtype=labels_zero_based.dtype)],
                    axis=0,
                )
                n = int(x.shape[1])

    used_clusters = np.unique(labels_zero_based)
    n_clusters_eff = int(used_clusters.size)
    map_arr = np.full(int(n_clusters), -1, dtype=np.int32)
    map_arr[used_clusters.astype(np.int64)] = np.arange(n_clusters_eff, dtype=np.int32)

    labels_aug_mapped = map_arr[labels_zero_based.astype(np.int64)]
    labels_orig_mapped = map_arr[labels_zero_based_orig.astype(np.int64)]
    if np.any(labels_aug_mapped < 0) or np.any(labels_orig_mapped < 0):
        raise RuntimeError("Internal remapping error while reproducing training partition.")

    if skip_kmeans and int(n_clusters_eff) == int(n_orig):
        cluster_centers = y[:, :n_orig].copy()
    else:
        d_out = int(y.shape[0])
        cluster_centers = np.zeros((d_out, n_clusters_eff), dtype=y.dtype)
        for new_c in range(n_clusters_eff):
            mask = labels_aug_mapped == new_c
            cluster_centers[:, new_c] = y[:, mask].mean(axis=1)

    cc_req = str(cluster_center_normalization).lower().strip()
    cc_applied = "none"
    if cc_req in ("l2", "auto_l2"):
        do_norm = (cc_req == "l2") or should_auto_l2_normalize_targets(y)
        if do_norm:
            cluster_centers = l2_normalize_columns(cluster_centers)
            cc_applied = "l2"

    counts_orig = np.bincount(labels_orig_mapped, minlength=n_clusters_eff)
    counts_aug = np.bincount(labels_aug_mapped, minlength=n_clusters_eff)

    return PartitionDiagnostics(
        x_aug=x,
        y_aug=y,
        labels_orig_mapped=labels_orig_mapped.astype(np.int64, copy=False),
        labels_aug_mapped=labels_aug_mapped.astype(np.int64, copy=False),
        counts_orig=counts_orig.astype(np.int64, copy=False),
        counts_aug=counts_aug.astype(np.int64, copy=False),
        cluster_centers_init=as_float_ndarray(cluster_centers),
        n_clusters_eff=n_clusters_eff,
        n_orig=n_orig,
        n_aug=int(x.shape[1]),
        singleton_count_before_aug=int(singletons.size),
        cluster_center_normalization_applied=cc_applied,
    )


# ----------------------------- Diagnostics helpers -----------------------------
def probabilities_for_inputs(
    model: dict,
    x_eval_col: np.ndarray,
    *,
    batch_size: int = 1024,
) -> np.ndarray:
    _, p = cast(
        Tuple[np.ndarray, np.ndarray],
        kahm_regress(
            model,
            x_eval_col,
            mode="soft",
            return_probabilities=True, # type: ignore[arg-type]
            batch_size=int(batch_size),
            show_progress=False,
        ),
    )
    p = as_float_ndarray(p)
    return p


def summarize_support(
    law: str,
    support_name: str,
    p: np.ndarray,
    counts: np.ndarray,
    n_support: int,
    *,
    alpha_used: Optional[float],
    topk_used: Optional[int],
    centers_alignment_ok: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    lhs = np.mean(np.square(p), axis=1)
    rhs = counts.astype(np.float64) / float(n_support)
    margin = lhs - rhs

    rows: List[Dict[str, Any]] = []
    for c in range(int(len(counts))):
        rows.append({
            "law": law,
            "support": support_name,
            "cluster_index_1based": c + 1,
            "lhs_empirical_l2_sq": float(lhs[c]),
            "rhs_count_fraction": float(rhs[c]),
            "margin": float(margin[c]),
            "Nc": int(counts[c]),
            "N": int(n_support),
            "violation": bool(margin[c] < 0.0),
            "soft_alpha": (None if alpha_used is None else float(alpha_used)),
            "soft_topk": (None if topk_used is None else int(topk_used)),
            "centers_alignment_ok": bool(centers_alignment_ok),
        })

    summary = {
        "law": law,
        "support": support_name,
        "n_clusters": int(len(counts)),
        "N": int(n_support),
        "min_margin": float(np.min(margin)) if margin.size else float("nan"),
        "mean_margin": float(np.mean(margin)) if margin.size else float("nan"),
        "median_margin": float(np.median(margin)) if margin.size else float("nan"),
        "p05_margin": float(np.percentile(margin, 5)) if margin.size else float("nan"),
        "num_violations": int(np.sum(margin < 0.0)),
        "violation_rate": float(np.mean(margin < 0.0)) if margin.size else float("nan"),
        "worst_cluster_index_1based": (None if margin.size == 0 else int(np.argmin(margin) + 1)),
        "soft_alpha": (None if alpha_used is None else float(alpha_used)),
        "soft_topk": (None if topk_used is None else int(topk_used)),
        "centers_alignment_ok": bool(centers_alignment_ok),
    }
    return rows, summary


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    import csv

    rows = list(rows)
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            f.write("")
        return

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_float(x: Any, digits: int = 6) -> str:
    try:
        return f"{float(x):.{digits}g}"
    except Exception:
        return str(x)


def write_single_report(
    path: Path,
    *,
    support: str,
    tolerance: float,
    global_summary: Dict[str, Any],
    law_rows: Sequence[Dict[str, Any]],
    config: Dict[str, Any],
) -> None:
    exact_ok = int(global_summary.get("num_violations", 0)) == 0
    worst_margin = float(global_summary.get("worst_margin", float("nan")))
    tol_ok = bool(worst_margin >= -float(tolerance))
    support_label = "original core training support" if support == "core_original" else "augmented core training support"

    lines: List[str] = []
    lines.append("# Assumption 1 empirical validation report")
    lines.append("")
    lines.append("This report summarizes the sample analogue of inequality (54) for the law-wise trained KAHM feature maps:")
    lines.append("")
    lines.append(r"$$\widehat{\|\Phi_c\|^2_{L_2}} = \frac{1}{N}\sum_{i=1}^{N} \Phi_c(x_i)^2 \ge \frac{N_c}{N}. $$")
    lines.append("")
    lines.append(f"Primary support: **{support}** ({support_label}).")
    lines.append("")
    if exact_ok:
        lines.append("## Headline result")
        lines.append("")
        lines.append("The sample analogue of (54) is satisfied **exactly** for all evaluated clusters on the selected support.")
    elif tol_ok:
        lines.append("## Headline result")
        lines.append("")
        lines.append(
            f"The law-wise trained KAHM feature maps satisfy the sample analogue of (54) **up to a very small absolute deviation** on the selected support. "
            f"Using absolute tolerance {tolerance:.6g}, the condition holds for all evaluated clusters because the worst-case margin is {worst_margin:.6g}."
        )
    else:
        lines.append("## Headline result")
        lines.append("")
        lines.append(
            f"The sample analogue of (54) is **not** satisfied within the requested tolerance {tolerance:.6g} on the selected support. "
            f"The worst-case margin is {worst_margin:.6g}."
        )
    lines.append("")
    lines.append("## Global summary")
    lines.append("")
    lines.append(f"- Laws evaluated: {int(config['n_laws'])}")
    lines.append(f"- Clusters evaluated: {int(global_summary['num_clusters'])}")
    lines.append(f"- Worst-case margin (LHS - RHS): {global_summary['worst_margin']:.12f}")
    lines.append(f"- Median margin: {global_summary['median_margin']:.12f}")
    lines.append(f"- Mean margin: {global_summary['mean_margin']:.12f}")
    lines.append(f"- 5th percentile margin: {global_summary['p05_margin']:.12f}")
    lines.append(f"- Worst case: law {global_summary['worst_law']}, cluster {int(global_summary['worst_cluster_index_1based'])}")
    lines.append(f"- Worst-case LHS: {global_summary['worst_lhs']:.12f}")
    lines.append(f"- Worst-case RHS: {global_summary['worst_rhs']:.12f}")
    lines.append(f"- Exact violations (margin < 0): {int(global_summary['num_violations'])} / {int(global_summary['num_clusters'])}")
    lines.append(f"- Tolerance check (margin >= -{tolerance:.6g}): {'PASS' if tol_ok else 'FAIL'}")
    lines.append("")
    lines.append("## Interpretation for the theory section")
    lines.append("")
    if tol_ok:
        lines.append(
            "This empirical diagnostic supports the statement that, in the studied deployment regime, the sample analogue of Assumption 1 is satisfied up to a very small absolute deviation. "
            "This does not prove the population-level premise exactly, but it shows that the realized law-wise feature maps are extremely close to the required lower bound on the selected support."
        )
    else:
        lines.append(
            "This empirical diagnostic does not support the exact lower-bound premise at the requested tolerance. "
            "The theoretical use of Assumption 1 should therefore be presented as a modeling premise rather than an empirically verified property of the trained feature maps."
        )
    lines.append("")
    lines.append("## Law-wise worst margins")
    lines.append("")
    lines.append("| Law | N | Clusters | Min margin | Median margin | Mean margin | Exact violations |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    law_rows_sorted = sorted(law_rows, key=lambda r: (float(r.get('min_margin', 0.0)), str(r.get('law', ''))))
    for row in law_rows_sorted:
        lines.append(
            f"| {row['law']} | {int(row['N'])} | {int(row['n_clusters'])} | {float(row['min_margin']):.12f} | {float(row['median_margin']):.12f} | {float(row['mean_margin']):.12f} | {int(row['num_violations'])} |"
        )
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- Requested clusters per law: {config['n_clusters_requested']}")
    lines.append(f"- Validation split fraction: {config['val_fraction']}")
    lines.append(f"- Soft-parameter tuning enabled: {config['tune_soft']}")
    lines.append("- NLMS tuning enabled: False")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Train law-specific KAHM query regressors and validate Assumption 1 / inequality (54) "
            "immediately after training, without saving models."
        )
    )

    p.add_argument("--idf_svd_model", default=DEFAULT_IDF_SVD_MODEL)
    p.add_argument("--queries_npz", default="")
    p.add_argument("--queries_npz_train", default=DEFAULT_QUERIES_NPZ_TRAIN)
    p.add_argument("--queries_npz_test", default=DEFAULT_QUERIES_NPZ_TEST)
    p.add_argument("--require_npz", action="store_true")
    p.add_argument("--out_prefix", default=DEFAULT_OUT_PREFIX)
    p.add_argument("--report_support", default=DEFAULT_REPORT_SUPPORT, choices=["core_original", "core_augmented"])
    p.add_argument("--report_tolerance", type=float, default=DEFAULT_REPORT_TOLERANCE)
    p.add_argument("--write_details", action="store_true", help="Also write detailed CSV/JSON audit files in addition to the single markdown report.")

    p.add_argument("--mb_model", default="mixedbread-ai/deepset-mxbai-embed-de-large-v1")
    p.add_argument("--mb_device", default="cpu")
    p.add_argument("--mb_batch", type=int, default=64)
    p.add_argument("--force_mb_on_the_fly", action="store_true")

    p.add_argument("--model_id", default=None)
    p.add_argument("--ae_cache_root", default=DEFAULT_AE_CACHE_ROOT)
    p.add_argument("--ae_dir", default=None)
    p.add_argument("--overwrite_ae_dir", action="store_true", default=DEFAULT_OVERWRITE_AE_DIR)

    p.add_argument("--n_clusters", type=int, default=DEFAULT_N_CLUSTERS)
    p.add_argument("--subspace_dim", type=int, default=DEFAULT_SUBSPACE_DIM)
    p.add_argument("--nb", type=int, default=DEFAULT_NB)
    p.add_argument("--random_state", type=int, default=DEFAULT_RANDOM_STATE)
    p.add_argument("--input_scale", type=float, default=DEFAULT_INPUT_SCALE)

    p.add_argument("--kmeans_kind", default=DEFAULT_KMEANS_KIND, choices=["auto", "full", "minibatch"])
    p.add_argument("--kmeans_batch_size", type=int, default=DEFAULT_KMEANS_BATCH_SIZE)
    p.add_argument("--max_train_per_cluster", type=int, default=DEFAULT_MAX_TRAIN_PER_CLUSTER)
    p.add_argument("--model_dtype", default=DEFAULT_MODEL_DTYPE, choices=["float32", "float64"])
    p.add_argument("--cluster_center_normalization", default=DEFAULT_CLUSTER_CENTER_NORMALIZATION, choices=["none", "l2", "auto_l2"])

    p.add_argument("--val_fraction", type=float, default=DEFAULT_VAL_FRACTION)
    p.add_argument("--val_max_samples", type=int, default=DEFAULT_VAL_MAX_SAMPLES)
    p.add_argument("--tune_soft", action="store_true", default=DEFAULT_TUNE_SOFT)
    p.add_argument("--soft_alphas", default=",".join(str(x) for x in DEFAULT_SOFT_ALPHAS))
    p.add_argument("--soft_topks", default=",".join("none" if x is None else str(x) for x in DEFAULT_SOFT_TOPKS))

    p.add_argument("--blocksafe", action="store_true", default=DEFAULT_BLOCKSAFE_ENABLED)
    p.add_argument("--blocksafe_backend", default=DEFAULT_BLOCKSAFE_BACKEND, choices=["threading", "multiprocessing"])
    p.add_argument("--blocksafe_jitter_std", type=float, default=DEFAULT_BLOCKSAFE_JITTER_STD)
    p.add_argument("--blocksafe_jitter_tries", type=int, default=DEFAULT_BLOCKSAFE_JITTER_TRIES)
    p.add_argument("--blocksafe_jitter_growth", type=float, default=DEFAULT_BLOCKSAFE_JITTER_GROWTH)
    p.add_argument("--blocksafe_eps_factor", type=float, default=DEFAULT_BLOCKSAFE_EPS_FACTOR)
    p.add_argument("--blocksafe_log_first", type=int, default=DEFAULT_BLOCKSAFE_LOG_FIRST)
    p.add_argument("--blocksafe_l2_normalized", action="store_true", default=DEFAULT_BLOCKSAFE_L2_NORMALIZED)

    p.add_argument(
        "--allow_center_mismatch",
        action="store_true",
        help="Continue with a warning instead of failing when reproduced cluster_centers_init do not match the trained model.",
    )
    p.add_argument("--center_atol", type=float, default=1e-6)
    p.add_argument("--center_rtol", type=float, default=1e-5)
    p.add_argument("--prob_batch_size", type=int, default=1024)
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

    out_prefix = Path(str(args.out_prefix)).expanduser()
    out_dir = out_prefix.parent if str(out_prefix.parent) else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_prefix.name

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
    x_train_all = embed_idf_svd_queries(idf_svd_model_path, train_texts)
    _ = embed_idf_svd_queries(idf_svd_model_path, test_texts)  # keep parity with train wrapper setup

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
        y_train_all = load_precomputed_mb_queries_npz(str(npz_train), train_ids)
        print(f"Loading precomputed Mixedbread TEST  query embeddings: {npz_test}")
        _ = load_precomputed_mb_queries_npz(str(npz_test), test_ids)  # keep parity with train wrapper
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
        y_train_all = embed_mb_queries_on_the_fly(
            str(args.mb_model),
            str(args.mb_device),
            train_texts,
            batch_size=int(args.mb_batch),
        )
        # TEST embeddings not needed for assumption validation, but retained for parity.
        _ = embed_mb_queries_on_the_fly(
            str(args.mb_model),
            str(args.mb_device),
            test_texts,
            batch_size=int(args.mb_batch),
        )

    train_idx_by_law: Dict[str, List[int]] = {}
    test_idx_by_law: Dict[str, List[int]] = {}
    for i, law in enumerate(train_laws):
        train_idx_by_law.setdefault(law, []).append(i)
    for i, law in enumerate(test_laws):
        test_idx_by_law.setdefault(law, []).append(i)

    laws = sorted(train_idx_by_law.keys())
    print(f"\nTraining {len(laws)} law-specific regressors for Assumption 1 diagnostics ...")

    cluster_rows: List[Dict[str, Any]] = []
    law_rows: List[Dict[str, Any]] = []
    law_meta_rows: List[Dict[str, Any]] = []

    global_margins_original: List[float] = []
    global_margins_augmented: List[float] = []

    for law in laws:
        idx_tr = np.asarray(train_idx_by_law[law], dtype=np.int64)
        idx_te = np.asarray(test_idx_by_law.get(law, []), dtype=np.int64)
        x_tr = x_train_all[idx_tr]
        y_tr = y_train_all[idx_tr]
        n_train = int(x_tr.shape[0])
        n_test = int(idx_te.size)

        print("\n" + "=" * 90)
        print(f"Law: {law} | N_train={n_train} | N_test={n_test}")

        rng = np.random.RandomState(int(args.random_state))
        idx = np.arange(n_train, dtype=np.int64)
        rng.shuffle(idx)

        n_val = 0
        if float(args.val_fraction) > 0:
            n_val = int(round(float(args.val_fraction) * n_train))
            n_val = min(n_val, int(args.val_max_samples))
        n_val = max(0, min(n_val, n_train - 2))

        val_idx = idx[:n_val]
        core_idx = idx[n_val:]

        x_val = x_tr[val_idx] if n_val > 0 else None
        y_val = y_tr[val_idx] if n_val > 0 else None
        x_core = x_tr[core_idx]
        y_core = y_tr[core_idx]

        x_train_col = x_core.T
        y_train_col = y_core.T

        n_clusters_eff_req = int(min(int(args.n_clusters), int(x_core.shape[0])))
        n_clusters_eff_req = max(1, n_clusters_eff_req)
        if n_clusters_eff_req != int(args.n_clusters):
            print(
                f"Clamping n_clusters: requested={int(args.n_clusters)} -> effective={n_clusters_eff_req} "
                f"(law train_core N={int(x_core.shape[0])})"
            )

        part = reproduce_training_partition(
            x_train_col,
            y_train_col,
            n_clusters=n_clusters_eff_req,
            random_state=int(args.random_state),
            kmeans_kind=str(args.kmeans_kind),
            kmeans_batch_size=int(args.kmeans_batch_size),
            cluster_center_normalization=str(args.cluster_center_normalization),
            singleton_strategy="augment",
            singleton_aux_mix=0.1,
        )

        safe_law = _sanitize_for_path(law)
        base_model_id = str(args.model_id) if args.model_id else f"assumption1__law={safe_law}"
        run_model_id = f"{base_model_id}__law={safe_law}"

        ae_dir = None
        if args.ae_dir is not None:
            ae_dir = str(Path(str(args.ae_dir)) / f"law={safe_law}")

        t0 = time.time()
        with _as_blocksafe_context(ctx):
            model = train_kahm_regressor(
                X=x_train_col,
                Y=y_train_col,
                n_clusters=n_clusters_eff_req,
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
        t_train = time.time() - t0
        print(f"Training time={t_train:.1f}s")

        model_centers_init = as_float_ndarray(model.get("cluster_centers_init"))
        centers_alignment_ok = (
            model_centers_init.shape == part.cluster_centers_init.shape
            and np.allclose(
                model_centers_init,
                part.cluster_centers_init,
                rtol=float(args.center_rtol),
                atol=float(args.center_atol),
            )
        )
        if int(model.get("n_clusters", -1)) != int(part.n_clusters_eff):
            centers_alignment_ok = False

        if not centers_alignment_ok:
            msg = (
                f"Reproduced cluster_centers_init do not match trained model for law={law}. "
                f"Cannot guarantee cluster-row alignment for Nc vs Phi_c."
            )
            if not bool(args.allow_center_mismatch):
                raise RuntimeError(msg)
            print(f"WARNING: {msg}", file=sys.stderr)

        tuning_result = None
        if bool(args.tune_soft):
            alphas = tuple(parse_float_list(str(args.soft_alphas)))
            topks = tuple(parse_topk_list(str(args.soft_topks)))
            topk_candidates_eff = [k for k in topks if (k is not None and k <= int(part.n_clusters_eff))]
            if not topk_candidates_eff:
                topk_candidates_eff = [1]
                print(f"[{law}] WARNING: no valid topk candidates <= n_clusters_eff={part.n_clusters_eff}; using topk=1 only.")
            if x_val is None or y_val is None:
                print("WARNING: tune_soft requested, but validation split is empty. Skipping tuning.")
            else:
                print("Tuning soft parameters on validation set...")
                tuning_result = tune_soft_params(
                    model,
                    x_val.T,
                    y_val.T,
                    alphas=alphas,
                    topks=topk_candidates_eff,
                    n_jobs=1,
                    verbose=True,
                )

        nlms_results = None

        alpha_used = model.get("soft_alpha", None)
        topk_used = model.get("soft_topk", None)

        p_orig = probabilities_for_inputs(model, x_train_col, batch_size=int(args.prob_batch_size))
        rows_orig, summary_orig = summarize_support(
            law,
            "core_original",
            p_orig,
            part.counts_orig,
            part.n_orig,
            alpha_used=alpha_used,
            topk_used=topk_used,
            centers_alignment_ok=centers_alignment_ok,
        )

        p_aug = probabilities_for_inputs(model, part.x_aug, batch_size=int(args.prob_batch_size))
        rows_aug, summary_aug = summarize_support(
            law,
            "core_augmented",
            p_aug,
            part.counts_aug,
            part.n_aug,
            alpha_used=alpha_used,
            topk_used=topk_used,
            centers_alignment_ok=centers_alignment_ok,
        )

        cluster_rows.extend(rows_orig)
        cluster_rows.extend(rows_aug)
        law_rows.append(summary_orig)
        law_rows.append(summary_aug)
        global_margins_original.extend([r["margin"] for r in rows_orig])
        global_margins_augmented.extend([r["margin"] for r in rows_aug])

        law_meta_rows.append({
            "law": law,
            "n_train_queries": int(n_train),
            "n_test_queries": int(n_test),
            "n_train_core_original": int(part.n_orig),
            "n_train_core_augmented": int(part.n_aug),
            "n_val": int(0 if x_val is None else x_val.shape[0]),
            "n_clusters_effective": int(part.n_clusters_eff),
            "singleton_clusters_before_augmentation": int(part.singleton_count_before_aug),
            "training_time_sec": float(t_train),
            "soft_alpha": (None if alpha_used is None else float(alpha_used)),
            "soft_topk": (None if topk_used is None else int(topk_used)),
            "centers_alignment_ok": bool(centers_alignment_ok),
            "cluster_center_normalization_applied": str(part.cluster_center_normalization_applied),
            "nlms_applied": False,
        })

        print(
            f"Assumption 1 margins | law={law} | core_original min={summary_orig['min_margin']:.6f} "
            f"| core_augmented min={summary_aug['min_margin']:.6f}"
        )

        del model, part, p_orig, p_aug
        gc.collect()

    def _global_summary_for_support(rows: Sequence[Dict[str, Any]], support: str) -> Dict[str, Any]:
        rows = [r for r in rows if r.get("support") == support]
        if not rows:
            return {"support": support, "num_clusters": 0}
        margins = np.asarray([float(r["margin"]) for r in rows], dtype=np.float64)
        worst_idx = int(np.argmin(margins))
        worst = rows[worst_idx]
        return {
            "support": support,
            "num_clusters": int(len(rows)),
            "min_margin": float(np.min(margins)),
            "mean_margin": float(np.mean(margins)),
            "median_margin": float(np.median(margins)),
            "p05_margin": float(np.percentile(margins, 5)),
            "num_violations": int(np.sum(margins < 0.0)),
            "violation_rate": float(np.mean(margins < 0.0)),
            "worst_law": str(worst["law"]),
            "worst_cluster_index_1based": int(worst["cluster_index_1based"]),
            "worst_lhs": float(worst["lhs_empirical_l2_sq"]),
            "worst_rhs": float(worst["rhs_count_fraction"]),
            "worst_margin": float(worst["margin"]),
        }

    support_selected = str(args.report_support)
    global_summary_selected = _global_summary_for_support(cluster_rows, support_selected)
    law_rows_selected = [r for r in law_rows if r.get("support") == support_selected]

    report_md = out_dir / f"{stem}_assumption1_report.md"
    config_for_report = {
        "n_laws": int(len(laws)),
        "n_clusters_requested": int(args.n_clusters),
        "val_fraction": float(args.val_fraction),
        "tune_soft": bool(args.tune_soft),
    }
    write_single_report(
        report_md,
        support=support_selected,
        tolerance=float(args.report_tolerance),
        global_summary=global_summary_selected,
        law_rows=law_rows_selected,
        config=config_for_report,
    )

    print("\n" + "=" * 90)
    print(
        f"Primary report {support_selected}: worst_margin={global_summary_selected['worst_margin']:.6f} | "
        f"median_margin={global_summary_selected['median_margin']:.6f} | "
        f"worst=({global_summary_selected['worst_law']}, c={global_summary_selected['worst_cluster_index_1based']})"
    )
    print(f"\nWrote: {report_md}")

    if bool(args.write_details):
        clusters_csv = out_dir / f"{stem}_clusters.csv"
        laws_csv = out_dir / f"{stem}_laws.csv"
        meta_csv = out_dir / f"{stem}_law_meta.csv"
        report_json = out_dir / f"{stem}_report.json"

        write_csv(clusters_csv, cluster_rows)
        write_csv(laws_csv, law_rows)
        write_csv(meta_csv, law_meta_rows)

        report = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "script": os.path.basename(__file__),
            "summary": {
                "n_laws": int(len(laws)),
                "global": [
                    _global_summary_for_support(cluster_rows, "core_original"),
                    _global_summary_for_support(cluster_rows, "core_augmented"),
                ],
            },
            "paths": {
                "clusters_csv": str(clusters_csv),
                "laws_csv": str(laws_csv),
                "law_meta_csv": str(meta_csv),
                "report_json": str(report_json),
                "assumption1_report_md": str(report_md),
            },
            "config": {
                "idf_svd_model": str(args.idf_svd_model),
                "queries_npz": (str(args.queries_npz).strip() or None),
                "queries_npz_train": (str(npz_train) if npz_train is not None else None),
                "queries_npz_test": (str(npz_test) if npz_test is not None else None),
                "n_clusters_requested": int(args.n_clusters),
                "subspace_dim": int(args.subspace_dim),
                "nb": int(args.nb),
                "random_state": int(args.random_state),
                "input_scale": float(args.input_scale),
                "kmeans_kind": str(args.kmeans_kind),
                "kmeans_batch_size": int(args.kmeans_batch_size),
                "max_train_per_cluster": (None if args.max_train_per_cluster is None else int(args.max_train_per_cluster)),
                "model_dtype": str(args.model_dtype),
                "cluster_center_normalization": str(args.cluster_center_normalization),
                "val_fraction": float(args.val_fraction),
                "val_max_samples": int(args.val_max_samples),
                "tune_soft": bool(args.tune_soft),
                "tune_nlms": False,
                "soft_alphas": list(parse_float_list(str(args.soft_alphas))),
                "soft_topks": list(parse_topk_list(str(args.soft_topks))),
                "allow_center_mismatch": bool(args.allow_center_mismatch),
                "center_atol": float(args.center_atol),
                "center_rtol": float(args.center_rtol),
                "report_support": support_selected,
                "report_tolerance": float(args.report_tolerance),
            },
        }

        with report_json.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"Wrote: {clusters_csv}")
        print(f"Wrote: {laws_csv}")
        print(f"Wrote: {meta_csv}")
        print(f"Wrote: {report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
