#!/usr/bin/env python3
"""ablate_kahm_query_regressors_by_law_n_clusters.py

Run an ablation study for the law-specific KAHM query-regressor trainer by
varying ``n_clusters`` across a fixed list of values, evaluating performance on
``TEST_QUERY_SET``, and saving a structured report.

This script mirrors the training/evaluation logic of
``train_kahm_query_regressors_by_law.py`` as closely as possible while adding:

- ablation over multiple ``n_clusters`` values (default: 100, 200, 300, 400)
- structured report generation (JSON, CSV, Markdown)
- optional metric plots (PNG)
- temporary artifact directories for saved regressors and caches that are
  deleted automatically after the report is written

Behavioral notes
----------------
- One KAHM regressor is trained per ``consensus_law``.
- For each law, ``n_clusters`` is clamped to the post-validation core-training
  sample count, matching the source trainer's behavior.
- Combined test-set evaluation uses the same distance-gated multi-model
  combination logic as the source trainer.
- Temporary model artifacts are written beneath a temporary directory by
  default; they are removed automatically once report generation finishes.
"""

from __future__ import annotations

import os

# Keep consistent with the source trainer defaults.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import csv
import gc
import hashlib
import io
import json
import re
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, ContextManager, Dict, Iterator, List, Optional, Protocol, Sequence, Tuple, cast, runtime_checkable

import numpy as np

from kahm_regression import (
    kahm_regress,
    save_kahm_regressor,
    train_kahm_regressor,
    tune_cluster_centers_nlms,
    tune_soft_params,
)
from query_set import TRAIN_QUERY_SET, TEST_QUERY_SET  # type: ignore
from combine_kahm_regressors_generalized import combine_kahm_regressors_distance_gated_multi

try:
    from otfl_blocksafe import enable_otfl_blocksafe, _BLOCKSAFE_STATS  # type: ignore
except Exception:
    enable_otfl_blocksafe = None
    _BLOCKSAFE_STATS = None


DEFAULT_IDF_SVD_MODEL = "idf_svd_model.joblib"
DEFAULT_QUERIES_NPZ = "queries_embedding_index.npz"  # optional combined file (back-compat)
DEFAULT_QUERIES_NPZ_TRAIN = "queries_embedding_index_train.npz"
DEFAULT_QUERIES_NPZ_TEST = "queries_embedding_index_test.npz"
DEFAULT_N_CLUSTERS_VALUES = (100, 200, 300, 400)

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

DEFAULT_EVAL_SOFT = True
DEFAULT_TUNE_SOFT = True
DEFAULT_TUNE_NLMS = True
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

DEFAULT_REPORT_DIR = "kahm_ablation_reports"
DEFAULT_REPORT_STEM = "kahm_query_regressors_by_law_n_clusters_ablation"
DEFAULT_COMBINED_MODE = "soft"
DEFAULT_COMBINED_BATCH_SIZE = 2048


@runtime_checkable
class _ContextManagerLike(Protocol):
    def __enter__(self) -> Any: ...

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any: ...


class Tee(io.TextIOBase):
    """Mirror stdout/stderr to multiple streams."""

    def __init__(self, *streams: io.TextIOBase) -> None:
        self._streams = streams

    def write(self, s: str) -> int:
        for stream in self._streams:
            stream.write(s)
            stream.flush()
        return len(s)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


@contextmanager
def tee_output(log_path: Path) -> Iterator[None]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        stdout_orig = sys.stdout
        stderr_orig = sys.stderr
        sys.stdout = cast(Any, Tee(cast(io.TextIOBase, stdout_orig), log_file))
        sys.stderr = cast(Any, Tee(cast(io.TextIOBase, stderr_orig), log_file))
        try:
            yield
        finally:
            sys.stdout = stdout_orig
            sys.stderr = stderr_orig


@contextmanager
def managed_temp_root(keep: bool, requested_root: Optional[str] = None) -> Iterator[Path]:
    if keep:
        if requested_root:
            root = Path(requested_root).expanduser().resolve()
            root.mkdir(parents=True, exist_ok=True)
        else:
            root = Path(tempfile.mkdtemp(prefix="kahm_ablation_keep_"))
        yield root
        return

    with tempfile.TemporaryDirectory(prefix="kahm_ablation_") as tmp_dir:
        yield Path(tmp_dir)


def maybe_matplotlib() -> ContextManager[Optional[Any]]:
    class _MaybeMatplotlibContext(ContextManager[Optional[Any]]):
        def __enter__(self) -> Optional[Any]:
            try:
                import matplotlib  # type: ignore

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt  # type: ignore
                return plt
            except Exception:
                return None

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
            return False

    return _MaybeMatplotlibContext()


@contextmanager
def _as_blocksafe_context(obj: Any) -> Iterator[None]:
    if isinstance(obj, _ContextManagerLike):
        with cast(ContextManager[None], obj):
            yield
        return

    if callable(obj):
        teardown = cast(Callable[[], Any], obj)
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
        return

    with nullcontext():
        yield


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


def parse_int_list(arg: str) -> List[int]:
    vals = [int(x.strip()) for x in arg.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one integer value.")
    return vals


def compute_embedding_metrics(Y_pred: np.ndarray, Y_true: np.ndarray) -> Dict[str, float]:
    if Y_pred.ndim != 2 or Y_true.ndim != 2:
        raise ValueError(f"Expected 2D arrays; got {Y_pred.shape}, {Y_true.shape}")
    if Y_pred.shape != Y_true.shape:
        raise ValueError(f"Shape mismatch: pred={Y_pred.shape} true={Y_true.shape}")

    D, N = Y_true.shape
    diff = Y_pred - Y_true
    mse = float(np.mean(diff * diff))

    y = Y_true.reshape(-1)
    y_mean = float(np.mean(y))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    ss_res = float(np.sum(diff.reshape(-1) ** 2))
    r2_overall = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")

    num = np.einsum("dn,dn->n", Y_pred, Y_true)
    den = np.linalg.norm(Y_pred, axis=0) * np.linalg.norm(Y_true, axis=0)
    cos = num / np.maximum(den, 1e-12)

    return {
        "mse": mse,
        "r2_overall": r2_overall,
        "cos_mean": float(np.mean(cos)),
        "cos_p10": float(np.percentile(cos, 10)),
        "cos_p50": float(np.percentile(cos, 50)),
        "cos_p90": float(np.percentile(cos, 90)),
        "n": int(N),
        "d": int(D),
    }


def embed_idf_svd_queries(idf_svd_model_path: str, texts: Sequence[str]) -> np.ndarray:
    import joblib

    pipe = joblib.load(idf_svd_model_path)
    X = pipe.transform(list(texts))
    X = as_float_ndarray(X)
    X = l2_normalize_rows(X)
    return X


def load_precomputed_mb_queries_npz(path: str, query_ids: Sequence[str]) -> np.ndarray:
    d = np.load(path, allow_pickle=False)
    if "query_id" not in d or "embeddings" not in d:
        raise ValueError(
            f"Queries NPZ '{path}' must contain keys 'query_id' and 'embeddings'. Keys: {list(d.keys())}"
        )

    qid_npz = np.asarray(d["query_id"])
    Y_npz = as_float_ndarray(d["embeddings"])

    if qid_npz.ndim != 1 or Y_npz.ndim != 2:
        raise ValueError(
            f"Queries NPZ '{path}': expected query_id (Q,), embeddings (Q,D); got {qid_npz.shape}, {Y_npz.shape}"
        )

    map_npz = {str(qid_npz[i]): i for i in range(qid_npz.shape[0])}
    missing = [qid for qid in query_ids if qid not in map_npz]
    if missing:
        raise ValueError(f"Queries NPZ '{path}' missing {len(missing)} query_ids. Example: {missing[:10]}")

    Y = np.vstack([Y_npz[map_npz[qid]] for qid in query_ids]).astype(Y_npz.dtype, copy=False)
    Y = l2_normalize_rows(Y)
    return Y


def embed_mb_queries_on_the_fly(model_name: str, device: str, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
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
    del model
    gc.collect()
    return Y


def _sanitize_for_path(s: str, *, max_len: int = 64) -> str:
    s0 = str(s).strip()
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", s0)
    cleaned = cleaned.strip("._-")
    if not cleaned:
        cleaned = "unknown"
    if len(cleaned) > max_len:
        h = hashlib.sha1(s0.encode("utf-8")).hexdigest()[:10]
        cleaned = cleaned[: max(1, max_len - 11)] + "_" + h
    return cleaned


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


def _resolve_npz(args: argparse.Namespace, which: str) -> Optional[str]:
    combined = str(args.queries_npz).strip()
    split_path = str(getattr(args, f"queries_npz_{which}", "")).strip()
    if split_path and Path(split_path).exists():
        return split_path
    if combined and Path(combined).exists():
        return combined
    return None


def _print_metrics(prefix: str, m: Dict[str, float]) -> None:
    print(prefix)
    print(f"  MSE:               {m['mse']:.6f}")
    print(f"  Overall R^2:       {m['r2_overall']:.4f}")
    print(f"  Cosine mean:       {m['cos_mean']:.4f}")
    print(f"  Cosine p10/p50/p90:{m['cos_p10']:.4f} / {m['cos_p50']:.4f} / {m['cos_p90']:.4f}")
    print(f"  N:                 {int(m['n'])}")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run an ablation study for the law-specific KAHM query-regressor trainer "
            "by varying n_clusters and saving a structured report."
        )
    )

    p.add_argument("--idf_svd_model", default=DEFAULT_IDF_SVD_MODEL, help="Path to idf_svd_model.joblib (required).")
    p.add_argument("--queries_npz", default="", help="Optional path to a combined precomputed Mixedbread query embeddings NPZ (backward compatible).")
    p.add_argument("--queries_npz_train", default=DEFAULT_QUERIES_NPZ_TRAIN, help="Path to precomputed Mixedbread TRAIN query embeddings NPZ.")
    p.add_argument("--queries_npz_test", default=DEFAULT_QUERIES_NPZ_TEST, help="Path to precomputed Mixedbread TEST query embeddings NPZ.")
    p.add_argument("--require_npz", action="store_true", help="If set, require NPZ targets and do not fall back to on-the-fly MB embedding.")

    p.add_argument("--report_dir", default=DEFAULT_REPORT_DIR, help="Directory where the report files will be saved.")
    p.add_argument("--report_stem", default=DEFAULT_REPORT_STEM, help="Base filename stem for saved report artifacts.")
    p.add_argument(
        "--n_clusters_values",
        default=",".join(str(x) for x in DEFAULT_N_CLUSTERS_VALUES),
        help="Comma-separated n_clusters values for the ablation study.",
    )
    p.add_argument(
        "--keep_temp_artifacts",
        action="store_true",
        help="Keep temporary model artifacts instead of deleting them after report generation.",
    )
    p.add_argument(
        "--temp_root",
        default=None,
        help="Optional root directory for temporary artifacts. Used only when provided.",
    )
    p.add_argument(
        "--skip_plot",
        action="store_true",
        help="Skip saving PNG metric plots in the report directory.",
    )

    p.add_argument("--mb_model", default="mixedbread-ai/deepset-mxbai-embed-de-large-v1", help="Mixedbread model name (fallback only).")
    p.add_argument("--mb_device", default="cpu", help="Device for fallback MB embedding (cpu/cuda/mps).")
    p.add_argument("--mb_batch", type=int, default=64, help="Batch size for fallback MB embedding.")
    p.add_argument("--force_mb_on_the_fly", action="store_true", help="Ignore NPZ targets and compute MB embeddings with sentence_transformers (torch required).")

    p.add_argument("--model_id", default=None, help="Identifier prefix used when creating per-run model ids.")
    p.add_argument("--ae_cache_root", default=DEFAULT_AE_CACHE_ROOT, help=f"Unused as persistent storage by default; temporary cache root will be created under the temp root (source default: {DEFAULT_AE_CACHE_ROOT}).")
    p.add_argument("--overwrite_ae_dir", action="store_true", default=DEFAULT_OVERWRITE_AE_DIR, help="Allow overwriting an existing AE directory.")

    p.add_argument("--subspace_dim", type=int, default=DEFAULT_SUBSPACE_DIM, help=f"Subspace dimension (default: {DEFAULT_SUBSPACE_DIM})")
    p.add_argument("--nb", type=int, default=DEFAULT_NB, help=f"Nb (default: {DEFAULT_NB})")
    p.add_argument("--random_state", type=int, default=DEFAULT_RANDOM_STATE, help=f"Random seed (default: {DEFAULT_RANDOM_STATE})")
    p.add_argument("--input_scale", type=float, default=DEFAULT_INPUT_SCALE, help=f"Input scaling (default: {DEFAULT_INPUT_SCALE})")

    p.add_argument("--kmeans_kind", default=DEFAULT_KMEANS_KIND, choices=["auto", "full", "minibatch"], help="KMeans implementation choice.")
    p.add_argument("--kmeans_batch_size", type=int, default=DEFAULT_KMEANS_BATCH_SIZE, help="MiniBatchKMeans batch size (if used).")
    p.add_argument("--max_train_per_cluster", type=int, default=DEFAULT_MAX_TRAIN_PER_CLUSTER, help="Cap training samples per cluster (optional).")
    p.add_argument("--model_dtype", default=DEFAULT_MODEL_DTYPE, choices=["float32", "float64"], help="Storage dtype inside the model.")
    p.add_argument("--cluster_center_normalization", default=DEFAULT_CLUSTER_CENTER_NORMALIZATION, choices=["none", "l2", "auto_l2"], help="Normalization for output cluster centers.")

    p.add_argument("--val_fraction", type=float, default=DEFAULT_VAL_FRACTION, help="Fraction of TRAIN queries used for validation/tuning.")
    p.add_argument("--val_max_samples", type=int, default=DEFAULT_VAL_MAX_SAMPLES, help="Max validation samples.")
    p.add_argument("--eval_soft", action="store_true", default=DEFAULT_EVAL_SOFT, help="Evaluate soft-mode regression.")
    p.add_argument("--tune_soft", action="store_true", default=DEFAULT_TUNE_SOFT, help="Tune soft-mode parameters (alpha/topk) on validation set.")
    p.add_argument("--tune_nlms", action="store_true", default=DEFAULT_TUNE_NLMS, help="Refine cluster centers with NLMS (optional).")
    p.add_argument("--soft_alphas", default=",".join(str(x) for x in DEFAULT_SOFT_ALPHAS), help="Comma-separated alphas for soft tuning.")
    p.add_argument("--soft_topks", default=",".join("none" if x is None else str(x) for x in DEFAULT_SOFT_TOPKS), help="Comma-separated topk values for soft tuning (use 'none').")

    p.add_argument("--blocksafe", action="store_true", default=DEFAULT_BLOCKSAFE_ENABLED, help="Enable OTFL BlockSafe (if available).")
    p.add_argument("--blocksafe_backend", default=DEFAULT_BLOCKSAFE_BACKEND, choices=["threading", "multiprocessing"], help="BlockSafe backend.")
    p.add_argument("--blocksafe_jitter_std", type=float, default=DEFAULT_BLOCKSAFE_JITTER_STD)
    p.add_argument("--blocksafe_jitter_tries", type=int, default=DEFAULT_BLOCKSAFE_JITTER_TRIES)
    p.add_argument("--blocksafe_jitter_growth", type=float, default=DEFAULT_BLOCKSAFE_JITTER_GROWTH)
    p.add_argument("--blocksafe_eps_factor", type=float, default=DEFAULT_BLOCKSAFE_EPS_FACTOR)
    p.add_argument("--blocksafe_log_first", type=int, default=DEFAULT_BLOCKSAFE_LOG_FIRST)
    p.add_argument("--blocksafe_l2_normalized", action="store_true", default=DEFAULT_BLOCKSAFE_L2_NORMALIZED)

    p.add_argument("--combined_mode", default=DEFAULT_COMBINED_MODE, choices=["soft", "hard"], help="Combination inference mode (default: soft).")
    p.add_argument("--combined_batch_size", type=int, default=DEFAULT_COMBINED_BATCH_SIZE, help="Batch size used by distance-gated combiner (default: 2048).")
    p.add_argument("--no_combined_progress", action="store_true", help="Disable progress bars during combined evaluation.")

    return p


def _prepare_embeddings(args: argparse.Namespace) -> Dict[str, Any]:
    idf_svd_model_path = str(args.idf_svd_model)
    if not Path(idf_svd_model_path).exists():
        raise FileNotFoundError(f"idf_svd_model not found: {idf_svd_model_path}")

    train_qs = list(TRAIN_QUERY_SET)
    test_qs = list(TEST_QUERY_SET)
    train_ids, train_texts, train_laws = _extract_ids_texts_laws(train_qs, "TRAIN_QUERY_SET")
    test_ids, test_texts, test_laws = _extract_ids_texts_laws(test_qs, "TEST_QUERY_SET")

    print(f"Embedding IDF–SVD queries using: {idf_svd_model_path}")
    X_train_all = embed_idf_svd_queries(idf_svd_model_path, train_texts)
    X_test_all = embed_idf_svd_queries(idf_svd_model_path, test_texts)

    npz_train = _resolve_npz(args, "train")
    npz_test = _resolve_npz(args, "test")

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
                "Missing required NPZ targets for: "
                + ", ".join(missing)
                + ". Provide --queries_npz_train/--queries_npz_test (or a combined --queries_npz) or unset --require_npz."
            )
        print("Computing Mixedbread query embeddings on-the-fly (torch required).")
        Y_train_all = embed_mb_queries_on_the_fly(
            str(args.mb_model), str(args.mb_device), train_texts, batch_size=int(args.mb_batch)
        )
        Y_test_all = embed_mb_queries_on_the_fly(
            str(args.mb_model), str(args.mb_device), test_texts, batch_size=int(args.mb_batch)
        )

    train_idx_by_law: Dict[str, List[int]] = {}
    for i, law in enumerate(train_laws):
        train_idx_by_law.setdefault(law, []).append(i)

    test_idx_by_law: Dict[str, List[int]] = {}
    for i, law in enumerate(test_laws):
        test_idx_by_law.setdefault(law, []).append(i)

    laws = sorted(train_idx_by_law.keys())
    unseen_test_laws = sorted(set(test_idx_by_law.keys()) - set(train_idx_by_law.keys()))
    if unseen_test_laws:
        print(
            "WARNING: test set contains consensus_law values not present in training set.\n"
            f"  Unseen laws (count={len(unseen_test_laws)}): {unseen_test_laws[:20]}"
            + (" ..." if len(unseen_test_laws) > 20 else "")
        )

    return {
        "train_ids": train_ids,
        "test_ids": test_ids,
        "train_laws": train_laws,
        "test_laws": test_laws,
        "X_train_all": X_train_all,
        "X_test_all": X_test_all,
        "Y_train_all": Y_train_all,
        "Y_test_all": Y_test_all,
        "train_idx_by_law": train_idx_by_law,
        "test_idx_by_law": test_idx_by_law,
        "laws": laws,
        "unseen_test_laws": unseen_test_laws,
        "npz_train": npz_train,
        "npz_test": npz_test,
        "use_npz": use_npz,
    }


def _make_blocksafe_context(args: argparse.Namespace) -> Any:
    if args.blocksafe and enable_otfl_blocksafe is not None:
        return enable_otfl_blocksafe(
            backend=str(args.blocksafe_backend),
            jitter_std=float(args.blocksafe_jitter_std),
            jitter_tries=int(args.blocksafe_jitter_tries),
            jitter_growth=float(args.blocksafe_jitter_growth),
            eps_factor=float(args.blocksafe_eps_factor),
            log_first=int(args.blocksafe_log_first),
            l2_normalized=bool(args.blocksafe_l2_normalized),
        )
    return nullcontext()


def run_single_ablation(
    *,
    args: argparse.Namespace,
    prepared: Dict[str, Any],
    requested_n_clusters: int,
    temp_root: Path,
) -> Dict[str, Any]:
    run_started = datetime.now(timezone.utc).isoformat()
    print("\n" + "#" * 100)
    print(f"Starting ablation run for requested n_clusters={requested_n_clusters}")

    models_by_law: Dict[str, dict] = {}
    saved_paths: Dict[str, str] = {}
    per_law_results: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

    X_train_all = cast(np.ndarray, prepared["X_train_all"])
    X_test_all = cast(np.ndarray, prepared["X_test_all"])
    Y_train_all = cast(np.ndarray, prepared["Y_train_all"])
    Y_test_all = cast(np.ndarray, prepared["Y_test_all"])
    train_laws = cast(List[str], prepared["train_laws"])
    test_laws = cast(List[str], prepared["test_laws"])
    train_idx_by_law = cast(Dict[str, List[int]], prepared["train_idx_by_law"])
    test_idx_by_law = cast(Dict[str, List[int]], prepared["test_idx_by_law"])
    laws = cast(List[str], prepared["laws"])

    run_temp_root = temp_root / f"n_clusters_{requested_n_clusters}"
    out_dir = run_temp_root / "out"
    ae_cache_root = run_temp_root / "ae_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    ae_cache_root.mkdir(parents=True, exist_ok=True)

    ctx = _make_blocksafe_context(args)

    print(f"Temporary artifact directory for this run: {run_temp_root}")
    print(f"Training {len(laws)} law-specific regressors ...")

    for law in laws:
        idx_tr = np.asarray(train_idx_by_law[law], dtype=np.int64)
        idx_te = np.asarray(test_idx_by_law.get(law, []), dtype=np.int64)

        X_tr = X_train_all[idx_tr]
        Y_tr = Y_train_all[idx_tr]
        X_te = X_test_all[idx_te] if idx_te.size else None
        Y_te = Y_test_all[idx_te] if idx_te.size else None

        N_train = int(X_tr.shape[0])
        N_test = int(0 if X_te is None else X_te.shape[0])

        print("\n" + "=" * 90)
        print(f"Law: {law} | N_train={N_train} | N_test={N_test}")

        rng = np.random.RandomState(int(args.random_state))
        idx = np.arange(N_train, dtype=np.int64)
        rng.shuffle(idx)

        n_val = 0
        if float(args.val_fraction) > 0:
            n_val = int(round(float(args.val_fraction) * N_train))
            n_val = min(n_val, int(args.val_max_samples))
        n_val = max(0, min(n_val, N_train - 2))

        val_idx = idx[:n_val]
        core_idx = idx[n_val:]

        X_val = X_tr[val_idx] if n_val > 0 else None
        Y_val = Y_tr[val_idx] if n_val > 0 else None
        X_core = X_tr[core_idx]
        Y_core = Y_tr[core_idx]

        X_train_col = X_core.T
        Y_train_col = Y_core.T

        n_clusters_eff = int(min(int(requested_n_clusters), int(X_core.shape[0])))
        n_clusters_eff = max(1, n_clusters_eff)
        was_clamped = n_clusters_eff != int(requested_n_clusters)

        if was_clamped:
            print(
                f"Clamping n_clusters: requested={int(requested_n_clusters)} -> effective={n_clusters_eff} "
                f"(law train_core N={int(X_core.shape[0])})"
            )

        safe_law = _sanitize_for_path(law)
        out_path = out_dir / f"kahm_query_regressor__law={safe_law}.joblib"
        base_model_id = str(args.model_id) if args.model_id else f"kahm_n_clusters_{requested_n_clusters}"
        run_model_id = f"{base_model_id}__law={safe_law}"
        ae_dir = str(ae_cache_root / f"law={safe_law}")

        t0 = time.time()
        try:
            with _as_blocksafe_context(ctx):
                model = train_kahm_regressor(
                    X=X_train_col,
                    Y=Y_train_col,
                    n_clusters=n_clusters_eff,
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
                    ae_cache_root=str(ae_cache_root),
                    ae_dir=ae_dir,
                    overwrite_ae_dir=True if (args.keep_temp_artifacts or args.overwrite_ae_dir) else bool(args.overwrite_ae_dir),
                    model_id=run_model_id,
                    singleton_strategy="augment",
                    singleton_aux_mix=0.1,
                )
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            print(f"ERROR: training failed for law={law}: {err}", file=sys.stderr)
            errors.append({"law": law, "stage": "train", "error": err})
            continue

        t_train = time.time() - t0
        try:
            print(f"Trained model_id={model.get('model_id')} | classifier_dir={model.get('classifier_dir')} | time={t_train:.1f}s")
        except Exception:
            print(f"Training time={t_train:.1f}s")

        tuning_result = None
        if bool(args.tune_soft):
            alphas = tuple(parse_float_list(str(args.soft_alphas)))
            topks = tuple(parse_topk_list(str(args.soft_topks)))
            topk_candidates_eff = [k for k in topks if (k is not None and k <= n_clusters_eff)]
            if not topk_candidates_eff:
                topk_candidates_eff = [1]
                print(f"[{law}] WARNING: no valid topk candidates <= n_clusters_eff={n_clusters_eff}; using topk=1 only.")
            if X_val is None or Y_val is None:
                print("WARNING: tune_soft requested, but validation split is empty. Skipping tuning.")
            else:
                print("Tuning soft parameters on validation set...")
                try:
                    tuning_result = tune_soft_params(
                        model,
                        X_val.T,
                        Y_val.T,
                        alphas=alphas,
                        topks=topk_candidates_eff,
                        n_jobs=1,
                        verbose=True,
                    )
                except Exception as exc:
                    err = f"{type(exc).__name__}: {exc}"
                    print(f"ERROR: tune_soft failed for law={law}: {err}", file=sys.stderr)
                    errors.append({"law": law, "stage": "tune_soft", "error": err})

        nlms_results = None
        if bool(args.tune_nlms):
            if X_val is None or Y_val is None:
                print("WARNING: tune_nlms requested, but validation split is empty. Skipping NLMS.")
            else:
                print("Refining cluster centers with NLMS...")
                try:
                    nlms_results = tune_cluster_centers_nlms(
                        model,
                        np.hstack([X_val.T, X_train_col]),
                        np.hstack([Y_val.T, Y_train_col]),
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
                except Exception as exc:
                    err = f"{type(exc).__name__}: {exc}"
                    print(f"ERROR: tune_nlms failed for law={law}: {err}", file=sys.stderr)
                    errors.append({"law": law, "stage": "tune_nlms", "error": err})

        metrics_soft = None
        if N_test > 0 and X_te is not None and Y_te is not None:
            X_eval_col = X_te.T
            Y_eval_col = Y_te.T
            if bool(args.eval_soft) or bool(args.tune_soft):
                try:
                    Y_pred_soft = kahm_regress(
                        model,
                        X_eval_col,
                        mode="soft",
                        return_probabilities=False,
                        batch_size=1024,
                    )
                    metrics_soft = compute_embedding_metrics(Y_pred_soft, Y_eval_col)
                    _print_metrics("Soft-mode metrics (law test subset):", metrics_soft)
                except Exception as exc:
                    err = f"{type(exc).__name__}: {exc}"
                    print(f"ERROR: per-law evaluation failed for law={law}: {err}", file=sys.stderr)
                    errors.append({"law": law, "stage": "eval_soft", "error": err})
        else:
            print("No test samples for this law; skipping per-law test evaluation.")

        created_at = datetime.now(timezone.utc).isoformat()
        try:
            tuning_payload = asdict(tuning_result) if tuning_result is not None else None
        except Exception:
            tuning_payload = tuning_result

        meta = {
            "created_at_utc": created_at,
            "script": os.path.basename(__file__),
            "consensus_law": law,
            "paths": {
                "idf_svd_model": args.idf_svd_model,
                "queries_npz": (str(args.queries_npz).strip() if (prepared["use_npz"] and str(args.queries_npz).strip()) else None),
                "queries_npz_train": prepared["npz_train"] if prepared["use_npz"] else None,
                "queries_npz_test": prepared["npz_test"] if prepared["use_npz"] else None,
                "out": str(out_path),
                "temporary_root": str(run_temp_root),
                "ae_cache_root": str(ae_cache_root),
            },
            "data": {
                "n_train_queries": int(N_train),
                "n_test_queries": int(N_test),
                "n_train_core": int(X_core.shape[0]),
                "n_val": int(0 if X_val is None else X_val.shape[0]),
                "d_in": int(X_tr.shape[1]),
                "d_out": int(Y_tr.shape[1]),
            },
            "hyperparams": {
                "n_clusters_requested": int(requested_n_clusters),
                "n_clusters_effective": int(n_clusters_eff),
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
                "eval_split": "law_test_subset",
            },
        }

        try:
            model["meta"] = meta
        except Exception:
            pass

        save_kahm_regressor(model, str(out_path))
        print(f"Saved temporary law regressor to: {out_path}")

        models_by_law[law] = model
        saved_paths[law] = str(out_path)
        per_law_results.append(
            {
                "law": law,
                "requested_n_clusters": int(requested_n_clusters),
                "effective_n_clusters": int(n_clusters_eff),
                "was_clamped": bool(was_clamped),
                "n_train_queries": int(N_train),
                "n_train_core": int(X_core.shape[0]),
                "n_val": int(0 if X_val is None else X_val.shape[0]),
                "n_test_queries": int(N_test),
                "train_time_seconds": float(t_train),
                "tuning": tuning_payload,
                "metrics_soft": metrics_soft,
                "saved_model_path": str(out_path),
            }
        )

    if not models_by_law:
        return {
            "requested_n_clusters": int(requested_n_clusters),
            "status": "failed",
            "error": "No law-specific models were trained successfully.",
            "errors": errors,
            "per_law": per_law_results,
            "saved_paths": saved_paths,
            "temporary_root": str(run_temp_root),
            "run_started_utc": run_started,
            "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        }

    print("\n" + "=" * 90)
    print(f"Evaluating combined regressor on full test set (N={int(X_test_all.shape[0])}) ...")

    if len(models_by_law) == 1:
        only_law = next(iter(models_by_law.keys()))
        print(f"Only one trained model (law={only_law}); skipping gating and using that model for all test queries.")
        Y_pred_col = kahm_regress(
            models_by_law[only_law],
            X_test_all.T,
            mode=str(args.combined_mode),
            return_probabilities=False,
            batch_size=1024,
        )
        chosen_idx = np.zeros((X_test_all.shape[0],), dtype=np.int16)
        names = [only_law]
        best_score = None
        all_scores = None
    else:
        Y_pred_row, chosen_idx, best_score, all_scores, names = combine_kahm_regressors_distance_gated_multi(
            X_test_all,
            models=models_by_law,
            input_layout="row",
            output_layout="row",
            mode=str(args.combined_mode),
            alpha=None,
            topk=None,
            batch_size=int(args.combined_batch_size),
            tie_break="first",
            show_progress=(not bool(args.no_combined_progress)),
            return_all_scores=True,
        )
        Y_pred_col = Y_pred_row.T

    metrics_combined = compute_embedding_metrics(Y_pred_col, Y_test_all.T)
    _print_metrics("Combined metrics (full test set):", metrics_combined)

    chosen_idx = np.asarray(chosen_idx, dtype=np.int64).reshape(-1)
    idx_to_name = {i: names[i] for i in range(len(names))}
    chosen_model_distribution = []
    print("\nChosen-model distribution (by gating index):")
    for i in range(len(names)):
        cnt = int(np.sum(chosen_idx == i))
        chosen_model_distribution.append({"gating_index": int(i), "law": idx_to_name[i], "count": cnt})
        print(f"  {i:>3d} | {idx_to_name[i]} | {cnt}")

    print("\nCombined metrics per-law (restricted to each law's test queries):")
    combined_per_law = []
    any_per_law = False
    for law, idxs in sorted(test_idx_by_law.items(), key=lambda kv: kv[0]):
        if not idxs:
            continue
        any_per_law = True
        idxs_np = np.asarray(idxs, dtype=np.int64)
        Y_pred_law = Y_pred_col[:, idxs_np]
        Y_true_law = Y_test_all.T[:, idxs_np]
        m = compute_embedding_metrics(Y_pred_law, Y_true_law)
        combined_per_law.append({"law": law, "metrics": m, "n_test_queries": int(len(idxs))})
        print(f"- {law} (N={len(idxs)})")
        print(f"    cos_mean={m['cos_mean']:.4f} | mse={m['mse']:.6f} | r2={m['r2_overall']:.4f}")
    if not any_per_law:
        print("  (no per-law test subsets found)")

    print("\nTemporary saved models:")
    for law in sorted(saved_paths.keys()):
        print(f"  {law}: {saved_paths[law]}")

    eff_values = [int(x["effective_n_clusters"]) for x in per_law_results]
    train_times = [float(x["train_time_seconds"]) for x in per_law_results]

    return {
        "requested_n_clusters": int(requested_n_clusters),
        "status": "ok",
        "run_started_utc": run_started,
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "temporary_root": str(run_temp_root),
        "temporary_out_dir": str(out_dir),
        "temporary_ae_cache_root": str(ae_cache_root),
        "models_trained": int(len(models_by_law)),
        "laws_in_train": list(laws),
        "unseen_test_laws": list(sorted(set(test_laws) - set(train_laws))),
        "saved_paths": saved_paths,
        "errors": errors,
        "combined_metrics": metrics_combined,
        "combined_per_law": combined_per_law,
        "chosen_model_distribution": chosen_model_distribution,
        "best_score_summary": None if best_score is None else {
            "min": float(np.min(best_score)),
            "mean": float(np.mean(best_score)),
            "max": float(np.max(best_score)),
        },
        "all_scores_shape": None if all_scores is None else list(np.asarray(all_scores).shape),
        "per_law": per_law_results,
        "summary": {
            "mean_effective_n_clusters": (float(np.mean(eff_values)) if eff_values else None),
            "min_effective_n_clusters": (int(np.min(eff_values)) if eff_values else None),
            "max_effective_n_clusters": (int(np.max(eff_values)) if eff_values else None),
            "num_clamped_laws": int(sum(1 for x in per_law_results if x["was_clamped"])),
            "total_train_time_seconds": float(np.sum(train_times)) if train_times else 0.0,
            "mean_train_time_seconds": float(np.mean(train_times)) if train_times else 0.0,
        },
    }


def write_summary_csv(path: Path, runs: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "requested_n_clusters",
        "status",
        "models_trained",
        "combined_cos_mean",
        "combined_cos_p10",
        "combined_cos_p50",
        "combined_cos_p90",
        "combined_mse",
        "combined_r2_overall",
        "combined_n",
        "num_clamped_laws",
        "mean_effective_n_clusters",
        "min_effective_n_clusters",
        "max_effective_n_clusters",
        "total_train_time_seconds",
        "mean_train_time_seconds",
        "num_errors",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            metrics = run.get("combined_metrics") or {}
            summary = run.get("summary") or {}
            writer.writerow(
                {
                    "requested_n_clusters": run.get("requested_n_clusters"),
                    "status": run.get("status"),
                    "models_trained": run.get("models_trained"),
                    "combined_cos_mean": metrics.get("cos_mean"),
                    "combined_cos_p10": metrics.get("cos_p10"),
                    "combined_cos_p50": metrics.get("cos_p50"),
                    "combined_cos_p90": metrics.get("cos_p90"),
                    "combined_mse": metrics.get("mse"),
                    "combined_r2_overall": metrics.get("r2_overall"),
                    "combined_n": metrics.get("n"),
                    "num_clamped_laws": summary.get("num_clamped_laws"),
                    "mean_effective_n_clusters": summary.get("mean_effective_n_clusters"),
                    "min_effective_n_clusters": summary.get("min_effective_n_clusters"),
                    "max_effective_n_clusters": summary.get("max_effective_n_clusters"),
                    "total_train_time_seconds": summary.get("total_train_time_seconds"),
                    "mean_train_time_seconds": summary.get("mean_train_time_seconds"),
                    "num_errors": len(run.get("errors") or []),
                }
            )


def write_per_law_csv(path: Path, runs: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "requested_n_clusters",
        "law",
        "effective_n_clusters",
        "was_clamped",
        "n_train_queries",
        "n_train_core",
        "n_val",
        "n_test_queries",
        "train_time_seconds",
        "soft_cos_mean",
        "soft_cos_p10",
        "soft_cos_p50",
        "soft_cos_p90",
        "soft_mse",
        "soft_r2_overall",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            for law_row in run.get("per_law", []):
                metrics = law_row.get("metrics_soft") or {}
                writer.writerow(
                    {
                        "requested_n_clusters": run.get("requested_n_clusters"),
                        "law": law_row.get("law"),
                        "effective_n_clusters": law_row.get("effective_n_clusters"),
                        "was_clamped": law_row.get("was_clamped"),
                        "n_train_queries": law_row.get("n_train_queries"),
                        "n_train_core": law_row.get("n_train_core"),
                        "n_val": law_row.get("n_val"),
                        "n_test_queries": law_row.get("n_test_queries"),
                        "train_time_seconds": law_row.get("train_time_seconds"),
                        "soft_cos_mean": metrics.get("cos_mean"),
                        "soft_cos_p10": metrics.get("cos_p10"),
                        "soft_cos_p50": metrics.get("cos_p50"),
                        "soft_cos_p90": metrics.get("cos_p90"),
                        "soft_mse": metrics.get("mse"),
                        "soft_r2_overall": metrics.get("r2_overall"),
                    }
                )


def rank_runs(runs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ok_runs = [run for run in runs if run.get("status") == "ok" and run.get("combined_metrics")]
    return sorted(
        ok_runs,
        key=lambda run: (
            float(run["combined_metrics"].get("cos_mean", float("-inf"))),
            -float(run["combined_metrics"].get("mse", float("inf"))),
        ),
        reverse=True,
    )


def write_markdown_report(path: Path, payload: Dict[str, Any]) -> None:
    runs = payload["runs"]
    ranked = rank_runs(runs)
    best = ranked[0] if ranked else None

    lines: List[str] = []
    lines.append(f"# {payload['report_stem']}")
    lines.append("")
    lines.append(f"Generated at: {payload['generated_at_utc']}")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- n_clusters values: {', '.join(str(x) for x in payload['n_clusters_values'])}")
    lines.append(f"- Temporary artifacts kept: {payload['keep_temp_artifacts']}")
    lines.append(f"- Combined inference mode: {payload['config']['combined_mode']}")
    lines.append(f"- Validation fraction: {payload['config']['val_fraction']}")
    lines.append(f"- Tune soft: {payload['config']['tune_soft']}")
    lines.append(f"- Tune NLMS: {payload['config']['tune_nlms']}")
    lines.append("")

    lines.append("## Aggregate results")
    lines.append("")
    lines.append("| requested_n_clusters | status | cos_mean | mse | r2_overall | num_clamped_laws | total_train_time_s |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for run in runs:
        m = run.get("combined_metrics") or {}
        s = run.get("summary") or {}
        lines.append(
            "| {n} | {status} | {cos_mean} | {mse} | {r2} | {clamped} | {tts} |".format(
                n=run.get("requested_n_clusters"),
                status=run.get("status"),
                cos_mean=(f"{m.get('cos_mean', float('nan')):.6f}" if m else "-"),
                mse=(f"{m.get('mse', float('nan')):.6f}" if m else "-"),
                r2=(f"{m.get('r2_overall', float('nan')):.6f}" if m else "-"),
                clamped=(s.get("num_clamped_laws", "-")),
                tts=(f"{s.get('total_train_time_seconds', float('nan')):.2f}" if s else "-"),
            )
        )
    lines.append("")

    if best is not None:
        m = best["combined_metrics"]
        lines.append("## Best run")
        lines.append("")
        lines.append(f"Best requested n_clusters: **{best['requested_n_clusters']}**")
        lines.append("")
        lines.append(f"- cos_mean: {m['cos_mean']:.6f}")
        lines.append(f"- mse: {m['mse']:.6f}")
        lines.append(f"- r2_overall: {m['r2_overall']:.6f}")
        lines.append(f"- clamped laws: {best['summary']['num_clamped_laws']}")
        lines.append("")

    lines.append("## Per-run notes")
    lines.append("")
    for run in runs:
        lines.append(f"### n_clusters = {run['requested_n_clusters']}")
        lines.append("")
        lines.append(f"- status: {run.get('status')}")
        lines.append(f"- models trained: {run.get('models_trained', 0)}")
        lines.append(f"- temporary root: `{run.get('temporary_root')}`")
        if run.get("combined_metrics"):
            m = run["combined_metrics"]
            lines.append(f"- combined cos_mean: {m['cos_mean']:.6f}")
            lines.append(f"- combined mse: {m['mse']:.6f}")
            lines.append(f"- combined r2_overall: {m['r2_overall']:.6f}")
        errors = run.get("errors") or []
        lines.append(f"- number of recorded errors: {len(errors)}")
        if errors:
            lines.append("- errors:")
            for err in errors:
                lines.append(f"  - {err.get('law', '?')} [{err.get('stage', '?')}]: {err.get('error', '')}")
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_plots(report_dir: Path, report_stem: str, runs: Sequence[Dict[str, Any]]) -> List[str]:
    saved: List[str] = []
    ok_runs = [run for run in runs if run.get("status") == "ok" and run.get("combined_metrics")]
    if not ok_runs:
        return saved

    xs = [int(run["requested_n_clusters"]) for run in ok_runs]
    cos_means = [float(run["combined_metrics"]["cos_mean"]) for run in ok_runs]
    mses = [float(run["combined_metrics"]["mse"]) for run in ok_runs]
    r2s = [float(run["combined_metrics"]["r2_overall"]) for run in ok_runs]

    with maybe_matplotlib() as plt:
        if plt is None:
            return saved

        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_subplot(111)
        ax.plot(xs, cos_means, marker="o")
        ax.set_xlabel("requested n_clusters")
        ax.set_ylabel("combined cosine mean")
        ax.set_title("KAHM ablation: cosine mean vs n_clusters")
        fig.tight_layout()
        path = report_dir / f"{report_stem}_cosine_mean.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        saved.append(str(path))

        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_subplot(111)
        ax.plot(xs, mses, marker="o")
        ax.set_xlabel("requested n_clusters")
        ax.set_ylabel("combined MSE")
        ax.set_title("KAHM ablation: MSE vs n_clusters")
        fig.tight_layout()
        path = report_dir / f"{report_stem}_mse.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        saved.append(str(path))

        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_subplot(111)
        ax.plot(xs, r2s, marker="o")
        ax.set_xlabel("requested n_clusters")
        ax.set_ylabel("combined R^2")
        ax.set_title("KAHM ablation: R^2 vs n_clusters")
        fig.tight_layout()
        path = report_dir / f"{report_stem}_r2.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        saved.append(str(path))

    return saved


def main() -> int:
    args = build_arg_parser().parse_args()
    n_clusters_values = parse_int_list(str(args.n_clusters_values))

    report_dir = Path(str(args.report_dir)).expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    report_stem = str(args.report_stem).strip() or DEFAULT_REPORT_STEM
    log_path = report_dir / f"{report_stem}.log"

    with tee_output(log_path):
        started = datetime.now(timezone.utc).isoformat()
        prepared = _prepare_embeddings(args)
        runs: List[Dict[str, Any]] = []

        with managed_temp_root(bool(args.keep_temp_artifacts), args.temp_root) as temp_root:
            print(f"Using temporary artifact root: {temp_root}")
            for requested_n_clusters in n_clusters_values:
                run = run_single_ablation(
                    args=args,
                    prepared=prepared,
                    requested_n_clusters=int(requested_n_clusters),
                    temp_root=temp_root,
                )
                runs.append(run)

            report_payload: Dict[str, Any] = {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "started_at_utc": started,
                "report_stem": report_stem,
                "report_dir": str(report_dir),
                "keep_temp_artifacts": bool(args.keep_temp_artifacts),
                "temp_root": str(temp_root),
                "n_clusters_values": list(int(x) for x in n_clusters_values),
                "config": {
                    "idf_svd_model": str(args.idf_svd_model),
                    "queries_npz": str(args.queries_npz),
                    "queries_npz_train": str(args.queries_npz_train),
                    "queries_npz_test": str(args.queries_npz_test),
                    "require_npz": bool(args.require_npz),
                    "force_mb_on_the_fly": bool(args.force_mb_on_the_fly),
                    "mb_model": str(args.mb_model),
                    "mb_device": str(args.mb_device),
                    "mb_batch": int(args.mb_batch),
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
                    "eval_soft": bool(args.eval_soft),
                    "tune_soft": bool(args.tune_soft),
                    "tune_nlms": bool(args.tune_nlms),
                    "soft_alphas": list(parse_float_list(str(args.soft_alphas))),
                    "soft_topks": list(parse_topk_list(str(args.soft_topks))),
                    "blocksafe": bool(args.blocksafe),
                    "combined_mode": str(args.combined_mode),
                    "combined_batch_size": int(args.combined_batch_size),
                },
                "dataset_summary": {
                    "n_train_queries": len(prepared["train_ids"]),
                    "n_test_queries": len(prepared["test_ids"]),
                    "n_train_laws": len(prepared["laws"]),
                    "unseen_test_laws": list(prepared["unseen_test_laws"]),
                    "use_npz": bool(prepared["use_npz"]),
                    "npz_train": prepared["npz_train"],
                    "npz_test": prepared["npz_test"],
                },
                "runs": runs,
            }

            json_path = report_dir / f"{report_stem}.json"
            csv_path = report_dir / f"{report_stem}.csv"
            per_law_csv_path = report_dir / f"{report_stem}_per_law.csv"
            md_path = report_dir / f"{report_stem}.md"

            json_path.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
            write_summary_csv(csv_path, runs)
            write_per_law_csv(per_law_csv_path, runs)
            write_markdown_report(md_path, report_payload)

            plot_paths: List[str] = []
            if not bool(args.skip_plot):
                plot_paths = save_plots(report_dir, report_stem, runs)

            report_payload["saved_report_files"] = {
                "json": str(json_path),
                "csv": str(csv_path),
                "per_law_csv": str(per_law_csv_path),
                "markdown": str(md_path),
                "log": str(log_path),
                "plots": plot_paths,
            }
            json_path.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")

            ranked = rank_runs(runs)
            print("\n" + "#" * 100)
            print("Ablation study finished.")
            print(f"Saved JSON report:      {json_path}")
            print(f"Saved summary CSV:      {csv_path}")
            print(f"Saved per-law CSV:      {per_law_csv_path}")
            print(f"Saved Markdown report:  {md_path}")
            if plot_paths:
                for plot_path in plot_paths:
                    print(f"Saved plot:             {plot_path}")
            if ranked:
                best = ranked[0]
                best_metrics = best["combined_metrics"]
                print(
                    "Best run by combined cosine mean: "
                    f"n_clusters={best['requested_n_clusters']} | "
                    f"cos_mean={best_metrics['cos_mean']:.6f} | "
                    f"mse={best_metrics['mse']:.6f} | "
                    f"r2={best_metrics['r2_overall']:.6f}"
                )
            else:
                print("No successful ablation run was available to rank.")

        if args.keep_temp_artifacts:
            print(f"Temporary artifacts were kept at: {args.temp_root or report_payload['temp_root']}")
        else:
            print("Temporary artifacts were deleted after the reports were saved.")
    try:
        log_path.unlink(missing_ok=True)
    except Exception as exc:
        print(f"WARNING: failed to delete log file {log_path}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
