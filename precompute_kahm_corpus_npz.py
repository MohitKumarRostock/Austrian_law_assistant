#!/usr/bin/env python3
"""
precompute_kahm_corpus_npz.py

Precompute KAHM-regressed approximate Mixedbread embeddings for ALL sentences
(from IDF–SVD sentence embeddings) and save to NPZ for fast retrieval.

This version is runnable WITHOUT passing CLI arguments (all inputs/outputs
have sensible defaults), but remains fully configurable via CLI.

Defaults:
  --idf_svd_npz   embedding_index_idf_svd.npz
  --kahm_model    kahm_regressor_idf_to_mixedbread.joblib
  --out_npz       embedding_index_kahm_mixedbread_approx.npz
  --kahm_mode     soft
  --batch         2048

Behavior if output exists:
  - If --overwrite is set: overwrite
  - Otherwise: print a message and exit successfully (no error)
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Optional dependency: tqdm for progress bars. If unavailable, we fall back
# to a no-op wrapper so the script remains runnable.
try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(it, *args, **kwargs):  # type: ignore
        return it

DEFAULT_IDF_SVD_NPZ = "embedding_index_idf_svd.npz"
DEFAULT_KAHM_MODEL = "kahm_regressor_idf_to_mixedbread.joblib"
DEFAULT_OUT_NPZ = "embedding_index_kahm_mixedbread_approx.npz"
DEFAULT_KAHM_MODE = "soft"
DEFAULT_BATCH = 2048


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps).astype(np.float32)
    return x / n


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


def kahm_regress_batched(
    model: dict,
    X: np.ndarray,
    *,
    mode: str,
    batch_size: int,
    show_progress: bool = True,
) -> np.ndarray:
    from kahm_regression import kahm_regress  # local module

    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (N,D); got {X.shape}")

    if "cluster_centers" not in model:
        raise KeyError("KAHM model missing 'cluster_centers' (cannot infer output dimension).")

    d_out = int(model["cluster_centers"].shape[0])
    n = int(X.shape[0])
    Y = np.empty((n, d_out), dtype=np.float32)

    Xt = X.T  # kahm_regress expects (D, N)
    bs = int(batch_size)
    if bs <= 0:
        raise ValueError(f"batch_size must be a positive integer; got {batch_size!r}")
    total_batches = (n + bs - 1) // bs if bs > 0 else 0
    iterator = range(0, n, bs)
    if show_progress:
        iterator = tqdm(iterator, total=total_batches, unit="batch", desc="KAHM regress")

    for j0 in iterator:
        j1 = min(n, j0 + bs)
        Yt = kahm_regress(model, Xt[:, j0:j1], mode=str(mode))
        Y[j0:j1, :] = np.asarray(Yt.T, dtype=np.float32)

    return Y


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Precompute KAHM-regressed corpus embeddings and save as NPZ.")

    # No longer required; defaults make it runnable without passing arguments
    p.add_argument("--idf_svd_npz", default=DEFAULT_IDF_SVD_NPZ, help=f"Path to IDF–SVD NPZ (default: {DEFAULT_IDF_SVD_NPZ})")
    p.add_argument("--kahm_model", default=DEFAULT_KAHM_MODEL, help=f"Path to trained KAHM regressor joblib (default: {DEFAULT_KAHM_MODEL})")
    p.add_argument("--out_npz", default=DEFAULT_OUT_NPZ, help=f"Output NPZ path (default: {DEFAULT_OUT_NPZ})")

    p.add_argument("--kahm_mode", choices=["soft", "hard"], default=DEFAULT_KAHM_MODE, help="KAHM inference mode (default: soft)")
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="Batch size for regression (default: 4096)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite out_npz if it exists (default: no)")
    p.add_argument("--no_progress", action="store_true", help="Disable the progress bar (default: show)")

    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    # Early validation for friendlier errors
    if not os.path.exists(args.idf_svd_npz):
        print(f"ERROR: IDF–SVD NPZ not found: {args.idf_svd_npz}", file=sys.stderr)
        return 2
    if not os.path.exists(args.kahm_model):
        print(f"ERROR: KAHM model not found: {args.kahm_model}", file=sys.stderr)
        return 2

    if os.path.exists(args.out_npz) and not args.overwrite:
        print(f"Output already exists: {args.out_npz}")
        print("Nothing to do (use --overwrite to recompute).")
        return 0

    print(f"Loading IDF–SVD NPZ: {args.idf_svd_npz}")
    ids, X, meta_x = load_npz_bundle(args.idf_svd_npz)
    print(f"  ids={ids.shape} X={X.shape}")

    print(f"Loading KAHM model: {args.kahm_model}")
    from kahm_regression import load_kahm_regressor  # local module

    model = load_kahm_regressor(args.kahm_model)

    # Normalize inputs (cosine retrieval assumes L2)
    Xn = l2_normalize_rows(X)

    print(f"Regressing corpus embeddings via KAHM ({args.kahm_mode}) in batches of {args.batch} ...")
    Y = kahm_regress_batched(
        model,
        Xn,
        mode=args.kahm_mode,
        batch_size=int(args.batch),
        show_progress=(not bool(args.no_progress)),
    )
    Yn = l2_normalize_rows(Y)

    created_at = datetime.now(timezone.utc).isoformat()
    fp_x = _npz_scalar_to_str(meta_x.get("dataset_fingerprint_sha256"))

    out_meta = dict(
        created_at_utc=created_at,
        source_idf_svd_npz=os.path.basename(args.idf_svd_npz),
        source_idf_svd_fingerprint_sha256=(fp_x or ""),
        source_kahm_model=os.path.basename(args.kahm_model),
        kahm_mode=str(args.kahm_mode),
        n_sentences=int(ids.shape[0]),
        d_in=int(X.shape[1]),
        d_out=int(Yn.shape[1]),
        note="KAHM-regressed approximate Mixedbread embeddings from IDF–SVD; L2-normalized for cosine/IP FAISS.",
    )

    print(f"Saving precomputed KAHM corpus NPZ: {args.out_npz}")
    np.savez_compressed(
        args.out_npz,
        sentence_id=ids.astype(np.int64, copy=False),
        embeddings=Yn.astype(np.float32, copy=False),
        **out_meta,
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
