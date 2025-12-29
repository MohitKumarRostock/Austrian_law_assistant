#!/usr/bin/env python3
"""
build_query_embedding_index_npz.py

Precompute query embeddings for QUERY_SET and save to NPZ.


Output NPZ contains:
  - query_id: (Q,) unicode strings
  - query_text: (Q,) unicode strings
  - embeddings: (Q, D) floating (dtype configurable; default preserves model output dtype)
  - meta scalars

Example:
  python build_query_embedding_index_npz.py \
    --out queries_embedding_index.npz \
    --model mixedbread-ai/deepset-mxbai-embed-de-large-v1 \
    --device cpu \
    --batch 64
"""

from __future__ import annotations

import os

# Must come after future import, before importing transformers/sentence_transformers
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Optional: reduce oversubscription during embedding (users can override via env vars)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


DEFAULT_OUT = "queries_embedding_index.npz"
DEFAULT_MODEL = "mixedbread-ai/deepset-mxbai-embed-de-large-v1"
DEFAULT_DEVICE = "cpu"
DEFAULT_BATCH = 64
DEFAULT_QUERY_PREFIX = "query: "
DEFAULT_DIM = 1024


def l2_normalize_rows_inplace(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize rows of a 2D array in-place when possible.

    Preserves dtype for floating arrays. Non-floating inputs are promoted to float32.
    """
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array for normalization; got shape={x.shape}")
    if not np.issubdtype(x.dtype, np.floating):
        x = x.astype(np.float32, copy=False)

    norms = np.linalg.norm(x, axis=1, keepdims=True)
    np.maximum(norms, eps, out=norms)
    x /= norms
    return x


def _load_query_set() -> List[Dict[str, Any]]:
    from query_set import TRAIN_QUERY_SET  # type: ignore
    qs = list(TRAIN_QUERY_SET)
    if not qs:
        raise RuntimeError("Imported TRAIN_QUERY_SET is empty.")
    return qs





def choose_device(requested: str) -> str:
    """Resolve device selection; supports 'auto' or explicit device strings."""
    if requested and requested.lower() != "auto":
        return requested
    try:
        import torch  # local import

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _iter_batches(n: int, batch_size: int) -> Iterable[Tuple[int, int]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for i in range(0, n, batch_size):
        yield i, min(n, i + batch_size)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Precompute query embeddings for QUERY_SET and save to NPZ.")
    p.add_argument("--out", default=DEFAULT_OUT, help=f"Output NPZ (default: {DEFAULT_OUT})")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"SentenceTransformer model (default: {DEFAULT_MODEL})")
    p.add_argument("--device", default=DEFAULT_DEVICE, help="cpu|cuda|mps|auto (default: cpu)")
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH, help=f"Batch size (default: {DEFAULT_BATCH})")
    p.add_argument("--query_prefix", default=DEFAULT_QUERY_PREFIX, help="Prefix for query encoding (default: 'query: ')")
    p.add_argument("--dim", type=int, default=DEFAULT_DIM, help=f"Embedding dim / truncate_dim (default: {DEFAULT_DIM})")
    p.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "float32", "float64"],
        help="Embedding dtype in output NPZ. 'auto' preserves model output dtype.",
    )
    p.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="L2-normalize embeddings (default: enabled).",
    )
    p.add_argument(
        "--compress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write compressed NPZ (default: enabled). Disable for faster saves.",
    )
    p.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show progress during encoding (default: enabled).",
    )
    p.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stream batches via repeated encode calls to reduce peak RAM (default: disabled).",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite output NPZ if it exists.")
    return p


def _validate_query_fields(qs: Sequence[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    query_ids = [str(q.get("query_id", "")).strip() for q in qs]
    query_texts = [str(q.get("query_text", "")).strip() for q in qs]

    if any(not qid for qid in query_ids):
        raise ValueError("QUERY_SET contains empty query_id entries.")
    if any(not qt for qt in query_texts):
        raise ValueError("QUERY_SET contains empty query_text entries.")
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("QUERY_SET contains duplicate query_id values (must be unique).")

    return query_ids, query_texts


def _coerce_dtype(arr: np.ndarray, dtype_arg: str) -> np.ndarray:
    if dtype_arg == "auto":
        return arr
    return arr.astype(np.dtype(dtype_arg), copy=False)


def _fix_dim(arr: np.ndarray, dim: int) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"encode output must be 2D; got {arr.shape}")
    if arr.shape[1] == dim:
        return arr
    if arr.shape[1] > dim:
        return arr[:, :dim]
    raise ValueError(f"Embedding dim mismatch: got {arr.shape[1]}, expected {dim}")


def _encode_single_call(
    model: Any,
    query_texts: Sequence[str],
    prefix: str,
    batch_size: int,
    dim: int,
    dtype_arg: str,
    normalize_embeddings: bool,
    show_progress: bool,
) -> np.ndarray:
    # Fast path: let SentenceTransformer handle internal batching + progress bar
    texts = [prefix + t for t in query_texts]
    Y = model.encode(
        texts,
        batch_size=int(batch_size),
        show_progress_bar=bool(show_progress),
        convert_to_numpy=True,
        normalize_embeddings=bool(normalize_embeddings),
    )
    Y = _fix_dim(np.asarray(Y), dim)
    Y = _coerce_dtype(Y, dtype_arg)
    return Y


def _encode_streaming(
    model: Any,
    query_texts: Sequence[str],
    prefix: str,
    batch_size: int,
    dim: int,
    dtype_arg: str,
    normalize_embeddings: bool,
    progress: bool,
) -> np.ndarray:
    # Lower peak RAM path: repeated encode calls with a preallocated output buffer.
    n = len(query_texts)
    if n == 0:
        raise ValueError("QUERY_SET is empty (no query_text entries).")

    def make_batch(s: int, e: int) -> List[str]:
        return [prefix + t for t in query_texts[s:e]]

    s0, e0 = next(iter(_iter_batches(n, batch_size)))
    first = model.encode(
        make_batch(s0, e0),
        batch_size=int(batch_size),
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=bool(normalize_embeddings),
    )
    first = _coerce_dtype(_fix_dim(np.asarray(first), dim), dtype_arg)
    out = np.empty((n, dim), dtype=first.dtype)
    out[s0:e0] = first

    if progress:
        print(f"Encoded {e0}/{n}")

    for s, e in _iter_batches(n, batch_size):
        if s == s0 and e == e0:
            continue
        chunk = model.encode(
            make_batch(s, e),
            batch_size=int(batch_size),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=bool(normalize_embeddings),
        )
        chunk = _coerce_dtype(_fix_dim(np.asarray(chunk), dim), dtype_arg)
        out[s:e] = chunk

        if progress and (e == n or (e // batch_size) % 10 == 0):
            print(f"Encoded {e}/{n}")

    return out


def _ensure_npz_suffix(path: Path) -> Path:
    # Match numpy behavior (adds .npz if missing) but keep our atomic temp naming reliable.
    if path.suffix == ".npz":
        return path
    if path.suffix:
        return path.with_suffix(path.suffix + ".npz")
    return path.with_suffix(".npz")


def _save_npz_atomic(path: Path, *, compress: bool, **arrays: Any) -> None:
    path = _ensure_npz_suffix(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure tmp ends with .npz, otherwise numpy may append .npz automatically.
    tmp = path.with_suffix(".tmp.npz")

    if compress:
        np.savez_compressed(tmp, **arrays)
    else:
        np.savez(tmp, **arrays)

    os.replace(tmp, path)


def main() -> int:
    args = build_arg_parser().parse_args()
    out_path = _ensure_npz_suffix(Path(args.out))

    if out_path.exists() and not args.overwrite:
        print(f"Output exists: {out_path} (use --overwrite to replace)." )
        return 0

    qs = _load_query_set()
    query_ids, query_texts = _validate_query_fields(qs)

    device = choose_device(str(args.device))
    model_name = str(args.model)
    dim = int(args.dim)
    prefix = str(args.query_prefix)
    batch = int(args.batch)

    print(f"Embedding {len(qs)} queries with model: {model_name} (device={device}, dim={dim}, batch={batch})")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device, truncate_dim=dim)

    if args.stream:
        Y = _encode_streaming(
            model=model,
            query_texts=query_texts,
            prefix=prefix,
            batch_size=batch,
            dim=dim,
            dtype_arg=str(args.dtype),
            normalize_embeddings=bool(args.normalize),
            progress=bool(args.progress),
        )
    else:
        Y = _encode_single_call(
            model=model,
            query_texts=query_texts,
            prefix=prefix,
            batch_size=batch,
            dim=dim,
            dtype_arg=str(args.dtype),
            normalize_embeddings=bool(args.normalize),
            show_progress=bool(args.progress),
        )

    # Safety normalization pass (cheap; ensures output is correct regardless of ST version behavior)
    if args.normalize:
        l2_normalize_rows_inplace(Y)

    created_at = datetime.now(timezone.utc).isoformat()
    out_meta = dict(
        created_at_utc=created_at,
        model=model_name,
        device=str(device),
        dim=int(dim),
        query_prefix=prefix,
        n_queries=int(len(qs)),
        dtype=str(Y.dtype),
        normalized=bool(args.normalize),
        note="L2-normalized query embeddings for QUERY_SET; generated separately to keep training torch-free.",
    )

    print(f"Saving: {out_path} (compress={bool(args.compress)})")
    _save_npz_atomic(
        out_path,
        compress=bool(args.compress),
        query_id=np.array(query_ids, dtype=np.str_),
        query_text=np.array(query_texts, dtype=np.str_),
        embeddings=Y,
        **out_meta,
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
