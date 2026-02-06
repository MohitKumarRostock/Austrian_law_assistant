#!/usr/bin/env python3
"""build_embedding_index_npz.py

Build a sentence embedding index from a RIS-derived corpus parquet.

This script intentionally mirrors the performance characteristics of
`build_query_embedding_index_npz.py`:
  - use SentenceTransformer.encode() for efficient CUDA batching
  - let the model decide its maximum sequence length by default

Output NPZ contains:
  - sentence_id: (N,) int64
  - embeddings: (N, D)
  - minimal metadata scalars
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
import hashlib
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


DEFAULT_CORPUS_PATH = Path("ris_sentences.parquet")
DEFAULT_OUT_NPZ = Path("embedding_index.npz")
DEFAULT_MODEL = "mixedbread-ai/deepset-mxbai-embed-de-large-v1"


DEFAULT_BATCH = 1


# ----------------------------- Utilities -----------------------------

def sha256_fingerprint_sentence_id_content(df: pd.DataFrame, *, id_col: str, text_col: str) -> str:
    """
    Compute a stable SHA256 over (sentence_id + content) for all rows in order.
    """
    if id_col not in df.columns or text_col not in df.columns:
        raise ValueError(f"Corpus is missing required column '{id_col}' or '{text_col}'. Columns: {list(df.columns)}")

    h = hashlib.sha256()
    # Use itertuples for speed; ensure consistent typing/encoding
    for row in df[[id_col, text_col]].itertuples(index=False, name=None):
        sid, content = row
        h.update(str(sid).encode("utf-8"))
        h.update(b"\0")
        h.update(str(content).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def choose_device(requested: str) -> torch.device:
    requested = (requested or "").strip().lower()
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("CUDA requested but not available; falling back to CPU.", file=sys.stderr)
        return torch.device("cpu")
    if requested == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        print("MPS requested but not available; falling back to CPU.", file=sys.stderr)
        return torch.device("cpu")
    if requested in ("cpu", ""):
        return torch.device("cpu")
    # Auto
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    # Unknown
    print(f"Unknown device '{requested}'; using CPU.", file=sys.stderr)
    return torch.device("cpu")


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean pooling over the sequence length using the attention mask.
    """
    # last_hidden_state: [B, T, H]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def atomic_save_npz(
    out_path: Path,
    *,
    compressed: bool,
    verify: bool = True,
    **arrays_and_meta: Any,
) -> None:
    """
    Atomically writes NPZ by writing to a temp .npz file in the same directory and then os.replace().

    Key detail: we write to an *open file handle* with suffix ".npz" so NumPy does not append ".npz"
    unexpectedly, and our verification path is correct.

    Windows fix: ensure any np.load() handle is closed before os.replace().
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(out_path.parent),
            prefix=out_path.stem + ".tmp.",
            suffix=".npz",
            delete=False,
        ) as f:
            tmp_path = Path(f.name)
            if compressed:
                np.savez_compressed(f, **arrays_and_meta)
            else:
                np.savez(f, **arrays_and_meta)

        if verify:
            # IMPORTANT: close handle before replace (Windows file-lock semantics)
            with np.load(tmp_path, allow_pickle=False) as d:
                if "sentence_id" not in d or "embeddings" not in d:
                    raise RuntimeError("NPZ verification failed: missing required keys.")
                if d["sentence_id"].ndim != 1 or d["embeddings"].ndim != 2:
                    raise RuntimeError("NPZ verification failed: unexpected array shapes.")

        os.replace(tmp_path, out_path)

    except Exception:
        # Cleanup temp file on failure
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        raise


# ----------------------------- Embedding -----------------------------

def _iter_batches(n: int, batch_size: int) -> Iterable[Tuple[int, int]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for i in range(0, n, batch_size):
        yield i, min(n, i + batch_size)


def _as_device_str(d: torch.device) -> str:
    # SentenceTransformer accepts strings like 'cpu', 'cuda', 'mps' (optionally with indices).
    # torch.device('cuda:0') stringifies as 'cuda:0' which ST accepts.
    return str(d)


@torch.inference_mode()
def encode_sentences(
    sentences: Sequence[str],
    *,
    model_name: str,
    device: torch.device,
    batch_size: int,
    max_length: int,
    normalize_embeddings: bool,
    truncate_dim: Optional[int],
    stream: bool,
    show_progress: bool,
) -> np.ndarray:
    # Match query script behavior: let SentenceTransformer handle batching/tokenization.
    from sentence_transformers import SentenceTransformer

    dev = _as_device_str(device)
    td = int(truncate_dim) if (truncate_dim is not None and truncate_dim > 0) else None

    # NOTE: SentenceTransformer supports truncate_dim (as used by build_query_embedding_index_npz.py).
    model = SentenceTransformer(model_name, device=dev, truncate_dim=(td or 0))

    # "Auto" max length behavior: if max_length <= 0, keep model.max_seq_length.
    # Otherwise, override it (useful for speed experiments).
    if int(max_length) > 0:
        model.max_seq_length = int(max_length)

    n = len(sentences)
    if n == 0:
        raise ValueError("No sentences to encode.")

    # Small robustness feature: if the requested batch size OOMs on CUDA, we automatically downshift.
    def _encode_list(texts: Sequence[str], bs: int, progress: bool) -> np.ndarray:
        return np.asarray(
            model.encode(
                list(texts),
                batch_size=int(bs),
                show_progress_bar=bool(progress),
                convert_to_numpy=True,
                normalize_embeddings=bool(normalize_embeddings),
            )
        )

    def _is_oom(e: BaseException) -> bool:
        s = str(e).lower()
        return "out of memory" in s or "cuda oom" in s or "cublas" in s

    bs = int(batch_size)
    if not stream:
        while True:
            try:
                Y = _encode_list(sentences, bs, progress=show_progress)
                break
            except RuntimeError as e:
                if _is_oom(e) and bs > 1:
                    bs = max(1, bs // 2)
                    if torch.cuda.is_available() and device.type == "cuda":
                        torch.cuda.empty_cache()
                    print(f"OOM during encode; retrying with batch_size={bs}", file=sys.stderr)
                    continue
                raise
        return Y.astype(np.float32, copy=False)

    # Streaming path: repeated encode calls into a preallocated output buffer.
    # This reduces peak RAM for very large corpora and still uses ST's internal tokenization.
    # We still do an OOM fallback by reducing bs.
    while True:
        try:
            s0, e0 = next(iter(_iter_batches(n, bs)))
            first = _encode_list(sentences[s0:e0], bs, progress=False)
            # Ensure float32 output for compatibility with existing downstream expectations.
            out = np.empty((n, first.shape[1]), dtype=np.float32)
            out[s0:e0] = first.astype(np.float32, copy=False)

            if show_progress:
                print(f"Encoded {e0}/{n}")

            for s, e in _iter_batches(n, bs):
                if s == s0 and e == e0:
                    continue
                chunk = _encode_list(sentences[s:e], bs, progress=False)
                out[s:e] = chunk.astype(np.float32, copy=False)
                if show_progress and (e == n or (e // bs) % 10 == 0):
                    print(f"Encoded {e}/{n}")

            return out
        except RuntimeError as e:
            if _is_oom(e) and bs > 1:
                bs = max(1, bs // 2)
                if torch.cuda.is_available() and device.type == "cuda":
                    torch.cuda.empty_cache()
                print(f"OOM during streaming encode; retrying with batch_size={bs}", file=sys.stderr)
                continue
            raise


# ----------------------------- Main pipeline -----------------------------

def build_embedding_index_npz(
    corpus_path: Path,
    out_npz: Path,
    *,
    model_name: str,
    device: torch.device,
    batch_size: int,
    max_length: int,
    normalize_embeddings: bool,
    truncate_dim: Optional[int],
    compressed: bool,
    show_progress: bool,
    stream: bool,
    id_col: str = "sentence_id",
    text_col: str = "sentence",
) -> Path:
    total_start = time.perf_counter()

    print(f"Loading corpus: {corpus_path}")
    df = pd.read_parquet(corpus_path)

    if id_col not in df.columns or text_col not in df.columns:
        raise ValueError(f"Corpus missing required column '{id_col}' or '{text_col}'. Columns: {list(df.columns)}")

    print("Computing dataset fingerprint (sha256 over sentence_id + content) ...")
    fp_start = time.perf_counter()
    fp = sha256_fingerprint_sentence_id_content(df, id_col=id_col, text_col=text_col)
    fp_end = time.perf_counter()
    print(f"Fingerprint computed in {fp_end - fp_start:.1f} s: {fp}")

    ids = df[id_col].to_numpy()
    sentences = df[text_col].astype(str).tolist()

    print(f"Using device: {device}")
    print(f"Loading model: {model_name}")

    print(f"Encoding {len(sentences):,} sentences ...")
    enc_start = time.perf_counter()
    embeddings = encode_sentences(
        sentences,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        normalize_embeddings=normalize_embeddings,
        truncate_dim=truncate_dim,
        stream=stream,
        show_progress=show_progress,
    )
    enc_end = time.perf_counter()

    N, d = embeddings.shape
    print(f"Embeddings shape: {embeddings.shape} ({embeddings.dtype})")

    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    print(f"Writing NPZ bundle: {out_npz} (compressed={compressed})")
    atomic_save_npz(
        out_npz,
        compressed=compressed,
        verify=True,
        sentence_id=ids.astype(np.int64, copy=False),
        embeddings=embeddings,
        # minimal metadata
        model=np.array(model_name),
        truncate_dim=np.array(0 if truncate_dim is None else truncate_dim, dtype=np.int32),
        truncate_dim_is_none=np.array(truncate_dim is None),
        normalize=np.array(bool(normalize_embeddings)),
        n_rows=np.array(int(N), dtype=np.int64),
        dim=np.array(int(d), dtype=np.int32),
        created_at_utc=np.array(created_at),
        corpus_path=np.array(str(corpus_path)),
        dataset_fingerprint_sha256=np.array(fp),
    )

    # Post-write sanity check (also close handle properly)
    with np.load(out_npz, allow_pickle=False) as d2:
        if d2["sentence_id"].shape[0] != N or d2["embeddings"].shape[0] != N:
            raise RuntimeError("Post-write verification failed: row counts mismatch.")
        if d2["embeddings"].shape[1] != d:
            raise RuntimeError("Post-write verification failed: embedding dim mismatch.")

    total_end = time.perf_counter()
    print(f"Done. Wrote: {out_npz.resolve()}")
    print(f"Encoding time: {enc_end - enc_start:.1f} s ({(enc_end - enc_start)/N:.4f} s/sentence)")
    print(f"Total runtime: {(total_end - total_start)/60:.1f} minutes")

    return out_npz


# ----------------------------- CLI -----------------------------

def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build embedding_index.npz with sentence_id + embeddings (atomic, robust).")
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH, help="Input parquet containing sentences.")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_NPZ, help="Output NPZ bundle path.")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL, help="HF model name for embeddings.")
    p.add_argument("--device", type=str, default="auto", help="Device: auto|cpu|cuda|mps")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH, help=f"Batch size for encoding (default: {DEFAULT_BATCH}).")
    p.add_argument(
        "--max-length",
        type=int,
        default=0,
        help="Max token length for encoder. Use 0 to keep the model's default (auto).",
    )
    p.add_argument("--normalize", action="store_true", help="L2-normalize embeddings (default: disabled).")
    p.add_argument("--truncate-dim", type=int, default=0, help="If >0, truncate embedding dimension to this value.")
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
    p.add_argument("--id-col", type=str, default="sentence_id", help="ID column name.")
    p.add_argument("--text-col", type=str, default="sentence", help="Text column name.")
    return p.parse_args(list(argv))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    device = choose_device(args.device)
    truncate_dim = args.truncate_dim if args.truncate_dim > 0 else None
    compressed = bool(args.compress)
    show_progress = bool(args.progress)
    stream = bool(args.stream)

    build_embedding_index_npz(
        corpus_path=args.corpus,
        out_npz=args.out,
        model_name=args.model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        normalize_embeddings=bool(args.normalize),
        truncate_dim=truncate_dim,
        compressed=compressed,
        show_progress=show_progress,
        stream=stream,
        id_col=args.id_col,
        text_col=args.text_col,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
