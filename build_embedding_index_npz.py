#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


DEFAULT_CORPUS_PATH = Path("ris_sentences.parquet")
DEFAULT_OUT_NPZ = Path("embedding_index.npz")
DEFAULT_MODEL = "mixedbread-ai/deepset-mxbai-embed-de-large-v1"


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
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    all_embeds: List[np.ndarray] = []

    for i in tqdm(range(0, len(sentences), batch_size), total=(len(sentences) + batch_size - 1) // batch_size, desc="Batches"):
        batch = sentences[i : i + batch_size]
        enc = tokenizer(
            list(batch),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        out = model(**enc)
        pooled = mean_pooling(out.last_hidden_state, enc["attention_mask"])  # [B, H]

        if truncate_dim is not None and truncate_dim > 0:
            pooled = pooled[:, :truncate_dim]

        pooled = pooled.detach().cpu().numpy().astype(np.float32, copy=False)
        all_embeds.append(pooled)

    embeddings = np.vstack(all_embeds).astype(np.float32, copy=False)
    if normalize_embeddings:
        embeddings = l2_normalize(embeddings).astype(np.float32, copy=False)
    return embeddings


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
    id_col: str = "sentence_id",
    text_col: str = "content",
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
    )
    enc_end = time.perf_counter()

    N, d = embeddings.shape
    print(f"Embeddings shape: {embeddings.shape} ({embeddings.dtype})")

    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    compressed = True
    print(f"Writing NPZ bundle: {out_npz} (compressed={compressed})")
    atomic_save_npz(
        out_npz,
        compressed=compressed,
        verify=True,
        sentence_id=ids.astype(np.int64, copy=False),
        embeddings=embeddings,
        # minimal metadata
        model=np.array(model_name),
        truncate_dim=np.array(int(truncate_dim), dtype=np.int32),
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
    p.add_argument("--batch-size", type=int, default=16, help="Batch size for encoding.")
    p.add_argument("--max-length", type=int, default=512, help="Max token length for encoder.")
    p.add_argument("--normalize", action="store_true", help="L2-normalize embeddings.")
    p.add_argument("--truncate-dim", type=int, default=0, help="If >0, truncate embedding dimension to this value.")
    p.add_argument("--id-col", type=str, default="sentence_id", help="ID column name.")
    p.add_argument("--text-col", type=str, default="content", help="Text column name.")
    return p.parse_args(list(argv))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    device = choose_device(args.device)
    truncate_dim = int(args.truncate_dim) if int(args.truncate_dim) > 0 else None

    build_embedding_index_npz(
        corpus_path=args.corpus,
        out_npz=args.out,
        model_name=args.model,
        device=device,
        batch_size=int(args.batch_size),
        max_length=int(args.max_length),
        normalize_embeddings=bool(args.normalize),
        truncate_dim=truncate_dim,
        id_col=args.id_col,
        text_col=args.text_col,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
