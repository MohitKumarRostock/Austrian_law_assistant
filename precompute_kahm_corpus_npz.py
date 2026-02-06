#!/usr/bin/env python3
"""
precompute_kahm_corpus_npz.py

Precompute KAHM-regressed approximate Mixedbread embeddings for ALL sentences
(from IDF–SVD sentence embeddings) and save to NPZ for fast retrieval.

Update (combined models)
------------------------
This version can optionally combine TWO already-trained KAHM regressors:
  - embedding regressor (trained on corpus embeddings)
  - query regressor     (trained on query embeddings)

Combination rule (distance gating):
  For each input sample x and each model M, compute:
      score_M(x) = min_c d_{M,c}(x)
  using unnormalized distances returned by
      kahm_regression._call_combine_multiple_autoencoders_extended
  (same distance type as kahm_regression.kahm_regress, i.e., "folding").

  Choose the prediction from the model with smaller score.

Defaults:
  --idf_svd_npz          embedding_index_idf_svd.npz
  --kahm_model           kahm_regressor_idf_to_mixedbread.joblib
  --kahm_query_model     (optional) kahm_query_regressor_idf_to_mixedbread.joblib
  --kahm_strategy        auto   (combine if query model exists, else single model)
  --out_npz              embedding_index_kahm_mixedbread_approx.npz
  --kahm_mode            soft
  --batch                1024

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

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(it, *args, **kwargs):  # type: ignore
        return it


DEFAULT_IDF_SVD_NPZ = "embedding_index_idf_svd.npz"
DEFAULT_KAHM_MODEL = "kahm_regressor_idf_to_mixedbread.joblib"
DEFAULT_KAHM_QUERY_MODEL = "kahm_query_regressor_idf_to_mixedbread.joblib"
DEFAULT_OUT_NPZ = "embedding_index_kahm_mixedbread_approx.npz"
DEFAULT_KAHM_MODE = "soft"
DEFAULT_BATCH = 1024


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
    X_row: np.ndarray,
    *,
    mode: str,
    batch_size: int,
    alpha: Optional[float] = None,
    topk: Optional[int | None] = None,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Run kahm_regress() once for a row-major X (N,D). Returns row-major (N,D_out).
    """
    from kahm_regression import kahm_regress

    X_row = np.asarray(X_row, dtype=np.float32)
    if X_row.ndim != 2:
        raise ValueError(f"X must be 2D (N,D); got {X_row.shape}")

    Xt = np.ascontiguousarray(X_row.T)  # (D_in, N)
    Yt = kahm_regress(
        model,
        Xt,
        mode=str(mode),
        batch_size=int(batch_size),
        alpha=alpha,
        topk=topk,
        show_progress=bool(show_progress),
    )
    return np.asarray(Yt.T, dtype=np.float32)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Precompute KAHM-regressed corpus embeddings and save as NPZ.")

    p.add_argument("--idf_svd_npz", default=DEFAULT_IDF_SVD_NPZ,
                   help=f"Path to IDF–SVD NPZ (default: {DEFAULT_IDF_SVD_NPZ})")

    # Keep legacy flag name for backward compatibility: --kahm_model refers to the "embedding regressor"
    p.add_argument("--kahm_model", default=DEFAULT_KAHM_MODEL,
                   help=f"Path to trained KAHM embedding regressor joblib (default: {DEFAULT_KAHM_MODEL})")

    p.add_argument("--kahm_query_model", default=DEFAULT_KAHM_QUERY_MODEL,
                   help=f"Optional path to trained KAHM query regressor joblib (default: none; suggested: {DEFAULT_KAHM_QUERY_MODEL})")

    p.add_argument("--kahm_embedding_base_dir", default=None,
                   help="Optional override for embedding regressor classifier_dir (if model was relocated).")
    p.add_argument("--kahm_query_base_dir", default=None,
                   help="Optional override for query regressor classifier_dir (if model was relocated).")

    p.add_argument("--kahm_strategy", default="embedding_only",
                   choices=["auto", "embedding_only", "query_only", "combined_distance"],
                   help="How to compute KAHM embeddings. 'auto' combines if kahm_query_model is provided, else uses embedding_only.")

    p.add_argument("--combine_tie_break", default="query", choices=["query", "embedding"],
                   help="When min distances are equal, choose which model (default: query).")

    p.add_argument("--out_npz", default=DEFAULT_OUT_NPZ,
                   help=f"Output NPZ path (default: {DEFAULT_OUT_NPZ})")

    p.add_argument("--kahm_mode", choices=["soft", "hard"], default=DEFAULT_KAHM_MODE,
                   help="KAHM inference mode (default: soft)")
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                   help=f"Batch size for regression/distance eval (default: {DEFAULT_BATCH})")
    p.add_argument("--overwrite", action="store_true", help="Overwrite out_npz if it exists (default: no)")
    p.add_argument("--no_progress", action="store_true", help="Disable the progress bar (default: show)")

    # Optional: align soft parameters across models (recommended when comparing/combining)
    p.add_argument("--alpha", type=float, default=None,
                   help="Soft alpha override (default: use model soft_alpha).")
    p.add_argument("--topk", type=int, default=None,
                   help="Soft topk override (default: use model soft_topk).")

    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    if not os.path.exists(args.idf_svd_npz):
        print(f"ERROR: IDF–SVD NPZ not found: {args.idf_svd_npz}", file=sys.stderr)
        return 2
    if not os.path.exists(args.kahm_model):
        print(f"ERROR: KAHM embedding model not found: {args.kahm_model}", file=sys.stderr)
        return 2
    if args.kahm_query_model is not None and not os.path.exists(args.kahm_query_model):
        print(f"ERROR: KAHM query model not found: {args.kahm_query_model}", file=sys.stderr)
        return 2

    if os.path.exists(args.out_npz) and not args.overwrite:
        print(f"Output already exists: {args.out_npz}")
        print("Nothing to do (use --overwrite to recompute).")
        return 0

    print(f"Loading IDF–SVD NPZ: {args.idf_svd_npz}")
    ids, X, meta_x = load_npz_bundle(args.idf_svd_npz)
    print(f"  ids={ids.shape} X={X.shape}")

    from kahm_regression import load_kahm_regressor

    print(f"Loading KAHM embedding model: {args.kahm_model}")
    emb_model = load_kahm_regressor(args.kahm_model, base_dir=args.kahm_embedding_base_dir)

    q_model = None
    if args.kahm_query_model is not None:
        print(f"Loading KAHM query model: {args.kahm_query_model}")
        q_model = load_kahm_regressor(args.kahm_query_model, base_dir=args.kahm_query_base_dir)

    # Normalize inputs (typical cosine/IP retrieval)
    Xn = l2_normalize_rows(X)

    strategy = str(args.kahm_strategy).lower().strip()
    if strategy == "auto":
        strategy = "combined_distance" if q_model is not None else "embedding_only"

    show_progress = (not bool(args.no_progress))

    gating_meta: Dict[str, Any] = {}
    if strategy == "combined_distance":
        if q_model is None:
            print("ERROR: kahm_strategy=combined_distance requires --kahm_query_model.", file=sys.stderr)
            return 2

        # Import combiner (supports either module name)
        try:
            from combine_kahm_regressors import combine_kahm_regressors_distance_gated  # type: ignore
        except Exception:
            from combine_kahm_regressors_distance_gated import combine_kahm_regressors_distance_gated  # type: ignore

        print(f"Computing combined KAHM corpus embeddings via distance gating (mode={args.kahm_mode}) ...")
        Y_row, chosen, d_emb, d_q = combine_kahm_regressors_distance_gated(
            Xn,
            embedding_model=emb_model,
            query_model=q_model,
            input_layout="row",
            output_layout="row",
            mode=str(args.kahm_mode),
            alpha=args.alpha,
            topk=args.topk,
            batch_size=int(args.batch),
            tie_break=str(args.combine_tie_break),
            show_progress=show_progress,
        )

        frac_q = float(np.mean(chosen))
        gating_meta = dict(
            combine_strategy="combined_distance",
            combine_tie_break=str(args.combine_tie_break),
            chosen_query_fraction=frac_q,
            chosen_embedding_fraction=(1.0 - frac_q),
            min_dist_embedding_mean=float(np.mean(d_emb)),
            min_dist_query_mean=float(np.mean(d_q)),
        )
        Y = np.asarray(Y_row, dtype=np.float32)

    elif strategy == "query_only":
        if q_model is None:
            print("ERROR: kahm_strategy=query_only requires --kahm_query_model.", file=sys.stderr)
            return 2
        print(f"Regressing corpus embeddings via KAHM query model ({args.kahm_mode}) ...")
        Y = kahm_regress_batched(
            q_model, Xn, mode=args.kahm_mode, batch_size=int(args.batch),
            alpha=args.alpha, topk=args.topk, show_progress=show_progress
        )

    else:
        # embedding_only
        print(f"Regressing corpus embeddings via KAHM embedding model ({args.kahm_mode}) ...")
        Y = kahm_regress_batched(
            emb_model, Xn, mode=args.kahm_mode, batch_size=int(args.batch),
            alpha=args.alpha, topk=args.topk, show_progress=show_progress
        )

    Yn = l2_normalize_rows(Y)

    created_at = datetime.now(timezone.utc).isoformat()
    fp_x = _npz_scalar_to_str(meta_x.get("dataset_fingerprint_sha256"))

    # Prepare metadata
    out_meta = dict(
        created_at_utc=created_at,
        source_idf_svd_npz=os.path.basename(args.idf_svd_npz),
        source_idf_svd_fingerprint_sha256=(fp_x or ""),
        kahm_strategy=str(strategy),
        source_kahm_embedding_model=os.path.basename(args.kahm_model),
        source_kahm_query_model=(os.path.basename(args.kahm_query_model) if args.kahm_query_model else ""),
        kahm_mode=str(args.kahm_mode),
        soft_alpha_override=("" if args.alpha is None else str(args.alpha)),
        soft_topk_override=("" if args.topk is None else str(args.topk)),
        n_sentences=int(ids.shape[0]),
        d_in=int(X.shape[1]),
        d_out=int(Yn.shape[1]),
        note="KAHM-regressed approximate Mixedbread embeddings from IDF–SVD; L2-normalized for cosine/IP FAISS.",
    )
    out_meta.update(gating_meta)

    print(f"Saving precomputed KAHM corpus NPZ: {args.out_npz}")
    np.savez_compressed(
        args.out_npz,
        sentence_id=ids.astype(np.int64, copy=False),
        embeddings=Yn.astype(np.float32, copy=False),

        created_at_utc=np.asarray(out_meta["created_at_utc"]),
        source_idf_svd_npz=np.asarray(out_meta["source_idf_svd_npz"]),
        source_idf_svd_fingerprint_sha256=np.asarray(out_meta["source_idf_svd_fingerprint_sha256"]),
        kahm_strategy=np.asarray(out_meta["kahm_strategy"]),
        source_kahm_embedding_model=np.asarray(out_meta["source_kahm_embedding_model"]),
        source_kahm_query_model=np.asarray(out_meta["source_kahm_query_model"]),
        kahm_mode=np.asarray(out_meta["kahm_mode"]),
        soft_alpha_override=np.asarray(out_meta["soft_alpha_override"]),
        soft_topk_override=np.asarray(out_meta["soft_topk_override"]),
        n_sentences=np.asarray(out_meta["n_sentences"]),
        d_in=np.asarray(out_meta["d_in"]),
        d_out=np.asarray(out_meta["d_out"]),
        note=np.asarray(out_meta["note"]),

        # Combined gating fields (empty defaults if not combined)
        combine_strategy=np.asarray(out_meta.get("combine_strategy", "")),
        combine_tie_break=np.asarray(out_meta.get("combine_tie_break", "")),
        chosen_query_fraction=np.asarray(float(out_meta.get("chosen_query_fraction", np.nan))),
        chosen_embedding_fraction=np.asarray(float(out_meta.get("chosen_embedding_fraction", np.nan))),
        min_dist_embedding_mean=np.asarray(float(out_meta.get("min_dist_embedding_mean", np.nan))),
        min_dist_query_mean=np.asarray(float(out_meta.get("min_dist_query_mean", np.nan))),
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
