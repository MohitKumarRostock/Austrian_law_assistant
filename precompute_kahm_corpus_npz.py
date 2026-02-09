#!/usr/bin/env python3
"""
precompute_kahm_corpus_npz.py

Precompute KAHM-regressed approximate Mixedbread embeddings for ALL sentences
(from IDF–SVD sentence embeddings) and save to NPZ for fast retrieval.

Update (combined models)
------------------------
This version can optionally combine already-trained KAHM regressors.

In addition to passing a single .joblib model file, you may pass a DIRECTORY
containing multiple per-law regressors (e.g., kahm_embedding_regressors_by_law/).
When a directory is provided, all '*.joblib' models inside are loaded and
combined via min-distance gating (same distance definition as KAHM itself).

If both an embedding-model family and a query-model family are provided and
--kahm_strategy=combined_distance is used, the script first selects the best
model within each family (min-distance), then gates between families.

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
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(it, *args, **kwargs):  # type: ignore
        return it


DEFAULT_IDF_SVD_NPZ = "embedding_index_idf_svd.npz"
DEFAULT_KAHM_MODEL = "kahm_embedding_regressors_by_law"
DEFAULT_KAHM_QUERY_MODEL = "kahm_query_regressors_by_law"
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


def _find_joblib_models(model_dir: str) -> List[Path]:
    p = Path(model_dir)
    if not p.is_dir():
        raise NotADirectoryError(f"Not a directory: {model_dir}")
    files = sorted([x for x in p.iterdir() if x.is_file() and x.suffix.lower() == ".joblib"])
    return files


def load_kahm_models(path_or_dir: str, *, base_dir_override: Optional[str] = None) -> Tuple[List[dict], List[str], str]:
    """Load one or many KAHM regressors.

    - If `path_or_dir` is a file: returns ([model], [name], "file")
    - If `path_or_dir` is a directory: loads all '*.joblib' inside and returns (models, names, "dir")

    Names are derived from file stems.
    """
    from kahm_regression import load_kahm_regressor

    p = Path(path_or_dir)
    if p.is_dir():
        files = _find_joblib_models(str(p))
        if len(files) == 0:
            raise FileNotFoundError(f"No .joblib models found in directory: {path_or_dir}")
        base_dir = base_dir_override if base_dir_override is not None else str(p)
        models: List[dict] = []
        names: List[str] = []
        for f in files:
            m = load_kahm_regressor(str(f), base_dir=base_dir)
            models.append(m)
            names.append(f.stem)
        return models, names, "dir"

    # file
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {path_or_dir}")
    m = load_kahm_regressor(str(p), base_dir=base_dir_override)
    return [m], [p.stem], "file"


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


def _format_choice_fractions(chosen_idx: np.ndarray, names: Sequence[str]) -> str:
    """Return a compact 'name=fraction;name=fraction;...' string."""
    chosen_idx = np.asarray(chosen_idx).reshape(-1)
    if chosen_idx.size == 0:
        return ""
    Q = len(list(names))
    if Q <= 0:
        return ""
    counts = np.bincount(chosen_idx.astype(np.int64, copy=False), minlength=Q).astype(np.float64)
    total = float(np.sum(counts))
    if total <= 0:
        return ""
    fracs = counts / total
    return ";".join([f"{names[i]}={fracs[i]:.6f}" for i in range(Q)])


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Precompute KAHM-regressed corpus embeddings and save as NPZ.")

    p.add_argument("--idf_svd_npz", default=DEFAULT_IDF_SVD_NPZ,
                   help=f"Path to IDF–SVD NPZ (default: {DEFAULT_IDF_SVD_NPZ})")

    # Keep legacy flag name for backward compatibility: --kahm_model refers to the "embedding regressor"
    p.add_argument(
        "--kahm_model",
        default=DEFAULT_KAHM_MODEL,
        help=(
            "Path to trained KAHM embedding regressor (.joblib) OR a directory containing multiple per-law regressors. "
            f"(default: {DEFAULT_KAHM_MODEL})"
        ),
    )

    p.add_argument(
        "--kahm_query_model",
        default=None,
        help=(
            "Optional path to trained KAHM query regressor (.joblib) OR a directory containing multiple per-law regressors. "
            f"(suggested: {DEFAULT_KAHM_QUERY_MODEL})"
        ),
    )

    p.add_argument("--kahm_embedding_base_dir", default=None,
                   help="Optional override for embedding regressor classifier_dir (if model was relocated).")
    p.add_argument("--kahm_query_base_dir", default=None,
                   help="Optional override for query regressor classifier_dir (if model was relocated).")

    p.add_argument("--kahm_strategy", default="embedding_only",
                   choices=["auto", "embedding_only", "query_only", "combined_distance"],
                   help="How to compute KAHM embeddings. 'auto' combines if kahm_query_model is provided, else uses embedding_only.")

    p.add_argument("--combine_tie_break", default="embedding", choices=["query", "embedding"],
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

    print(f"Loading KAHM embedding model(s): {args.kahm_model}")
    emb_models, emb_names, emb_kind = load_kahm_models(args.kahm_model, base_dir_override=args.kahm_embedding_base_dir)
    print(f"  loaded {len(emb_models)} embedding model(s) ({emb_kind})")

    q_models: Optional[List[dict]] = None
    q_names: Optional[List[str]] = None
    q_kind: Optional[str] = None
    if args.kahm_query_model is not None:
        print(f"Loading KAHM query model(s): {args.kahm_query_model}")
        q_models, q_names, q_kind = load_kahm_models(args.kahm_query_model, base_dir_override=args.kahm_query_base_dir)
        print(f"  loaded {len(q_models)} query model(s) ({q_kind})")

    # Normalize inputs (typical cosine/IP retrieval)
    Xn = l2_normalize_rows(X)

    strategy = str(args.kahm_strategy).lower().strip()
    if strategy == "auto":
        strategy = "combined_distance" if q_models is not None else "embedding_only"

    show_progress = (not bool(args.no_progress))

    gating_meta: Dict[str, Any] = {}

    # Lazy import: only needed for multi-model or family gating
    def _import_combiner():
        from combine_kahm_regressors_generalized import (
                combine_kahm_regressors_distance_gated,
                combine_kahm_regressors_distance_gated_multi,
                kahm_predict_with_min_distance,
            )
        return combine_kahm_regressors_distance_gated, combine_kahm_regressors_distance_gated_multi, kahm_predict_with_min_distance


    if strategy == "combined_distance":
        if q_models is None or q_names is None:
            print("ERROR: kahm_strategy=combined_distance requires --kahm_query_model.", file=sys.stderr)
            return 2

        combine_2, combine_multi, kahm_predict_with_min_distance = _import_combiner()

        print(f"Computing combined KAHM corpus embeddings via distance gating (mode={args.kahm_mode}) ...")

        # Fast path: single embedding model vs single query model
        if len(emb_models) == 1 and len(q_models) == 1:
            Y_row, chosen_family, d_emb, d_q = combine_2(
                Xn,
                embedding_model=emb_models[0],
                query_model=q_models[0],
                input_layout="row",
                output_layout="row",
                mode=str(args.kahm_mode),
                alpha=args.alpha,
                topk=args.topk,
                batch_size=int(args.batch),
                tie_break=str(args.combine_tie_break),
                show_progress=show_progress,
            )
            Y = np.asarray(Y_row, dtype=np.float32)
            frac_q = float(np.mean(chosen_family))
            gating_meta = dict(
                combine_strategy="combined_distance",
                combine_tie_break=str(args.combine_tie_break),
                chosen_query_fraction=frac_q,
                chosen_embedding_fraction=(1.0 - frac_q),
                min_dist_embedding_mean=float(np.mean(d_emb)),
                min_dist_query_mean=float(np.mean(d_q)),
            )

        else:
            # Family gating: pick best model within each family, then gate between families.
            X_col = np.ascontiguousarray(Xn.T)

            # --- embedding family ---
            if len(emb_models) == 1:
                Y_emb_col, d_emb_best = kahm_predict_with_min_distance(
                    emb_models[0],
                    X_col,
                    mode=str(args.kahm_mode),
                    alpha=args.alpha,
                    topk=args.topk,
                    batch_size=int(args.batch),
                    show_progress=show_progress,
                )
                Y_emb = np.ascontiguousarray(Y_emb_col.T)
                chosen_emb = np.zeros((Xn.shape[0],), dtype=np.int16)
            else:
                Y_emb, chosen_emb, d_emb_best, _all_scores, _names = combine_multi(
                    Xn,
                    models=emb_models,
                    model_names=emb_names,
                    input_layout="row",
                    output_layout="row",
                    mode=str(args.kahm_mode),
                    alpha=args.alpha,
                    topk=args.topk,
                    batch_size=int(args.batch),
                    tie_break="first",
                    show_progress=show_progress,
                    return_all_scores=False,
                )

            # --- query family ---
            if len(q_models) == 1:
                Y_q_col, d_q_best = kahm_predict_with_min_distance(
                    q_models[0],
                    X_col,
                    mode=str(args.kahm_mode),
                    alpha=args.alpha,
                    topk=args.topk,
                    batch_size=int(args.batch),
                    show_progress=show_progress,
                )
                Y_q = np.ascontiguousarray(Y_q_col.T)
                chosen_q = np.zeros((Xn.shape[0],), dtype=np.int16)
            else:
                Y_q, chosen_q, d_q_best, _all_scores2, _names2 = combine_multi(
                    Xn,
                    models=q_models,
                    model_names=q_names,
                    input_layout="row",
                    output_layout="row",
                    mode=str(args.kahm_mode),
                    alpha=args.alpha,
                    topk=args.topk,
                    batch_size=int(args.batch),
                    tie_break="first",
                    show_progress=show_progress,
                    return_all_scores=False,
                )

            d_emb_best = np.asarray(d_emb_best, dtype=np.float32).reshape(-1)
            d_q_best = np.asarray(d_q_best, dtype=np.float32).reshape(-1)
            if d_emb_best.shape != d_q_best.shape:
                raise ValueError(f"Distance shape mismatch: embedding={d_emb_best.shape} query={d_q_best.shape}")

            choose_q = d_q_best < d_emb_best
            if str(args.combine_tie_break).lower().strip() == "query":
                choose_q = np.logical_or(choose_q, d_q_best == d_emb_best)

            chosen_family = choose_q.astype(np.int8)
            Y = np.where(choose_q[:, None], Y_q, Y_emb).astype(np.float32, copy=False)

            frac_q = float(np.mean(chosen_family))
            gating_meta = dict(
                combine_strategy="combined_distance",
                combine_tie_break=str(args.combine_tie_break),
                chosen_query_fraction=frac_q,
                chosen_embedding_fraction=(1.0 - frac_q),
                min_dist_embedding_mean=float(np.mean(d_emb_best)),
                min_dist_query_mean=float(np.mean(d_q_best)),
                embedding_family_n_models=int(len(emb_models)),
                query_family_n_models=int(len(q_models)),
                embedding_family_choice_fractions=_format_choice_fractions(chosen_emb, emb_names),
                query_family_choice_fractions=_format_choice_fractions(chosen_q, q_names),
            )

    elif strategy == "query_only":
        if q_models is None or q_names is None:
            print("ERROR: kahm_strategy=query_only requires --kahm_query_model.", file=sys.stderr)
            return 2

        if len(q_models) == 1:
            print(f"Regressing corpus embeddings via KAHM query model ({args.kahm_mode}) ...")
            Y = kahm_regress_batched(
                q_models[0], Xn, mode=args.kahm_mode, batch_size=int(args.batch),
                alpha=args.alpha, topk=args.topk, show_progress=show_progress
            )
        else:
            _, combine_multi, _ = _import_combiner()
            print(f"Regressing corpus embeddings via combined query models ({len(q_models)} models; {args.kahm_mode}) ...")
            Y, chosen_idx, best_score, _all_scores, _names = combine_multi(
                Xn,
                models=q_models,
                model_names=q_names,
                input_layout="row",
                output_layout="row",
                mode=str(args.kahm_mode),
                alpha=args.alpha,
                topk=args.topk,
                batch_size=int(args.batch),
                tie_break="first",
                show_progress=show_progress,
                return_all_scores=False,
            )
            gating_meta = dict(
                combine_strategy="query_only_multi",
                query_family_n_models=int(len(q_models)),
                query_family_choice_fractions=_format_choice_fractions(chosen_idx, q_names),
                query_family_min_dist_mean=float(np.mean(best_score)),
            )

    else:
        # embedding_only
        if len(emb_models) == 1:
            print(f"Regressing corpus embeddings via KAHM embedding model ({args.kahm_mode}) ...")
            Y = kahm_regress_batched(
                emb_models[0], Xn, mode=args.kahm_mode, batch_size=int(args.batch),
                alpha=args.alpha, topk=args.topk, show_progress=show_progress
            )
        else:
            _, combine_multi, _ = _import_combiner()
            print(f"Regressing corpus embeddings via combined embedding models ({len(emb_models)} models; {args.kahm_mode}) ...")
            Y, chosen_idx, best_score, _all_scores, _names = combine_multi(
                Xn,
                models=emb_models,
                model_names=emb_names,
                input_layout="row",
                output_layout="row",
                mode=str(args.kahm_mode),
                alpha=args.alpha,
                topk=args.topk,
                batch_size=int(args.batch),
                tie_break="first",
                show_progress=show_progress,
                return_all_scores=False,
            )
            gating_meta = dict(
                combine_strategy="embedding_only_multi",
                embedding_family_n_models=int(len(emb_models)),
                embedding_family_choice_fractions=_format_choice_fractions(chosen_idx, emb_names),
                embedding_family_min_dist_mean=float(np.mean(best_score)),
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
        source_kahm_embedding_model=os.path.basename(str(args.kahm_model)),
        source_kahm_query_model=(os.path.basename(str(args.kahm_query_model)) if args.kahm_query_model else ""),
        source_kahm_embedding_model_kind=str(emb_kind),
        source_kahm_query_model_kind=(str(q_kind) if q_kind is not None else ""),
        source_kahm_embedding_models=(";".join(list(emb_names)) if emb_names else ""),
        source_kahm_query_models=(";".join(list(q_names)) if q_names else ""),
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
        source_kahm_embedding_model_kind=np.asarray(out_meta.get("source_kahm_embedding_model_kind", "")),
        source_kahm_query_model_kind=np.asarray(out_meta.get("source_kahm_query_model_kind", "")),
        source_kahm_embedding_models=np.asarray(out_meta.get("source_kahm_embedding_models", "")),
        source_kahm_query_models=np.asarray(out_meta.get("source_kahm_query_models", "")),
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

        # Per-family multi-model diagnostics (when directory models are used)
        embedding_family_n_models=np.asarray(int(out_meta.get("embedding_family_n_models", 0))),
        query_family_n_models=np.asarray(int(out_meta.get("query_family_n_models", 0))),
        embedding_family_choice_fractions=np.asarray(out_meta.get("embedding_family_choice_fractions", "")),
        query_family_choice_fractions=np.asarray(out_meta.get("query_family_choice_fractions", "")),
        embedding_family_min_dist_mean=np.asarray(float(out_meta.get("embedding_family_min_dist_mean", np.nan))),
        query_family_min_dist_mean=np.asarray(float(out_meta.get("query_family_min_dist_mean", np.nan))),
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
