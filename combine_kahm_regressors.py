#!/usr/bin/env python3
"""
combine_kahm_regressors.py

Distance-gated combination of two already-trained KAHM regressors.

Rationale
---------
You have two KAHM regressors trained on different data regimes (corpus vs queries).
Their clusterings (and thus cluster counts) may differ, which makes probability
scores (e.g., max soft assignment probability) not directly comparable.

This module combines the two models by computing, for each input sample x, the
*minimum unnormalized distance* to any cluster in each model:

    score_M(x) = min_c d_M,c(x)

Lower scores indicate the input is well-explained by (close to) at least one
cluster of that model. The combined predictor selects per-sample the prediction
from the model with the smaller score.

Key properties
--------------
- Uses distances from kahm_regression._call_combine_multiple_autoencoders_extended
  with the same distance_type used by kahm_regression.kahm_regress (currently "folding").
- Supports both 'hard' and 'soft' prediction:
    - hard: choose nearest cluster center
    - soft: compute weights from distances via (1 - d)^alpha with optional top-k sparsification,
            consistent with kahm_regression.distances_to_probabilities_one_minus_sharp.

API
---
The primary importable function is:

    combine_kahm_regressors_distance_gated(...)

The script can also be run as a CLI to combine predictions for an input matrix
stored as .npy or .npz.

Notes
-----
- Distance gating avoids the obvious bias from different numbers of clusters, but
  it can still be biased if one model has much denser cluster coverage. If this
  becomes material, calibrate scores per-model on a held-out set (e.g., map
  min-distance to a percentile per model and gate on the percentile).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(it, *args, **kwargs):  # type: ignore
        return it


# ----------------------------
# Helpers: I/O, normalization
# ----------------------------

def _l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, eps).astype(np.float32)
    return X / n


def _l2_normalize_cols(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    n = np.linalg.norm(X, axis=0, keepdims=True)
    n = np.maximum(n, eps).astype(np.float32)
    return X / n


def load_array(path: str, key: Optional[str] = None) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    if p.suffix.lower() == ".npy":
        arr = np.load(str(p))
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"Expected ndarray in {path}")
        return arr

    if p.suffix.lower() == ".npz":
        data = np.load(str(p), allow_pickle=False)
        if key is None:
            # Heuristic: prefer 'embeddings', else 'X', else first key
            for k in ("embeddings", "X"):
                if k in data:
                    key = k
                    break
            if key is None:
                key = list(data.keys())[0]
        if key not in data:
            raise KeyError(f"Key '{key}' not found in {path}. Available: {list(data.keys())}")
        return np.asarray(data[key])

    raise ValueError(f"Unsupported input extension: {p.suffix}. Use .npy or .npz")


def save_array(path: str, arr: np.ndarray, *, output_layout: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if output_layout not in {"row", "col"}:
        raise ValueError("--output_layout must be 'row' or 'col'")

    A = np.asarray(arr, dtype=np.float32)
    if p.suffix.lower() == ".npy":
        np.save(str(p), A)
        return

    if p.suffix.lower() == ".npz":
        np.savez_compressed(str(p), embeddings=A)
        return

    raise ValueError(f"Unsupported output extension: {p.suffix}. Use .npy or .npz")


def _as_col_major(X: np.ndarray, input_layout: str) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if input_layout == "auto":
        # Heuristic: if rows >> cols, likely row-major; otherwise ambiguous
        if X.ndim != 2:
            raise ValueError(f"X must be 2D; got shape={X.shape}")
        if X.shape[0] >= X.shape[1]:
            input_layout = "row"
        else:
            input_layout = "col"

    if input_layout == "row":
        return np.ascontiguousarray(X.T)
    if input_layout == "col":
        return np.ascontiguousarray(X)
    raise ValueError("--input_layout must be one of: auto,row,col")


def _as_row_major(Y_col: np.ndarray, output_layout: str) -> np.ndarray:
    if output_layout == "col":
        return np.ascontiguousarray(Y_col)
    if output_layout == "row":
        return np.ascontiguousarray(Y_col.T)
    raise ValueError("--output_layout must be 'row' or 'col'")


# ----------------------------
# Core: distance + prediction
# ----------------------------

def _resolve_soft_params(model: dict, alpha: Optional[float], topk: Optional[int | None]) -> Tuple[float, Optional[int]]:
    a = float(alpha) if alpha is not None else float(model.get("soft_alpha", 10.0))
    k = topk if topk is not None else model.get("soft_topk", 10)
    if k is not None:
        k = int(k)
        if k <= 0:
            raise ValueError("topk must be positive or None")
    return a, k


def _load_ae_ref(model: dict, ae_ref: Any) -> Tuple[list, bool]:
    """
    Return (AE_list, from_disk). AE_list is always a list suitable for passing to
    _call_combine_multiple_autoencoders_extended.
    """
    # In-memory classifier entries may already be a list/tuple of AE objects
    if isinstance(ae_ref, (list, tuple)):
        return list(ae_ref), False

    # Disk-backed classifier entries are saved as relative joblib paths (strings)
    if isinstance(ae_ref, (str, os.PathLike)):
        from joblib import load as joblib_load  # local import
        base_dir = model.get("classifier_dir", None)
        p = Path(ae_ref)
        if not p.is_absolute():
            if base_dir is None:
                raise ValueError("Model AE ref is relative, but model['classifier_dir'] is not set.")
            p = Path(base_dir) / p
        obj = joblib_load(str(p))
        if isinstance(obj, (list, tuple)):
            return list(obj), True
        return [obj], True

    # Fallback: single AE object
    return [ae_ref], False


def kahm_predict_with_min_distance(
    model: dict,
    X_new_col: np.ndarray,
    *,
    mode: str = "soft",
    alpha: Optional[float] = None,
    topk: Optional[int | None] = None,
    batch_size: int = 2048,
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute (Y_pred_col, min_dist) for a KAHM model on inputs X_new_col=(D_in,N).

    - Y_pred_col has shape (D_out, N)
    - min_dist has shape (N,), the minimum cluster distance per sample
    """
    from kahm_regression import _call_combine_multiple_autoencoders_extended, l2_normalize_columns, distances_to_probabilities_one_minus_sharp

    X_new_col = np.asarray(X_new_col, dtype=np.float32)
    if X_new_col.ndim != 2:
        raise ValueError(f"X_new_col must be 2D (D_in,N); got {X_new_col.shape}")

    # Apply model input scaling (matches kahm_regress behavior)
    input_scale = float(model.get("input_scale", 1.0))
    if input_scale != 1.0:
        X_new_col = np.asarray(X_new_col * np.float32(input_scale), dtype=np.float32)

    cluster_centers = np.asarray(model.get("cluster_centers", None))
    if cluster_centers is None:
        raise KeyError("Model missing 'cluster_centers'.")
    cluster_centers = np.asarray(cluster_centers, dtype=np.float32)
    if cluster_centers.ndim != 2:
        raise ValueError(f"cluster_centers must be 2D (D_out,C); got {cluster_centers.shape}")

    # Optional normalization of centers (matches kahm_regress behavior)
    if str(model.get("cluster_centers_normalization", "none")).lower().strip() == "l2":
        cluster_centers = l2_normalize_columns(cluster_centers)

    AE_arr = model.get("_classifier_cache", model.get("classifier", None))
    if not isinstance(AE_arr, (list, tuple)) or len(AE_arr) == 0:
        raise TypeError("Expected model['classifier'] (or model['_classifier_cache']) to be a non-empty list/tuple.")
    C_eff = int(cluster_centers.shape[1])
    if len(AE_arr) != C_eff:
        raise ValueError(f"Mismatch: got {len(AE_arr)} autoencoders but cluster_centers has C_eff={C_eff} clusters.")

    N_new = int(X_new_col.shape[1])
    bs = int(batch_size) if (batch_size is not None and int(batch_size) > 0) else None

    # Distance_type is currently fixed in kahm_regress
    distance_type = "folding"

    # Track min distance for gating
    best_dist = np.full((N_new,), np.inf, dtype=np.float32)

    # HARD: track best idx
    best_idx = np.zeros((N_new,), dtype=np.int64)

    mode = str(mode).lower().strip()
    if mode not in {"hard", "soft"}:
        raise ValueError("mode must be 'hard' or 'soft'.")

    alpha_resolved, topk_resolved = _resolve_soft_params(model, alpha, topk)

    # SOFT: maintain top-k distances per sample (smallest distances)
    if mode == "soft" and topk_resolved is not None:
        k = int(topk_resolved)
        topk_dist = np.full((k, N_new), np.inf, dtype=np.float32)
        topk_idx = np.full((k, N_new), -1, dtype=np.int64)
    else:
        topk_dist = None
        topk_idx = None

    iterator = enumerate(AE_arr)
    if show_progress:
        iterator = tqdm(list(iterator), desc="KAHM: distance eval", unit="cluster")

    for c, ae_ref in iterator:
        AE_list, from_disk = _load_ae_ref(model, ae_ref)
        try:
            if bs is None:
                d = _call_combine_multiple_autoencoders_extended(X_new_col, AE_list, distance_type, n_jobs=1)
                d = np.asarray(d, dtype=np.float32).reshape(-1)
                if d.size != N_new:
                    raise ValueError(f"Distance vector from cluster {c+1} has shape {d.shape}; expected ({N_new},).")

                # update hard best
                mask = d < best_dist
                best_dist[mask] = d[mask]
                best_idx[mask] = int(c)

                # update soft topk
                if topk_dist is not None and topk_idx is not None:
                    # update only where d is better than current worst
                    worst_pos = np.argmax(topk_dist, axis=0)
                    worst_val = topk_dist[worst_pos, np.arange(N_new)]
                    m2 = d < worst_val
                    if np.any(m2):
                        cols = np.where(m2)[0]
                        rp = worst_pos[cols]
                        topk_dist[rp, cols] = d[cols]
                        topk_idx[rp, cols] = int(c)
            else:
                for s in range(0, N_new, bs):
                    e = min(s + bs, N_new)
                    Xb = X_new_col[:, s:e]
                    d = _call_combine_multiple_autoencoders_extended(Xb, AE_list, distance_type, n_jobs=1)
                    d = np.asarray(d, dtype=np.float32).reshape(-1)
                    if d.size != (e - s):
                        raise ValueError(f"Distance vector from cluster {c+1} has shape {d.shape}; expected ({e-s},).")

                    # update hard best on slice
                    cur = best_dist[s:e]
                    mask = d < cur
                    if np.any(mask):
                        idxs = np.where(mask)[0]
                        cur[idxs] = d[idxs]
                        best_dist[s:e] = cur
                        best_idx[s:e][idxs] = int(c)

                    # update soft topk on slice
                    if topk_dist is not None and topk_idx is not None:
                        td = topk_dist[:, s:e]
                        ti = topk_idx[:, s:e]
                        worst_pos = np.argmax(td, axis=0)
                        worst_val = td[worst_pos, np.arange(e - s)]
                        m2 = d < worst_val
                        if np.any(m2):
                            cols = np.where(m2)[0]
                            rp = worst_pos[cols]
                            td[rp, cols] = d[cols]
                            ti[rp, cols] = int(c)
                            topk_dist[:, s:e] = td
                            topk_idx[:, s:e] = ti
        finally:
            # Free disk-loaded AE objects promptly to control peak memory
            if from_disk:
                del AE_list

    # Compute prediction
    if mode == "hard":
        Y_pred = cluster_centers[:, best_idx]  # (D_out, N)
        return np.asarray(Y_pred, dtype=np.float32), best_dist

    # SOFT MODE
    if topk_resolved is None:
        # Streaming full mixture would be very expensive; fall back to kahm_regress behavior by building full D.
        # For practical setups, keep topk_resolved not None.
        D = np.empty((C_eff, N_new), dtype=np.float32)
        # Re-run distances to build D (rare path). This is intentionally explicit.
        iterator = enumerate(AE_arr)
        if show_progress:
            iterator = tqdm(list(iterator), desc="KAHM: building full D", unit="cluster")
        for c, ae_ref in iterator:
            AE_list, from_disk = _load_ae_ref(model, ae_ref)
            try:
                if bs is None:
                    d = _call_combine_multiple_autoencoders_extended(X_new_col, AE_list, distance_type, n_jobs=1)
                    d = np.asarray(d, dtype=np.float32).reshape(-1)
                    D[c, :] = d
                else:
                    for s in range(0, N_new, bs):
                        e = min(s + bs, N_new)
                        Xb = X_new_col[:, s:e]
                        d = _call_combine_multiple_autoencoders_extended(Xb, AE_list, distance_type, n_jobs=1)
                        d = np.asarray(d, dtype=np.float32).reshape(-1)
                        D[c, s:e] = d
            finally:
                if from_disk:
                    del AE_list
        P = distances_to_probabilities_one_minus_sharp(D, alpha=float(alpha_resolved), topk=None, inplace=True)
        Y_pred = cluster_centers @ P
        return np.asarray(Y_pred, dtype=np.float32), best_dist

    # Use kept top-k distances to compute weights (no need to store full D)
    k = int(topk_resolved)
    assert topk_dist is not None and topk_idx is not None

    # Ensure no missing indices (can happen only if k > C_eff)
    if k > C_eff:
        raise ValueError(f"topk={k} exceeds number of clusters C_eff={C_eff}.")

    # Convert distances to scores and normalize per sample
    one_minus = np.clip(1.0 - topk_dist, 0.0, 1.0)
    S = np.power(one_minus, float(alpha_resolved), dtype=np.float32)
    denom = np.sum(S, axis=0, keepdims=True)
    # uniform fallback where denom==0
    zero = denom <= 0
    if np.any(zero):
        denom = np.where(zero, 1.0, denom)
        S = np.where(zero, 1.0 / float(k), S)

    W = S / denom  # (k,N)

    # Weighted sum of corresponding cluster centers
    D_out = int(cluster_centers.shape[0])
    Y_pred = np.zeros((D_out, N_new), dtype=np.float32)
    for j in range(k):
        idx = topk_idx[j, :].astype(np.int64, copy=False)
        Y_pred += cluster_centers[:, idx] * W[j, :][None, :]

    return Y_pred, best_dist


def combine_kahm_regressors_distance_gated(
    X: np.ndarray,
    *,
    embedding_model: dict,
    query_model: dict,
    input_layout: str = "row",
    output_layout: str = "row",
    mode: str = "soft",
    alpha: Optional[float] = None,
    topk: Optional[int | None] = None,
    batch_size: int = 2048,
    tie_break: str = "query",
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Combine two KAHM models via min-distance gating.

    Returns:
      Y_pred: combined predictions (layout controlled by output_layout)
      chosen: int8 array shape (N,) with 0=embedding_model, 1=query_model
      score_embedding: min-distance per sample for embedding_model
      score_query:     min-distance per sample for query_model
    """
    X_col = _as_col_major(X, input_layout)
    # Normalize inputs for cosine/IP retrieval pipelines (optional but typical).
    # We do NOT normalize here by default; callers should do so explicitly if desired.

    Y_emb, d_emb = kahm_predict_with_min_distance(
        embedding_model, X_col, mode=mode, alpha=alpha, topk=topk, batch_size=batch_size, show_progress=show_progress
    )
    Y_q, d_q = kahm_predict_with_min_distance(
        query_model, X_col, mode=mode, alpha=alpha, topk=topk, batch_size=batch_size, show_progress=show_progress
    )

    tb = str(tie_break).lower().strip()
    if tb not in {"query", "embedding"}:
        raise ValueError("tie_break must be 'query' or 'embedding'")

    # Choose model with smaller distance (better). Ties resolved deterministically.
    if tb == "query":
        choose_query = d_q <= d_emb
    else:
        choose_query = d_q < d_emb

    chosen = choose_query.astype(np.int8)

    Y_comb_col = np.empty_like(Y_emb, dtype=np.float32)
    # choose_query True => use query model
    Y_comb_col[:, choose_query] = Y_q[:, choose_query]
    Y_comb_col[:, ~choose_query] = Y_emb[:, ~choose_query]

    return _as_row_major(Y_comb_col, output_layout), chosen, d_emb.astype(np.float32), d_q.astype(np.float32)


# ----------------------------
# CLI
# ----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Combine two KAHM regressors via min-distance gating (cluster proximity).")
    ap.add_argument("--embedding_model", required=True, help="Path to embedding regressor joblib (Model A).")
    ap.add_argument("--query_model", required=True, help="Path to query regressor joblib (Model B).")
    ap.add_argument("--embedding_base_dir", default=None, help="Optional override for Model A classifier_dir.")
    ap.add_argument("--query_base_dir", default=None, help="Optional override for Model B classifier_dir.")

    ap.add_argument("--x", required=True, help="Input embeddings (.npy or .npz).")
    ap.add_argument("--x_key", default=None, help="If --x is .npz, the key to load (default: embeddings/X/first).")

    ap.add_argument("--input_layout", default="auto", choices=["auto", "row", "col"],
                    help="Layout of input X: row=(N,D), col=(D,N), auto=infer (default).")
    ap.add_argument("--output_layout", default="row", choices=["row", "col"],
                    help="Layout of output embeddings: row=(N,D_out), col=(D_out,N) (default: row).")

    ap.add_argument("--mode", default="soft", choices=["soft", "hard"], help="Prediction mode (default: soft).")
    ap.add_argument("--alpha", type=float, default=None, help="Soft alpha override (default: use model soft_alpha).")
    ap.add_argument("--topk", type=int, default=None, help="Soft topk override (default: use model soft_topk).")
    ap.add_argument("--batch_size", type=int, default=2048, help="Distance batch size (default: 2048).")
    ap.add_argument("--tie_break", default="query", choices=["query", "embedding"],
                    help="Tie-break when distances are equal (default: query).")

    ap.add_argument("--normalize_input", action="store_true", help="L2-normalize input rows/cols before inference.")
    ap.add_argument("--normalize_output", action="store_true", help="L2-normalize output embeddings.")
    ap.add_argument("--no_progress", action="store_true", help="Disable progress bars.")
    ap.add_argument("--out", required=True, help="Output path (.npy or .npz).")
    ap.add_argument("--save_aux", action="store_true", help="If output is .npz, also store gating diagnostics.")

    return ap


def main() -> int:
    args = build_arg_parser().parse_args()

    from kahm_regression import load_kahm_regressor

    X = load_array(args.x, key=args.x_key)

    # Layout handling + optional input normalization
    if args.input_layout == "auto":
        # try to infer
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        inferred = "row" if X.shape[0] >= X.shape[1] else "col"
        input_layout = inferred
    else:
        input_layout = args.input_layout

    if args.normalize_input:
        if input_layout == "row":
            X = _l2_normalize_rows(np.asarray(X, dtype=np.float32))
        else:
            X = _l2_normalize_cols(np.asarray(X, dtype=np.float32))

    emb_model = load_kahm_regressor(args.embedding_model, base_dir=args.embedding_base_dir)
    q_model = load_kahm_regressor(args.query_model, base_dir=args.query_base_dir)

    Y, chosen, d_emb, d_q = combine_kahm_regressors_distance_gated(
        X,
        embedding_model=emb_model,
        query_model=q_model,
        input_layout=input_layout,
        output_layout=args.output_layout,
        mode=args.mode,
        alpha=args.alpha,
        topk=args.topk,
        batch_size=int(args.batch_size),
        tie_break=args.tie_break,
        show_progress=(not args.no_progress),
    )

    if args.normalize_output:
        if args.output_layout == "row":
            Y = _l2_normalize_rows(Y)
        else:
            Y = _l2_normalize_cols(Y)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix.lower() == ".npz":
        if args.save_aux:
            np.savez_compressed(
                str(out_path),
                embeddings=np.asarray(Y, dtype=np.float32),
                chosen_model=np.asarray(chosen, dtype=np.int8),
                min_dist_embedding=np.asarray(d_emb, dtype=np.float32),
                min_dist_query=np.asarray(d_q, dtype=np.float32),
            )
        else:
            np.savez_compressed(str(out_path), embeddings=np.asarray(Y, dtype=np.float32))
    elif out_path.suffix.lower() == ".npy":
        np.save(str(out_path), np.asarray(Y, dtype=np.float32))
    else:
        raise ValueError("Output must be .npy or .npz")

    # Print brief diagnostics
    frac_q = float(np.mean(chosen))
    print(f"Saved combined embeddings to: {out_path}")
    print(f"Gating: query chosen for {frac_q*100:.1f}% of samples; embedding for {(1-frac_q)*100:.1f}%.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
