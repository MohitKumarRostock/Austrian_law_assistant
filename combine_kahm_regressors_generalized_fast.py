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
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, overload, Literal

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



def materialize_kahm_classifier_cache(model: dict, *, force: bool = False, show_progress: bool = True) -> None:
    """Load any disk-backed AE refs in model['classifier'] into RAM once.

    Pure speed optimization: arithmetic is unchanged. After materialization,
    kahm_predict_with_min_distance will reuse in-memory AE objects via
    model['_classifier_cache'] and avoid repeated joblib loads.
    """
    if (not force) and isinstance(model.get("_classifier_cache", None), (list, tuple)) and len(model["_classifier_cache"]) > 0:
        return

    AE_arr = model.get("classifier", None)
    if not isinstance(AE_arr, (list, tuple)) or len(AE_arr) == 0:
        raise TypeError("Expected model['classifier'] to be a non-empty list/tuple.")

    try:
        from joblib import load as joblib_load  # local import
    except Exception as e:  # pragma: no cover
        raise ImportError("joblib is required to materialize classifier cache") from e

    base_dir = model.get("classifier_dir", None)

    it = enumerate(AE_arr)
    if show_progress:
        it = tqdm(it, total=len(AE_arr), desc="KAHM: materialize AEs", unit="cluster")

    cache: list[list[Any]] = []
    for _c, ae_ref in it:
        if isinstance(ae_ref, (list, tuple)):
            cache.append(list(ae_ref))
            continue
        if isinstance(ae_ref, (str, os.PathLike)):
            p = Path(ae_ref)
            if not p.is_absolute():
                if base_dir is None:
                    raise ValueError("Model AE ref is relative, but model['classifier_dir'] is not set.")
                p = Path(base_dir) / p
            obj = joblib_load(str(p))
            if isinstance(obj, (list, tuple)):
                cache.append(list(obj))
            else:
                cache.append([obj])
            continue
        cache.append([ae_ref])

    model["_classifier_cache"] = cache


def prepare_kahm_model_for_inference(
    model: dict,
    *,
    materialize_classifier: bool = True,
    cache_cluster_centers: bool = True,
    show_progress: bool = False,
) -> None:
    """Warm caches used during inference (no arithmetic changes)."""
    if materialize_classifier:
        materialize_kahm_classifier_cache(model, force=False, show_progress=bool(show_progress))

    if cache_cluster_centers:
        if model.get("_cluster_centers_cache", None) is None:
            cc_raw = model.get("cluster_centers", None)
            if cc_raw is None:
                raise KeyError("Model missing 'cluster_centers'.")
            cc = np.asarray(cc_raw, dtype=np.float32)
            if cc.ndim != 2:
                raise ValueError(f"cluster_centers must be 2D (D_out,C); got {cc.shape}")
            if str(model.get("cluster_centers_normalization", "none")).lower().strip() == "l2":
                from kahm_regression import l2_normalize_columns  # local import
                cc = l2_normalize_columns(cc)
            model["_cluster_centers_cache"] = cc


def _kahm_pred_from_topk(
    *,
    cluster_centers: np.ndarray,
    topk_dist: np.ndarray,
    topk_idx: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Compute Y_pred_col from stored top-k distances/indices (mirrors kahm_predict_with_min_distance)."""
    topk_dist = np.asarray(topk_dist, dtype=np.float32)
    topk_idx = np.asarray(topk_idx, dtype=np.int64)
    one_minus = np.clip(1.0 - topk_dist, 0.0, 1.0)
    S = np.power(one_minus, float(alpha), dtype=np.float32)
    denom = np.sum(S, axis=0, keepdims=True)
    zero = denom <= 0
    k = int(topk_dist.shape[0])
    if np.any(zero):
        denom = np.where(zero, 1.0, denom)
        S = np.where(zero, 1.0 / float(k), S)
    W = S / denom  # (k,N)

    D_out = int(cluster_centers.shape[0])
    N = int(topk_idx.shape[1])
    Y_pred = np.zeros((D_out, N), dtype=np.float32)
    for j in range(k):
        idx = topk_idx[j, :]
        Y_pred += cluster_centers[:, idx] * W[j, :][None, :]
    return Y_pred



@overload
def kahm_predict_with_min_distance(
    model: dict,
    X_new_col: np.ndarray,
    *,
    mode: str = "soft",
    alpha: Optional[float] = None,
    topk: Optional[int | None] = None,
    batch_size: int = 2048,
    show_progress: bool = True,
    return_pred: bool = True,
    return_topk_state: Literal[False] = False,
) -> Tuple[np.ndarray | None, np.ndarray]:
    ...

@overload
def kahm_predict_with_min_distance(
    model: dict,
    X_new_col: np.ndarray,
    *,
    mode: str = "soft",
    alpha: Optional[float] = None,
    topk: Optional[int | None] = None,
    batch_size: int = 2048,
    show_progress: bool = True,
    return_pred: bool = True,
    return_topk_state: Literal[True],
) -> Tuple[np.ndarray | None, np.ndarray, Dict[str, Any]]:
    ...

def kahm_predict_with_min_distance(
    model: dict,
    X_new_col: np.ndarray,
    *,
    mode: str = "soft",
    alpha: Optional[float] = None,
    topk: Optional[int | None] = None,
    batch_size: int = 2048,
    show_progress: bool = True,
    return_pred: bool = True,
    return_topk_state: bool = False,
) -> Tuple[np.ndarray | None, np.ndarray] | Tuple[np.ndarray | None, np.ndarray, dict]:
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

    # Cluster centers (cache the post-normalized float32 array to avoid repeating work)
    cluster_centers_raw = model.get("_cluster_centers_cache", None)
    if cluster_centers_raw is None:
        cc_raw = model.get("cluster_centers", None)
        if cc_raw is None:
            raise KeyError("Model missing 'cluster_centers'.")
        cc = np.asarray(cc_raw, dtype=np.float32)
        if cc.ndim != 2:
            raise ValueError(f"cluster_centers must be 2D (D_out,C); got {cc.shape}")
    
        # Optional normalization of centers (matches kahm_regress behavior)
        if str(model.get("cluster_centers_normalization", "none")).lower().strip() == "l2":
            cc = l2_normalize_columns(cc)
    
        model["_cluster_centers_cache"] = cc
        cluster_centers_raw = cc
    
    cluster_centers: np.ndarray = np.asarray(cluster_centers_raw, dtype=np.float32)
    if cluster_centers.ndim != 2:
        raise ValueError(f"_cluster_centers_cache must be 2D (D_out,C); got {cluster_centers.shape}")
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
        iterator = tqdm(iterator, total=len(AE_arr), desc="KAHM: distance eval", unit="cluster")

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
        if return_pred:
            Y_pred = cluster_centers[:, best_idx]  # (D_out, N)
            Y_out = np.asarray(Y_pred, dtype=np.float32)
        else:
            Y_out = None
        if return_topk_state:
            state = {"mode": "hard", "best_idx": np.asarray(best_idx, dtype=np.int64), "alpha": float(alpha_resolved), "topk": None}
            return Y_out, best_dist, state
        return Y_out, best_dist

    # SOFT MODE
    if topk_resolved is None:
        # Streaming full mixture would be very expensive; fall back to kahm_regress behavior by building full D.
        # For practical setups, keep topk_resolved not None.
        D = np.empty((C_eff, N_new), dtype=np.float32)
        # Re-run distances to build D (rare path). This is intentionally explicit.
        iterator = enumerate(AE_arr)
        if show_progress:
            iterator = tqdm(iterator, total=len(AE_arr), desc="KAHM: building full D", unit="cluster")
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
        if not return_pred:
            Y_out = None
        else:
            Y_out = np.asarray(Y_pred, dtype=np.float32)
        if return_topk_state:
            state = {"mode": "soft_full", "D": np.asarray(D, dtype=np.float32), "alpha": float(alpha_resolved), "topk": None}
            return Y_out, best_dist, state
        return Y_out, best_dist

    # Use kept top-k distances to compute weights (no need to store full D)
    k = int(topk_resolved)
    assert topk_dist is not None and topk_idx is not None

    # Ensure no missing indices (can happen only if k > C_eff)
    if k > C_eff:
        raise ValueError(f"topk={k} exceeds number of clusters C_eff={C_eff}.")

    # If the caller only needs distances (plus top-k state), skip the prediction construction.
    if (not return_pred) and return_topk_state:
        state = {"mode": "soft_topk", "topk_dist": np.asarray(topk_dist, dtype=np.float32), "topk_idx": np.asarray(topk_idx, dtype=np.int64), "alpha": float(alpha_resolved), "topk": int(topk_resolved)}
        return None, best_dist, state
    if (not return_pred) and (not return_topk_state):
        return None, best_dist

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

    if not return_pred:
        Y_out = None
    else:
        Y_out = Y_pred
    if return_topk_state:
        state = {"mode": "soft_topk", "topk_dist": np.asarray(topk_dist, dtype=np.float32), "topk_idx": np.asarray(topk_idx, dtype=np.int64), "alpha": float(alpha_resolved), "topk": int(topk_resolved)}
        return Y_out, best_dist, state
    return Y_out, best_dist



def combine_kahm_regressors_distance_gated_multi(
    X: np.ndarray,
    *,
    models: Union[Sequence[dict], Dict[str, dict]],
    model_names: Optional[Sequence[str]] = None,
    input_layout: str = "row",
    output_layout: str = "row",
    mode: str = "soft",
    alpha: Optional[float] = None,
    topk: Optional[int | None] = None,
    batch_size: int = 2048,
    tie_break: str = "first",
    show_progress: bool = True,
    return_all_scores: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], List[str]]:
    """
    Combine Q>=2 KAHM models via min-distance gating (cluster proximity).

    For each model m and input sample x, compute:
        score_m(x) = min_c d_{m,c}(x)
    and select the model with the smallest score.

    Args:
      X: input matrix, layout controlled by input_layout
      models: either a sequence of model dicts, or a dict {name: model}
      model_names: optional names aligned with `models` when `models` is a sequence
      tie_break:
        - 'first': choose the smallest model index on exact ties (default)
        - 'last' : choose the largest model index on exact ties
        - a model name (requires return_all_scores=True): prefer this model on exact ties

      return_all_scores: if True, returns per-model min-distances as (Q,N). If False,
                         returns None for that output.

    Returns:
      Y_pred: combined predictions (layout controlled by output_layout)
      chosen: int16 array shape (N,), chosen model index in [0, Q-1]
      best_score: float32 array shape (N,), min score across models per sample
      all_scores: optional float32 array shape (Q,N), per-model min score per sample
      names: list[str] length Q, model names in the same order as indices
    """
    # Canonicalize models + names
    if isinstance(models, dict):
        names = list(models.keys())
        model_list = [models[n] for n in names]
    else:
        model_list = list(models)
        if model_names is None:
            names = [f"model_{i}" for i in range(len(model_list))]
        else:
            names = list(model_names)

    Q = len(model_list)
    if Q < 2:
        raise ValueError(f"Need at least 2 models to combine; got Q={Q}.")
    if len(names) != Q:
        raise ValueError(f"model_names length {len(names)} does not match number of models Q={Q}.")

    # Normalize + validate tie_break
    tb = str(tie_break).strip()
    tb_lower = tb.lower()
    tb_mode: str
    prefer_name: Optional[str] = None
    if tb_lower in {"first", "min", "smallest"}:
        tb_mode = "first"
    elif tb_lower in {"last", "max", "largest"}:
        tb_mode = "last"
    else:
        tb_mode = "prefer"
        prefer_name = tb

    if tb_mode == "prefer" and not return_all_scores:
        raise ValueError("tie_break by model name requires return_all_scores=True.")

    X_col = _as_col_major(X, input_layout)

    # Allocate outputs
    N = int(X_col.shape[1])
    all_scores = np.empty((Q, N), dtype=np.float32) if return_all_scores else None

    Y_best_col: Optional[np.ndarray] = None
    best_score = np.full((N,), np.inf, dtype=np.float32)
    chosen = np.full((N,), -1, dtype=np.int16)

    # Evaluate models sequentially; update best per sample
    for i, model in enumerate(model_list):
        # Compute distances (and retain minimal state needed to build predictions only where required).
        Y_i_col, d_i, state = kahm_predict_with_min_distance(
            model,
            X_col,
            mode=mode,
            alpha=alpha,
            topk=topk,
            batch_size=batch_size,
            show_progress=show_progress,
            return_pred=False,
            return_topk_state=True,
        )
        d_i = np.asarray(d_i, dtype=np.float32).reshape(-1)
        if d_i.size != N:
            raise ValueError(f"Model {names[i]} produced distance shape {d_i.shape}; expected ({N},).")

        if all_scores is not None:
            all_scores[i, :] = d_i

        # Ensure cluster centers are available (kahm_predict_with_min_distance caches them on first use)
        centers_raw = model.get("_cluster_centers_cache", None)
        if centers_raw is None:
            cc_raw = model.get("cluster_centers", None)
            if cc_raw is None:
                raise KeyError("Model missing 'cluster_centers'.")
            cc = np.asarray(cc_raw, dtype=np.float32)
            if cc.ndim != 2:
                raise ValueError(f"cluster_centers must be 2D (D_out,C); got {cc.shape}")
            if str(model.get("cluster_centers_normalization", "none")).lower().strip() == "l2":
                from kahm_regression import l2_normalize_columns  # local import
                cc = l2_normalize_columns(cc)
            model["_cluster_centers_cache"] = cc
            centers_raw = cc
        centers: np.ndarray = np.asarray(centers_raw, dtype=np.float32)

        def _pred_full_from_state(st: dict) -> np.ndarray:
            m = str(st.get("mode", "")).lower().strip()
            if m == "hard":
                idx = np.asarray(st["best_idx"], dtype=np.int64)
                return centers[:, idx]
            if m == "soft_topk":
                return _kahm_pred_from_topk(
                    cluster_centers=centers,
                    topk_dist=np.asarray(st["topk_dist"], dtype=np.float32),
                    topk_idx=np.asarray(st["topk_idx"], dtype=np.int64),
                    alpha=float(st["alpha"]),
                )
            if m == "soft_full":
                from kahm_regression import distances_to_probabilities_one_minus_sharp  # local import
                D = np.asarray(st["D"], dtype=np.float32)
                P = distances_to_probabilities_one_minus_sharp(D, alpha=float(st["alpha"]), topk=None, inplace=True)
                return centers @ P
            raise ValueError(f"Unknown KAHM state mode: {st.get('mode')}")

        def _pred_subset_from_state(st: dict, cols: np.ndarray) -> np.ndarray:
            m = str(st.get("mode", "")).lower().strip()
            if m == "hard":
                idx = np.asarray(st["best_idx"], dtype=np.int64)[cols]
                return centers[:, idx]
            if m == "soft_topk":
                td = np.asarray(st["topk_dist"], dtype=np.float32)[:, cols]
                ti = np.asarray(st["topk_idx"], dtype=np.int64)[:, cols]
                return _kahm_pred_from_topk(cluster_centers=centers, topk_dist=td, topk_idx=ti, alpha=float(st["alpha"]))
            if m == "soft_full":
                from kahm_regression import distances_to_probabilities_one_minus_sharp  # local import
                D = np.asarray(st["D"], dtype=np.float32)[:, cols]
                P = distances_to_probabilities_one_minus_sharp(D, alpha=float(st["alpha"]), topk=None, inplace=True)
                return centers @ P
            raise ValueError(f"Unknown KAHM state mode: {st.get('mode')}")

        if Y_best_col is None:
            Y_best_col = np.asarray(_pred_full_from_state(state), dtype=np.float32)
            best_score[:] = d_i
            chosen[:] = np.int16(i)
        else:
            if int(centers.shape[0]) != int(Y_best_col.shape[0]):
                raise ValueError(
                    f"Output dim mismatch: model {names[i]} has D_out={centers.shape[0]}, "
                    f"but previous models have D_out={Y_best_col.shape[0]}."
                )

            # Update by strictly better (or also tie, depending on mode)
            if tb_mode == "last":
                better = d_i <= best_score
            else:
                better = d_i < best_score

            if np.any(better):
                cols = np.where(better)[0]
                best_score[cols] = d_i[cols]
                chosen[cols] = np.int16(i)
                Y_best_col[:, cols] = _pred_subset_from_state(state, cols)

    assert Y_best_col is not None

    # If we want to prefer a specific model name on exact ties, resolve after collecting all_scores.
    if tb_mode == "prefer" and prefer_name is not None:
        if prefer_name not in names:
            raise ValueError(f"tie_break='{prefer_name}' not among model names: {names}")
        p = names.index(prefer_name)
        assert all_scores is not None
        # For each sample, find min score and tie set, then prefer p if tied.
        minv = np.min(all_scores, axis=0)
        tied_with_prefer = all_scores[p, :] == minv
        # Only flip where current choice is also tied but not preferred and preferred is tied.
        # This preserves deterministic behavior while honoring preference.
        flip = tied_with_prefer & (chosen != p)
        if np.any(flip):
            cols = np.where(flip)[0]
            # Recompute Y for preferred model only on the flipped subset to avoid storing all Y_i.
            # This is the rare tie path; correctness > speed.
            # NOTE: this will re-run distance eval for the preferred model; acceptable because ties are rare.
            Y_p_col, _ = kahm_predict_with_min_distance(
                model_list[p],
                X_col[:, cols],
                mode=mode,
                alpha=alpha,
                topk=topk,
                batch_size=batch_size,
                show_progress=False,
                return_pred=True,
                return_topk_state=False,
            )
            if Y_p_col is None:
                raise RuntimeError("Unexpected: preferred-model prediction is None.")
            Y_best_col[:, cols] = Y_p_col
            chosen[cols] = np.int16(p)

    return _as_row_major(Y_best_col, output_layout), chosen, best_score, all_scores, names


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
    Backwards-compatible 2-model wrapper around combine_kahm_regressors_distance_gated_multi.

    Returns:
      Y_pred: combined predictions (layout controlled by output_layout)
      chosen: int8 array shape (N,) with 0=embedding_model, 1=query_model
      score_embedding: min-distance per sample for embedding_model
      score_query:     min-distance per sample for query_model
    """
    tb = str(tie_break).lower().strip()
    if tb not in {"query", "embedding"}:
        raise ValueError("tie_break must be 'query' or 'embedding'")

    # Map to multi-model tie_break semantics by naming the models.
    models = [embedding_model, query_model]
    names = ["embedding", "query"]
    preferred = "query" if tb == "query" else "embedding"

    Y, chosen_idx, _best, all_scores, _names = combine_kahm_regressors_distance_gated_multi(
        X,
        models=models,
        model_names=names,
        input_layout=input_layout,
        output_layout=output_layout,
        mode=mode,
        alpha=alpha,
        topk=topk,
        batch_size=batch_size,
        tie_break=preferred,  # prefer on ties, matching legacy behavior
        show_progress=show_progress,
        return_all_scores=True,
    )

    assert all_scores is not None and all_scores.shape[0] == 2
    # Legacy chosen type is int8 with 0/1
    chosen = np.asarray(chosen_idx, dtype=np.int8)
    d_emb = np.asarray(all_scores[0, :], dtype=np.float32)
    d_q = np.asarray(all_scores[1, :], dtype=np.float32)
    return Y, chosen, d_emb, d_q


# ----------------------------
# CLI
# ----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Combine Q KAHM regressors via min-distance gating (cluster proximity)."
    )

    # New multi-model interface (preferred)
    ap.add_argument(
        "--model",
        action="append",
        default=None,
        help=(
            "Model spec. Use 'name=path.joblib' or just 'path.joblib'. "
            "May be repeated; at least two models are required."
        ),
    )
    ap.add_argument(
        "--model_base_dir",
        action="append",
        default=None,
        help=(
            "Optional base_dir override for each --model (repeat to match count). "
            "If provided once, the same base_dir is applied to all models."
        ),
    )

    # Backwards-compatible 2-model interface
    ap.add_argument("--embedding_model", default=None, help="(Legacy) Path to embedding regressor joblib (Model A).")
    ap.add_argument("--query_model", default=None, help="(Legacy) Path to query regressor joblib (Model B).")
    ap.add_argument("--embedding_base_dir", default=None, help="(Legacy) Optional override for Model A classifier_dir.")
    ap.add_argument("--query_base_dir", default=None, help="(Legacy) Optional override for Model B classifier_dir.")

    ap.add_argument("--x", required=True, help="Input embeddings (.npy or .npz).")
    ap.add_argument("--x_key", default=None, help="If --x is .npz, the key to load (default: embeddings/X/first).")

    ap.add_argument(
        "--input_layout",
        default="auto",
        choices=["auto", "row", "col"],
        help="Layout of input X: row=(N,D), col=(D,N), auto=infer (default).",
    )
    ap.add_argument(
        "--output_layout",
        default="row",
        choices=["row", "col"],
        help="Layout of output embeddings: row=(N,D_out), col=(D_out,N) (default: row).",
    )

    ap.add_argument("--mode", default="soft", choices=["soft", "hard"], help="Prediction mode (default: soft).")
    ap.add_argument("--alpha", type=float, default=None, help="Soft alpha override (default: use model soft_alpha).")
    ap.add_argument("--topk", type=int, default=None, help="Soft topk override (default: use model soft_topk).")
    ap.add_argument("--batch_size", type=int, default=2048, help="Distance batch size (default: 2048).")

    ap.add_argument(
        "--tie_break",
        default="first",
        help=(
            "Tie-break policy when per-sample min distances are exactly equal. "
            "Use 'first' (default), 'last', or a model name provided via --model name=path."
        ),
    )

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

    # ----------------------------
    # Resolve model specs
    # ----------------------------
    model_specs: List[Tuple[str, str, Optional[str]]] = []

    if args.model:
        # New multi-model interface
        raw = list(args.model)
        base_dirs = list(args.model_base_dir) if args.model_base_dir else []
        if len(base_dirs) == 1 and len(raw) > 1:
            base_dirs = base_dirs * len(raw)
        if base_dirs and len(base_dirs) != len(raw):
            raise ValueError("--model_base_dir must be provided either once, or the same number of times as --model.")

        seen = set()
        for i, spec in enumerate(raw):
            spec = str(spec).strip()
            if "=" in spec:
                name, path = spec.split("=", 1)
                name = name.strip()
                path = path.strip()
            else:
                path = spec
                name = Path(path).stem

            if not name:
                raise ValueError(f"Invalid --model spec: {spec!r} (empty name).")
            # Ensure unique names
            base = name
            k = 2
            while name in seen:
                name = f"{base}_{k}"
                k += 1
            seen.add(name)

            bd = base_dirs[i] if base_dirs else None
            model_specs.append((name, path, bd))
    else:
        # Legacy interface (2-model)
        if not args.embedding_model or not args.query_model:
            raise ValueError("Provide either repeated --model entries (>=2) OR both --embedding_model and --query_model.")
        model_specs = [
            ("embedding", str(args.embedding_model), args.embedding_base_dir),
            ("query", str(args.query_model), args.query_base_dir),
        ]
        # Preserve legacy default tie-break (query) unless the user explicitly provided something else.
        if str(args.tie_break).strip().lower() == "first":
            args.tie_break = "query"

    if len(model_specs) < 2:
        raise ValueError(f"Need at least 2 models to combine; got {len(model_specs)}.")

    # Load models
    models_dict: Dict[str, dict] = {}
    for name, p, bd in model_specs:
        models_dict[name] = load_kahm_regressor(p, base_dir=bd)

    # Combine
    Y, chosen_idx, best_score, all_scores, names = combine_kahm_regressors_distance_gated_multi(
        X,
        models=models_dict,
        input_layout=input_layout,
        output_layout=args.output_layout,
        mode=args.mode,
        alpha=args.alpha,
        topk=args.topk,
        batch_size=int(args.batch_size),
        tie_break=args.tie_break,
        show_progress=(not args.no_progress),
        return_all_scores=True,
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
            save_dict = {
                "embeddings": np.asarray(Y, dtype=np.float32),
                "chosen_model_idx": np.asarray(chosen_idx, dtype=np.int16),
                "model_names": np.asarray(names, dtype=object),
                "min_dist_best": np.asarray(best_score, dtype=np.float32),
            }
            if all_scores is not None:
                save_dict["min_dist_all"] = np.asarray(all_scores, dtype=np.float32)

            np.savez_compressed(str(out_path), **save_dict)

        else:
            np.savez_compressed(str(out_path), embeddings=np.asarray(Y, dtype=np.float32))
    elif out_path.suffix.lower() == ".npy":
        np.save(str(out_path), np.asarray(Y, dtype=np.float32))
    else:
        raise ValueError("Output must be .npy or .npz")

    # Print brief diagnostics
    print(f"Saved combined embeddings to: {out_path}")
    chosen_idx_np = np.asarray(chosen_idx, dtype=np.int16)
    for i, n in enumerate(names):
        frac = float(np.mean(chosen_idx_np == i))
        print(f"Gating: {n} chosen for {frac*100:.1f}% of samples.")

    return 0


def demo_three_kahm_models() -> None:
    """
    End-to-end demo:
      1) Generate 3 synthetic data regimes
      2) Train 3 KAHM regressors (one per regime)
      3) Combine them with min-distance gating for prediction on a mixed test set

    Notes
    -----
    - This demo assumes the OTFL dependencies used by kahm_regression.py are available
      in your environment (parallel_autoencoders, combine_multiple_autoencoders_extended, etc.).
    - Shapes follow kahm_regression.py conventions:
        X: (D_in, N)  i.e., column-major samples
        Y: (D_out, N)
    """
    import numpy as _np

    # Prefer the user's patched module if present; fall back to kahm_regression.py

    from kahm_regression import train_kahm_regressor, kahm_regress, tune_soft_params  # type: ignore

    rng = _np.random.default_rng(0)

    # ----------------------------
    # Synthetic data regimes
    # ----------------------------
    D_in, D_out = 6, 3
    N_train_per_regime = 1000
    N_test_per_regime = 100
    noise = 0.10

    def _make_regime(regime_id: int, N: int) -> tuple[_np.ndarray, _np.ndarray]:
        r = _np.random.default_rng(100 + int(regime_id))
        if regime_id == 0:
            X = _np.tanh(r.normal(loc=0.0, scale=1.0, size=(D_in, N))).astype(_np.float32)
            Y = _np.vstack(
                [
                    2.0 * X[0, :] + 0.5 * X[1, :] ** 2,
                    -1.0 * X[2, :] + _np.sin(X[3, :]),
                    1.5 * X[4, :] + 0.3 * X[5, :],
                ]
            )
        elif regime_id == 1:
            X = _np.tanh(r.normal(loc=2.0, scale=1.0, size=(D_in, N))).astype(_np.float32)
            Y = _np.vstack(
                [
                    -1.6 * X[0, :] + 0.2 * X[1, :] + _np.cos(X[2, :]),
                    0.6 * X[3, :] ** 2 - 0.4 * X[4, :],
                    X[5, :] - 0.8 * X[2, :],
                ]
            )
        elif regime_id == 2:
            X = _np.tanh(r.normal(loc=-2.0, scale=1.0, size=(D_in, N))).astype(_np.float32)
            Y = _np.vstack(
                [
                    0.8 * X[0, :] ** 3 - 0.9 * X[1, :],
                    X[2, :] + _np.tanh(2.0 * X[3, :]),
                    0.5 * X[4, :] + _np.exp(-X[5, :] ** 2),
                ]
            )
        else:
            raise ValueError("regime_id must be 0, 1, or 2")

        Y = (Y + noise * r.normal(size=Y.shape)).astype(_np.float32)
        return X, Y
    
    

    # ----------------------------
    # Train three separate models
    # ----------------------------
    # Keep clusters moderate for a quick demo; increase for better fidelity.
    train_specs = [
        (0, "regime_A", 1000),
        (1, "regime_B", 1000),
        (2, "regime_C", 1000),
    ]
        # ----------------------------
    # Build a mixed test set (samples from all regimes)
    # ----------------------------
    X_parts: list[_np.ndarray] = []
    Y_parts: list[_np.ndarray] = []
    true_regime: list[_np.ndarray] = []

    for rid, name, _ in train_specs:
        Xt, Yt = _make_regime(rid, N_test_per_regime)
        X_parts.append(Xt)
        Y_parts.append(Yt)
        true_regime.append(_np.full((N_test_per_regime,), rid, dtype=_np.int16))

    X_test = _np.concatenate(X_parts, axis=1)
    Y_test = _np.concatenate(Y_parts, axis=1)
    true_regime_idx = _np.concatenate(true_regime, axis=0)

    # Shuffle columns so regimes are interleaved
    perm = rng.permutation(X_test.shape[1])
    X_test = X_test[:, perm]
    Y_test = Y_test[:, perm]
    true_regime_idx = true_regime_idx[perm]
    models: dict[str, dict] = {}

    print("\n=== Training 3 KAHM regressors (one per regime) ===")
    for rid, name, n_clusters in train_specs:
        Xtr, Ytr = _make_regime(rid, N_train_per_regime)
        print(f"\n--- Training {name} (regime_id={rid}, n_clusters={n_clusters}) ---")
        models[name] = train_kahm_regressor(
            Xtr,
            Ytr,
            n_clusters=int(n_clusters),
            subspace_dim=15,
            Nb=80,
            random_state=rid,
            verbose=True,
            input_scale=0.5,
            save_ae_to_disk=False,  # demo: keep everything in-memory
            cluster_center_normalization="none",
            
        )
        tune_res = tune_soft_params(
            models[name],
            X_test,
            Y_test,
            alphas=(2.0, 5.0, 8.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 18.0, 20.0),
            topks=(2, 5, 10, 11, 12, 13, 14, 15, 20, 25, 50, 100, 150),
            n_jobs=-1,
            verbose=True
            )



    # ----------------------------
    # Combine via min-distance gating
    # ----------------------------
    # Soft mode uses the same (1 - d)^alpha mapping as kahm_regression.py

    print("\n=== Combining models via min-distance gating ===")
    Y_comb, chosen_idx, best_score, _all_scores, names = combine_kahm_regressors_distance_gated_multi(
        X_test,
        models=models,
        input_layout="col",    # X is (D_in, N)
        output_layout="col",   # return (D_out, N)
        mode="soft",
        batch_size=1024,
        show_progress=True,
        return_all_scores=False,
    )

    def _mse(a: _np.ndarray, b: _np.ndarray) -> float:
        return float(_np.mean((a - b) ** 2))

    # Overall metrics
    mse_comb = _mse(Y_comb, Y_test)

    # Per-model baselines
    mse_per_model: dict[str, float] = {}
    for name in names:
        yhat = kahm_regress(models[name], X_test, mode="soft", batch_size=1024)
        mse_per_model[name] = _mse(yhat, Y_test)

    # Because the data is *generated* from 3 known regimes, we can also compute
    # how often the gating chooses the "correct" regime model.
    # Map model-name order to regime id order:
    name_to_rid = {"regime_A": 0, "regime_B": 1, "regime_C": 2}
    chosen_rid = _np.asarray([name_to_rid[names[int(i)]] for i in chosen_idx], dtype=_np.int16)
    gate_acc = float(_np.mean(chosen_rid == true_regime_idx))

    # Chosen counts
    counts = _np.bincount(chosen_idx.astype(int), minlength=len(names))

    print("\n=== Results (toy demo) ===")
    print(f"Test set: N={X_test.shape[1]} samples | D_in={D_in} | D_out={D_out}")
    print(f"Combined MSE: {mse_comb:.6f}")
    print("Single-model MSEs:")
    for name in names:
        print(f"  {name}: {mse_per_model[name]:.6f}")
    print("Chosen model counts:")
    for i, name in enumerate(names):
        print(f"  {name}: {int(counts[i])}")
    print(f"Gating accuracy vs true regime (toy): {gate_acc:.3%}")

    print("\nDone. In real use, train each model on its own data regime and call:")
    print("  combine_kahm_regressors_distance_gated_multi(X, models={...}, input_layout='col', output_layout='col', ...)")


if __name__ == "__main__":
    import sys as _sys
    # If run without arguments (or with --demo), execute the self-contained demo.
    if len(_sys.argv) == 1 or (len(_sys.argv) == 2 and _sys.argv[1] == "--demo"):
        demo_three_kahm_models()
    else:
        raise SystemExit(main())
