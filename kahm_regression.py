"""
kahm_regression.py

KAHM-based multivariate regression via output clustering.

Soft / "true" regression (distance-matrix based):
- combine_multiple_autoencoders_extended is used with one autoencoder per output cluster
  to obtain per-cluster distances D of shape (C_eff, N_new).
- Distances are assumed to lie in [0, 1] and be monotone with (1 - probability).

Sharpened + truncated probability mapping (recommended when C_eff is large):
    S = (1 - D) ** alpha
    keep only top-k scores per sample (optional)
    P = S / sum_c S

Prediction:
    Y_hat = cluster_centers @ P

Autotuning support:
- tune_soft_params(...) evaluates a grid over (alpha, topk) on a validation set,
  stores best values in the model dict as:
      model['soft_alpha'], model['soft_topk']

Inference defaults:
- kahm_regress(..., mode='soft', alpha=None, topk=None) will automatically use
  stored model values if present, otherwise falls back to alpha=10, topk=10.

Requires:
    pip install scikit-learn numpy joblib
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, Literal, overload

import numpy as np
from numpy.typing import DTypeLike
from sklearn.cluster import KMeans, MiniBatchKMeans
from joblib import dump, load

from parallel_autoencoders import parallel_autoencoders
from combine_multiple_autoencoders_extended import combine_multiple_autoencoders_extended


# ----------------------------
# Precision helpers
# ----------------------------

def _as_float_ndarray(x: Any, *, min_dtype: DTypeLike = np.float32) -> np.ndarray:
    """Convert input to a floating ndarray without downcasting precision.

    - Integer/bool inputs are promoted to float64.
    - Floating inputs keep their existing precision unless below `min_dtype`,
      in which case they are promoted to `min_dtype`.
    """
    arr = np.asarray(x)
    if arr.dtype.kind not in "fc":
        # Promote integers/bools to a sensible floating default.
        arr = arr.astype(np.float64, copy=False)

    dtype = np.result_type(arr.dtype, min_dtype)
    return arr.astype(dtype, copy=False)

def _scale_like(X: np.ndarray, scale: float, *, inplace: bool = False) -> np.ndarray:
    """Scale X by `scale` without unintentionally upcasting dtype.

    - If X is float32 and scale is a Python float, `X * scale` would upcast to float64.
      This helper ensures the scale is cast to X.dtype first.
    - If `inplace=True` and X is writeable, scaling is performed in-place to avoid a full copy.
    """
    if scale == 1.0:
        return X
    if X.dtype.kind not in 'fc':
        X = X.astype(np.float32, copy=False)
    s = np.asarray(scale, dtype=X.dtype)
    if inplace and X.flags.writeable:
        np.multiply(X, s, out=X, casting='unsafe')
        return X
    # out-of-place but dtype-stable
    return (X * s).astype(X.dtype, copy=False)



def _ae_as_list(ae: Any) -> list:
    """Ensure an autoencoder spec is passed as a flat list.

    The OTFL helper `combine_multiple_autoencoders_extended` expects a sequence of
    autoencoder components. In some training pipelines each cluster stores:
      - a list/tuple of components, or
      - a single component (dict/ndarray-like).

    This normalizes both cases and also flattens a common accidental nesting
    pattern: [ [component, ...] ].
    """
    if isinstance(ae, (list, tuple)):
        if len(ae) == 1 and isinstance(ae[0], (list, tuple)):
            return list(ae[0])
        return list(ae)
    return [ae]

def l2_normalize_columns(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize columns of a 2D matrix (D, N). Safe for non-directional data when not used."""
    M = _as_float_ndarray(M)
    if M.ndim != 2:
        raise ValueError(f"Expected 2D matrix for l2_normalize_columns; got shape={M.shape}")
    norms = np.linalg.norm(M, axis=0, keepdims=True)
    norms = np.maximum(norms, eps).astype(M.dtype, copy=False)
    return M / norms


def _should_auto_l2_normalize_targets(Y: np.ndarray) -> bool:
    """Heuristic: decide if Y columns are approximately unit-norm (directional data)."""
    Y = _as_float_ndarray(Y)
    if Y.ndim != 2 or Y.shape[1] == 0:
        return False
    norms = np.linalg.norm(Y, axis=0)
    # Robust statistics
    p10, p50, p90 = np.percentile(norms, [10, 50, 90]).tolist()
    # Treat as directional if most norms are near 1
    return (0.90 <= p50 <= 1.10) and (p10 >= 0.80) and (p90 <= 1.20)


# ----------------------------
# Training helpers
# ----------------------------

def train_kahm_regressor(
    X: np.ndarray,
    Y: np.ndarray,
    n_clusters: int,
    subspace_dim: int = 20,
    Nb: int = 100,
    random_state: int | None = 0,
    verbose: bool = True,
    input_scale: float = 1.0,
    # ----------------
    # Scalability controls
    # ----------------
    # For large n_clusters, scikit-learn's full KMeans can be both slow and memory hungry.
    # Use MiniBatchKMeans to reduce peak memory and make training practical.
    kmeans_kind: str = "auto",  # {'auto','full','minibatch'}
    kmeans_batch_size: int = 4096,
    # Limit the number of training samples used by the KAHM classifier per output cluster.
    # This directly limits the size of the stored classifier model and is the highest-ROI
    # lever when C is large.
    max_train_per_cluster: int | None = None,
    # Downcast arrays inside the trained classifier to reduce RAM.
    # (Does not change the external model API.)
    model_dtype: str = "auto",
    # Optional: normalize output cluster centroids (useful for directional/unit-norm embedding targets)
    #   - "none": do nothing (default; fully general)
    #   - "l2": always L2-normalize centroids
    #   - "auto_l2": normalize only if Y columns appear approximately unit-norm
    cluster_center_normalization: str = "none",
) -> dict:
    """Train a KAHM-based regressor via output clustering."""
    X = _as_float_ndarray(X)
    Y = _as_float_ndarray(Y)

    # Choose a shared working dtype.
    # - If model_dtype is "auto"/"none": preserve input precision (up to float32 minimum from _as_float_ndarray).
    # - If model_dtype is "float32"/"float64": use that dtype during training to reduce peak RAM.
    md0 = str(model_dtype).lower().strip()
    if md0 in ("auto", "none", ""):
        work_dtype = np.result_type(X.dtype, Y.dtype)
    elif md0 in ("float32", "f32"):
        work_dtype = np.float32
    elif md0 in ("float64", "f64"):
        work_dtype = np.float64
    else:
        raise ValueError("model_dtype must be one of {'auto','float32','float64'}")

    X = X.astype(work_dtype, copy=False)
    Y = Y.astype(work_dtype, copy=False)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D arrays shaped (D, N).")

    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"X and Y must have the same number of samples (columns). "
            f"Got X.shape={X.shape}, Y.shape={Y.shape}."
        )

    D_in, N = X.shape
    D_out, _ = Y.shape

    if N < 2 * n_clusters:
        raise ValueError(
            f"Not enough samples ({N}) to ensure at least 2 samples per cluster "
            f"for n_clusters={n_clusters}. Need N >= 2 * n_clusters."
        )

    if verbose:
        print(f"Training KAHM regressor on {N} samples.")
        print(f"Input dim:  {D_in}, Output dim: {D_out}")
        print(f"Requested number of clusters: {n_clusters}")
        print(f"Input scaling factor (input_scale): {input_scale}")

    # Apply input scaling (for AE-based cluster assignment)
    if input_scale != 1.0:
        X = _scale_like(X, float(input_scale), inplace=False)

    # 1) K-means on outputs
    if verbose:
        print("Running K-means on outputs...")

    Y_T = Y.T  # (N, D_out)

    kind = str(kmeans_kind).lower().strip()
    if kind not in ("auto", "full", "minibatch"):
        raise ValueError("kmeans_kind must be one of {'auto','full','minibatch'}")

    use_minibatch = (kind == "minibatch") or (kind == "auto" and int(n_clusters) >= 2000)

    if use_minibatch:
        if verbose:
            print(
                f"Using MiniBatchKMeans (n_clusters={n_clusters}, batch_size={int(kmeans_batch_size)}) "
                "to reduce peak memory."
            )
        kmeans = MiniBatchKMeans(
            n_clusters=int(n_clusters),
            random_state=random_state,
            batch_size=int(kmeans_batch_size),
            n_init="auto",
            reassignment_ratio=0.01,
        )
        kmeans.fit(Y_T)
    else:
        kmeans = KMeans(n_clusters=int(n_clusters), random_state=random_state, n_init="auto")
        kmeans.fit(Y_T)
    labels_zero_based = kmeans.labels_.astype(int)

    # 1a) Merge singleton clusters into nearest non-singleton cluster
    counts = np.bincount(labels_zero_based, minlength=n_clusters)
    singletons = np.where(counts == 1)[0]

    if singletons.size > 0 and verbose:
        print(f"Merging {singletons.size} singleton cluster(s)...")

    if singletons.size > 0:
        centers = kmeans.cluster_centers_
        # Precompute squared norms of centers once (avoids large temporaries during singleton merging).
        c2 = np.einsum("ij,ij->i", centers, centers)
        for cl in singletons:
            sample_indices = np.where(labels_zero_based == cl)[0]
            if sample_indices.size != 1:
                continue
            s_idx = sample_indices[0]
            y_sample = Y_T[s_idx]

            # Squared distance: ||c - y||^2 = ||c||^2 + ||y||^2 - 2 c·y
            y2 = float(np.dot(y_sample, y_sample))
            d2 = c2 + y2 - 2.0 * centers.dot(y_sample)

            candidates = np.where(counts >= 2)[0]
            if candidates.size == 0:
                candidates = np.where(counts >= 1)[0]

            candidates = candidates[candidates != cl]
            if candidates.size == 0:
                continue

            target = candidates[np.argmin(d2[candidates])]
            labels_zero_based[s_idx] = target
            counts[target] += 1
            counts[cl] -= 1


    # Free the transposed matrix used for KMeans to reduce peak RAM.
    try:
        del Y_T
    except Exception:
        pass
    # 1c) Drop empty clusters, remap to contiguous labels
    used_clusters = np.unique(labels_zero_based)
    n_clusters_eff = used_clusters.size

    if verbose:
        print(f"Effective number of clusters after preprocessing: {n_clusters_eff}")

        # Vectorized remap (avoids Python-object overhead for large N)
    map_arr = np.full(int(n_clusters), -1, dtype=np.int32)
    map_arr[used_clusters.astype(np.int64)] = np.arange(n_clusters_eff, dtype=np.int32)
    labels_mapped_zero = map_arr[labels_zero_based.astype(np.int64)]
    if np.any(labels_mapped_zero < 0):
        raise RuntimeError("Internal error while remapping cluster labels (found unmapped label).")
    labels_one_based = labels_mapped_zero + 1  # OTFL expects labels 1..C_eff
  # OTFL expects labels 1..C_eff

    cluster_centers = np.zeros((D_out, n_clusters_eff), dtype=work_dtype)
    for new_c in range(n_clusters_eff):
        mask = labels_mapped_zero == new_c
        cluster_centers[:, new_c] = Y[:, mask].mean(axis=1)

    # Optional centroid normalization for directional targets (e.g., unit-norm embeddings)
    cc_req = str(cluster_center_normalization).lower().strip()
    cc_applied = "none"
    if cc_req in ("l2", "auto_l2"):
        do_norm = (cc_req == "l2") or _should_auto_l2_normalize_targets(Y)
        if do_norm:
            cluster_centers = l2_normalize_columns(cluster_centers)
            cc_applied = "l2"
    print(f"cluster center normaization: {cc_applied}")

    final_counts = np.bincount(labels_mapped_zero, minlength=n_clusters_eff)
    if verbose:
        print(f"Min cluster size after preprocessing: {final_counts.min()}")

    # 2) Optionally subsample per output cluster for autoencoder training
    #    This is the most effective way to control memory when n_clusters is large.
    X_clf = X
    labels_one_based_clf = labels_one_based

    if max_train_per_cluster is not None:
        m = int(max_train_per_cluster)
        if m <= 0:
            raise ValueError("max_train_per_cluster must be a positive integer or None")
        if verbose:
            print(f"Subsampling autoencoder training data: max_train_per_cluster={m}")

        rng = np.random.RandomState(int(random_state) if random_state is not None else 0)
        keep_idx_parts: list[np.ndarray] = []
        for c in range(n_clusters_eff):
            idx = np.where(labels_mapped_zero == c)[0]
            if idx.size <= m:
                keep_idx_parts.append(idx)
            else:
                keep_idx_parts.append(rng.choice(idx, size=m, replace=False))

        keep_idx = np.concatenate(keep_idx_parts).astype(np.int64, copy=False)
        keep_idx.sort()
        X_clf = X[:, keep_idx]
        labels_one_based_clf = labels_one_based[keep_idx]
        if verbose:
            print(f"Autoencoder training samples: {X_clf.shape[1]} (was {X.shape[1]})")

    # 3) Train per-cluster autoencoders

    # Y is no longer needed after computing cluster_centers; free it before classifier training.
    try:
        del Y
    except Exception:
        pass
    import gc as _gc
    _gc.collect()

    if verbose:
        print("Training per-cluster autoencoders (parallel_autoencoders)...")

    # Ensure float32 to minimize peak memory inside OTFL autoencoder code
    if X_clf.dtype != np.float32:
        X_clf = X_clf.astype(np.float32, copy=False)
    labels_one_based_clf = labels_one_based_clf.astype(np.int32, copy=False)

    # Train one autoencoder per output cluster.
    # We call `parallel_autoencoders` on each cluster with Nb == N_cluster so it returns a single AE.
    AE_arr: list[Any] = []
    for c in range(n_clusters_eff):
        idx_c = np.where(labels_one_based_clf == (c + 1))[0]
        if idx_c.size < 2:
            raise ValueError(
                f"Cluster {c + 1} has only {idx_c.size} sample(s) after subsampling; "
                "need at least 2 to train an autoencoder. "
                "Increase max_train_per_cluster or reduce n_clusters."
            )

        X_c = X_clf[:, idx_c]
        # Ensure subspace_dim is feasible for this cluster.
        print(f" Cluster {c + 1}/{n_clusters_eff}: training autoencoder on {X_c.shape[1]} samples ...")
        AE_c_list = parallel_autoencoders(
            X_c,
            subspace_dim=subspace_dim,
            Nb=Nb,
            n_jobs=1,
            verbose=False,
        )
        AE_arr.append(AE_c_list)

    # For backward compatibility with the rest of this file, we keep the name `clf`.
    # In the per-cluster-AE variant, `clf` is a list of length C_eff (one autoencoder per cluster).
    clf = AE_arr

    # 4) Downcast model arrays to reduce RAM footprint
    #    NOTE: We keep this conservative: float32 for all numeric arrays.
    #    If you need even more savings, you can change PC to float16,
    #    but that may affect numerical stability in some OTFL implementations.
    md = str(model_dtype).lower().strip()
    if md in ("auto", "none", ""):
        _dtype = work_dtype
    elif md in ("float32", "f32"):
        _dtype = np.float32
    elif md in ("float64", "f64"):
        _dtype = np.float64
    else:
        raise ValueError("model_dtype must be one of {'auto','float32','float64'}")

    def _downcast_obj(obj):
        if isinstance(obj, np.ndarray) and obj.dtype.kind in "fc":
            return obj.astype(_dtype, copy=False)
        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                obj[k] = _downcast_obj(v)
            return obj
        if isinstance(obj, (list, tuple)):
            out = [ _downcast_obj(v) for v in obj ]
            return out if isinstance(obj, list) else tuple(out)
        return obj

    try:
        clf = _downcast_obj(clf)
    except Exception:
        # If OTFL returns unexpected structures, keep the model unmodified.
        pass

    if verbose:
        print("Training finished.")

    return {
        "classifier": clf,
        "cluster_centers": np.asarray(cluster_centers, dtype=_dtype),  # (D_out, C_eff)
        "n_clusters": int(n_clusters_eff),
        "input_scale": float(input_scale),
        # soft params (optional; filled by tune_soft_params)
        "soft_alpha": None,
        "soft_topk": None,
        "cluster_centers_normalization_requested": cc_req,
        "cluster_centers_normalization": cc_applied,
    }


# ----------------------------
# Soft probability mapping
# ----------------------------

def _topk_truncate_inplace(S: np.ndarray, k: int, *, chunk_cols: int = 64) -> None:
    """Zero all but the top-k entries per column of S in-place (memory-aware).

    This helper is designed to avoid allocating an index matrix of shape (C, N),
    which can dominate peak RAM when C and N are large.

    Implementation detail:
      - processes columns in chunks, limiting temporary index arrays to (C, chunk_cols)
    """
    if S.ndim != 2:
        raise ValueError("S must be 2D shaped (C, N).")
    C, N = S.shape
    k = int(k)
    if k <= 0 or k >= C or N == 0:
        return

    # Cap temporary index memory (~32 MiB by default).
    max_index_bytes = 32 * 1024 * 1024
    max_chunk = max(1, int(max_index_bytes // (8 * max(1, C))))
    chunk_cols = max(1, min(int(chunk_cols), max_chunk))

    for start in range(0, N, chunk_cols):
        end = min(N, start + chunk_cols)
        sub = S[:, start:end]  # (C, B)
        # argpartition returns full indices for the submatrix, but B is bounded.
        idx = np.argpartition(sub, -k, axis=0)
        idx_top = idx[-k:, :]  # (k, B)
        vals_top = np.take_along_axis(sub, idx_top, axis=0).copy()
        sub.fill(S.dtype.type(0.0))
        np.put_along_axis(sub, idx_top, vals_top, axis=0)


def distances_to_probabilities_one_minus_sharp(
    distance_matrix: np.ndarray,
    *,
    alpha: float = 10.0,
    topk: int | None = 10,
    eps: float = 1e-12,
    inplace: bool = False,
) -> np.ndarray:
    """
    Convert distances D (C, N) into probabilities P (C, N):

        S = (1 - D) ** alpha
        (optional) keep only top-k scores per column
        P = S / sum_c S

    Notes on memory
    ---------------
    - This implementation avoids creating an additional full-size `P` matrix.
      The returned array *is* the score/probability buffer.
    - If `inplace=True`, the input matrix is overwritten (or copied if not writeable).

    Assumes D in [0,1]. Uses uniform fallback if sum_c S == 0 for a sample.
    """
    D = _as_float_ndarray(distance_matrix)
    if D.ndim != 2:
        raise ValueError("distance_matrix must be 2D shaped (C, N).")

    C, N = D.shape
    dtype = D.dtype
    one = dtype.type(1.0)
    zero = dtype.type(0.0)
    eps_t = dtype.type(eps)

    if inplace:
        S = D if D.flags.writeable else D.copy()
        np.subtract(one, S, out=S)
    else:
        # Allocate a single full-size buffer for scores/probabilities.
        S = np.subtract(one, D)

    # Clamp to [0, 1] to avoid negative scores if D has minor numerical drift.
    np.clip(S, zero, one, out=S)

    a = float(alpha)
    if a != 1.0:
        np.power(S, dtype.type(a), out=S)

    if topk is not None:
        k = int(topk)
        if 0 < k < C:
            _topk_truncate_inplace(S, k)

    denom = S.sum(axis=0, dtype=dtype)  # (N,)
    zero_cols = denom <= eps_t
    if np.any(zero_cols):
        denom = denom.copy()
        denom[zero_cols] = one

    # Normalize in-place (broadcast along rows).
    np.divide(S, denom, out=S)

    if np.any(zero_cols):
        S[:, zero_cols] = dtype.type(1.0 / C)

    return S

def _ensure_distance_matrix_shape(
    D: np.ndarray,
    C_eff: int,
    N: int,
    labels: Optional[np.ndarray] = None,
    *,
    square_sample: int = 512,
) -> np.ndarray:
    """Ensure distance matrix D is shaped (C_eff, N).

    For non-square shapes this is unambiguous:
      - (C_eff, N): returned as-is
      - (N, C_eff): transposed

    For the corner case C_eff == N (square matrix), shape alone is ambiguous.
    If `labels` (length N) are provided (as returned by OTFL), we disambiguate by
    choosing the orientation whose argmin assignments best match the labels.
    """
    if D.ndim != 2:
        raise ValueError(f"distance_matrix must be 2D; got shape {D.shape}.")

    # Unambiguous cases
    if D.shape == (C_eff, N) and (C_eff != N or D.shape != (N, C_eff)):
        return D
    if D.shape == (N, C_eff) and (C_eff != N or D.shape != (C_eff, N)):
        return D.T

    # If we reach here, shapes are either square (C_eff == N) or otherwise mismatched.
    if D.shape != (C_eff, N) and D.shape != (N, C_eff):
        raise ValueError(
            f"distance_matrix shape mismatch. Expected {(C_eff, N)} or {(N, C_eff)}, got {D.shape}."
        )

    # Square ambiguity: C_eff == N and D is (N, N)
    if C_eff != N:
        # Should not happen due to checks above, but keep safe.
        return D if D.shape == (C_eff, N) else D.T

    if labels is None:
        # Cannot disambiguate; keep backward-compatible behavior:
        # treat D as already (C, N). (Downstream code assumes axis-0 is clusters.)
        return D

    lab = np.asarray(labels).reshape(-1)
    if lab.size != N:
        raise ValueError(f"labels length mismatch for square distance matrix: labels={lab.size}, N={N}")

    # Sample indices to keep this cheap for very large N
    if square_sample is not None and int(square_sample) > 0 and int(square_sample) < N:
        idx = np.linspace(0, N - 1, int(square_sample), dtype=np.int64)
    else:
        idx = np.arange(N, dtype=np.int64)

    # Hypothesis A: D is (C, N)
    pred_cn = np.argmin(D[:, idx], axis=0).astype(np.int64)
    acc_cn = float((pred_cn == lab[idx].astype(np.int64)).mean())

    # Hypothesis B: D is (N, C) (so D.T is (C, N))
    pred_nc = np.argmin(D[idx, :], axis=1).astype(np.int64)
    acc_nc = float((pred_nc == lab[idx].astype(np.int64)).mean())

    return D if acc_cn >= acc_nc else D.T


def _get_soft_params_from_model(
    model: dict,
    alpha: Optional[float],
    topk: Optional[int | None],
) -> Tuple[float, int | None]:
    """Resolve soft parameters: prefer explicit args, else model values, else defaults."""
    if alpha is None:
        alpha = model.get("soft_alpha", None)
    if topk is None:
        topk = model.get("soft_topk", None)

    alpha_resolved = float(alpha) if alpha is not None else 10.0
    topk_resolved = topk if topk is not None else 10
    return alpha_resolved, topk_resolved


# ----------------------------
# Prediction
# ----------------------------

@overload
def kahm_regress(
    model: dict,
    X_new: np.ndarray,
    n_jobs: int = -1,
    *,
    mode: str = "hard",
    return_probabilities: Literal[False] = False,
    # Soft-mode parameters (None => use model['soft_*'] if set)
    alpha: Optional[float] = None,
    topk: Optional[int | None] = None,
    # For large N_new, compute in batches to control memory usage
    batch_size: Optional[int] = None,
) -> np.ndarray: ...

@overload
def kahm_regress(
    model: dict,
    X_new: np.ndarray,
    n_jobs: int = -1,
    *,
    mode: str = "hard",
    return_probabilities: Literal[True],
    # Soft-mode parameters (None => use model['soft_*'] if set)
    alpha: Optional[float] = None,
    topk: Optional[int | None] = None,
    # For large N_new, compute in batches to control memory usage
    batch_size: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]: ...

def kahm_regress(
    model: dict,
    X_new: np.ndarray,
    n_jobs: int = -1,
    *,
    mode: str = "hard",
    return_probabilities: bool = False,
    # Soft-mode parameters (None => use model['soft_*'] if set)
    alpha: Optional[float] = None,
    topk: Optional[int | None] = None,
    # For large N_new, compute in batches to control memory usage
    batch_size: Optional[int] = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Predict outputs for new inputs using a trained KAHM regressor.

    This variant assumes the trained model stores **one autoencoder per output cluster**
    in `model['classifier']` (kept for backward compatibility with the earlier API).

    - Hard mode:
        * compute distance-to-cluster for each cluster autoencoder (reconstruction distance)
        * predict cluster label by argmin distance
        * output the corresponding cluster center

    - Soft mode:
        * build a distance matrix D of shape (C_eff, N_new)
        * convert to probabilities with `distances_to_probabilities_one_minus_sharp`
        * return Y_hat = cluster_centers @ P

    Distance computation uses:
        combine_multiple_autoencoders_extended(X, AE_c_list, distance_type="folding")
        where AE_c_list is a list/tuple of autoencoder components (or a single autoencoder wrapped in a list).
    """
    X_new = _as_float_ndarray(X_new)
    if X_new.ndim != 2:
        raise ValueError("X_new must be 2D shaped (D_in, N_new).")

    input_scale = float(model.get("input_scale", 1.0))
    if input_scale != 1.0:
        X_new = _scale_like(X_new, float(input_scale), inplace=False)

    AE_arr = model["classifier"]
    if not isinstance(AE_arr, (list, tuple)) or len(AE_arr) == 0:
        raise TypeError(
            "This kahm_regression variant expects model['classifier'] to be a non-empty list of "
            "per-cluster autoencoders (trained with parallel_autoencoders on each cluster)."
        )

    cluster_centers = _as_float_ndarray(model["cluster_centers"])  # (D_out, C_eff)
    # If the trained model indicates centroid normalization, enforce it here (idempotent).
    if str(model.get("cluster_centers_normalization", "none")).lower().strip() == "l2":
        cluster_centers = l2_normalize_columns(cluster_centers)

    C_eff = int(cluster_centers.shape[1])
    if len(AE_arr) != C_eff:
        raise ValueError(
            f"Mismatch: got {len(AE_arr)} autoencoders but cluster_centers has C_eff={C_eff} clusters."
        )

    N_new = int(X_new.shape[1])
    out_dtype = np.result_type(cluster_centers.dtype, np.float32)

    distance_type = "folding"

    if mode == "hard":
        # Streaming argmin over clusters (does not materialize a full (C_eff, N_new) matrix).
        best_dist = np.full((N_new,), np.inf, dtype=np.float64)
        best_idx = np.zeros((N_new,), dtype=np.int64)

        for c, AE_c in enumerate(AE_arr):
            d = combine_multiple_autoencoders_extended(X_new, _ae_as_list(AE_c), distance_type)
            d = np.asarray(d).reshape(-1)
            if d.size != N_new:
                raise ValueError(
                    f"Distance vector from cluster {c + 1} has shape {d.shape}; expected ({N_new},)."
                )
            mask = d < best_dist
            best_dist[mask] = d[mask]
            best_idx[mask] = c

        Y_pred = cluster_centers[:, best_idx]

        if return_probabilities:
            # One-hot probabilities for hard assignments.
            P_hard = np.zeros((C_eff, N_new), dtype=out_dtype)
            P_hard[best_idx, np.arange(N_new)] = 1.0
            return Y_pred, P_hard

        return Y_pred

    if mode != "soft":
        raise ValueError("mode must be either 'hard' or 'soft'.")

    alpha_resolved, topk_resolved = _get_soft_params_from_model(model, alpha, topk)

    # Batching for memory safety
    if batch_size is not None and int(batch_size) > 0:
        bs = int(batch_size)
        Y_pred = np.empty((cluster_centers.shape[0], N_new), dtype=out_dtype)

        for start in range(0, N_new, bs):
            end = min(start + bs, N_new)
            X_batch = X_new[:, start:end]
            N_b = int(X_batch.shape[1])

            # Build distance matrix for the batch: (C_eff, N_b)
            D_batch = np.empty((C_eff, N_b), dtype=np.float64)
            for c, AE_c in enumerate(AE_arr):
                d = combine_multiple_autoencoders_extended(X_batch, _ae_as_list(AE_c), distance_type)
                d = np.asarray(d).reshape(-1)
                if d.size != N_b:
                    raise ValueError(
                        f"Distance vector from cluster {c + 1} has shape {d.shape}; expected ({N_b},)."
                    )
                D_batch[c, :] = d

            D_batch = _ensure_distance_matrix_shape(_as_float_ndarray(D_batch), C_eff, N_b, labels=None)
            P_batch = distances_to_probabilities_one_minus_sharp(D_batch, alpha=alpha_resolved, topk=topk_resolved, inplace=True)
            Y_pred[:, start:end] = cluster_centers @ P_batch

        if return_probabilities:
            raise ValueError("return_probabilities=True is not supported when batch_size is set.")
        return Y_pred

    # Non-batched (returns full P if requested)
    D = np.empty((C_eff, N_new), dtype=np.float64)
    for c, AE_c in enumerate(AE_arr):
        d = combine_multiple_autoencoders_extended(X_new, _ae_as_list(AE_c), distance_type)
        d = np.asarray(d).reshape(-1)
        if d.size != N_new:
            raise ValueError(
                f"Distance vector from cluster {c + 1} has shape {d.shape}; expected ({N_new},)."
            )
        D[c, :] = d

    D = _ensure_distance_matrix_shape(_as_float_ndarray(D), C_eff, N_new, labels=None)
    P = distances_to_probabilities_one_minus_sharp(D, alpha=alpha_resolved, topk=topk_resolved, inplace=True)
    Y_pred = cluster_centers @ P

    if return_probabilities:
        return Y_pred, P
    return Y_pred

# ----------------------------
# Autotuning
# ----------------------------

@dataclass(frozen=True)
class SoftTuningResult:
    best_alpha: float
    best_topk: int | None
    best_mse: float


def tune_soft_params(
    model: dict,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    *,
    alphas: Sequence[float] = (5.0, 10.0, 15.0, 20.0),
    topks: Sequence[int | None] = (5, 10, 15, 20),
    n_jobs: int = -1,
    verbose: bool = True,
) -> SoftTuningResult:
    """
    Tune (alpha, topk) on a validation set and store the best choice in `model`.

    Notes
    -----
    - Computes the full distance matrix once for X_val and reuses it for all grid points.
    - Objective: minimize mean squared error (MSE) over all output dimensions and samples.
    """
    X_val = _as_float_ndarray(X_val)
    Y_val = _as_float_ndarray(Y_val)

    if X_val.ndim != 2 or Y_val.ndim != 2:
        raise ValueError("X_val and Y_val must be 2D shaped (D, N).")

    if X_val.shape[1] != Y_val.shape[1]:
        raise ValueError("X_val and Y_val must have the same number of samples (columns).")

    # Apply same scaling used during training
    input_scale = float(model.get("input_scale", 1.0))
    Xv = _scale_like(X_val, float(input_scale), inplace=False) if input_scale != 1.0 else X_val

    clf = model["classifier"]
    cluster_centers = _as_float_ndarray(model["cluster_centers"])
    C_eff = cluster_centers.shape[1]
    N_val = Xv.shape[1]

    # This variant expects per-cluster autoencoders in model['classifier'].
    AE_arr = clf
    if not isinstance(AE_arr, (list, tuple)) or len(AE_arr) == 0:
        raise TypeError(
            "This kahm_regression variant expects model['classifier'] to be a non-empty list of "
            "per-cluster autoencoders."
        )
    if len(AE_arr) != C_eff:
        raise ValueError(
            f"Mismatch: got {len(AE_arr)} autoencoders but cluster_centers has C_eff={C_eff} clusters."
        )

    # Compute full distance matrix once for X_val and reuse it for all grid points.
    D_val = np.empty((C_eff, N_val), dtype=np.float64)
    for c, AE_c in enumerate(AE_arr):
        d = combine_multiple_autoencoders_extended(Xv, _ae_as_list(AE_c), "folding")
        d = np.asarray(d).reshape(-1)
        if d.size != N_val:
            raise ValueError(
                f"Distance vector from cluster {c + 1} has shape {d.shape}; expected ({N_val},)."
            )
        D_val[c, :] = d

    D_val = _ensure_distance_matrix_shape(_as_float_ndarray(D_val), C_eff, N_val, labels=None)

    # Materialize sequences to allow validation and stable repr in verbose output
    alphas = tuple(alphas)
    topks = tuple(topks)
    if len(alphas) == 0:
        raise ValueError("alphas must contain at least one value.")
    if len(topks) == 0:
        raise ValueError("topks must contain at least one value.")

    best_mse = float("inf")
    best_alpha: float = float(alphas[0])
    best_topk: int | None = topks[0]

    if verbose:
        print("Tuning soft parameters on validation set...")
        print(f"Grid: alphas={list(alphas)}, topks={list(topks)}")
        print(f"Validation samples: {N_val}, clusters: {C_eff}")

    # Reusable probability buffer (one full C_eff x N_val matrix)
    work = np.empty_like(D_val, dtype=D_val.dtype)
    dtype = work.dtype
    one = dtype.type(1.0)
    zero = dtype.type(0.0)
    eps_t = dtype.type(1e-12)

    for a in alphas:
        a_f = float(a)
        for k in topks:
            # work = (1 - D_val) ** alpha
            np.subtract(one, D_val, out=work)
            np.clip(work, zero, one, out=work)
            if a_f != 1.0:
                np.power(work, dtype.type(a_f), out=work)

            if k is not None:
                kk = int(k)
                if 0 < kk < C_eff:
                    _topk_truncate_inplace(work, kk)

            denom = work.sum(axis=0, dtype=dtype)
            zero_cols = denom <= eps_t
            if np.any(zero_cols):
                denom = denom.copy()
                denom[zero_cols] = one

            np.divide(work, denom, out=work)

            if np.any(zero_cols):
                work[:, zero_cols] = dtype.type(1.0 / C_eff)

            Y_hat = cluster_centers @ work
            mse = float(np.mean((Y_hat - Y_val) ** 2))

            if verbose:
                print(f"  alpha={a_f:g}, topk={k}: MSE={mse:.6f}")

            if mse < best_mse:
                best_mse = mse
                best_alpha = a_f
                best_topk = k

    # Persist into model for later inference defaults
    model["soft_alpha"] = best_alpha
    model["soft_topk"] = best_topk

    if verbose:
        print(f"Best soft params: alpha={best_alpha}, topk={best_topk}, val MSE={best_mse:.6f}")

    return SoftTuningResult(best_alpha=best_alpha, best_topk=best_topk, best_mse=best_mse)


# ----------------------------
# Persistence
# ----------------------------

def save_kahm_regressor(model: dict, path: str) -> None:
    dump(model, path)
    print(f"KAHM regressor saved to {path}")


def load_kahm_regressor(path: str) -> dict:
    model = load(path)
    print(f"KAHM regressor loaded from {path}")
    return model


# ----------------------------
# Example usage
# ----------------------------

if __name__ == "__main__":
    np.random.seed(0)

    MODEL_PATH = "kahm_regressor_example.joblib"

    # Toy data
    D_in, D_out, N = 5, 3, 10_000
    X = np.tanh(np.random.randn(D_in, N))
    Y = np.vstack(
        [
            2 * X[0, :] + 0.5 * X[1, :] ** 2 + 0.1 * np.random.randn(N),
            -X[2, :] + np.sin(X[3, :]) + 0.1 * np.random.randn(N),
            X[4, :] * 1.5 + 0.1 * np.random.randn(N),
        ]
    )

    # Split: train / val / test
    N_train = int(0.7 * N)
    N_val = int(0.15 * N)
    X_train = X[:, :N_train]
    Y_train = Y[:, :N_train]

    X_val = X[:, N_train:N_train + N_val]
    Y_val = Y[:, N_train:N_train + N_val]

    X_test = X[:, N_train + N_val:]
    Y_test = Y[:, N_train + N_val:]

    model = train_kahm_regressor(
        X_train,
        Y_train,
        n_clusters=1000,
        subspace_dim=20,
        Nb=50,
        random_state=0,
        verbose=False,
        input_scale=0.5,
        cluster_center_normalization="none"
    )

    # Tune alpha/topk on validation and store into model
    tune_soft_params(
        model,
        X_val,
        Y_val,
        alphas=(5.0, 10.0, 15.0, 20.0),
        topks=(5, 10, 15, 20),
        n_jobs=-1,
        verbose=True,
    )

    save_kahm_regressor(model, MODEL_PATH)
    loaded_model = load_kahm_regressor(MODEL_PATH)

    # Hard prediction
    Y_pred_hard = kahm_regress(loaded_model, X_test, mode="hard")

    # Soft prediction: uses stored (soft_alpha, soft_topk) automatically
    Y_pred_soft, P = kahm_regress(loaded_model, X_test, mode="soft", return_probabilities=True)

    def _mse(yhat: np.ndarray, ytrue: np.ndarray) -> float:
        return float(np.mean((yhat - ytrue) ** 2))

    def _r2_overall(yhat: np.ndarray, ytrue: np.ndarray) -> float:
        residual_ss = float(np.sum((yhat - ytrue) ** 2))
        total_ss = float(np.sum((ytrue - ytrue.mean(axis=1, keepdims=True)) ** 2))
        return 1.0 - residual_ss / total_ss

    print(f"\nStored soft_alpha={loaded_model.get('soft_alpha')}, soft_topk={loaded_model.get('soft_topk')}")
    print(f"Test MSE (hard): {_mse(Y_pred_hard, Y_test):.6f} | R^2 (hard): {_r2_overall(Y_pred_hard, Y_test):.4f}")
    print(f"Test MSE (soft): {_mse(Y_pred_soft, Y_test):.6f} | R^2 (soft): {_r2_overall(Y_pred_soft, Y_test):.4f}")

    col_sums = P.sum(axis=0)
    print("Probability column sums (min/mean/max):",
          float(col_sums.min()), float(col_sums.mean()), float(col_sums.max()))
    p_max = P.max(axis=0)
    print("Avg max prob (soft):", float(p_max.mean()),
          "| min/median/max:", float(p_max.min()), float(np.median(p_max)), float(p_max.max()))