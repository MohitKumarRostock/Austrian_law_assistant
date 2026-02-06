"""
kahm_classification.py

KAHM-based classification using per-class autoencoder distances.

Training:
- For each class label c, train an autoencoder (via `parallel_autoencoders`) on the
  subset of inputs X belonging to class c.

Inference:
- For a new sample x, compute distances d_c(x) to each class autoencoder using
  `combine_multiple_autoencoders_extended`.
- Predict the class with minimum distance:
      y_hat = argmin_c d_c(x)

Notes vs. kahm_regression.py
---------------------------
- No output clustering; labels are provided directly.
- No distance->probability mapping (softmax/sharpening/top-k) is performed.
- No NLMS tuning or output-center refinement is required.

Requires:
    pip install scikit-learn numpy joblib

Also requires the OTFL helpers used in kahm_regression.py:
    - parallel_autoencoders
    - combine_multiple_autoencoders_extended
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, overload, Literal

import os
import tempfile
import gc as _gc

import numpy as np
from numpy.typing import DTypeLike

from sklearn.neighbors import NearestNeighbors
from joblib import dump, load


# OTFL helpers (same imports as kahm_regression.py)
from parallel_autoencoders import parallel_autoencoders
from combine_multiple_autoencoders_extended import combine_multiple_autoencoders_extended


# ----------------------------
# Precision helpers
# ----------------------------

def _as_float_ndarray(x: Any, *, min_dtype: DTypeLike = np.float32) -> np.ndarray:
    """Convert input to a floating ndarray without downcasting precision."""
    arr = np.asarray(x)
    if arr.dtype.kind not in "fc":
        arr = arr.astype(np.float64, copy=False)
    dtype = np.result_type(arr.dtype, min_dtype)
    return arr.astype(dtype, copy=False)


def _scale_like(X: np.ndarray, scale: float, *, inplace: bool = False) -> np.ndarray:
    """Scale X by `scale` without unintentionally upcasting dtype."""
    if scale == 1.0:
        return X
    if X.dtype.kind not in "fc":
        X = X.astype(np.float32, copy=False)
    s = np.asarray(scale, dtype=X.dtype)
    if inplace and X.flags.writeable:
        np.multiply(X, s, out=X, casting="unsafe")
        return X
    return (X * s).astype(X.dtype, copy=False)


def _ae_as_list(ae: Any) -> list:
    """Ensure an autoencoder spec is passed as a flat list."""
    if isinstance(ae, (list, tuple)):
        if len(ae) == 1 and isinstance(ae[0], (list, tuple)):
            return list(ae[0])
        return list(ae)
    return [ae]


def _call_combine_multiple_autoencoders_extended(
    X: np.ndarray,
    AE_list: Sequence[Any],
    distance_type: str,
    *,
    n_jobs: int | None = None,
):
    """Call `combine_multiple_autoencoders_extended` with best-effort support for `n_jobs`."""
    fn: Any = combine_multiple_autoencoders_extended
    try:
        return fn(X, AE_list, distance_type, n_jobs=n_jobs)
    except TypeError:
        return fn(X, AE_list, distance_type)


# ----------------------------
# Training
# ----------------------------

def train_kahm_classifier(
    X: np.ndarray,
    y: Any,
    *,
    subspace_dim: int = 20,
    Nb: int = 100,
    random_state: int | None = 0,
    verbose: bool = True,
    input_scale: float = 1.0,
    # Limit per-class training set size (RAM/time lever)
    max_train_per_class: int | None = None,
    # Downcast arrays during training and inside the stored model to reduce RAM.
    model_dtype: str = "auto",
    # Disk-backed classifier storage (recommended when number of classes is large)
    save_ae_to_disk: bool = True,
    ae_dir: str | Path | None = None,
    ae_cache_root: str | Path = "kahm_ae_cache",
    overwrite_ae_dir: bool = False,
    model_id: str | None = None,
    ae_compress: int = 3,
    # Singleton class handling
    # When a class has only 1 sample, OTFL AE training can fail.
    # Strategy:
    #   - "augment" (default): add one synthetic sample by mixing with the global nearest neighbor in input space.
    #   - "duplicate": add a tiny-jitter copy of the singleton sample.
    singleton_strategy: str = "augment",  # {'augment','duplicate'}
    singleton_aux_mix: float = 0.1,
    singleton_jitter_std: float = 1e-3,
) -> dict:
    """
    Train a KAHM classifier using per-class autoencoders.

    Parameters
    ----------
    X:
        Input matrix shaped (D_in, N).
    y:
        Labels (array-like) of length N (or shape (N,1)/(1,N)).
    """
    X = _as_float_ndarray(X)

    # Normalize labels to a 1D array of length N (dtype object to allow strings)
    y_arr = np.asarray(y)
    if y_arr.ndim == 2 and 1 in y_arr.shape:
        y_arr = y_arr.reshape(-1)
    elif y_arr.ndim != 1:
        raise ValueError("y must be 1D (length N) or 2D with one singleton dimension (N,1) or (1,N).")
    y_arr = y_arr.astype(object, copy=False)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array shaped (D, N).")
    D_in, N = X.shape
    if y_arr.shape[0] != N:
        raise ValueError(f"X and y must have the same number of samples. Got X.shape={X.shape}, y.shape={y_arr.shape}.")

    # Choose working dtype
    md0 = str(model_dtype).lower().strip()
    if md0 in ("auto", "none", ""):
        work_dtype = X.dtype
    elif md0 in ("float32", "f32"):
        work_dtype = np.float32
    elif md0 in ("float64", "f64"):
        work_dtype = np.float64
    else:
        raise ValueError("model_dtype must be one of {'auto','float32','float64'}")
    X = X.astype(work_dtype, copy=False)

    if verbose:
        classes_preview = ", ".join([repr(c) for c in np.unique(y_arr)[:10]])
        print(f"Training KAHM classifier on {N} samples. Input dim={D_in}.")
        print(f"Number of classes: {len(np.unique(y_arr))}. Classes (preview): {classes_preview}")
        print(f"Input scaling factor (input_scale): {input_scale}")

    # Apply input scaling (for AE-based class assignment)
    if input_scale != 1.0:
        X = _scale_like(X, float(input_scale), inplace=False)

    # Encode classes to indices 0..C-1
    classes = np.unique(y_arr)
    C = int(classes.shape[0])
    class_to_index = {cls: i for i, cls in enumerate(classes)}
    y_idx = np.asarray([class_to_index[cls] for cls in y_arr], dtype=np.int64)

    # Handle singleton classes if present
    counts = np.bincount(y_idx, minlength=C)
    singletons = np.where(counts == 1)[0]
    if singletons.size > 0 and verbose:
        print(f"Handling {singletons.size} singleton class(es) using strategy='{singleton_strategy}'...")

    if singletons.size > 0:
        strategy = str(singleton_strategy).lower().strip()
        if strategy not in ("augment", "duplicate"):
            raise ValueError("singleton_strategy must be one of {'augment','duplicate'}")

        rng = np.random.RandomState(int(random_state) if random_state is not None else 0)

        if strategy == "augment":
            mix = float(singleton_aux_mix)
            mix = 0.0 if not np.isfinite(mix) else max(0.0, min(1.0, mix))

            # Global nearest neighbors in input space
            X_nn = np.asarray(X.T, dtype=np.float32 if X.dtype == np.float64 else X.dtype)
            N_nn = int(X_nn.shape[0])
            if N_nn < 2:
                raise RuntimeError("Cannot augment singleton classes with fewer than 2 total training points.")

            k_nn = 5 if N_nn >= 5 else N_nn
            nn = NearestNeighbors(n_neighbors=k_nn, metric="euclidean", algorithm="auto")
            nn.fit(X_nn)
            _, nn_idx = nn.kneighbors(X_nn, return_distance=True)  # (N, k_nn)

            ar = np.arange(N_nn, dtype=nn_idx.dtype)
            nearest_global = nn_idx[:, 1].astype(np.int64, copy=False)
            self_mask = nearest_global == ar
            if np.any(self_mask):
                for kk in range(2, nn_idx.shape[1]):
                    repl = self_mask & (nn_idx[:, kk] != ar)
                    if np.any(repl):
                        nearest_global[repl] = nn_idx[repl, kk]
                    self_mask = nearest_global == ar
                    if not np.any(self_mask):
                        break
            if np.any(self_mask):
                bad = np.where(self_mask)[0]
                for i in bad:
                    nearest_global[int(i)] = int((int(i) + 1) % N_nn)

            X_new_cols: list[np.ndarray] = []
            y_new: list[int] = []

            for cls_i in singletons:
                # locate singleton sample index
                s_idx = int(np.where(y_idx == int(cls_i))[0][0])
                x_singleton = X[:, s_idx]
                nn_local = int(nearest_global[s_idx])
                if nn_local == s_idx:
                    nn_local = int((s_idx + 1) % int(X.shape[1]))
                x_neighbor = X[:, nn_local]

                x_aux = (1.0 - mix) * x_singleton + mix * x_neighbor
                # preserve norm of singleton point
                x_norm = float(np.linalg.norm(x_singleton)) + 1e-12
                aux_norm = float(np.linalg.norm(x_aux)) + 1e-12
                x_aux = x_aux * (x_norm / aux_norm)

                X_new_cols.append(x_aux.reshape(-1, 1))
                y_new.append(int(cls_i))

            if X_new_cols:
                X = np.concatenate([X] + X_new_cols, axis=1)
                y_idx = np.concatenate([y_idx, np.asarray(y_new, dtype=y_idx.dtype)], axis=0)
                y_arr = np.concatenate([y_arr, np.asarray([classes[i] for i in y_new], dtype=object)], axis=0)

                # Update counts
                N = int(X.shape[1])
                counts = np.bincount(y_idx, minlength=C)
                remaining = np.where(counts == 1)[0]
                if remaining.size > 0:
                    raise RuntimeError(
                        f"Augmentation failed to eliminate {remaining.size} singleton class(es). "
                        "Consider checking X for degenerate duplicates or using singleton_strategy='duplicate'."
                    )

        else:
            # duplicate with jitter
            std = float(singleton_jitter_std)
            std = 0.0 if not np.isfinite(std) else max(0.0, std)

            X_new_cols: list[np.ndarray] = []
            y_new: list[int] = []

            for cls_i in singletons:
                s_idx = int(np.where(y_idx == int(cls_i))[0][0])
                x_singleton = X[:, s_idx]
                noise = rng.randn(*x_singleton.shape).astype(X.dtype, copy=False) * np.asarray(std, dtype=X.dtype)
                x_aux = x_singleton + noise
                X_new_cols.append(x_aux.reshape(-1, 1))
                y_new.append(int(cls_i))

            if X_new_cols:
                X = np.concatenate([X] + X_new_cols, axis=1)
                y_idx = np.concatenate([y_idx, np.asarray(y_new, dtype=y_idx.dtype)], axis=0)
                y_arr = np.concatenate([y_arr, np.asarray([classes[i] for i in y_new], dtype=object)], axis=0)

                N = int(X.shape[1])
                counts = np.bincount(y_idx, minlength=C)
                remaining = np.where(counts == 1)[0]
                if remaining.size > 0:
                    raise RuntimeError(
                        f"Duplicate/jitter failed to eliminate {remaining.size} singleton class(es). "
                        "Consider increasing singleton_jitter_std."
                    )

    # Per-class subsampling (optional)
    if max_train_per_class is not None:
        m = int(max_train_per_class)
        if m < 2:
            raise ValueError("max_train_per_class must be >= 2 (or None).")

        rng = np.random.RandomState(int(random_state) if random_state is not None else 0)
        keep = np.zeros((int(X.shape[1]),), dtype=bool)

        for c in range(C):
            idx_c = np.where(y_idx == c)[0]
            if idx_c.size <= m:
                keep[idx_c] = True
            else:
                chosen = rng.choice(idx_c, size=m, replace=False)
                keep[chosen] = True

        X = X[:, keep]
        y_idx = y_idx[keep]
        y_arr = y_arr[keep]
        N = int(X.shape[1])
        if verbose:
            print(f"Subsampled to {N} total training points using max_train_per_class={m}.")

    # Prepare storage (disk-backed or memory)
    if save_ae_to_disk:
        run_id = model_id if model_id is not None else f"kahm_cls_{os.getpid()}_{int(np.random.randint(0, int(1e9)))}"
        root = Path(ae_cache_root)
        root.mkdir(parents=True, exist_ok=True)

        if ae_dir is None:
            ae_dir_resolved = root / run_id
        else:
            ae_dir_resolved = Path(ae_dir)

        if ae_dir_resolved.exists():
            if not overwrite_ae_dir:
                raise FileExistsError(
                    f"ae_dir '{ae_dir_resolved}' already exists. "
                    "Use overwrite_ae_dir=True or specify a different ae_dir/model_id."
                )
        else:
            ae_dir_resolved.mkdir(parents=True, exist_ok=True)

        AE_arr: list[str] = []
        if verbose:
            print(f"Saving per-class autoencoders under: {ae_dir_resolved}")

        for c in range(C):
            idx_c = np.where(y_idx == c)[0]
            if idx_c.size < 2:
                raise RuntimeError(
                    f"Class {classes[c]!r} has {idx_c.size} sample(s) after preprocessing; "
                    "need at least 2 for AE training."
                )
            X_c = X[:, idx_c]
            if verbose:
                print(f" Class {c + 1}/{C} ({classes[c]!r}): training autoencoder on {X_c.shape[1]} samples ...")

            AE_c_list = parallel_autoencoders(
                X_c,
                subspace_dim=subspace_dim,
                Nb=Nb,
                n_jobs=1,
                verbose=False,
            )

            shard = ae_dir_resolved / f"{c // 1000:03d}"
            shard.mkdir(parents=True, exist_ok=True)

            ae_path = shard / f"ae_class_{c + 1:05d}.joblib"
            dump(AE_c_list, ae_path, compress=int(ae_compress))
            AE_arr.append(os.path.relpath(str(ae_path), str(ae_dir_resolved)))

            del X_c, idx_c, AE_c_list
            _gc.collect()

        classifier = AE_arr
        classifier_dir_for_model = str(ae_dir_resolved)
        model_id_for_model = run_id

    else:
        classifier = []
        classifier_dir_for_model = None
        model_id_for_model = None

        for c in range(C):
            idx_c = np.where(y_idx == c)[0]
            if idx_c.size < 2:
                raise RuntimeError(
                    f"Class {classes[c]!r} has {idx_c.size} sample(s) after preprocessing; need at least 2."
                )
            X_c = X[:, idx_c]
            if verbose:
                print(f" Class {c + 1}/{C} ({classes[c]!r}): training autoencoder on {X_c.shape[1]} samples ...")

            AE_c_list = parallel_autoencoders(
                X_c,
                subspace_dim=subspace_dim,
                Nb=Nb,
                n_jobs=1,
                verbose=False,
            )
            classifier.append(AE_c_list)

            del X_c, idx_c, AE_c_list
            _gc.collect()

    model: dict[str, Any] = {
        "kind": "kahm_classifier",
        "input_dim": int(D_in),
        "n_classes": int(C),
        "classes": classes.tolist(),  # JSON/joblib friendly
        "classifier": classifier,
        "classifier_dir": classifier_dir_for_model,
        "model_id": model_id_for_model,
        "subspace_dim": int(subspace_dim),
        "Nb": int(Nb),
        "input_scale": float(input_scale),
        "model_dtype": str(np.dtype(work_dtype).name),
        # runtime-only cache (not saved)
        "_classifier_cache": {},
    }
    return model


# ----------------------------
# Inference
# ----------------------------

@overload
def kahm_predict(
    model: dict,
    X_new: np.ndarray,
    n_jobs: int = -1,
    *,
    return_distances: Literal[False] = False,
    batch_size: Optional[int] = None,
    show_progress: bool = True,
) -> np.ndarray: ...


@overload
def kahm_predict(
    model: dict,
    X_new: np.ndarray,
    n_jobs: int = -1,
    *,
    return_distances: Literal[True],
    batch_size: Optional[int] = None,
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray]: ...


def kahm_predict(
    model: dict,
    X_new: np.ndarray,
    n_jobs: int = -1,
    *,
    return_distances: bool = False,
    batch_size: Optional[int] = None,
    show_progress: bool = True,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """
    Predict class labels for new inputs using a trained KAHM classifier.

    Parameters
    ----------
    model:
        Trained model dict from `train_kahm_classifier`.
    X_new:
        Input matrix shaped (D_in, N_new).
    return_distances:
        If True, also return the per-class distance matrix of shape (C, N_new).
    batch_size:
        Compute distances in batches over N_new to reduce peak memory.
    show_progress:
        If True and tqdm is installed, show a progress bar.

    Returns
    -------
    y_pred:
        Predicted labels (dtype inferred from training labels), shape (N_new,).
    D (optional):
        Distance matrix (C, N_new) as float64.
    """
    if model.get("kind") != "kahm_classifier":
        raise ValueError("Expected a kahm_classifier model dict.")

    X_new = _as_float_ndarray(X_new, min_dtype=np.float32)

    if X_new.ndim != 2:
        raise ValueError("X_new must be 2D shaped (D, N_new).")
    D_in, N_new = X_new.shape
    if int(D_in) != int(model["input_dim"]):
        raise ValueError(f"Input dimension mismatch: X_new has D={D_in}, model expects {model['input_dim']}.")

    # Apply the same scaling used in training
    input_scale = float(model.get("input_scale", 1.0))
    if input_scale != 1.0:
        X_new = _scale_like(X_new, input_scale, inplace=False)

    classes = np.asarray(model["classes"])  # infer dtype; avoids sklearn "unknown" targets for numeric labels
    C = int(model["n_classes"])
    if classes.shape[0] != C:
        raise ValueError("Model is inconsistent: n_classes != len(classes).")

    classifier = model["classifier"]
    if len(classifier) != C:
        raise ValueError(f"Model is inconsistent: len(classifier)={len(classifier)} but n_classes={C}.")

    # progress bar helper
    def _maybe_tqdm_total(total: int, desc: str, unit: str):
        if not show_progress:
            return None
        try:
            from tqdm import tqdm  # type: ignore
            return tqdm(total=total, desc=desc, unit=unit, leave=False)
        except Exception:
            return None

    # Resolve AE references (disk-backed or in-memory)
    def _is_pathlike(x) -> bool:
        return isinstance(x, (str, os.PathLike, Path))

    def _resolve_ae_path(ae_ref):
        if not _is_pathlike(ae_ref):
            return ae_ref
        p = Path(ae_ref)
        base_dir = model.get("classifier_dir", None)
        if not p.is_absolute() and base_dir is not None:
            p = Path(base_dir) / p
        return str(p)

    def _load_ae_maybe(ae_ref, class_idx: int):
        cache = model.setdefault("_classifier_cache", {})
        if class_idx in cache:
            return cache[class_idx], False
        ref = _resolve_ae_path(ae_ref)
        if _is_pathlike(ref):
            ae_obj = load(ref)
        else:
            ae_obj = ref
        cache[class_idx] = ae_obj
        return ae_obj, _is_pathlike(ref)

    distance_type = "folding"
    bs = int(batch_size) if (batch_size is not None and int(batch_size) > 0) else None

    # If the user asks for distances, allocate full matrix; otherwise keep only best.
    if return_distances:
        Dmat = np.empty((C, N_new), dtype=np.float64)
    else:
        Dmat = None

    best_dist = np.full((N_new,), np.inf, dtype=np.float64)
    best_idx = np.zeros((N_new,), dtype=np.int64)

    pbar = _maybe_tqdm_total(C * N_new, desc="KAHM classify: distance eval", unit="sample")

    try:
        for c, AE_ref in enumerate(classifier):
            AE_c, from_disk = _load_ae_maybe(AE_ref, class_idx=c)
            try:
                if bs is None:
                    d = _call_combine_multiple_autoencoders_extended(X_new, _ae_as_list(AE_c), distance_type, n_jobs=n_jobs)
                    d = np.asarray(d, dtype=np.float64).reshape(-1)
                    if d.size != N_new:
                        raise ValueError(f"Distance vector for class {c} has shape {d.shape}; expected ({N_new},).")
                    if Dmat is not None:
                        Dmat[c, :] = d
                    mask = d < best_dist
                    best_dist[mask] = d[mask]
                    best_idx[mask] = c
                    if pbar is not None:
                        pbar.update(N_new)
                else:
                    for start in range(0, N_new, bs):
                        end = min(start + bs, N_new)
                        Xb = X_new[:, start:end]
                        d = _call_combine_multiple_autoencoders_extended(Xb, _ae_as_list(AE_c), distance_type, n_jobs=n_jobs)
                        d = np.asarray(d, dtype=np.float64).reshape(-1)
                        if d.size != (end - start):
                            raise ValueError(
                                f"Distance vector for class {c} has shape {d.shape}; expected ({end-start},)."
                            )
                        if Dmat is not None:
                            Dmat[c, start:end] = d
                        cols = np.arange(start, end)
                        mask = d < best_dist[cols]
                        upd = cols[mask]
                        best_dist[upd] = d[mask]
                        best_idx[upd] = c
                        if pbar is not None:
                            pbar.update(end - start)
            finally:
                if from_disk:
                    # If loaded from disk, keep it cached by default (speed). To avoid RAM growth,
                    # user can clear model['_classifier_cache'] manually between calls.
                    pass
    finally:
        if pbar is not None:
            pbar.close()

    y_pred = np.asarray(classes[best_idx]).reshape(-1)
    if return_distances:
        assert Dmat is not None
        return y_pred, Dmat
    return y_pred


# ----------------------------
# Persistence
# ----------------------------

def _strip_runtime_keys(model: dict) -> dict:
    """Remove runtime-only keys (leading underscore) before saving."""
    return {k: v for k, v in model.items() if not str(k).startswith("_")}


def save_kahm_classifier(model: dict, path: str) -> None:
    """
    Save a KAHM classifier to disk.

    Portability:
    - If model['classifier_dir'] is absolute, attempt to store it relative to the model file directory.
    """
    model_to_save = _strip_runtime_keys(model)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    cdir = model_to_save.get("classifier_dir", None)
    if isinstance(cdir, str) and cdir:
        try:
            cdir_p = Path(cdir)
            if cdir_p.is_absolute():
                rel = os.path.relpath(str(cdir_p), str(p.parent))
                model_to_save["classifier_dir"] = rel
        except Exception:
            pass

    dump(model_to_save, str(p))


def load_kahm_classifier(path: str, *, base_dir: str | None = None) -> dict:
    """
    Load a KAHM classifier from disk.

    Parameters
    ----------
    path:
        Path to the saved classifier (joblib).
    base_dir:
        Optional override for model['classifier_dir'].
    """
    p = Path(path)
    model = load(str(p))
    if not isinstance(model, dict):
        raise ValueError("Loaded object is not a dict; expected a saved KAHM classifier dict.")

    # Restore runtime cache
    model.setdefault("_classifier_cache", {})

    # Resolve classifier_dir portability
    if base_dir is not None:
        model["classifier_dir"] = str(base_dir)
    else:
        cdir = model.get("classifier_dir", None)
        if isinstance(cdir, str) and cdir:
            cdir_p = Path(cdir)
            if not cdir_p.is_absolute():
                model["classifier_dir"] = str((p.parent / cdir_p).resolve())

    return model


# ----------------------------
# Demo
# ----------------------------

if __name__ == "__main__":
    np.random.seed(0)

    MODEL_PATH = "kahm_classifier_example.joblib"

    # Synthetic classification demo (2D, 3 classes)
    from sklearn.datasets import make_blobs
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix

    X2, y2 = make_blobs(
        n_samples=1000,
        centers=3,
        n_features=2,
        cluster_std=0.5,
        random_state=0,
        return_centers=False,
    )[:2]
    # KAHM expects (D, N)
    X = X2.T
    y = y2

    # Split train/test
    X_train2, X_test2, y_train, y_test = train_test_split(
        X.T, y, test_size=0.25, random_state=0, stratify=y
    )
    X_train = X_train2.T
    X_test = X_test2.T

    model = train_kahm_classifier(
        X_train,
        y_train,
        subspace_dim=20,
        Nb=100,
        random_state=0,
        verbose=True,
        input_scale=1.0,
        save_ae_to_disk=False,  # keep demo self-contained
        singleton_strategy="augment",
        singleton_aux_mix = 0.1,
    )

    # Predict (min-distance over classes)
    y_pred = kahm_predict(model, X_test, batch_size=1024)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))

    print(f"\nTest accuracy: {acc:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    # Save/load smoke test
    save_kahm_classifier(model, MODEL_PATH)
    loaded = load_kahm_classifier(MODEL_PATH)
    y_pred2 = kahm_predict(loaded, X_test, batch_size=1024, show_progress=False)
    acc2 = accuracy_score(y_test, y_pred2)
    print(f"Reloaded model accuracy: {acc2:.4f}")
