# otfl_blocksafe.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import numpy as np


# ----------------------------- OTFL BlockSafe Patch -----------------------------

@dataclass
class BlockSafeStats:
    dimreduce_hits: int = 0
    dimreduce_repaired: int = 0
    dimreduce_fallbacks: int = 0
    autoencoder_singletons_augmented: int = 0


_BLOCKSAFE_STATS = BlockSafeStats()
_BLOCKSAFE_ENABLED = False


def _blocksafe_colwise_l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=0, keepdims=True)
    norms = np.maximum(norms, eps)
    return X / norms


def _blocksafe_jitter_copy(
    y: np.ndarray,
    *,
    std: float,
    rng: np.random.Generator,
    l2_normalized: bool,
) -> np.ndarray:
    noise = rng.normal(loc=0.0, scale=float(std), size=y.shape).astype(y.dtype, copy=False)
    y2 = y + noise
    if l2_normalized:
        y2 = _blocksafe_colwise_l2_normalize(y2)
    return y2


def _blocksafe_trivial_reduce(y: np.ndarray, requested_k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return a stable reduction with at least 1 component (if d>0).
    This avoids OTFL code paths that compute reductions over empty arrays.
    """
    d, n = int(y.shape[0]), int(y.shape[1])
    if d <= 0:
        return np.zeros((1, n), dtype=y.dtype), np.zeros((d, 1), dtype=y.dtype)

    k = max(1, int(requested_k))
    k = min(k, d)

    y_red = y[:k, :].astype(y.dtype, copy=False)
    PC = np.zeros((d, k), dtype=y.dtype)
    PC[:k, :k] = np.eye(k, dtype=y.dtype)
    return y_red, PC


def enable_otfl_blocksafe(
    *,
    jitter_std: float = 1e-5,
    jitter_tries: int = 6,
    jitter_growth: float = 2.0,
    pca_eps: Optional[float] = None,
    eps_factor: float = 10.0,
    backend: str = "threading",
    l2_normalized: bool = True,
    log_first: int = 20,
    random_seed: int = 0,
) -> Callable[[], Any]:
    """
    Installs monkey-patches into OTFL to handle degenerate Nb-blocks.

    IMPORTANT:
      - Use backend="threading" (or n_jobs=1) so patches apply. With loky
        subprocesses, patches will NOT propagate on macOS.
    """
    global _BLOCKSAFE_ENABLED

    import importlib
    from joblib import parallel_backend

    dr = importlib.import_module("dim_reduce")
    ae_mod = importlib.import_module("autoencoder")
    pa_mod = importlib.import_module("parallel_autoencoders")
    clf_mod = importlib.import_module("classifier")

    orig_dim_reduce = getattr(dr, "dim_reduce", None)
    orig_autoencoder = getattr(ae_mod, "autoencoder", None)
    if orig_dim_reduce is None or orig_autoencoder is None:
        raise AttributeError("blocksafe patch: could not locate dim_reduce.dim_reduce or autoencoder.autoencoder")

    # Infer eps (best-effort)
    eps_used = pca_eps
    if eps_used is None:
        for name in ("eps", "EPS", "_eps"):
            if hasattr(dr, name):
                try:
                    eps_used = float(getattr(dr, name))
                    break
                except Exception:
                    pass
    if eps_used is None:
        eps_used = 1e-12

    rng = np.random.default_rng(int(random_seed))
    log_budget = {"remaining": int(log_first)}

    def _log(msg: str) -> None:
        if log_budget["remaining"] > 0:
            print(msg)
            log_budget["remaining"] -= 1

    def dim_reduce_safe(y_data: np.ndarray, subspace_dim: int, *args, **kwargs):
        if not np.isfinite(y_data).all():
            raise ValueError("[blocksafe] dim_reduce: non-finite values detected in y_data")

        d, n = int(y_data.shape[0]), int(y_data.shape[1])
        k_req = int(subspace_dim)

        # OTFL can clamp to k=0 for n=1; never allow empty reductions.
        if k_req < 1:
            _BLOCKSAFE_STATS.dimreduce_hits += 1
            _BLOCKSAFE_STATS.dimreduce_fallbacks += 1
            _log(
                f"[blocksafe] dim_reduce: requested subspace_dim={k_req} (<1) for shape=({d},{n}); "
                "using safe 1D trivial reduction."
            )
            return _blocksafe_trivial_reduce(y_data, 1)

        try:
            return orig_dim_reduce(y_data, k_req, *args, **kwargs)
        except ValueError as e:
            if "No eigenvalues" not in str(e):
                raise

        _BLOCKSAFE_STATS.dimreduce_hits += 1

        # Ensure jitter variance clears eps
        std0 = max(float(jitter_std), float(eps_used) * float(eps_factor))
        std = std0
        for _ in range(int(jitter_tries)):
            yd2 = _blocksafe_jitter_copy(y_data, std=std, rng=rng, l2_normalized=l2_normalized)
            try:
                out = orig_dim_reduce(yd2, k_req, *args, **kwargs)
                _BLOCKSAFE_STATS.dimreduce_repaired += 1
                _log(f"[blocksafe] dim_reduce: repaired degenerate subset shape=({d},{n}) with jitter_std={std:.2e}")
                return out
            except ValueError as e2:
                if "No eigenvalues" not in str(e2):
                    raise
                std *= float(jitter_growth)

        _BLOCKSAFE_STATS.dimreduce_fallbacks += 1
        _log(f"[blocksafe] dim_reduce: fallback trivial reduction for shape=({d},{n}) after {jitter_tries} tries.")
        return _blocksafe_trivial_reduce(y_data, k_req)

    def autoencoder_safe(y_data: np.ndarray, subspace_dim: int, *args, **kwargs):
        # Ensure n>=2 so OTFL's covariance computation is well-defined.
        d, n = int(y_data.shape[0]), int(y_data.shape[1])
        if n < 2:
            std0 = max(float(jitter_std), float(eps_used) * float(eps_factor))
            y_dup = _blocksafe_jitter_copy(y_data, std=std0, rng=rng, l2_normalized=l2_normalized)
            y_aug = np.concatenate([y_data, y_dup], axis=1)  # (d,2)
            _BLOCKSAFE_STATS.autoencoder_singletons_augmented += 1
            _log(f"[blocksafe] autoencoder: augmented singleton block shape=({d},{n}) -> ({d},2) with jitter_std={std0:.2e}")
            return orig_autoencoder(y_aug, int(subspace_dim), *args, **kwargs)
        return orig_autoencoder(y_data, int(subspace_dim), *args, **kwargs)

    # Monkey-patch OTFL entry points (module attributes)
    dr.dim_reduce = dim_reduce_safe  # type: ignore[attr-defined]

    ae_mod.autoencoder = autoencoder_safe  # type: ignore[attr-defined]
    # IMPORTANT: autoencoder.py typically imported dim_reduce into its namespace
    if hasattr(ae_mod, "dim_reduce"):
        ae_mod.dim_reduce = dim_reduce_safe  # type: ignore[attr-defined]

    # Patch common direct-import aliases in other OTFL modules
    if hasattr(pa_mod, "autoencoder"):
        pa_mod.autoencoder = autoencoder_safe  # type: ignore[attr-defined]
    if hasattr(pa_mod, "dim_reduce"):
        pa_mod.dim_reduce = dim_reduce_safe  # type: ignore[attr-defined]

    if hasattr(clf_mod, "autoencoder"):
        clf_mod.autoencoder = autoencoder_safe  # type: ignore[attr-defined]
    if hasattr(clf_mod, "dim_reduce"):
        clf_mod.dim_reduce = dim_reduce_safe  # type: ignore[attr-defined]

    def _ctx():
        return parallel_backend(str(backend))

    jitter_eff = max(float(jitter_std), float(eps_used) * float(eps_factor))
    if not _BLOCKSAFE_ENABLED:
        print(f"[blocksafe] Enabled. eps_used={eps_used:.2e} jitter_std_effective={jitter_eff:.2e} backend={backend}")
        _BLOCKSAFE_ENABLED = True
    else:
        print(f"[blocksafe] Already enabled. backend={backend}")

    return _ctx
