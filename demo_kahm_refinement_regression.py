#!/usr/bin/env python3
"""
demo_kahm_refinement_regression.py

Toy demonstrator for KAHM multivariate regression with and without output refinement.

What this demo shows
--------------------
1) Baseline regression: predict Y by mixing output-cluster centers with soft weights P
   derived from input-side distances (alpha/topk).
2) Refinement: keep the SAME weights P, but project the mixed prediction onto each
   output cluster's learned output manifold (OTFL output autoencoders), then mix:
       Y_ref = sum_c P_c * Pi_c(Y0)
   (This corresponds to output_refinement="project_then_mix" in kahm_regression.py.)

3) Comparison across mechanisms implemented in kahm_regression.py:
   - hard (center lookup) vs hard + projector
   - soft (center mixing) vs soft + projector (1 vs 2 refinement iterations)

Prerequisites
-------------
- You must have your project installed such that `import kahm_regression` resolves.
  Typically, place kahm_regression.py in the same directory or install your package.
- OTFL must be importable because kahm_regression uses:
    - otfl.parallel_autoencoders (training output projectors)
    - otfl.output_parallel_autoencoders_extended (inference projection)
    - otfl.classifier_new / prediction_classifier_extended_v2 (input-side gating)

Run
---
python demo_kahm_refinement_regression.py

Notes
-----
This demo is constructed to produce *measurable* improvement from refinement.
It does so by:
- making clusters in output space highly separated (KMeans recovers them),
- inducing classifier ambiguity for a fraction of samples (soft weights include wrong clusters),
- ensuring wrong-cluster components are largely orthogonal to the correct output manifold,
  so projection reduces off-manifold interference.
"""

from __future__ import annotations

import numpy as np

# Import your KAHM regressor implementation.
# If your file is named differently, change this import.
import kahm_regression as kr


def r2_overall(Y_hat: np.ndarray, Y_true: np.ndarray) -> float:
    resid = float(np.sum((Y_hat - Y_true) ** 2))
    tot = float(np.sum((Y_true - Y_true.mean(axis=1, keepdims=True)) ** 2))
    return 1.0 - resid / tot if tot > 0 else float("nan")


def mse(Y_hat: np.ndarray, Y_true: np.ndarray) -> float:
    return float(np.mean((Y_hat - Y_true) ** 2))


def make_toy_regression(
    *,
    N: int = 12000,
    Din: int = 12,
    Dout: int = 24,
    C: int = 4,
    latent_dim: int = 2,
    mean_scale: float = 4.0,
    within_scale: float = 0.25,
    noise_out: float = 0.05,
    amb_frac: float = 0.35,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a toy dataset (X, Y, z_true) where:
    - Outputs Y form C well-separated clusters.
    - Each cluster lies near a low-dimensional subspace (output "manifold").
    - Inputs X contain a noisy (sometimes ambiguous) cluster signal.

    Shapes:
      X: (Din, N)
      Y: (Dout, N)
      z_true: (N,)
    """
    rng = np.random.default_rng(seed)

    # Cluster means: orthogonal directions in output space (first C coordinates).
    means = np.zeros((Dout, C), dtype=float)
    means[:C, :] = np.eye(C) * mean_scale

    # Cluster-specific low-dim directions (nearly orthogonal across clusters).
    # We build one orthonormal basis per cluster.
    bases = []
    for c in range(C):
        A = rng.normal(size=(Dout, latent_dim))
        Q, _ = np.linalg.qr(A)
        bases.append(Q[:, :latent_dim])

    # True cluster labels
    z = rng.integers(0, C, size=N)

    # Latent within-cluster variation
    U = rng.normal(size=(latent_dim, N))

    # Output generation: mean + low-dim variation + noise
    Y = np.empty((Dout, N), dtype=float)
    for c in range(C):
        idx = np.where(z == c)[0]
        if idx.size == 0:
            continue
        Y[:, idx] = (
            means[:, c:c+1]
            + within_scale * (bases[c] @ U[:, idx])
            + noise_out * rng.normal(size=(Dout, idx.size))
        )

    # Input generation:
    # Encode cluster label in first C dims, but make a fraction ambiguous by blending with another cluster.
    X = rng.normal(size=(Din, N)) * 0.20
    onehot = np.zeros((C, N), dtype=float)
    onehot[z, np.arange(N)] = 1.0

    amb = rng.random(N) < amb_frac
    z2 = rng.integers(0, C, size=N)
    z2 = np.where(z2 == z, (z2 + 1) % C, z2)
    onehot2 = np.zeros((C, N), dtype=float)
    onehot2[z2, np.arange(N)] = 1.0

    mix = rng.uniform(0.35, 0.65, size=N)  # degree of ambiguity
    label_signal = onehot.copy()
    label_signal[:, amb] = (1.0 - mix[amb]) * onehot[:, amb] + mix[amb] * onehot2[:, amb]

    # Place the label signal into X and add small noise.
    X[:C, :] += 2.0 * label_signal
    X[C:2*C, :] += 0.5 * label_signal  # redundancy helps the classifier learn

    return X, Y, z


def main() -> None:
    # --------------------
    # 1) Data
    # --------------------
    X, Y, z_true = make_toy_regression()

    N = X.shape[1]
    n_train = int(0.70 * N)
    n_val = int(0.15 * N)

    X_train, Y_train = X[:, :n_train], Y[:, :n_train]
    X_val, Y_val = X[:, n_train:n_train + n_val], Y[:, n_train:n_train + n_val]
    X_test, Y_test = X[:, n_train + n_val:], Y[:, n_train + n_val:]

    print(f"Data: N={N} | train={X_train.shape[1]} val={X_val.shape[1]} test={X_test.shape[1]}")
    print(f"Shapes: X={X.shape} Y={Y.shape}")

    # --------------------
    # 2) Train KAHM regressor with output projectors enabled
    # --------------------
    # Choose n_clusters equal to the true number of clusters in this toy.
    # In your real setting you likely use a large n_clusters.
    model = kr.train_kahm_regressor(
        X_train,
        Y_train,
        n_clusters=1200,
        subspace_dim=20,      # input-side classifier subspace dim
        Nb=100,               # input-side block size
        random_state=0,
        verbose=True,
        # Output-domain projectors:
        train_output_projectors=True,
        output_subspace_dim=20,     # output AE subspace dim
        output_Nb=100,          # large => ~1 AE per output-cluster in this toy
        output_distance_type="folding",
        max_train_output_per_cluster=None,
        output_projector_prefer_otfl=True,
    )

    # --------------------
    # 3) Tune alpha/topk on validation set (NO refinement during tuning)
    #    This ensures refined vs baseline use the exact same weights P.
    # --------------------
    tuning = kr.tune_soft_params(
        model,
        X_val,
        Y_val,
        alphas=(2.0, 5.0, 8.0, 10.0, 13.0, 15.0, 18.0, 20.0, 25.0),
        topks=(2, 5, 10, 15, 20, 25),
        n_jobs=-1,
        verbose=True,
        output_refinement="project_then_mix",
        refinement_iters=1
    )

    print("\nChosen soft params (stored into model):")
    print(f"  soft_alpha = {model.get('soft_alpha')}")
    print(f"  soft_topk  = {model.get('soft_topk')}")
    print(f"  best_mse   = {tuning.best_mse:.6f}")

    # --------------------
    # 3b) Sanity check: refinement uses EXACTLY the same soft weights P
    #     (no extra sparsification / renormalization).
    #     We do this on a small subset so we can request return_probabilities=True.
    # --------------------
    rng = np.random.default_rng(123)
    idx_small = rng.choice(X_test.shape[1], size=200, replace=False)
    X_small = X_test[:, idx_small]
    Y_base_small, P_base = kr.kahm_regress(
        model, X_small, mode="soft", output_refinement="none", return_probabilities=True
    )
    Y_ref_small, P_ref = kr.kahm_regress(
        model, X_small, mode="soft", output_refinement="project_then_mix", refinement_iters=1, return_probabilities=True
    )
    max_abs = float(np.max(np.abs(P_base - P_ref)))
    print(f"\nSanity check: max |P_base - P_ref| on 200 samples = {max_abs:.3e}")

    # --------------------
    # 4) Evaluate mechanisms on test set
    # --------------------
    results = []

    # Hard: center lookup (no refinement)
    Y_hat = kr.kahm_regress(model, X_test, mode="hard", output_refinement="none")
    results.append(("hard", "none", 0, mse(Y_hat, Y_test), r2_overall(Y_hat, Y_test)))

    # Hard: project chosen center through output projector (cluster-wise)
    Y_hat = kr.kahm_regress(model, X_test, mode="hard", output_refinement="project_then_mix")
    results.append(("hard", "project(center)", 1, mse(Y_hat, Y_test), r2_overall(Y_hat, Y_test)))

    # Soft: center mixing (baseline)
    Y_hat = kr.kahm_regress(model, X_test, mode="soft", output_refinement="none", batch_size=2000)
    results.append(("soft", "none", 0, mse(Y_hat, Y_test), r2_overall(Y_hat, Y_test)))

    # Soft + refinement (1 iter)
    Y_hat = kr.kahm_regress(
        model,
        X_test,
        mode="soft",
        output_refinement="project_then_mix",
        refinement_iters=1,
        batch_size=2000,
    )
    results.append(("soft", "project_then_mix", 1, mse(Y_hat, Y_test), r2_overall(Y_hat, Y_test)))

    # Soft + refinement (2 iters)
    Y_hat = kr.kahm_regress(
        model,
        X_test,
        mode="soft",
        output_refinement="project_then_mix",
        refinement_iters=2,
        batch_size=2000,
    )
    results.append(("soft", "project_then_mix", 2, mse(Y_hat, Y_test), r2_overall(Y_hat, Y_test)))

    # --------------------
    # 5) Print results
    # --------------------
    print("\n=== Test results (multivariate regression) ===")
    print("mode | refinement | iters | MSE | R2")
    for mode, ref, iters, m, r2 in results:
        print(f"{mode:4s} | {ref:16s} | {iters:5d} | {m:10.6f} | {r2: .4f}")

    # Highlight improvement of soft refinement over baseline soft
    soft_base = [x for x in results if x[0] == "soft" and x[1] == "none"][0]
    soft_ref1 = [x for x in results if x[0] == "soft" and x[1] == "project_then_mix" and x[2] == 1][0]

    print("\nSoft refinement delta (iters=1):")
    print(f"  ΔMSE = {soft_ref1[3] - soft_base[3]: .6f} (negative is better)")
    print(f"  ΔR2  = {soft_ref1[4] - soft_base[4]: .4f} (positive is better)")

    print("\nDone.")


if __name__ == "__main__":
    main()
