#!/usr/bin/env python3
"""\
train_kahm_query_regressors_by_law.py

Train *law-specific* KAHM query regressors (IDF–SVD -> Mixedbread) per
`consensus_law`, then evaluate a distance-gated multi-model combination
(using combine_kahm_regressors_generalized.py) on the full TEST_QUERY_SET.

Design goals
------------
- Reuse the exact same pipeline + hyperparameters as train_kahm_query_regressor.py.
- Only law-specific change: if n_clusters > N_train_for_law, clamp to N_train_for_law.
  (Practically, we clamp to the *core* training count after any validation split,
   to avoid KMeans errors.)

Data
----
Reads TRAIN_QUERY_SET and TEST_QUERY_SET from query_set.py. Each query must have:
  - query_id (str)
  - query_text (str)
  - consensus_law (str)

Outputs
-------
- Saves one regressor per law to an output directory.
- Prints per-law metrics (if test samples exist) and overall combined metrics.

Example
-------
python train_kahm_query_regressors_by_law.py \
  --idf_svd_model idf_svd_model.joblib \
  --queries_npz_train queries_embedding_index_train.npz \
  --queries_npz_test  queries_embedding_index_test.npz \
  --out kahm_query_regressors_by_law/

Notes
-----
- This script intentionally does *not* create a single serialized "combined" model.
  Combination is evaluated via distance-gating at inference time.
"""

from __future__ import annotations

import os

# Keep consistent with train_kahm_query_regressor.py
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import hashlib
import re
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Reuse the full training pipeline, metrics, and CLI defaults.
import train_kahm_query_regressor as base

from query_set import TRAIN_QUERY_SET, TEST_QUERY_SET  # type: ignore

from combine_kahm_regressors_generalized import combine_kahm_regressors_distance_gated_multi


def _sanitize_for_path(s: str, *, max_len: int = 64) -> str:
    s0 = str(s).strip()
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", s0)
    cleaned = cleaned.strip("._-")
    if not cleaned:
        cleaned = "unknown"
    if len(cleaned) > max_len:
        h = hashlib.sha1(s0.encode("utf-8")).hexdigest()[:10]
        cleaned = cleaned[: max(1, max_len - 11)] + "_" + h
    return cleaned


def _extract_ids_texts_laws(qs: Sequence[Dict[str, Any]], name: str) -> Tuple[List[str], List[str], List[str]]:
    ids: List[str] = []
    texts: List[str] = []
    laws: List[str] = []
    for i, q in enumerate(qs):
        qid = str(q.get("query_id", "")).strip()
        txt = str(q.get("query_text", "")).strip()
        law = str(q.get("consensus_law", "")).strip()
        if not qid:
            raise ValueError(f"{name}[{i}] has empty query_id")
        if not txt:
            raise ValueError(f"{name}[{i}] has empty query_text")
        if not law:
            raise ValueError(f"{name}[{i}] has empty consensus_law")
        ids.append(qid)
        texts.append(txt)
        laws.append(law)

    if len(set(ids)) != len(ids):
        raise ValueError(f"{name} has duplicate query_id values")

    return ids, texts, laws


def _resolve_out_dir_and_prefix(out_arg: str) -> Tuple[Path, str]:
    p = Path(str(out_arg)).expanduser()
    # If user passes a joblib, treat it as a template base file.
    if p.suffix.lower() == ".joblib":
        out_dir = p.parent if str(p.parent) else Path(".")
        prefix = p.stem
        return out_dir, prefix
    # Otherwise treat as directory.
    return p, "kahm_query_regressor"


def _print_metrics(prefix: str, m: Dict[str, float]) -> None:
    print(prefix)
    print(f"  MSE:               {m['mse']:.6f}")
    print(f"  Overall R^2:       {m['r2_overall']:.4f}")
    print(f"  Cosine mean:       {m['cos_mean']:.4f}")
    print(f"  Cosine p10/p50/p90:{m['cos_p10']:.4f} / {m['cos_p50']:.4f} / {m['cos_p90']:.4f}")
    print(f"  N:                 {int(m['n'])}")


def build_arg_parser() -> argparse.ArgumentParser:
    # Start from the original CLI and keep all parameters identical.
    p = base.build_arg_parser()

    # For this multi-model script, a directory default is more convenient.
    # Users can still pass a *.joblib path to use it as a naming template.
    p.set_defaults(out="kahm_query_regressors_by_law")

    # Interpret --out as either a directory or a base file template.
    p.description = (
        "Train a separate KAHM query regressor per consensus_law (same pipeline as "
        "train_kahm_query_regressor.py), then evaluate a distance-gated combination "
        "across all laws on the full test set."
    )

    p.add_argument(
        "--combined_mode",
        default="soft",
        choices=["soft", "hard"],
        help="Combination inference mode (default: soft).",
    )
    p.add_argument(
        "--combined_batch_size",
        type=int,
        default=2048,
        help="Batch size used by distance-gated combiner (default: 2048).",
    )
    p.add_argument(
        "--no_combined_progress",
        action="store_true",
        help="Disable progress bars during combined evaluation.",
    )

    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    idf_svd_model_path = str(args.idf_svd_model)
    if not Path(idf_svd_model_path).exists():
        raise FileNotFoundError(f"idf_svd_model not found: {idf_svd_model_path}")

    train_qs = list(TRAIN_QUERY_SET)
    test_qs = list(TEST_QUERY_SET)

    train_ids, train_texts, train_laws = _extract_ids_texts_laws(train_qs, "TRAIN_QUERY_SET")
    test_ids, test_texts, test_laws = _extract_ids_texts_laws(test_qs, "TEST_QUERY_SET")

    out_dir, out_prefix = _resolve_out_dir_and_prefix(str(args.out))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optional BlockSafe enablement (reused)
    ctx = base.nullcontext()
    if args.blocksafe and base.enable_otfl_blocksafe is not None:
        ctx = base.enable_otfl_blocksafe(
            backend=str(args.blocksafe_backend),
            jitter_std=float(args.blocksafe_jitter_std),
            jitter_tries=int(args.blocksafe_jitter_tries),
            jitter_growth=float(args.blocksafe_jitter_growth),
            eps_factor=float(args.blocksafe_eps_factor),
            log_first=int(args.blocksafe_log_first),
            l2_normalized=bool(args.blocksafe_l2_normalized),
        )

    # Precompute embeddings once (fast + consistent)
    print(f"Embedding IDF–SVD queries using: {idf_svd_model_path}")
    X_train_all = base.embed_idf_svd_queries(idf_svd_model_path, train_texts)
    X_test_all = base.embed_idf_svd_queries(idf_svd_model_path, test_texts)

    if float(args.input_scale) != 1.0:
        X_train_all = X_train_all * float(args.input_scale)
        X_test_all = X_test_all * float(args.input_scale)

    # Targets (Mixedbread)
    def _resolve_npz(which: str) -> Optional[str]:
        combined = str(args.queries_npz).strip()
        split_path = str(getattr(args, f"queries_npz_{which}", "")).strip()
        if split_path and Path(split_path).exists():
            return split_path
        if combined and Path(combined).exists():
            return combined
        return None

    npz_train = _resolve_npz("train")
    npz_test = _resolve_npz("test")

    if bool(args.force_mb_on_the_fly):
        npz_train = None
        npz_test = None

    use_npz = (npz_train is not None) and (npz_test is not None)

    if use_npz:
        print(f"Loading precomputed Mixedbread TRAIN query embeddings: {npz_train}")
        Y_train_all = base.load_precomputed_mb_queries_npz(str(npz_train), train_ids)
        print(f"Loading precomputed Mixedbread TEST  query embeddings: {npz_test}")
        Y_test_all = base.load_precomputed_mb_queries_npz(str(npz_test), test_ids)
    else:
        if bool(getattr(args, "require_npz", False)):
            missing = []
            if npz_train is None:
                missing.append("train")
            if npz_test is None:
                missing.append("test")
            raise FileNotFoundError(
                "Missing required NPZ targets for: "
                + ", ".join(missing)
                + ". Provide --queries_npz_train/--queries_npz_test (or a combined --queries_npz) or unset --require_npz."
            )
        print("Computing Mixedbread query embeddings on-the-fly (torch required).")
        Y_train_all = base.embed_mb_queries_on_the_fly(
            str(args.mb_model), str(args.mb_device), train_texts, batch_size=int(args.mb_batch)
        )
        Y_test_all = base.embed_mb_queries_on_the_fly(
            str(args.mb_model), str(args.mb_device), test_texts, batch_size=int(args.mb_batch)
        )

    # Group indices by law
    train_idx_by_law: Dict[str, List[int]] = {}
    for i, law in enumerate(train_laws):
        train_idx_by_law.setdefault(law, []).append(i)

    test_idx_by_law: Dict[str, List[int]] = {}
    for i, law in enumerate(test_laws):
        test_idx_by_law.setdefault(law, []).append(i)

    laws = sorted(train_idx_by_law.keys())

    # Warn if test contains unseen laws
    unseen_test_laws = sorted(set(test_idx_by_law.keys()) - set(train_idx_by_law.keys()))
    if unseen_test_laws:
        print("WARNING: test set contains consensus_law values not present in training set.\n"
              f"  Unseen laws (count={len(unseen_test_laws)}): {unseen_test_laws[:20]}" + (" ..." if len(unseen_test_laws) > 20 else ""))

    print(f"\nTraining {len(laws)} law-specific regressors ...")

    models_by_law: Dict[str, dict] = {}
    saved_paths: Dict[str, str] = {}

    # Train each law model
    for law in laws:
        idx_tr = np.asarray(train_idx_by_law[law], dtype=np.int64)
        idx_te = np.asarray(test_idx_by_law.get(law, []), dtype=np.int64)

        X_tr = X_train_all[idx_tr]
        Y_tr = Y_train_all[idx_tr]
        X_te = X_test_all[idx_te] if idx_te.size else None
        Y_te = Y_test_all[idx_te] if idx_te.size else None

        N_train = int(X_tr.shape[0])
        N_test = int(0 if X_te is None else X_te.shape[0])

        print("\n" + "=" * 90)
        print(f"Law: {law} | N_train={N_train} | N_test={N_test}")

        # Build validation split exactly like the base script
        rng = np.random.RandomState(int(args.random_state))
        idx = np.arange(N_train, dtype=np.int64)
        rng.shuffle(idx)

        n_val = 0
        if float(args.val_fraction) > 0:
            n_val = int(round(float(args.val_fraction) * N_train))
            n_val = min(n_val, int(args.val_max_samples))
        n_val = max(0, min(n_val, N_train - 2))  # keep at least 2 for training (base behavior)

        val_idx = idx[:n_val]
        core_idx = idx[n_val:]

        X_val = X_tr[val_idx] if n_val > 0 else None
        Y_val = Y_tr[val_idx] if n_val > 0 else None
        X_core = X_tr[core_idx]
        Y_core = Y_tr[core_idx]

        # KAHM expects (D, N)
        X_train_col = X_core.T
        Y_train_col = Y_core.T

        # Clamp clusters to training samples for this law (effective training samples after val split)
        n_clusters_eff = int(min(int(args.n_clusters), int(X_core.shape[0])))
        n_clusters_eff = max(1, n_clusters_eff)

        if n_clusters_eff != int(args.n_clusters):
            print(f"Clamping n_clusters: requested={int(args.n_clusters)} -> effective={n_clusters_eff} (law train_core N={int(X_core.shape[0])})")

        # Unique IDs / paths
        safe_law = _sanitize_for_path(law)
        out_path = out_dir / f"{out_prefix}__law={safe_law}.joblib"

        base_model_id = str(args.model_id) if args.model_id else out_path.stem
        run_model_id = f"{base_model_id}__law={safe_law}"

        # If user provided an explicit ae_dir, create a per-law subdirectory to avoid collisions.
        ae_dir = None
        if args.ae_dir is not None:
            ae_dir = str(Path(str(args.ae_dir)) / f"law={safe_law}")

        t0 = time.time()
        try:
            with base._as_blocksafe_context(ctx):
                model = base.train_kahm_regressor(
                    X=X_train_col,
                    Y=Y_train_col,
                    n_clusters=n_clusters_eff,
                    subspace_dim=int(args.subspace_dim),
                    Nb=int(args.nb),
                    random_state=int(args.random_state),
                    verbose=True,
                    input_scale=1.0,  # already applied
                    kmeans_kind=str(args.kmeans_kind),
                    kmeans_batch_size=int(args.kmeans_batch_size),
                    max_train_per_cluster=(None if args.max_train_per_cluster is None else int(args.max_train_per_cluster)),
                    model_dtype=str(args.model_dtype),
                    cluster_center_normalization=str(args.cluster_center_normalization),
                    save_ae_to_disk=False,
                    ae_cache_root=str(args.ae_cache_root),
                    ae_dir=ae_dir,
                    overwrite_ae_dir=bool(args.overwrite_ae_dir),
                    model_id=run_model_id,
                    singleton_strategy="augment",
                    singleton_aux_mix=0.1,
                )
        except Exception as exc:
            print(f"ERROR: training failed for law={law}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue

        t_train = time.time() - t0
        try:
            print(f"Trained model_id={model.get('model_id')} | classifier_dir={model.get('classifier_dir')} | time={t_train:.1f}s")
        except Exception:
            print(f"Training time={t_train:.1f}s")

        # Optional soft tuning (unchanged)
        tuning_result = None
        if bool(args.tune_soft):
            alphas = tuple(base.parse_float_list(str(args.soft_alphas)))
            topks = tuple(base.parse_topk_list(str(args.soft_topks)))
            topk_candidates_eff = [k for k in topks if (k is not None and k <= n_clusters_eff)]
            if not topk_candidates_eff:
                topk_candidates_eff = [1]
                print(f"[{law}] WARNING: no valid topk candidates <= n_clusters_eff={n_clusters_eff}; using topk=1 only.")
            if X_val is None or Y_val is None:
                print("WARNING: tune_soft requested, but validation split is empty. Skipping tuning.")
            else:
                print("Tuning soft parameters on validation set...")
                tuning_result = base.tune_soft_params(
                    model,
                    X_val.T,
                    Y_val.T,
                    alphas=alphas,
                    topks=topk_candidates_eff,
                    n_jobs=1,
                    verbose=True,
                )

        # Optional NLMS refinement (unchanged)
        nlms_results = None
        if bool(args.tune_nlms):
            if X_val is None or Y_val is None:
                print("WARNING: tune_nlms requested, but validation split is empty. Skipping NLMS.")
            else:
                print("Refining cluster centers with NLMS...")
                nlms_results = base.tune_cluster_centers_nlms(
                    model,
                    np.hstack([X_val.T, X_train_col]),
                    np.hstack([Y_val.T, Y_train_col]),
                    mu=0.1,
                    epsilon=1,
                    epochs=20,
                    batch_size=1024,
                    shuffle=True,
                    random_state=int(args.random_state),
                    anchor_lambda=0.0,
                    n_jobs=1,
                    preload_classifier=True,
                    verbose=True,
                    alpha=(tuning_result.best_alpha if tuning_result is not None else None),
                    topk=(tuning_result.best_topk if tuning_result is not None else None),
                )

        # Per-law evaluation on its own test subset (optional but helpful)
        metrics_soft = None
        if N_test > 0 and X_te is not None and Y_te is not None:
            X_eval_col = X_te.T
            Y_eval_col = Y_te.T
            if bool(args.eval_soft) or bool(args.tune_soft):
                Y_pred_soft = base.kahm_regress(
                    model,
                    X_eval_col,
                    mode="soft",
                    return_probabilities=False,
                    batch_size=1024,
                )
                metrics_soft = base.compute_embedding_metrics(Y_pred_soft, Y_eval_col)
                _print_metrics("Soft-mode metrics (law test subset):", metrics_soft)
        else:
            print("No test samples for this law; skipping per-law test evaluation.")

        # Save model with minimal-but-useful metadata
        created_at = datetime.now(timezone.utc).isoformat()
        try:
            tuning_payload = asdict(tuning_result) if tuning_result is not None else None
        except Exception:
            tuning_payload = tuning_result

        meta = {
            "created_at_utc": created_at,
            "script": os.path.basename(__file__),
            "consensus_law": law,
            "paths": {
                "idf_svd_model": args.idf_svd_model,
                "queries_npz": (str(args.queries_npz).strip() if (use_npz and str(args.queries_npz).strip()) else None),
                "queries_npz_train": (npz_train if use_npz else None),
                "queries_npz_test": (npz_test if use_npz else None),
                "out": str(out_path),
            },
            "data": {
                "n_train_queries": int(N_train),
                "n_test_queries": int(N_test),
                "n_train_core": int(X_core.shape[0]),
                "n_val": int(0 if X_val is None else X_val.shape[0]),
                "d_in": int(X_tr.shape[1]),
                "d_out": int(Y_tr.shape[1]),
            },
            "hyperparams": {
                "n_clusters_requested": int(args.n_clusters),
                "n_clusters_effective": int(n_clusters_eff),
                "subspace_dim": int(args.subspace_dim),
                "Nb": int(args.nb),
                "random_state": int(args.random_state),
                "input_scale": float(args.input_scale),
                "kmeans_kind": str(args.kmeans_kind),
                "kmeans_batch_size": int(args.kmeans_batch_size),
                "max_train_per_cluster": (None if args.max_train_per_cluster is None else int(args.max_train_per_cluster)),
                "model_dtype": str(args.model_dtype),
                "cluster_center_normalization": str(args.cluster_center_normalization),
                "val_fraction": float(args.val_fraction),
                "val_max_samples": int(args.val_max_samples),
                "eval_soft": bool(args.eval_soft),
                "tune_soft": bool(args.tune_soft),
                "tune_nlms": bool(args.tune_nlms),
                "soft_alphas": list(base.parse_float_list(str(args.soft_alphas))),
                "soft_topks": list(base.parse_topk_list(str(args.soft_topks))),
            },
            "tuning": tuning_payload,
            "nlms": (None if nlms_results is None else str(nlms_results)),
            "metrics": {
                "soft": metrics_soft,
                "eval_split": "law_test_subset",
            },
        }

        try:
            model["meta"] = meta
        except Exception:
            pass

        base.save_kahm_regressor(model, str(out_path))
        print(f"Saved law regressor to: {out_path}")

        models_by_law[law] = model
        saved_paths[law] = str(out_path)

    if not models_by_law:
        raise RuntimeError("No law-specific models were trained successfully.")

    # ----------------------
    # Combined evaluation
    # ----------------------
    print("\n" + "=" * 90)
    print(f"Evaluating combined regressor on full test set (N={int(X_test_all.shape[0])}) ...")

    # If only one model exists, fall back to direct inference
    if len(models_by_law) == 1:
        only_law = next(iter(models_by_law.keys()))
        print(f"Only one trained model (law={only_law}); skipping gating and using that model for all test queries.")
        Y_pred_col = base.kahm_regress(models_by_law[only_law], X_test_all.T, mode=str(args.combined_mode), return_probabilities=False, batch_size=1024)
        chosen_idx = np.zeros((X_test_all.shape[0],), dtype=np.int16)
        names = [only_law]
    else:
        Y_pred_row, chosen_idx, best_score, all_scores, names = combine_kahm_regressors_distance_gated_multi(
            X_test_all,
            models=models_by_law,  # dict preserves mapping to names
            input_layout="row",
            output_layout="row",
            mode=str(args.combined_mode),
            alpha=None,
            topk=None,
            batch_size=int(args.combined_batch_size),
            tie_break="first",
            show_progress=(not bool(args.no_combined_progress)),
            return_all_scores=True,
        )
        Y_pred_col = Y_pred_row.T

    metrics_combined = base.compute_embedding_metrics(Y_pred_col, Y_test_all.T)
    _print_metrics("Combined metrics (full test set):", metrics_combined)

    # Diagnostics: chosen model distribution
    print("\nChosen-model distribution (by gating index):")
    chosen_idx = np.asarray(chosen_idx, dtype=np.int64).reshape(-1)
    # Map idx->name
    idx_to_name = {i: names[i] for i in range(len(names))}
    for i in range(len(names)):
        cnt = int(np.sum(chosen_idx == i))
        print(f"  {i:>3d} | {idx_to_name[i]} | {cnt}")

    # Optional: per-law combined metrics on each law's test subset
    print("\nCombined metrics per-law (restricted to each law's test queries):")
    any_per_law = False
    for law, idxs in sorted(test_idx_by_law.items(), key=lambda kv: kv[0]):
        if not idxs:
            continue
        any_per_law = True
        idxs_np = np.asarray(idxs, dtype=np.int64)
        Y_pred_law = Y_pred_col[:, idxs_np]
        Y_true_law = Y_test_all.T[:, idxs_np]
        m = base.compute_embedding_metrics(Y_pred_law, Y_true_law)
        print(f"- {law} (N={len(idxs)})")
        print(f"    cos_mean={m['cos_mean']:.4f} | mse={m['mse']:.6f} | r2={m['r2_overall']:.4f}")
    if not any_per_law:
        print("  (no per-law test subsets found)")

    print("\nSaved models:")
    for law in sorted(saved_paths.keys()):
        print(f"  {law}: {saved_paths[law]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
