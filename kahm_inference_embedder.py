#!/usr/bin/env python3
"""
kahm_inference_embedder.py

Fast, exact KAHM query embedding extraction for inference.

Key speed-ups (no approximation, identical outputs):
1) Load IDF–SVD pipeline once.
2) Load all KAHM regressors once.
3) Materialize disk-backed autoencoders into RAM (optional, default ON).
4) Use the optimized distance-gated combiner (combine_kahm_regressors_generalized_fast.py).

You can either import KahmQueryEmbedder in your inference code, or run as a CLI.

CLI example:
  python kahm_inference_embedder.py \
    --idf_svd_model idf_svd_model.joblib \
    --kahm_query_model_dir kahm_query_regressors_by_law \
    --input_txt queries.txt \
    --output_npz query_kahm_embeddings.npz
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import joblib

def l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, eps).astype(np.float32)
    return X / n

def load_kahm_model(path: str) -> dict:
    from kahm_regression import load_kahm_regressor
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return load_kahm_regressor(path)

def load_kahm_models_from_dir(dir_path: str) -> Dict[str, dict]:
    d = Path(str(dir_path)).expanduser()
    if not d.exists() or not d.is_dir():
        raise FileNotFoundError(dir_path)
    paths = sorted(d.glob("*.joblib"))
    if not paths:
        raise FileNotFoundError(f"No *.joblib models found in {dir_path}")
    models: Dict[str, dict] = {}
    for fp in paths:
        models[fp.stem] = load_kahm_model(str(fp))
    return models

class KahmQueryEmbedder:
    """
    Stateful embedder for *exact* KAHM query embeddings.

    Intended usage: construct once at process startup, then call .embed(...)
    repeatedly for inference batches.
    """
    def __init__(
        self,
        *,
        idf_svd_model_path: str,
        kahm_query_model_dir: str,
        kahm_mode: str = "soft",
        batch_size: int = 2048,
        materialize_classifier: bool = True,
        cache_cluster_centers: bool = True,
        tie_break: str = "first",
        show_progress: bool = False,
    ) -> None:
        self.idf_pipe = joblib.load(idf_svd_model_path)
        self.models = load_kahm_models_from_dir(kahm_query_model_dir)
        self.kahm_mode = str(kahm_mode)
        self.batch_size = int(batch_size)
        self.tie_break = str(tie_break)
        self.show_progress = bool(show_progress)

        try:
            from combine_kahm_regressors_generalized_fast import (
                combine_kahm_regressors_distance_gated_multi,
                prepare_kahm_model_for_inference,
            )
            self._combine = combine_kahm_regressors_distance_gated_multi
            if materialize_classifier or cache_cluster_centers:
                for m in self.models.values():
                    prepare_kahm_model_for_inference(
                        m,
                        materialize_classifier=bool(materialize_classifier),
                        cache_cluster_centers=bool(cache_cluster_centers),
                        show_progress=False,
                    )
        except Exception:
            from combine_kahm_regressors_generalized import combine_kahm_regressors_distance_gated_multi
            self._combine = combine_kahm_regressors_distance_gated_multi

    def embed(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        if not texts:
            raise ValueError("texts is empty")

        X = self.idf_pipe.transform(texts)
        X = l2_normalize_rows(np.asarray(X, dtype=np.float32))

        Y, chosen, best_score, _all_scores, names = self._combine(
            X,
            models=self.models,
            input_layout="row",
            output_layout="row",
            mode=self.kahm_mode,
            batch_size=self.batch_size,
            tie_break=self.tie_break,
            show_progress=self.show_progress,
            return_all_scores=False,
        )
        Y = l2_normalize_rows(np.asarray(Y, dtype=np.float32))
        return Y, np.asarray(chosen), np.asarray(best_score, dtype=np.float32), list(names)

def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]
    return [x for x in lines if x.strip()]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--idf_svd_model", required=True)
    ap.add_argument("--kahm_query_model_dir", required=True)
    ap.add_argument("--input_txt", required=True, help="One query per line (UTF-8).")
    ap.add_argument("--output_npz", required=True)
    ap.add_argument("--kahm_mode", default="soft", choices=["soft", "hard"])
    ap.add_argument("--kahm_batch", type=int, default=2048)
    ap.add_argument("--no_materialize_classifier", action="store_true", help="Disable preloading AEs into RAM.")
    args = ap.parse_args()

    texts = _read_lines(args.input_txt)
    emb = KahmQueryEmbedder(
        idf_svd_model_path=args.idf_svd_model,
        kahm_query_model_dir=args.kahm_query_model_dir,
        kahm_mode=args.kahm_mode,
        batch_size=args.kahm_batch,
        materialize_classifier=(not args.no_materialize_classifier),
        cache_cluster_centers=True,
        show_progress=False,
    )
    Y, chosen, best_score, names = emb.embed(texts)

    out = Path(args.output_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        embeddings=Y,
        chosen_model=np.asarray(chosen, dtype=np.int16),
        min_distance=np.asarray(best_score, dtype=np.float32),
        model_names=np.asarray(list(names), dtype=object),
        texts=np.asarray(texts, dtype=object),
    )
    print(f"Saved: {out}  embeddings={Y.shape}")

if __name__ == "__main__":
    main()
