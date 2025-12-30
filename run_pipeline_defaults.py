#!/usr/bin/env python3
"""
run_pipeline_defaults.py

Runs the full pipeline with default arguments:
  1) train_kahm_embedding_regressor.py
  2) precompute_kahm_corpus_npz.py
  3) evaluate_three_embeddings.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print("\n" + "=" * 120)
    print("RUN:", " ".join(cmd))
    print("=" * 120, flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> int:
    project_dir = Path(__file__).resolve().parent
    py = sys.executable  # ensures we use the current venv python

    scripts = [
        "extract_sentences_from_German_pdfs.py",
        "build_embedding_index_npz.py",
        "build_query_embedding_index_npz.py",
        "build_embedding_index_idf_svd_npz.py",
        "train_kahm_embedding_regressor.py",
        "precompute_kahm_corpus_npz.py",
        "evaluate_three_embeddings.py",
    ]

    for s in scripts:
        p = project_dir / s
        if not p.exists():
            raise FileNotFoundError(f"Missing script: {p}")
    run([py, str(project_dir / scripts[0])], cwd=project_dir)
    run([py, str(project_dir / scripts[1])], cwd=project_dir)
    run([py, str(project_dir / scripts[2])], cwd=project_dir)
    run([py, str(project_dir / scripts[3])], cwd=project_dir)
    run([py, str(project_dir / scripts[4])], cwd=project_dir)
    run([py, str(project_dir / scripts[5])], cwd=project_dir)
    run([py, str(project_dir / scripts[6])], cwd=project_dir)
    print("\nPipeline finished successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
