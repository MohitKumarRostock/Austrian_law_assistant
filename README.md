# Austrian Law Assistant: KAHM-Based Compute-Efficient Encoders for Legal Retrieval

This repository contains the code and artifacts used for the experiments in the paper **“Geometric Kernel Machines as Compute-Efficient Encoders”**.

The main experimental question is whether a **KAHM-based query encoder** can replace online transformer query inference by mapping inexpensive **IDF–SVD lexical query features** into a high-quality semantic embedding space while keeping a strong transformer corpus index fixed.

## Repository contents

The repository currently includes:

- corpus and query artifacts used in the paper:
  - `ris_sentences.parquet`
  - `train.jsonl`, `test.jsonl`, `meta.json`
  - `embedding_index.npz`
  - `embedding_index_idf_svd.npz`
  - `queries_embedding_index_train.npz`
  - `queries_embedding_index_test.npz`
  - `kahm_query_regressors_by_law/`
  - `kahm_evaluation_report.md`
- model-building and training scripts:
  - `build_embedding_index_npz.py`
  - `build_embedding_index_idf_svd_npz_mxbai_fixed_vocab.py`
  - `train_kahm_query_regressors_by_law.py`
  - `kahm_regression.py`
  - `combine_kahm_regressors_generalized.py`
- evaluation and figure-generation scripts:
  - `evaluate_three_embeddings_storylines.py`
  - `generate_kahm_result_figures_matlab.m`
- corpus-construction utilities:
  - `download_ris_pdfs_all_laws.py`
  - `extract_sentences_from_German_pdfs.py`
  - `generate_query_set_austrian_law.py`

## Reproducibility quick start

### Option A: reproduce the reported evaluation from the included artifacts

Create an environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Then run the evaluation script on the included benchmark artifacts:

```bash
python evaluate_three_embeddings_storylines.py   --ks 3,5,10,15,20   --report_path kahm_evaluation_report_reproduced.md
```

This reproduces the retrieval evaluation from the stored corpus embeddings, lexical embeddings, query embeddings, and law-specific KAHM regressors already present in the repository.

### Option B: regenerate the figures used in the paper

From the generated or included evaluation report:

```matlab
generate_kahm_result_figures_matlab('kahm_evaluation_report.md', 'kahm_figures_matlab')
```

This writes publication figures to `kahm_figures_matlab/`.

### Option C: retrain the law-specific KAHM query regressors

```bash
python train_kahm_query_regressors_by_law.py   --idf_svd_model idf_svd_model.joblib   --queries_npz_train queries_embedding_index_train.npz   --queries_npz_test queries_embedding_index_test.npz   --out kahm_query_regressors_by_law
```

### Option D: rebuild the lexical IDF–SVD embedding index

```bash
python build_embedding_index_idf_svd_npz_mxbai_fixed_vocab.py   --corpus ris_sentences.parquet   --out_npz embedding_index_idf_svd.npz   --out_model idf_svd_model.joblib   --no_char_ngrams   --dim 512
```

## End-to-end pipeline (from raw legal texts)

If you want to rebuild the dataset and all downstream artifacts from raw RIS documents, the high-level pipeline is:

1. Download the RIS law PDFs:
   ```bash
   python download_ris_pdfs_all_laws.py --out_dir ris_pdfs
   ```

2. Extract passage-level retrieval units:
   ```bash
   python extract_sentences_from_German_pdfs.py --help
   ```

3. Build the semantic corpus index:
   ```bash
   python build_embedding_index_npz.py --help
   ```

4. Build the lexical IDF–SVD index:
   ```bash
   python build_embedding_index_idf_svd_npz_mxbai_fixed_vocab.py --help
   ```

5. Generate the synthetic query benchmark:
   ```bash
   python generate_query_set_austrian_law.py --help
   ```

6. Build the query embedding bundles:
   ```bash
   python build_query_embedding_index_npz.py --help
   ```

7. Train the law-specific KAHM regressors:
   ```bash
   python train_kahm_query_regressors_by_law.py --help
   ```

8. Run the evaluation:
   ```bash
   python evaluate_three_embeddings_storylines.py --help
   ```

## Determinism and reported environment

The evaluation report records the hardware/software setup used for the reported results, including the Python runtime, FAISS, PyTorch, SentenceTransformers, scikit-learn, NumPy, pandas, and the bootstrap configuration. The repository also stores the generated `kahm_evaluation_report.md`, which serves as the direct source for the result tables and figures in the paper.

To maximize reproducibility:

- keep the random seeds and default hyperparameters unchanged,
- use the included artifact files when reproducing the reported tables,
- and prefer a fixed tagged release for paper-linked reruns.

## Notes on licensing

Unless otherwise noted, the **source code** in this repository is made available under the Apache License 2.0.

Pretrained models, downloaded legal texts, and third-party resources may remain subject to their own original terms. If you redistribute the repository or publish a release archive, review the licensing status of included data files, pretrained models, and derived artifacts.


