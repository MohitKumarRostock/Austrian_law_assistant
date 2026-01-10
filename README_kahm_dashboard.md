# KAHM Dashboard (Gradio)

This dashboard visualizes your **publication-ready evaluation report** and provides **interactive retrieval** for:

- **KAHM(query→MB corpus)**: KAHM-regressed query embedding searched against the *Mixedbread* corpus embedding index.
- **Full-KAHM**: same KAHM query embedding searched against the *KAHM-transformed* corpus index.

It is designed for **evidence retrieval** where the top results can span multiple law topics (not a single-topic classifier).

## Files

- `kahm_dashboard_gradio.py` — the Gradio dashboard app.

## Install

From your project root (recommended):

```bash
pip install gradio faiss-cpu numpy pandas plotly joblib pyarrow
```

If you are on Apple Silicon and run into FAISS issues, consider installing FAISS via conda or using a CPU-only wheel compatible with your Python version.

## Run

```bash
python kahm_dashboard_gradio.py
```

Open the displayed local URL (usually `http://127.0.0.1:7860`).

## Expected project files (defaults)

The app defaults to the same filenames used by `evaluate_three_embeddings.py`:

- `kahm_evaluation_report.md`
- `ris_sentences.parquet`
- `embedding_index.npz`
- `embedding_index_kahm_mixedbread_approx.npz`
- `idf_svd_model.joblib`
- `kahm_regressor_idf_to_mixedbread.joblib`
- Query set: `query_set.TEST_QUERY_SET`

If your paths differ, set them in the **Configuration** panel.

## Notes on metadata columns

The app tries to display:

- `law_type` (required for a meaningful display)
- page number (any of: `page_no`, `page`, `pageno`, `page_number`, `pdf_page`)
- sentence text (any of: `sentence`, `sentence_text`, `text`, `content`, `passage`, `segment`)

If your parquet uses different column names, either rename the columns or extend the candidate lists in the script.

## Troubleshooting

### 1) `Could not import evaluate_three_embeddings.py`
Ensure `kahm_dashboard_gradio.py` is in the same directory as `evaluate_three_embeddings.py` OR that your project root is on `PYTHONPATH`.

### 2) `faiss is not installed`
Install `faiss-cpu`. If you cannot, you can adapt the code to use brute-force cosine similarity via NumPy (slower).

### 3) Slow first load
The first run builds FAISS indices from the stored embedding matrices; subsequent interactions are cached.
