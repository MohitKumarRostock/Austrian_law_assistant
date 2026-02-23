# KAHM Embeddings Dashboard (Streamlit)

This dashboard has three goals:

1) **Communicate effectiveness** of KAHM (query-adapter → Mixedbread corpus) using the evaluation report:
   - KPI cards (Top-1 / MRR / Hit@k / Lift)
   - Quality curves across k
   - Compute profile and routing table (when present)

2) **Interactive retrieval demo**:
   - User inputs a query
   - Retrieves top-k Austrian-law sentences
   - Shows `sentence_id`, `law_type`, and (if present) a text snippet

3) **Professional query input UX**:
   - Type-ahead examples from TRAIN/TEST query sets
   - Next-word suggestions via a simple trigram model trained on those queries

## Files

- `kahm_dashboard_app.py` — Streamlit app

## Run

```bash
pip install streamlit plotly pandas numpy pyarrow faiss-cpu sentence-transformers joblib
streamlit run kahm_dashboard_app.py
```

## Required artifacts (configure in sidebar)

- `kahm_evaluation_report.md` (the Markdown report)
- `ris_sentences.parquet` with at least columns:
  - `sentence_id` (int)
  - `law_type` (str)
  - ideally one of: `text`, `sentence`, `content`, `paragraph`, `body`
- embeddings bundles (NPZ) containing `emb` and `sentence_ids`:
  - Mixedbread corpus: `embedding_index.npz`
  - IDF–SVD corpus: `embedding_index_idf_svd.npz` (optional)
- models for online query embedding:
  - IDF–SVD pipeline: `idf_svd_model.joblib` (required for IDF and for KAHM)
  - KAHM model dir: `kahm_query_regressors_by_law` (required for KAHM)
  - Mixedbread model name (for transformer query encoding)

## Query autocomplete sources

Choose one:

- **Python module**: `query_set.TRAIN_QUERY_SET` / `query_set.TEST_QUERY_SET`
- **JSONL**: `train.jsonl` / `test.jsonl` (with `query_text` fields)

