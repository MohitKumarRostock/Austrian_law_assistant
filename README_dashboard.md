# KAHM Customer Dashboard (Streamlit)

This dashboard turns the generated Markdown report into a customer-facing, graphical pitch **and** includes a live retrieval demo.

## Quick start

```bash
pip install -r requirements_dashboard.txt
streamlit run kahm_dashboard_app.py
```

## What you need on disk (same defaults as the evaluation scripts)

- `kahm_evaluation_report.md` (the generated report)
- `ris_sentences.parquet` (corpus metadata; must include `sentence_id`, `law_type`; ideally includes a sentence text column like `text` or `sentence`)
- `embedding_index.npz` (Mixedbread corpus embeddings; must include keys `sentence_ids` and `embeddings`)
- `idf_svd_model.joblib`
- `kahm_query_regressors_by_law/` (directory with `*.joblib` query models)

## Live demo dependencies

The live demo uses `KahmQueryEmbedder` from `kahm_inference_embedder.py`. That file depends on additional modules (e.g. `kahm_regression`,
`combine_kahm_regressors_generalized_fast`), so make sure your project environment includes them (same environment you used for evaluation).

## Query suggestions

If you want typeahead suggestions from the train/test queries:
- Place `train.jsonl` and `test.jsonl` next to `query_set.py`, **or**
- Set `AUSTLAW_QUERYSET_DIR=/path/to/queryset_dir` where that directory contains `train.jsonl` and `test.jsonl`, **or**
- Use the other environment variables described in `query_set.py`.

