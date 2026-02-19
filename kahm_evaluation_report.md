# KAHM embeddings: retrieval evaluation on Austrian laws

Generated: 2026-02-19 22:53:26 | script=evaluate_three_embeddings_storylines.py | version=2026-02-19-scientific-q2mb-v2

## Experimental design

This report evaluates Austrian law retrieval with a fixed sentence corpus. KAHM is evaluated strictly as a *query adapter* into a frozen transformer corpus space (Mixedbread).

### Systems compared

- **IDF–SVD**: IDF–SVD query encoder → IDF–SVD corpus index (low-cost baseline).
- **Mixedbread (true)**: transformer query encoder → Mixedbread corpus index (upper baseline).
- **KAHM(query→MB corpus)**: IDF–SVD features → KAHM adapter → Mixedbread corpus index (no transformer on query path).

### Protocol

- Queries: 5000
- Aligned corpus sentences: 10762
- Mixedbread embedding dim: 1024
- Cutoffs k: 1, 5, 10, 20
- Bootstrap: paired nonparametric, n=5000, seed=0

## Retrieval quality

### Micro-average (per query) at k=1
| Method | hit@k | MRR@k (unique laws) | top1 | majority-acc | consensus frac | lift (prior) |
| --- | --- | --- | --- | --- | --- | --- |
| Mixedbread (true) | 0.378 [0.364, 0.391] | 0.378 [0.365, 0.392] | 0.378 [0.365, 0.391] | 0.378 [0.364, 0.391] | 0.378 [0.365, 0.391] | 69.335 [64.417, 74.750] |
| KAHM(query→MB corpus) | 0.403 [0.390, 0.417] | 0.403 [0.389, 0.417] | 0.403 [0.390, 0.417] | 0.403 [0.390, 0.417] | 0.403 [0.389, 0.417] | 72.448 [67.642, 77.466] |
| IDF–SVD | 0.287 [0.274, 0.300] | 0.287 [0.274, 0.299] | 0.287 [0.274, 0.299] | 0.287 [0.274, 0.300] | 0.287 [0.275, 0.299] | 45.492 [41.112, 50.412] |

Δ at k=1 (paired bootstrap, mean differences)
| Comparison | Δhit | ΔMRR_ul | Δtop1 | Δmajority | Δcons_frac | Δlift |
| --- | --- | --- | --- | --- | --- | --- |
| KAHM − IDF | +0.116 [+0.103, +0.130] | +0.116 [+0.103, +0.130] | +0.116 [+0.102, +0.130] | +0.116 [+0.103, +0.130] | +0.116 [+0.103, +0.129] | +26.957 [+20.910, +33.171] |
| KAHM − MB | +0.025 [+0.014, +0.036] | +0.025 [+0.015, +0.036] | +0.025 [+0.014, +0.036] | +0.025 [+0.014, +0.036] | +0.025 [+0.014, +0.036] | +3.113 [-1.177, +7.379] |

### Micro-average (per query) at k=5
| Method | hit@k | MRR@k (unique laws) | top1 | majority-acc | consensus frac | lift (prior) |
| --- | --- | --- | --- | --- | --- | --- |
| Mixedbread (true) | 0.560 [0.546, 0.573] | 0.454 [0.441, 0.466] | 0.378 [0.365, 0.392] | 0.392 [0.378, 0.405] | 0.330 [0.320, 0.340] | 57.062 [53.795, 60.438] |
| KAHM(query→MB corpus) | 0.572 [0.559, 0.586] | 0.474 [0.461, 0.487] | 0.403 [0.390, 0.417] | 0.420 [0.406, 0.434] | 0.355 [0.344, 0.365] | 60.940 [57.804, 64.088] |
| IDF–SVD | 0.433 [0.420, 0.447] | 0.349 [0.337, 0.361] | 0.287 [0.274, 0.299] | 0.302 [0.289, 0.315] | 0.271 [0.261, 0.282] | 38.900 [36.585, 41.255] |

Δ at k=5 (paired bootstrap, mean differences)
| Comparison | Δhit | ΔMRR_ul | Δtop1 | Δmajority | Δcons_frac | Δlift |
| --- | --- | --- | --- | --- | --- | --- |
| KAHM − IDF | +0.139 [+0.125, +0.153] | +0.126 [+0.114, +0.137] | +0.116 [+0.103, +0.130] | +0.117 [+0.104, +0.131] | +0.084 [+0.075, +0.093] | +22.040 [+19.056, +25.166] |
| KAHM − MB | +0.013 [+0.002, +0.023] | +0.021 [+0.012, +0.030] | +0.025 [+0.014, +0.036] | +0.028 [+0.018, +0.039] | +0.025 [+0.018, +0.031] | +3.878 [+1.640, +6.048] |

### Micro-average (per query) at k=10
| Method | hit@k | MRR@k (unique laws) | top1 | majority-acc | consensus frac | lift (prior) |
| --- | --- | --- | --- | --- | --- | --- |
| Mixedbread (true) | 0.625 [0.612, 0.639] | 0.468 [0.456, 0.480] | 0.378 [0.364, 0.392] | 0.402 [0.388, 0.416] | 0.303 [0.293, 0.312] | 49.011 [46.555, 51.580] |
| KAHM(query→MB corpus) | 0.634 [0.620, 0.648] | 0.489 [0.477, 0.501] | 0.403 [0.390, 0.416] | 0.428 [0.414, 0.441] | 0.332 [0.323, 0.342] | 54.709 [52.042, 57.407] |
| IDF–SVD | 0.497 [0.483, 0.511] | 0.364 [0.351, 0.376] | 0.287 [0.275, 0.299] | 0.305 [0.292, 0.318] | 0.256 [0.246, 0.266] | 35.715 [33.796, 37.715] |

Δ at k=10 (paired bootstrap, mean differences)
| Comparison | Δhit | ΔMRR_ul | Δtop1 | Δmajority | Δcons_frac | Δlift |
| --- | --- | --- | --- | --- | --- | --- |
| KAHM − IDF | +0.137 [+0.123, +0.151] | +0.125 [+0.114, +0.136] | +0.116 [+0.102, +0.130] | +0.123 [+0.110, +0.136] | +0.076 [+0.068, +0.084] | +18.994 [+16.648, +21.412] |
| KAHM − MB | +0.009 [-0.002, +0.019] | +0.021 [+0.013, +0.029] | +0.025 [+0.014, +0.036] | +0.026 [+0.015, +0.036] | +0.029 [+0.024, +0.035] | +5.698 [+4.076, +7.182] |

### Micro-average (per query) at k=20
| Method | hit@k | MRR@k (unique laws) | top1 | majority-acc | consensus frac | lift (prior) |
| --- | --- | --- | --- | --- | --- | --- |
| Mixedbread (true) | 0.688 [0.675, 0.701] | 0.478 [0.466, 0.489] | 0.378 [0.364, 0.392] | 0.397 [0.384, 0.411] | 0.270 [0.262, 0.279] | 40.339 [38.558, 42.143] |
| KAHM(query→MB corpus) | 0.689 [0.676, 0.701] | 0.497 [0.485, 0.510] | 0.403 [0.390, 0.417] | 0.420 [0.406, 0.433] | 0.301 [0.292, 0.310] | 46.072 [44.145, 48.073] |
| IDF–SVD | 0.572 [0.558, 0.586] | 0.376 [0.364, 0.387] | 0.287 [0.274, 0.299] | 0.305 [0.292, 0.318] | 0.238 [0.230, 0.248] | 32.062 [30.501, 33.693] |

Δ at k=20 (paired bootstrap, mean differences)
| Comparison | Δhit | ΔMRR_ul | Δtop1 | Δmajority | Δcons_frac | Δlift |
| --- | --- | --- | --- | --- | --- | --- |
| KAHM − IDF | +0.117 [+0.104, +0.130] | +0.121 [+0.110, +0.133] | +0.116 [+0.103, +0.130] | +0.115 [+0.102, +0.127] | +0.063 [+0.056, +0.070] | +14.010 [+12.259, +15.756] |
| KAHM − MB | +0.001 [-0.009, +0.011] | +0.020 [+0.011, +0.028] | +0.025 [+0.015, +0.036] | +0.023 [+0.013, +0.032] | +0.031 [+0.026, +0.035] | +5.734 [+4.675, +6.810] |

## Robustness: macro-average (per law)

Macro averages resample laws (labels) rather than queries; this reduces sensitivity to label imbalance. Reported at k=10.

| Method | hit@k | MRR_ul@k | top1 | majority-acc |
| --- | --- | --- | --- | --- |
| Mixedbread (true) | 0.625 [0.560, 0.685] | 0.468 [0.410, 0.525] | 0.378 [0.324, 0.432] | 0.402 [0.345, 0.461] |
| KAHM(query→MB corpus) | 0.634 [0.570, 0.695] | 0.489 [0.430, 0.547] | 0.403 [0.347, 0.461] | 0.428 [0.368, 0.488] |
| IDF–SVD | 0.497 [0.436, 0.560] | 0.364 [0.312, 0.416] | 0.287 [0.239, 0.336] | 0.305 [0.251, 0.359] |

| Comparison | Δhit | ΔMRR_ul | Δtop1 | Δmajority |
| --- | --- | --- | --- | --- |
| KAHM − IDF | +0.137 [+0.102, +0.173] | +0.125 [+0.097, +0.154] | +0.116 [+0.086, +0.148] | +0.123 [+0.091, +0.155] |
| KAHM − MB | +0.009 [-0.002, +0.020] | +0.021 [+0.010, +0.032] | +0.025 [+0.012, +0.039] | +0.026 [+0.013, +0.038] |

## Compute profile (measured wall-time proxies)

Per-query numbers below are steady-state proxies (one-time initialization and warm-up are reported separately).

### One-time initialization and warm-up (cold-start)

| Component | Wall time |
| --- | --- |
| IDF–SVD pipeline load | 2512.777 ms |
| KAHM init (models + caches) | 6996.325 ms |
| KAHM warm-up | 7328.049 ms |
| Mixedbread model load | 7702.626 ms |
| Mixedbread warm-up encode | 2034.939 ms |

| Path | Query source | Query embed / q | FAISS search / q | Total online / q |
| --- | --- | --- | --- | --- |
| IDF–SVD | model | 0.832 ms | 0.303 ms | 1.136 ms |
| KAHM(q→MB) | model | 140.102 ms | 0.580 ms | 140.682 ms |
| Mixedbread (true) | online | 799.383 ms | 0.580 ms | 799.963 ms |

### Memory footprint proxies

- MB corpus embeddings: 44.1 MB
- IDF corpus embeddings: 22.0 MB

## Majority-vote routing (tau recommendation)

Coverage constraint: coverage ≥ 0.50

| Method | tau* | coverage | acc|covered | majority-acc |
| --- | --- | --- | --- | --- |
| Mixedbread (true) | 0.00 | 1.000 | 0.402 | 0.402 |
| KAHM(query→MB corpus) | 0.00 | 1.000 | 0.428 | 0.428 |
| IDF–SVD | 0.11 | 0.999 | 0.305 | 0.305 |

## Reproducibility

Command-line arguments:

```json
{
  "bootstrap_samples": 5000,
  "bootstrap_seed": 0,
  "corpus_parquet": "ris_sentences.parquet",
  "device": "cpu",
  "drop_empty_queries": true,
  "idf_svd_model": "idf_svd_model.joblib",
  "idf_svd_npz": "embedding_index_idf_svd.npz",
  "k": 10,
  "kahm_batch": 1024,
  "kahm_mode": "soft",
  "kahm_query_embeddings_npz": "",
  "kahm_query_model": "kahm_query_regressors_by_law",
  "kahm_query_strategy": "query_model",
  "kahm_show_progress": true,
  "ks": "1,5,10,20",
  "majority_thresholds": "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8",
  "mb_force_online": true,
  "mb_query_batch": 1,
  "mb_query_npz": "",
  "mb_query_npz_required": false,
  "mb_query_npz_test": "queries_embedding_index_test.npz",
  "mb_query_npz_train": "queries_embedding_index_train.npz",
  "min_routing_coverage": 0.5,
  "mixedbread_model": "mixedbread-ai/deepset-mxbai-embed-de-large-v1",
  "predominance_fraction": 0.1,
  "query_prefix": "query: ",
  "query_set": "query_set.TEST_QUERY_SET",
  "report_overwrite": true,
  "report_path": "kahm_evaluation_report.md",
  "report_show_transformer_context": true,
  "report_show_transformer_deltas": true,
  "report_title": "KAHM embeddings: retrieval evaluation on Austrian laws",
  "results_json_path": "",
  "semantic_npz": "embedding_index.npz",
  "threads": 1,
  "topk_dump_path": ""
}
```
