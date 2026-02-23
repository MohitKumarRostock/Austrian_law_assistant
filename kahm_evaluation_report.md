# KAHM embeddings: retrieval evaluation on Austrian laws

Generated: 2026-02-23 13:07:46 | script=evaluate_three_embeddings_storylines.py | version=2026-02-23-scientific-pubreport-v1

## Summary

This evaluation compares three retrieval pipelines for mapping natural-language queries to Austrian-law labels via sentence-level retrieval on a fixed corpus:

- **IDF–SVD:** IDF–SVD query embeddings → IDF–SVD corpus embeddings.
- **Mixedbread (true) (reference):** transformer query embeddings → transformer corpus embeddings.
- **KAHM(query→MB corpus):** gradient-free query adapter (IDF–SVD features mapped into the transformer embedding space) → frozen transformer corpus embeddings.

Uncertainty is quantified with a paired nonparametric bootstrap across queries (5000 resamples; seed=0).

## Data and provenance

### Corpus

- Corpus file: `ris_sentences.parquet`
- Aligned sentences (intersection of embedding indices): **10762**
- Embedding space dimension (transformer index): **1024**
- Label universe size (laws present in aligned corpus): **84**

Top-10 corpus law priors (count and prior probability):

| Law | Count | Prior |
| --- | --- | --- |
| ASVG | 1394 | 0.130 |
| GewO | 560 | 0.052 |
| BWG | 528 | 0.049 |
| StPO | 421 | 0.039 |
| ABGB | 405 | 0.038 |
| DSGVO | 389 | 0.036 |
| AWG | 367 | 0.034 |
| EO | 334 | 0.031 |
| ZPO | 316 | 0.029 |
| UG | 285 | 0.026 |

### Queries

- Evaluated query set: `query_set.TEST_QUERY_SET`
- TRAIN query set (diagnostics only): `query_set.TRAIN_QUERY_SET`
- Evaluated queries after filtering: **5000**
- Evaluated cutoffs: **k = 3, 5, 10, 15, 20**

Test query-set composition (after filtering):

- Unique topic IDs: **3228**
- Unique query texts: **5000** (duplicates=0)

| Style | Count | Frac |
| --- | --- | --- |
| authority | 723 | 0.145 |
| keyword | 719 | 0.144 |
| nl_long | 717 | 0.143 |
| nl_short | 717 | 0.143 |
| procedural | 717 | 0.143 |
| scenario | 706 | 0.141 |
| fragment | 701 | 0.140 |

### Synthetic query generation (metadata)

- Metadata source: `file:/Users/mohit/Pythonprojects/Austrian_law_assistant/meta.json`
- seed: **19**
- split_mode: **iid**
- train_n: **40000**
- test_n: **5000**
- n_laws: **84**
- variants_per_style: **3**
- queries_per_topic: **21**
- candidate_oversupply: **2.0**
- law_mention_prob: **0.12**
- keyword_law_mention_prob: **0.25**
- surface_noise_prob: **0.06**
- law_context_prob: **0.65**
- topic_term_prob: **0.3**
- issue_term_prob: **0.35**
- keyword_term_prob: **0.35**
- test_topics_subset_of_train: **True**

Split semantics (from the generator):
- `iid` (default): TRAIN/TEST are stratified; TEST draws only from topics seen in TRAIN (per-law).
- `iid_unrestricted`: TRAIN/TEST are stratified partitions of a shared topic pool (topics may be unseen in TRAIN).
- `topic_disjoint`: no topic appears in both splits (hardest generalization).

### Split hygiene diagnostics

- Exact-text overlap (TRAIN ∩ TEST): **0** queries
- Topic overlap (TRAIN ∩ TEST): **3228** topics
- Topic overlap fraction of TEST: **1.000**

### Label-leakage diagnostics (test)

Boundary match rule: `(?<!\w)LABEL(?!\w) (case-insensitive)`. These diagnostics estimate how often law abbreviations appear verbatim in query text.

- P(any law label mentioned): **0.160**
- P(gold law label mentioned): **0.156**
- P(other (non-gold) label mentioned): **0.005**

## Retrieval protocol

All embeddings are L2-normalized and indexed with FAISS `IndexFlatIP` (inner product on normalized vectors, i.e., cosine similarity). For each query, we retrieve the top-*k* sentences and aggregate their law labels to compute metrics.

Majority-vote predominance threshold for majority-accuracy: **τ = 0.10**.

## Metrics

All metrics are computed **per query** at cutoff *k* and then averaged across queries. We report 95% confidence intervals via paired bootstrap.

- **Hit@k:** 1 if at least one retrieved sentence is labeled with the gold law, else 0.
- **MRR@k (unique laws):** reciprocal rank of the first occurrence of the gold law when the top-*k* list is collapsed to unique laws.
- **Top-1 accuracy:** 1 if the top-ranked sentence law equals the gold law, else 0.
- **Majority-accuracy:** 1 if the plurality law in top-*k* equals gold **and** its fraction ≥ τ; otherwise 0 (abstentions count as 0).
- **Mean consensus fraction:** fraction of the top-*k* sentences that belong to the gold law.
- **Mean lift (prior):** consensus fraction divided by the corpus prior of the gold law (enrichment over chance).

## Results

### Micro-averaged quality (mean ± 95% CI)

**MRR@k (unique laws)**

| k | IDF–SVD | KAHM(query→MB corpus) | Mixedbread (true) |
| --- | --- | --- | --- |
| 3 | 0.334 [0.322, 0.347] | 0.460 [0.447, 0.472] | 0.436 [0.423, 0.449] |
| 5 | 0.349 [0.337, 0.361] | 0.474 [0.461, 0.487] | 0.454 [0.441, 0.466] |
| 10 | 0.364 [0.351, 0.376] | 0.489 [0.477, 0.501] | 0.468 [0.456, 0.480] |
| 15 | 0.371 [0.359, 0.383] | 0.495 [0.483, 0.507] | 0.474 [0.462, 0.486] |
| 20 | 0.376 [0.364, 0.387] | 0.497 [0.485, 0.510] | 0.478 [0.466, 0.489] |

**Hit@k**

| k | IDF–SVD | KAHM(query→MB corpus) | Mixedbread (true) |
| --- | --- | --- | --- |
| 3 | 0.389 [0.375, 0.402] | 0.525 [0.511, 0.538] | 0.503 [0.488, 0.516] |
| 5 | 0.433 [0.420, 0.447] | 0.572 [0.559, 0.586] | 0.560 [0.546, 0.573] |
| 10 | 0.497 [0.483, 0.511] | 0.634 [0.620, 0.648] | 0.625 [0.612, 0.639] |
| 15 | 0.539 [0.525, 0.553] | 0.672 [0.658, 0.684] | 0.662 [0.649, 0.676] |
| 20 | 0.572 [0.558, 0.586] | 0.689 [0.676, 0.701] | 0.688 [0.675, 0.701] |

**Top-1 accuracy**

| k | IDF–SVD | KAHM(query→MB corpus) | Mixedbread (true) |
| --- | --- | --- | --- |
| 3 | 0.287 [0.274, 0.299] | 0.403 [0.390, 0.417] | 0.378 [0.365, 0.392] |
| 5 | 0.287 [0.274, 0.299] | 0.403 [0.390, 0.417] | 0.378 [0.365, 0.392] |
| 10 | 0.287 [0.275, 0.299] | 0.403 [0.390, 0.416] | 0.378 [0.364, 0.392] |
| 15 | 0.287 [0.274, 0.299] | 0.403 [0.390, 0.417] | 0.378 [0.365, 0.392] |
| 20 | 0.287 [0.274, 0.299] | 0.403 [0.390, 0.417] | 0.378 [0.364, 0.392] |

**Majority-accuracy** (τ=0.10)

| k | IDF–SVD | KAHM(query→MB corpus) | Mixedbread (true) |
| --- | --- | --- | --- |
| 3 | 0.298 [0.286, 0.311] | 0.414 [0.400, 0.427] | 0.379 [0.366, 0.393] |
| 5 | 0.302 [0.289, 0.315] | 0.420 [0.406, 0.434] | 0.392 [0.378, 0.405] |
| 10 | 0.305 [0.292, 0.318] | 0.428 [0.414, 0.441] | 0.402 [0.388, 0.416] |
| 15 | 0.311 [0.298, 0.324] | 0.423 [0.409, 0.437] | 0.400 [0.387, 0.414] |
| 20 | 0.305 [0.292, 0.318] | 0.420 [0.406, 0.433] | 0.397 [0.384, 0.411] |

**Mean consensus fraction**

| k | IDF–SVD | KAHM(query→MB corpus) | Mixedbread (true) |
| --- | --- | --- | --- |
| 3 | 0.279 [0.268, 0.290] | 0.374 [0.362, 0.385] | 0.345 [0.334, 0.356] |
| 5 | 0.271 [0.261, 0.282] | 0.355 [0.344, 0.365] | 0.330 [0.320, 0.340] |
| 10 | 0.256 [0.246, 0.266] | 0.332 [0.323, 0.342] | 0.303 [0.293, 0.312] |
| 15 | 0.247 [0.238, 0.256] | 0.315 [0.306, 0.324] | 0.285 [0.277, 0.294] |
| 20 | 0.238 [0.230, 0.248] | 0.301 [0.292, 0.310] | 0.270 [0.262, 0.279] |

**Mean lift (prior)**

| k | IDF–SVD | KAHM(query→MB corpus) | Mixedbread (true) |
| --- | --- | --- | --- |
| 3 | 42.062 [39.219, 45.081] | 65.820 [62.222, 69.582] | 60.549 [56.883, 64.307] |
| 5 | 38.900 [36.585, 41.255] | 60.940 [57.804, 64.088] | 57.062 [53.795, 60.438] |
| 10 | 35.715 [33.796, 37.715] | 54.709 [52.042, 57.407] | 49.011 [46.555, 51.580] |
| 15 | 33.889 [32.142, 35.688] | 49.923 [47.704, 52.133] | 44.308 [42.225, 46.400] |
| 20 | 32.062 [30.501, 33.693] | 46.072 [44.145, 48.073] | 40.339 [38.558, 42.143] |

### Paired deltas (KAHM − IDF–SVD)

| k | Δhit@k | ΔMRR@k | ΔTop-1 | ΔMajority-acc | ΔMean cons frac | ΔMean lift |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | +0.136 [+0.123, +0.151] | +0.125 [+0.113, +0.137] | +0.116 [+0.103, +0.130] | +0.115 [+0.102, +0.129] | +0.095 [+0.085, +0.105] | +23.758 [+19.900, +27.636] |
| 5 | +0.139 [+0.125, +0.153] | +0.126 [+0.114, +0.137] | +0.116 [+0.103, +0.130] | +0.117 [+0.104, +0.131] | +0.084 [+0.075, +0.093] | +22.040 [+19.056, +25.166] |
| 10 | +0.137 [+0.123, +0.151] | +0.125 [+0.114, +0.136] | +0.116 [+0.102, +0.130] | +0.123 [+0.110, +0.136] | +0.076 [+0.068, +0.084] | +18.994 [+16.648, +21.412] |
| 15 | +0.133 [+0.120, +0.147] | +0.124 [+0.113, +0.135] | +0.116 [+0.103, +0.130] | +0.112 [+0.099, +0.125] | +0.068 [+0.061, +0.075] | +16.034 [+14.027, +18.055] |
| 20 | +0.117 [+0.104, +0.130] | +0.121 [+0.110, +0.133] | +0.116 [+0.103, +0.130] | +0.115 [+0.102, +0.127] | +0.063 [+0.056, +0.070] | +14.010 [+12.259, +15.756] |

### Paired deltas vs transformer-query baseline (context; KAHM − Mixedbread)

| k | Δhit@k | ΔMRR@k | ΔTop-1 | ΔMajority-acc | ΔMean cons frac | ΔMean lift |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | +0.022 [+0.011, +0.033] | +0.024 [+0.015, +0.033] | +0.025 [+0.014, +0.036] | +0.034 [+0.023, +0.045] | +0.029 [+0.022, +0.036] | +5.271 [+2.654, +7.915] |
| 5 | +0.013 [+0.002, +0.023] | +0.021 [+0.012, +0.030] | +0.025 [+0.014, +0.036] | +0.028 [+0.018, +0.039] | +0.025 [+0.018, +0.031] | +3.878 [+1.640, +6.048] |
| 10 | +0.009 [-0.002, +0.019] | +0.021 [+0.013, +0.029] | +0.025 [+0.014, +0.036] | +0.026 [+0.015, +0.036] | +0.029 [+0.024, +0.035] | +5.698 [+4.076, +7.182] |
| 15 | +0.009 [+0.000, +0.019] | +0.021 [+0.013, +0.029] | +0.025 [+0.014, +0.036] | +0.023 [+0.013, +0.034] | +0.030 [+0.025, +0.034] | +5.615 [+4.364, +6.853] |
| 20 | +0.001 [-0.009, +0.011] | +0.020 [+0.011, +0.028] | +0.025 [+0.015, +0.036] | +0.023 [+0.013, +0.032] | +0.031 [+0.026, +0.035] | +5.734 [+4.675, +6.810] |

### Macro-averaged quality (per-law average; robustness)

Macro-averaging computes metrics per law and then averages across laws (each law has equal weight). This is a robustness check against label-frequency skew.

**Macro MRR@k (unique laws)**

| k | IDF–SVD | KAHM(query→MB corpus) | Mixedbread (true) |
| --- | --- | --- | --- |
| 3 | 0.334 [0.284, 0.385] | 0.460 [0.400, 0.520] | 0.436 [0.382, 0.491] |
| 5 | 0.349 [0.297, 0.402] | 0.475 [0.416, 0.531] | 0.454 [0.400, 0.511] |
| 10 | 0.364 [0.312, 0.416] | 0.489 [0.430, 0.547] | 0.468 [0.410, 0.525] |
| 15 | 0.371 [0.319, 0.425] | 0.495 [0.438, 0.553] | 0.474 [0.417, 0.530] |
| 20 | 0.376 [0.326, 0.428] | 0.497 [0.439, 0.555] | 0.478 [0.422, 0.534] |

**Macro Hit@k**

| k | IDF–SVD | KAHM(query→MB corpus) | Mixedbread (true) |
| --- | --- | --- | --- |
| 3 | 0.389 [0.333, 0.446] | 0.525 [0.462, 0.585] | 0.503 [0.443, 0.561] |
| 5 | 0.434 [0.375, 0.492] | 0.573 [0.511, 0.632] | 0.560 [0.496, 0.621] |
| 10 | 0.497 [0.436, 0.560] | 0.634 [0.570, 0.695] | 0.625 [0.560, 0.685] |
| 15 | 0.539 [0.478, 0.598] | 0.672 [0.606, 0.734] | 0.662 [0.594, 0.725] |
| 20 | 0.572 [0.511, 0.631] | 0.689 [0.621, 0.751] | 0.688 [0.619, 0.751] |

**Macro Top-1 accuracy**

| k | IDF–SVD | KAHM(query→MB corpus) | Mixedbread (true) |
| --- | --- | --- | --- |
| 3 | 0.287 [0.241, 0.336] | 0.403 [0.345, 0.461] | 0.378 [0.323, 0.433] |
| 5 | 0.287 [0.239, 0.336] | 0.403 [0.346, 0.461] | 0.378 [0.324, 0.430] |
| 10 | 0.287 [0.239, 0.336] | 0.403 [0.347, 0.461] | 0.378 [0.324, 0.432] |
| 15 | 0.287 [0.239, 0.338] | 0.403 [0.346, 0.460] | 0.378 [0.324, 0.430] |
| 20 | 0.287 [0.240, 0.336] | 0.403 [0.347, 0.460] | 0.378 [0.325, 0.433] |

**Macro Majority-accuracy** (τ=0.10)

| k | IDF–SVD | KAHM(query→MB corpus) | Mixedbread (true) |
| --- | --- | --- | --- |
| 3 | 0.298 [0.249, 0.350] | 0.414 [0.356, 0.473] | 0.379 [0.323, 0.437] |
| 5 | 0.303 [0.251, 0.356] | 0.420 [0.362, 0.479] | 0.392 [0.336, 0.450] |
| 10 | 0.305 [0.251, 0.359] | 0.428 [0.368, 0.488] | 0.402 [0.345, 0.461] |
| 15 | 0.311 [0.255, 0.367] | 0.423 [0.361, 0.484] | 0.400 [0.340, 0.461] |
| 20 | 0.305 [0.250, 0.364] | 0.420 [0.357, 0.486] | 0.397 [0.336, 0.460] |

### Macro paired deltas (KAHM − IDF–SVD)

| k | Δhit@k | ΔMRR@k | ΔTop-1 | ΔMajority-acc | ΔMean cons frac | ΔMean lift |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | +0.136 [+0.105, +0.167] | +0.125 [+0.097, +0.155] | +0.116 [+0.087, +0.147] | +0.115 [+0.085, +0.147] | +0.095 [+0.068, +0.124] | +23.788 [+11.338, +37.146] |
| 5 | +0.139 [+0.107, +0.171] | +0.126 [+0.098, +0.155] | +0.116 [+0.087, +0.147] | +0.117 [+0.086, +0.149] | +0.084 [+0.058, +0.110] | +22.076 [+11.553, +33.578] |
| 10 | +0.137 [+0.102, +0.173] | +0.125 [+0.097, +0.154] | +0.116 [+0.086, +0.148] | +0.123 [+0.091, +0.155] | +0.076 [+0.052, +0.101] | +19.026 [+10.166, +28.245] |
| 15 | +0.133 [+0.096, +0.171] | +0.124 [+0.097, +0.153] | +0.116 [+0.086, +0.146] | +0.112 [+0.081, +0.144] | +0.068 [+0.045, +0.092] | +16.055 [+8.412, +24.426] |
| 20 | +0.117 [+0.081, +0.153] | +0.121 [+0.094, +0.151] | +0.116 [+0.086, +0.147] | +0.115 [+0.085, +0.148] | +0.063 [+0.041, +0.086] | +14.026 [+6.874, +21.526] |

### Macro paired deltas vs transformer-query baseline (context; KAHM − Mixedbread)

| k | Δhit@k | ΔMRR@k | ΔTop-1 | ΔMajority-acc | ΔMean cons frac | ΔMean lift |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | +0.022 [+0.009, +0.035] | +0.024 [+0.012, +0.036] | +0.025 [+0.012, +0.039] | +0.034 [+0.020, +0.050] | +0.029 [+0.019, +0.039] | +5.277 [+1.918, +8.928] |
| 5 | +0.013 [+0.002, +0.024] | +0.021 [+0.010, +0.032] | +0.025 [+0.012, +0.039] | +0.028 [+0.016, +0.041] | +0.025 [+0.016, +0.032] | +3.881 [+0.787, +7.052] |
| 10 | +0.009 [-0.002, +0.020] | +0.021 [+0.010, +0.032] | +0.025 [+0.012, +0.039] | +0.026 [+0.013, +0.038] | +0.030 [+0.023, +0.036] | +5.706 [+3.343, +8.205] |
| 15 | +0.009 [-0.003, +0.023] | +0.021 [+0.011, +0.032] | +0.025 [+0.013, +0.039] | +0.023 [+0.011, +0.036] | +0.030 [+0.023, +0.036] | +5.619 [+3.658, +7.641] |
| 20 | +0.001 [-0.011, +0.013] | +0.020 [+0.009, +0.030] | +0.025 [+0.012, +0.039] | +0.023 [+0.010, +0.037] | +0.031 [+0.024, +0.037] | +5.738 [+3.896, +7.772] |

## Majority-vote routing (coverage/precision)

We report a coverage–precision sweep over routing thresholds τ′ (distinct from the predominance threshold used in the majority metric). Coverage is the fraction of queries that meet τ′; precision is accuracy conditioned on being covered.

Recommended τ′ maximizes precision subject to coverage ≥ **0.50**.

| Method | τ′ | Coverage | Majority-acc | Precision (acc|covered) |
| --- | --- | --- | --- | --- |
| IDF–SVD | 0.41 | 0.538 | 0.251 | 0.468 |
| Mixedbread (true) | 0.41 | 0.525 | 0.308 | 0.587 |
| KAHM(query→MB corpus) | 0.41 | 0.555 | 0.339 | 0.611 |

## Reproducibility

- Bootstrap: B=5000, seed=0
- Thread cap: 1 (0 means no override)

### Software / environment

- Python: `3.11.14`
- Platform: `macOS-26.3-x86_64-i386-64bit`
- numpy: `1.26.4`
- pandas: `2.3.3`
- faiss-cpu: `1.13.2`
- torch: `2.2.2`
- sentence-transformers: `5.2.0`
- scikit-learn: `1.8.0`
- joblib: `1.5.3`

### Artifacts

| Artifact | Path | Exists | Bytes |
| --- | --- | --- | --- |
| corpus_parquet | /Users/mohit/Pythonprojects/Austrian_law_assistant/ris_sentences.parquet | yes | 7989643 |
| semantic_npz | /Users/mohit/Pythonprojects/Austrian_law_assistant/embedding_index.npz | yes | 40970814 |
| idf_svd_npz | /Users/mohit/Pythonprojects/Austrian_law_assistant/embedding_index_idf_svd.npz | yes | 20526733 |
| idf_svd_model | /Users/mohit/Pythonprojects/Austrian_law_assistant/idf_svd_model.joblib | yes | 68502082 |
| kahm_query_model | /Users/mohit/Pythonprojects/Austrian_law_assistant/kahm_query_regressors_by_law | yes | 0 |
| mb_query_npz_test | /Users/mohit/Pythonprojects/Austrian_law_assistant/queries_embedding_index_test.npz | yes | 19502404 |
| mb_query_npz_train | /Users/mohit/Pythonprojects/Austrian_law_assistant/queries_embedding_index_train.npz | yes | 156095067 |

## Notes and limitations

- Query sets appear to follow the synthetic schema (`query_text`, `consensus_law`, `topic_id`, `style`) when such fields are present; interpretation of results should consider the split mode (topic overlap vs disjoint topics).
- This report focuses on retrieval quality and does not benchmark end-to-end latency or energy use.
- The transformer-query baseline is reported as a reference; KAHM may outperform it if the adapter is supervised/tuned for this label set.
