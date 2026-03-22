# KAHM embeddings: retrieval evaluation on Austrian laws

Generated: 2026-03-21 22:02:39 | script=evaluate_three_embeddings_storylines.py | version=2026-02-23-scientific-pubreport-v1

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
| 3 | 0.334 [0.322, 0.347] | 0.466 [0.453, 0.479] | 0.436 [0.423, 0.449] |
| 5 | 0.349 [0.337, 0.361] | 0.481 [0.469, 0.494] | 0.454 [0.441, 0.466] |
| 10 | 0.364 [0.351, 0.376] | 0.496 [0.484, 0.508] | 0.468 [0.456, 0.480] |
| 15 | 0.371 [0.359, 0.383] | 0.501 [0.489, 0.513] | 0.474 [0.462, 0.486] |
| 20 | 0.376 [0.364, 0.387] | 0.504 [0.492, 0.516] | 0.478 [0.466, 0.489] |

**Hit@k**

| k | IDF–SVD | KAHM(query→MB corpus) | Mixedbread (true) |
| --- | --- | --- | --- |
| 3 | 0.389 [0.375, 0.402] | 0.529 [0.516, 0.543] | 0.503 [0.488, 0.516] |
| 5 | 0.433 [0.420, 0.447] | 0.579 [0.566, 0.593] | 0.560 [0.546, 0.573] |
| 10 | 0.497 [0.483, 0.511] | 0.643 [0.629, 0.656] | 0.625 [0.612, 0.639] |
| 15 | 0.539 [0.525, 0.553] | 0.673 [0.659, 0.685] | 0.662 [0.649, 0.676] |
| 20 | 0.572 [0.558, 0.586] | 0.694 [0.681, 0.707] | 0.688 [0.675, 0.701] |

**Top-1 accuracy**

| k | IDF–SVD | KAHM(query→MB corpus) | Mixedbread (true) |
| --- | --- | --- | --- |
| 3 | 0.287 [0.274, 0.299] | 0.411 [0.397, 0.424] | 0.378 [0.365, 0.392] |
| 5 | 0.287 [0.274, 0.299] | 0.411 [0.397, 0.424] | 0.378 [0.365, 0.392] |
| 10 | 0.287 [0.275, 0.299] | 0.411 [0.397, 0.424] | 0.378 [0.364, 0.392] |
| 15 | 0.287 [0.274, 0.299] | 0.411 [0.398, 0.424] | 0.378 [0.365, 0.392] |
| 20 | 0.287 [0.274, 0.299] | 0.411 [0.397, 0.424] | 0.378 [0.364, 0.392] |

**Majority-accuracy** (τ=0.10)

| k | IDF–SVD | KAHM(query→MB corpus) | Mixedbread (true) |
| --- | --- | --- | --- |
| 3 | 0.298 [0.286, 0.311] | 0.422 [0.408, 0.435] | 0.379 [0.366, 0.393] |
| 5 | 0.302 [0.289, 0.315] | 0.430 [0.417, 0.444] | 0.392 [0.378, 0.405] |
| 10 | 0.305 [0.292, 0.318] | 0.436 [0.422, 0.450] | 0.402 [0.388, 0.416] |
| 15 | 0.311 [0.298, 0.324] | 0.427 [0.414, 0.440] | 0.400 [0.387, 0.414] |
| 20 | 0.305 [0.292, 0.318] | 0.424 [0.411, 0.438] | 0.397 [0.384, 0.411] |

**Mean consensus fraction**

| k | IDF–SVD | KAHM(query→MB corpus) | Mixedbread (true) |
| --- | --- | --- | --- |
| 3 | 0.279 [0.268, 0.290] | 0.378 [0.367, 0.390] | 0.345 [0.334, 0.356] |
| 5 | 0.271 [0.261, 0.282] | 0.362 [0.352, 0.373] | 0.330 [0.320, 0.340] |
| 10 | 0.256 [0.246, 0.266] | 0.338 [0.328, 0.348] | 0.303 [0.293, 0.312] |
| 15 | 0.247 [0.238, 0.256] | 0.319 [0.310, 0.328] | 0.285 [0.277, 0.294] |
| 20 | 0.238 [0.230, 0.248] | 0.305 [0.296, 0.314] | 0.270 [0.262, 0.279] |

**Mean lift (prior)**

| k | IDF–SVD | KAHM(query→MB corpus) | Mixedbread (true) |
| --- | --- | --- | --- |
| 3 | 42.062 [39.219, 45.081] | 67.163 [63.525, 70.954] | 60.549 [56.883, 64.307] |
| 5 | 38.900 [36.585, 41.255] | 62.404 [59.138, 65.685] | 57.062 [53.795, 60.438] |
| 10 | 35.715 [33.796, 37.715] | 55.824 [53.157, 58.595] | 49.011 [46.555, 51.580] |
| 15 | 33.889 [32.142, 35.688] | 50.812 [48.591, 53.038] | 44.308 [42.225, 46.400] |
| 20 | 32.062 [30.501, 33.693] | 47.025 [45.047, 49.044] | 40.339 [38.558, 42.143] |

### Paired deltas (KAHM − IDF–SVD)

| k | Δhit@k | ΔMRR@k | ΔTop-1 | ΔMajority-acc | ΔMean cons frac | ΔMean lift |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | +0.141 [+0.127, +0.155] | +0.132 [+0.119, +0.144] | +0.124 [+0.110, +0.137] | +0.123 [+0.110, +0.137] | +0.100 [+0.090, +0.110] | +25.101 [+21.323, +28.982] |
| 5 | +0.146 [+0.132, +0.159] | +0.133 [+0.121, +0.144] | +0.124 [+0.110, +0.138] | +0.128 [+0.115, +0.141] | +0.091 [+0.082, +0.100] | +23.504 [+20.436, +26.677] |
| 10 | +0.145 [+0.132, +0.159] | +0.132 [+0.121, +0.143] | +0.124 [+0.110, +0.137] | +0.131 [+0.117, +0.144] | +0.081 [+0.073, +0.089] | +20.109 [+17.773, +22.528] |
| 15 | +0.134 [+0.120, +0.148] | +0.130 [+0.119, +0.141] | +0.124 [+0.110, +0.137] | +0.116 [+0.103, +0.129] | +0.072 [+0.065, +0.079] | +16.922 [+14.899, +18.916] |
| 20 | +0.122 [+0.109, +0.136] | +0.128 [+0.117, +0.139] | +0.124 [+0.110, +0.137] | +0.119 [+0.106, +0.132] | +0.067 [+0.060, +0.074] | +14.963 [+13.196, +16.715] |

### Paired deltas vs transformer-query baseline (context; KAHM − Mixedbread)

| k | Δhit@k | ΔMRR@k | ΔTop-1 | ΔMajority-acc | ΔMean cons frac | ΔMean lift |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | +0.027 [+0.016, +0.037] | +0.030 [+0.021, +0.040] | +0.033 [+0.021, +0.044] | +0.042 [+0.031, +0.053] | +0.033 [+0.026, +0.041] | +6.614 [+3.911, +9.331] |
| 5 | +0.019 [+0.009, +0.030] | +0.028 [+0.019, +0.037] | +0.033 [+0.022, +0.043] | +0.038 [+0.028, +0.049] | +0.032 [+0.026, +0.039] | +5.341 [+3.146, +7.481] |
| 10 | +0.017 [+0.007, +0.028] | +0.028 [+0.019, +0.036] | +0.033 [+0.022, +0.044] | +0.034 [+0.023, +0.044] | +0.035 [+0.029, +0.040] | +6.813 [+5.177, +8.336] |
| 15 | +0.010 [+0.001, +0.020] | +0.027 [+0.019, +0.035] | +0.033 [+0.021, +0.043] | +0.028 [+0.018, +0.038] | +0.034 [+0.029, +0.039] | +6.503 [+5.246, +7.718] |
| 20 | +0.006 [-0.003, +0.016] | +0.026 [+0.018, +0.035] | +0.033 [+0.022, +0.043] | +0.027 [+0.018, +0.037] | +0.035 [+0.031, +0.039] | +6.687 [+5.625, +7.776] |

### Macro-averaged quality (per-law average; robustness)

Macro-averaging computes metrics per law and then averages across laws (each law has equal weight). This is a robustness check against label-frequency skew.

**Macro MRR@k (unique laws)**

| k | IDF–SVD | KAHM(query→MB corpus) | Mixedbread (true) |
| --- | --- | --- | --- |
| 3 | 0.334 [0.284, 0.385] | 0.466 [0.406, 0.526] | 0.436 [0.382, 0.491] |
| 5 | 0.349 [0.297, 0.402] | 0.482 [0.423, 0.538] | 0.454 [0.400, 0.511] |
| 10 | 0.364 [0.312, 0.416] | 0.496 [0.437, 0.554] | 0.468 [0.410, 0.525] |
| 15 | 0.371 [0.319, 0.425] | 0.501 [0.444, 0.559] | 0.474 [0.417, 0.530] |
| 20 | 0.376 [0.326, 0.428] | 0.504 [0.446, 0.562] | 0.478 [0.422, 0.534] |

**Macro Hit@k**

| k | IDF–SVD | KAHM(query→MB corpus) | Mixedbread (true) |
| --- | --- | --- | --- |
| 3 | 0.389 [0.333, 0.446] | 0.530 [0.467, 0.590] | 0.503 [0.443, 0.561] |
| 5 | 0.434 [0.375, 0.492] | 0.579 [0.517, 0.639] | 0.560 [0.496, 0.621] |
| 10 | 0.497 [0.436, 0.560] | 0.643 [0.578, 0.703] | 0.625 [0.560, 0.685] |
| 15 | 0.539 [0.478, 0.598] | 0.673 [0.607, 0.735] | 0.662 [0.594, 0.725] |
| 20 | 0.572 [0.511, 0.631] | 0.694 [0.625, 0.756] | 0.688 [0.619, 0.751] |

**Macro Top-1 accuracy**

| k | IDF–SVD | KAHM(query→MB corpus) | Mixedbread (true) |
| --- | --- | --- | --- |
| 3 | 0.287 [0.241, 0.336] | 0.411 [0.352, 0.468] | 0.378 [0.323, 0.433] |
| 5 | 0.287 [0.239, 0.336] | 0.411 [0.354, 0.467] | 0.378 [0.324, 0.430] |
| 10 | 0.287 [0.239, 0.336] | 0.411 [0.354, 0.469] | 0.378 [0.324, 0.432] |
| 15 | 0.287 [0.239, 0.338] | 0.411 [0.353, 0.468] | 0.378 [0.324, 0.430] |
| 20 | 0.287 [0.240, 0.336] | 0.411 [0.354, 0.468] | 0.378 [0.325, 0.433] |

**Macro Majority-accuracy** (τ=0.10)

| k | IDF–SVD | KAHM(query→MB corpus) | Mixedbread (true) |
| --- | --- | --- | --- |
| 3 | 0.298 [0.249, 0.350] | 0.422 [0.364, 0.481] | 0.379 [0.323, 0.437] |
| 5 | 0.303 [0.251, 0.356] | 0.430 [0.371, 0.489] | 0.392 [0.336, 0.450] |
| 10 | 0.305 [0.251, 0.359] | 0.436 [0.374, 0.497] | 0.402 [0.345, 0.461] |
| 15 | 0.311 [0.255, 0.367] | 0.428 [0.365, 0.489] | 0.400 [0.340, 0.461] |
| 20 | 0.305 [0.250, 0.364] | 0.424 [0.361, 0.491] | 0.397 [0.336, 0.460] |

### Macro paired deltas (KAHM − IDF–SVD)

| k | Δhit@k | ΔMRR@k | ΔTop-1 | ΔMajority-acc | ΔMean cons frac | ΔMean lift |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | +0.141 [+0.109, +0.172] | +0.132 [+0.103, +0.162] | +0.124 [+0.093, +0.156] | +0.123 [+0.091, +0.156] | +0.100 [+0.073, +0.129] | +25.136 [+12.537, +38.554] |
| 5 | +0.146 [+0.114, +0.178] | +0.133 [+0.104, +0.162] | +0.124 [+0.092, +0.155] | +0.128 [+0.096, +0.160] | +0.091 [+0.065, +0.118] | +23.541 [+12.724, +35.273] |
| 10 | +0.145 [+0.110, +0.182] | +0.132 [+0.104, +0.162] | +0.124 [+0.092, +0.156] | +0.131 [+0.099, +0.165] | +0.081 [+0.057, +0.106] | +20.141 [+11.253, +29.602] |
| 15 | +0.134 [+0.098, +0.171] | +0.130 [+0.103, +0.160] | +0.124 [+0.093, +0.155] | +0.117 [+0.086, +0.148] | +0.072 [+0.048, +0.096] | +16.945 [+9.248, +25.482] |
| 20 | +0.122 [+0.086, +0.159] | +0.128 [+0.100, +0.157] | +0.124 [+0.093, +0.156] | +0.119 [+0.089, +0.152] | +0.067 [+0.045, +0.090] | +14.980 [+7.698, +22.644] |

### Macro paired deltas vs transformer-query baseline (context; KAHM − Mixedbread)

| k | Δhit@k | ΔMRR@k | ΔTop-1 | ΔMajority-acc | ΔMean cons frac | ΔMean lift |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | +0.027 [+0.013, +0.041] | +0.030 [+0.019, +0.042] | +0.033 [+0.020, +0.046] | +0.042 [+0.028, +0.059] | +0.033 [+0.023, +0.044] | +6.625 [+3.169, +10.335] |
| 5 | +0.019 [+0.008, +0.031] | +0.028 [+0.017, +0.039] | +0.033 [+0.020, +0.046] | +0.038 [+0.025, +0.052] | +0.032 [+0.023, +0.041] | +5.346 [+2.044, +8.660] |
| 10 | +0.017 [+0.006, +0.028] | +0.028 [+0.017, +0.038] | +0.033 [+0.019, +0.046] | +0.034 [+0.020, +0.047] | +0.035 [+0.027, +0.042] | +6.821 [+4.254, +9.603] |
| 15 | +0.010 [-0.002, +0.023] | +0.027 [+0.017, +0.037] | +0.033 [+0.020, +0.046] | +0.028 [+0.016, +0.040] | +0.034 [+0.027, +0.041] | +6.509 [+4.378, +8.704] |
| 20 | +0.006 [-0.006, +0.019] | +0.026 [+0.016, +0.036] | +0.033 [+0.020, +0.046] | +0.027 [+0.014, +0.042] | +0.035 [+0.028, +0.042] | +6.692 [+4.704, +8.841] |

## Majority-vote routing (coverage/precision)

We report a coverage–precision sweep over routing thresholds τ′ (distinct from the predominance threshold used in the majority metric). Coverage is the fraction of queries that meet τ′; precision is accuracy conditioned on being covered.

Recommended τ′ maximizes precision subject to coverage ≥ **0.50**.

| Method | τ′ | Coverage | Majority-acc | Precision (acc|covered) |
| --- | --- | --- | --- | --- |
| IDF–SVD | 0.41 | 0.538 | 0.251 | 0.468 |
| Mixedbread (true) | 0.41 | 0.525 | 0.308 | 0.587 |
| KAHM(query→MB corpus) | 0.41 | 0.562 | 0.346 | 0.615 |

## Computational profile

This section reports query-time computational profiles for the three retrieval paths. The primary comparison target is **online per-query time** (query embedding + FAISS search). If a query embedding source was loaded from a precomputed NPZ in this run, the corresponding online embedding time is reported as `n/a` and the load time is reported separately.

### Per-query online path comparison

| Path | Query source | Query embed / q | FAISS search / q | Total online / q | Observed step sum / q | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| IDF–SVD | model | 0.844 ms | 0.305 ms | 1.149 ms | 1.149 ms | IDF–SVD model load shown in component table (cold-start). |
| KAHM(query→MB corpus) | model | 93.250 ms | 0.584 ms | 93.834 ms | 93.834 ms | Online total only available when KAHM queries were embedded in this run (not precomputed NPZ). |
| Mixedbread (true) | online | 800.083 ms | 0.580 ms | 800.663 ms | 800.663 ms | Online total only available when Mixedbread queries were encoded on the fly (not precomputed NPZ). |

### Measured components (wall-clock)

| Component | Wall time | Per query | Notes |
| --- | --- | --- | --- |
| IDF–SVD query pipeline init (cold-start) | 2.634 s | 0.527 ms | One-time pipeline/materialization cost. |
| IDF–SVD query embedding (batch) | 4.220 s | 0.844 ms |  |
| KAHM query load (precomputed NPZ) | n/a | n/a | Only present when --kahm_query_embeddings_npz is used. |
| KAHM query model init (cold-start) | 5.126 s | 1.025 ms | Only present for online KAHM embedding. |
| KAHM query warm-up (excluded from online total) | 5.665 s | n/a |  |
| KAHM query embedding (batch) | 466.249 s | 93.250 ms |  |
| Mixedbread query load (precomputed NPZ) | n/a | n/a | Only present when precomputed Mixedbread query embeddings are used. |
| Mixedbread model init (cold-start) | 7.606 s | 1.521 ms | Only present for online transformer query encoding. |
| Mixedbread query warm-up (excluded from online total) | 1.721 s | n/a |  |
| Mixedbread query embedding (batch) | 4000.415 s | 800.083 ms |  |
| FAISS build (IDF corpus index) | 0.316 s | n/a |  |
| FAISS search (IDF path) | 1.526 s | 0.305 ms |  |
| FAISS build (MB corpus index) | 0.127 s | n/a | Shared by Mixedbread and KAHM(query→MB) paths. |
| FAISS search (Mixedbread path) | 2.898 s | 0.580 ms |  |
| FAISS search (KAHM→MB path) | 2.919 s | 0.584 ms |  |
| Corpus embedding memory (IDF matrix) | 22,040,576 bytes | n/a | NumPy array nbytes (aligned corpus embeddings used in this run). |
| Corpus embedding memory (MB matrix) | 44,081,152 bytes | n/a | NumPy array nbytes (aligned corpus embeddings used in this run). |

### Derived online speedups (per-query)

| Comparison | Speedup | Definition |
| --- | --- | --- |
| IDF–SVD vs KAHM(query→MB corpus) | 0.01× | IDF online / KAHM online |
| Mixedbread (true) vs KAHM(query→MB corpus) | 8.53× | MB online / KAHM online |
| Mixedbread (true) vs IDF–SVD | 696.67× | MB online / IDF online |

### Machine profile (auto-detected; best effort)

| Field | Value |
| --- | --- |
| Hostname | iMac-von-Mohit.local |
| Platform | macOS-26.3.1-x86_64-i386-64bit |
| System | Darwin |
| Machine / arch | x86_64 |
| Processor | i386 |
| CPU logical cores | 8 |
| RAM total | 8.00 GiB |
| Python | 3.11.14 |
| Torch runtime | 2.2.2 |
| Accelerator type | mps |
| Accelerator name | Apple Metal (MPS) |
| CUDA available | False |
| MPS available | True |
| Requested device arg | cpu |
| Auto-resolved device | cpu |
| Thread cap arg | 1 |
| KAHM query source | model |
| Mixedbread query source | online |
| n_queries | 5000 |
| n_corpus | 10762 |
| embedding_dim | 1024 |
| retrieval_k_max | 20 |

## Reproducibility

- Bootstrap: B=5000, seed=0
- Thread cap: 1 (0 means no override)

### Software / environment

- Python: `3.11.14`
- Platform: `macOS-26.3.1-x86_64-i386-64bit`
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
- This report focuses on retrieval quality, with added wall-clock query-time profiling; it does not benchmark end-to-end serving latency under concurrency or energy use.
- The transformer-query baseline is reported as a reference; KAHM may outperform it if the adapter is supervised/tuned for this label set.
