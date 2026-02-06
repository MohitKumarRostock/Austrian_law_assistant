# KAHM embeddings: retrieval evaluation on Austrian laws

**Generated (UTC):** 2026-01-28T13:15:19Z  
**Source script:** `evaluate_three_embeddings.py` (version `2026-01-25-combined-distance-gate-v1`)  

**KAHM query embedding strategy:** `query_model`  
**KAHM combine tie-break:** `query`  

## Abstract

We study whether KAHM can replace transformer-based query embedding at retrieval time by learning a lightweight mapping from an IDF–SVD representation into Mixedbread embedding space, enabling search against a fixed Mixedbread corpus index. On 5000 human-labeled queries over 7,120 aligned sentences (k=10), KAHM(query→MB corpus) achieved Hit@10=0.820 (0.810, 0.831) and MRR@10=0.644 (0.632, 0.655). Compared with IDF–SVD, KAHM changed Hit@10 by +0.174 (+0.161, +0.187) and MRR@10 by +0.178 (+0.166, +0.190) under paired bootstrap. Against Mixedbread, paired deltas for Hit@10, Top-1 accuracy had 95% CIs excluding 0, whereas MRR@10 had 95% CIs that included 0 (differences not resolved under this bootstrap; not a formal equivalence claim). Mean lift vs prior changed by +3.012 (+2.302, +3.722).  Majority-vote accuracy was higher by +0.040 (+0.028, +0.053) versus Mixedbread.Finally, Full-KAHM embeddings showed high cosine alignment with Mixedbread geometry (corpus cosine=0.9973 (0.9971, 0.9976), query cosine=0.9415 (0.9409, 0.9421)) and recovered similar law-level neighborhoods (law-set Jaccard@10=0.409 (0.403, 0.415); Δ vs IDF=+0.043 (+0.036, +0.051)), while sentence-level neighbor identity remained modest (sentence Jaccard@10=0.069 (0.066, 0.072); Δ vs IDF=-0.015 (-0.018, -0.012)).

## Experimental configuration

The following settings were recorded in the evaluation run:

- Queries: **5000**
- Corpus sentences (aligned): **7120**
- Embedding dimension: **1024**
- Retrieval depth: **k=10**
- Bootstrap: **paired nonparametric**, 5000 samples, seed=0
- Majority-vote routing thresholds: 0.5,0.6,0.7,0.8
- Predominance fraction (per-query majority-vote accuracy metric): **0.50**

### Implementation details

- Retrieval uses FAISS inner-product search over **L2-normalized** embeddings (equivalent to cosine similarity).
- Mixedbread baseline: `mixedbread-ai/deepset-mxbai-embed-de-large-v1` with query prefix `query: `.
- IDF–SVD: pipeline loaded from `idf_svd_model.joblib`; corpus index from `embedding_index_idf_svd.npz`.
- KAHM: regressor loaded from `kahm_regressor_idf_to_mixedbread.joblib` (mode `soft`), mapping IDF–SVD query embeddings into Mixedbread space.
- Methods compared:
  - **Mixedbread (true):** Mixedbread queries → Mixedbread corpus.
  - **IDF–SVD:** IDF–SVD queries → IDF–SVD corpus.
  - **KAHM(query→MB corpus):** KAHM-regressed queries → Mixedbread corpus.
  - **Full-KAHM (query→KAHM corpus):** KAHM-regressed queries → KAHM-transformed corpus (`embedding_index_kahm_mixedbread_approx.npz`).

## Methods

### Data and alignment

Each embedding index provides sentence identifiers that are aligned against a shared sentence-level corpus. All reported retrieval metrics are computed over the intersection of sentence identifiers present in each embedding index, yielding 7,120 common sentences.

### Retrieval protocol

For each query, we compute a query embedding for each method, retrieve the top-k nearest neighbors from the corresponding index, and map retrieved sentence identifiers to their associated law identifiers to compute law-level retrieval metrics.

### Evaluation metrics

Each query is labeled by human annotation with a single reference law identifier (the *consensus law*). Metrics are computed per query and then averaged:

- **Hit@k:** 1 if the consensus law appears at least once among the top-k retrieved laws; else 0.
- **Top-1 accuracy:** 1 if the top-ranked retrieved law equals the consensus law; else 0.
- **MRR@k (unique laws):** form the ordered list of *unique* laws appearing in the top-k list; if the consensus law occurs at rank r in this list, score 1/r; otherwise 0.
- **Majority-vote accuracy (predominance ≥ 0.50):** let the majority law be the most frequent law in the top-k list and let f be its fraction. Score 1 if (i) the majority law equals the consensus law and (ii) f ≥ 0.50; otherwise 0.
- **Mean consensus fraction:** fraction of the top-k items belonging to the consensus law (count/k).
- **Mean lift vs prior:** for each query, divide the consensus fraction by the marginal probability of the consensus law in the aligned corpus; report the mean across queries.
Note: lift vs prior can be sensitive to rare-law priors; interpret it as complementary to Hit/MRR/Top-1.

### Majority-vote diagnostics and routing

We summarize the law vote distribution over each query’s top-k neighborhood using:
- **Top-law fraction:** fraction of the top-k belonging to the most frequent law.
- **Vote margin:** (top-law fraction) minus (runner-up law fraction).
- **Vote entropy:** Shannon entropy of the empirical law distribution in the top-k list.
- **#Unique laws:** number of distinct laws appearing in the top-k list.
For routing, we evaluate a threshold rule that accepts the majority-law prediction when the top-law fraction ≥ τ. We report (i) **coverage** P(covered), (ii) **accuracy among covered** P(correct | covered), and (iii) **routing accuracy** P(correct ∩ covered).

### Statistical analysis

We estimate 95% confidence intervals (CIs) using a paired, nonparametric bootstrap over queries (n_boot=5,000, seed=0). CIs are percentile intervals computed from the 2.5th and 97.5th bootstrap quantiles. For paired deltas, the statistic is computed on per-query differences; a delta is treated as statistically different from 0 when its 95% CI excludes 0.

## Results

### Main retrieval performance

**Table 1.** Main retrieval metrics (mean over queries; 95% bootstrap CI).

| Method | Hit@10 | MRR@10 (unique laws) | Top-1 accuracy | Majority-vote accuracy (predominance ≥ 0.50) | Mean consensus fraction | Mean lift vs prior |
| --- | --- | --- | --- | --- | --- | --- |
| Mixedbread (true) | 0.845 (0.835, 0.855) | 0.644 (0.634, 0.656) | 0.517 (0.504, 0.532) | 0.449 (0.435, 0.463) | 0.443 (0.434, 0.453) | 31.659 (30.503, 32.875) |
| IDF–SVD | 0.646 (0.633, 0.659) | 0.465 (0.453, 0.477) | 0.352 (0.339, 0.365) | 0.305 (0.292, 0.318) | 0.314 (0.304, 0.323) | 20.509 (19.592, 21.429) |
| KAHM(query→MB corpus) | 0.820 (0.810, 0.831) | 0.644 (0.632, 0.655) | 0.532 (0.519, 0.546) | 0.489 (0.475, 0.503) | 0.463 (0.453, 0.473) | 34.671 (33.350, 35.940) |
| Full-KAHM (query→KAHM corpus) | 0.780 (0.769, 0.792) | 0.542 (0.531, 0.554) | 0.410 (0.396, 0.424) | 0.267 (0.255, 0.279) | 0.308 (0.300, 0.316) | 19.138 (18.476, 19.820) |

### Comparative analyses

**Table 2.** Paired deltas for KAHM(query→MB corpus) vs IDF–SVD (95% bootstrap CI).

| Metric | Δ (KAHM − IDF) | 95% CI excludes 0? | Superiority (lower CI > 0)? |
| --- | --- | --- | --- |
| Hit@10 | +0.174 (+0.161, +0.187) | Yes | Yes |
| MRR@10 (unique laws) | +0.178 (+0.166, +0.190) | Yes | Yes |
| Top-1 accuracy | +0.180 (+0.165, +0.195) | Yes | Yes |
| Majority-vote accuracy (predominance ≥ 0.50) | +0.184 (+0.170, +0.198) | Yes | Yes |
| Mean consensus fraction | +0.150 (+0.141, +0.159) | Yes | Yes |
| Mean lift vs prior | +14.163 (+13.140, +15.217) | Yes | Yes |

Interpretation: the superiority column reflects the one-sided criterion used in the evaluation (paired 95% CI lower bound > 0).

**Table 3.** Paired deltas for KAHM(query→MB corpus) vs Mixedbread (95% bootstrap CI).

| Metric | Δ (KAHM − Mixedbread) | 95% CI excludes 0? |
| --- | --- | --- |
| Hit@10 | -0.025 (-0.035, -0.015) | Yes |
| MRR@10 (unique laws) | -0.001 (-0.010, +0.009) | No |
| Top-1 accuracy | +0.015 (+0.001, +0.028) | Yes |
| Majority-vote accuracy (predominance ≥ 0.50) | +0.040 (+0.028, +0.053) | Yes |
| Mean consensus fraction | +0.020 (+0.013, +0.027) | Yes |
| Mean lift vs prior | +3.012 (+2.302, +3.722) | Yes |

### Majority-vote behavior

**Table 4.** Vote-distribution diagnostics over the top-k neighborhood (k=10; 95% bootstrap CI).

| Method | Top-law fraction | Vote margin | Vote entropy | #Unique laws | P(all 10 one law) |
| --- | --- | --- | --- | --- | --- |
| Mixedbread (true) | 0.597 (0.590, 0.603) | 0.398 (0.390, 0.407) | 0.972 (0.957, 0.986) | 3.632 (3.585, 3.679) | 0.118 (0.109, 0.127) |
| IDF–SVD | 0.607 (0.601, 0.613) | 0.407 (0.398, 0.416) | 0.940 (0.926, 0.955) | 3.509 (3.464, 3.553) | 0.119 (0.110, 0.128) |
| KAHM(query→MB corpus) | 0.609 (0.602, 0.616) | 0.415 (0.406, 0.424) | 0.938 (0.922, 0.953) | 3.532 (3.483, 3.579) | 0.138 (0.129, 0.148) |
| Full-KAHM (query→KAHM corpus) | 0.440 (0.434, 0.446) | 0.239 (0.232, 0.246) | 1.389 (1.375, 1.403) | 5.142 (5.094, 5.190) | 0.032 (0.027, 0.037) |

### Vote-based routing

**Table 5.** Routing sweep comparing KAHM(query→MB corpus) and Mixedbread at different predominance thresholds τ (95% bootstrap CI).

| τ | Coverage (KAHM) | Accuracy among covered (KAHM) | Routing accuracy (KAHM) | Coverage (Mixedbread) | Accuracy among covered (Mixedbread) | Routing accuracy (Mixedbread) |
| --- | --- | --- | --- | --- | --- | --- |
| 0.50 | 0.688 (0.675, 0.701) | 0.711 (0.695, 0.726) | 0.489 (0.475, 0.503) | 0.672 (0.659, 0.685) | 0.669 (0.652, 0.684) | 0.449 (0.435, 0.463) |
| 0.60 | 0.526 (0.512, 0.540) | 0.790 (0.774, 0.806) | 0.416 (0.402, 0.429) | 0.502 (0.488, 0.516) | 0.751 (0.734, 0.768) | 0.377 (0.363, 0.391) |
| 0.70 | 0.401 (0.388, 0.415) | 0.864 (0.850, 0.879) | 0.347 (0.334, 0.360) | 0.376 (0.363, 0.390) | 0.821 (0.804, 0.839) | 0.309 (0.297, 0.322) |
| 0.80 | 0.301 (0.289, 0.314) | 0.931 (0.918, 0.944) | 0.280 (0.268, 0.293) | 0.275 (0.263, 0.288) | 0.893 (0.877, 0.909) | 0.246 (0.234, 0.258) |

**Table 6.** Paired deltas vs Mixedbread for routing (KAHM(query→MB corpus) − Mixedbread).

| τ | ΔCoverage | ΔRouting accuracy |
| --- | --- | --- |
| 0.50 | +0.017 (+0.002, +0.032) | +0.040 (+0.028, +0.053) |
| 0.60 | +0.024 (+0.009, +0.039) | +0.039 (+0.027, +0.051) |
| 0.70 | +0.025 (+0.011, +0.038) | +0.038 (+0.026, +0.049) |
| 0.80 | +0.026 (+0.014, +0.038) | +0.035 (+0.023, +0.045) |

**Table 7.** Decomposition of ΔRouting accuracy into coverage and precision effects.

Point estimates:
| τ | ΔRouting accuracy | Coverage contribution | Precision contribution |
| --- | --- | --- | --- |
| 0.50 | +0.040 | +0.011 | +0.029 |
| 0.60 | +0.039 | +0.019 | +0.020 |
| 0.70 | +0.038 | +0.021 | +0.017 |
| 0.80 | +0.035 | +0.024 | +0.011 |

With paired-bootstrap CIs:
| τ | ΔRouting accuracy | Coverage contribution | Precision contribution |
| --- | --- | --- | --- |
| 0.50 | +0.040 (+0.028, +0.053) | +0.011 (+0.001, +0.022) | +0.029 (+0.018, +0.040) |
| 0.60 | +0.039 (+0.027, +0.051) | +0.019 (+0.007, +0.030) | +0.020 (+0.011, +0.029) |
| 0.70 | +0.038 (+0.026, +0.049) | +0.021 (+0.009, +0.033) | +0.017 (+0.010, +0.024) |
| 0.80 | +0.035 (+0.024, +0.046) | +0.024 (+0.013, +0.035) | +0.011 (+0.006, +0.016) |

**Table 8.** Suggested routing thresholds (coverage constraint and objectives).

Coverage constraint: **coverage ≥ 0.50**
Threshold search grid: **0.00–1.00 (step 0.01)**


| Method | τ* | Coverage | Accuracy among covered | Routing accuracy | Objective |
| --- | --- | --- | --- | --- | --- |
| Mixedbread (true) | 0.00 | 1.000 | 0.549 | 0.549 | Maximize precision |
| KAHM(query→MB corpus) | 0.11 | 1.000 | 0.568 | 0.568 | Maximize precision |
| Full-KAHM (query→KAHM corpus) | 0.11 | 0.999 | 0.461 | 0.461 | Maximize precision |
| IDF–SVD | 0.00 | 1.000 | 0.359 | 0.359 | Maximize precision |
| Mixedbread (true) | 0.00 | 1.000 | 0.549 | 0.549 | Maximize routing accuracy |
| KAHM(query→MB corpus) | 0.11 | 1.000 | 0.568 | 0.568 | Maximize routing accuracy |
| Full-KAHM (query→KAHM corpus) | 0.11 | 0.999 | 0.461 | 0.461 | Maximize routing accuracy |
| IDF–SVD | 0.00 | 1.000 | 0.359 | 0.359 | Maximize routing accuracy |

### Embedding-space alignment and neighborhood overlap

**Table 9.** Alignment and overlap statistics for Full-KAHM vs Mixedbread (95% bootstrap CI).

| Quantity | Estimate (95% CI) |
| --- | --- |
| Cosine alignment (corpus) | 0.9973 (0.9971, 0.9976) |
| Cosine alignment (queries) | 0.9415 (0.9409, 0.9421) |
| Sentence Jaccard@10 | 0.069 (0.066, 0.072) |
| Sentence overlap fraction@10 | 0.115 (0.111, 0.119) |
| Law-set Jaccard@10 | 0.409 (0.403, 0.415) |
| Δ sentence Jaccard (Full − IDF) | -0.015 (-0.018, -0.012) |
| Δ law-set Jaccard (Full − IDF) | +0.043 (+0.036, +0.051) |

## Reproducibility checklist

- Query set: `query_set.TEST_QUERY_SET`
- Corpus parquet: `ris_sentences.parquet`
- Mixedbread model: `mixedbread-ai/deepset-mxbai-embed-de-large-v1`
- Query prefix: `query: `
- IDF–SVD model: `idf_svd_model.joblib`
- KAHM model: `kahm_regressor_idf_to_mixedbread.joblib` (mode `soft`)
- Indices: MB `embedding_index.npz`, IDF `embedding_index_idf_svd.npz`, KAHM `embedding_index_kahm_mixedbread_approx.npz`
- Device: `auto`
- Threads cap: 1
- Bootstrap: samples=5000, seed=0

---
## Summary paragraph

Across 5000 queries (k=10), KAHM(query→MB corpus) achieved Hit@10=0.820 (0.810, 0.831) and MRR@10=0.644 (0.632, 0.655). Paired-bootstrap deltas supported KAHM(query→MB corpus) superiority over IDF–SVD on Hit@10, MRR@10, Top-1 accuracy (Table 2). Compared to Mixedbread, paired deltas indicated differences for Hit@10, Top-1 accuracy (95% CIs exclude 0), while MRR@10 were not resolved (95% CIs include 0; Table 3). Majority-vote behavior differed depending on the routing threshold τ (Tables 5–8). Majority-vote accuracy was higher by +0.040 (+0.028, +0.053) relative to Mixedbread. Full-KAHM embeddings showed high cosine alignment with Mixedbread in embedding space (mean corpus cosine 0.9973) and recovered similar law-level neighborhoods (law-set Jaccard@10=0.409 (0.403, 0.415); Δ vs IDF=+0.043 (+0.036, +0.051); Table 9), while sentence-level neighbor identity remained modest (sentence Jaccard@10=0.069 (0.066, 0.072); Δ vs IDF=-0.015 (-0.018, -0.012); Table 9).
