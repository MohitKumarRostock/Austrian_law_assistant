# KAHM embeddings: retrieval evaluation on Austrian laws

**Generated (UTC):** 2026-02-09T10:05:24Z  
**Source script:** `evaluate_three_embeddings.py` (version `2026-02-09-routing-threshold-objectives-fix-v1`)  

**KAHM query embedding strategy:** `query_model`  
**KAHM combine tie-break:** `query`  

## Abstract

We study whether KAHM can replace transformer-based query embedding at retrieval time by learning a lightweight mapping from an IDF–SVD representation into Mixedbread embedding space, enabling search against a fixed Mixedbread corpus index. On 5000 human-labeled queries over 14,487 aligned sentences (k=100), KAHM(query→MB corpus) achieved Hit@100=0.785 (0.774, 0.796) and MRR@100=0.497 (0.485, 0.509). Compared with IDF–SVD, KAHM changed Hit@100 by +0.062 (+0.051, +0.074) and MRR@100 by +0.113 (+0.102, +0.123) under paired bootstrap. Against Mixedbread, paired deltas for Hit@100, MRR@100, Top-1 accuracy had 95% CIs excluding 0. Mean lift vs prior changed by +2.987 (+2.506, +3.470).  Majority-vote accuracy was higher by +0.032 (+0.022, +0.041) versus Mixedbread.Finally, Full-KAHM embeddings showed high cosine alignment with Mixedbread geometry (corpus cosine=0.9940 (0.9937, 0.9942), query cosine=0.9547 (0.9538, 0.9556)) and recovered similar law-level neighborhoods (law-set Jaccard@100=0.447 (0.444, 0.450); Δ vs IDF=+0.094 (+0.090, +0.098)), while sentence-level neighbor identity remained modest (sentence Jaccard@100=0.074 (0.072, 0.076); Δ vs IDF=-0.045 (-0.047, -0.043)).

## Experimental configuration

The following settings were recorded in the evaluation run:

- Queries: **5000**
- Corpus sentences (aligned): **14487**
- Embedding dimension: **1024**
- Retrieval depth: **k=100**
- Bootstrap: **paired nonparametric**, 5000 samples, seed=0
- Majority-vote routing thresholds: 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8
- Predominance fraction (per-query majority-vote accuracy metric): **0.10**

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

Each embedding index provides sentence identifiers that are aligned against a shared sentence-level corpus. All reported retrieval metrics are computed over the intersection of sentence identifiers present in each embedding index, yielding 14,487 common sentences.

### Retrieval protocol

For each query, we compute a query embedding for each method, retrieve the top-k nearest neighbors from the corresponding index, and map retrieved sentence identifiers to their associated law identifiers to compute law-level retrieval metrics.

### Evaluation metrics

Each query is labeled by human annotation with a single reference law identifier (the *consensus law*). Metrics are computed per query and then averaged:

- **Hit@k:** 1 if the consensus law appears at least once among the top-k retrieved laws; else 0.
- **Top-1 accuracy:** 1 if the top-ranked retrieved law equals the consensus law; else 0.
- **MRR@k (unique laws):** form the ordered list of *unique* laws appearing in the top-k list; if the consensus law occurs at rank r in this list, score 1/r; otherwise 0.
- **Majority-vote accuracy (predominance ≥ 0.10):** let the majority law be the most frequent law in the top-k list and let f be its fraction. Score 1 if (i) the majority law equals the consensus law and (ii) f ≥ 0.10; otherwise 0.
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

| Method | Hit@100 | MRR@100 (unique laws) | Top-1 accuracy | Majority-vote accuracy (predominance ≥ 0.10) | Mean consensus fraction | Mean lift vs prior |
| --- | --- | --- | --- | --- | --- | --- |
| Mixedbread (true) | 0.797 (0.785, 0.808) | 0.475 (0.464, 0.487) | 0.362 (0.349, 0.375) | 0.349 (0.336, 0.362) | 0.196 (0.189, 0.202) | 24.273 (23.505, 25.078) |
| IDF–SVD | 0.723 (0.710, 0.735) | 0.384 (0.373, 0.396) | 0.284 (0.271, 0.296) | 0.273 (0.260, 0.285) | 0.176 (0.169, 0.183) | 20.489 (19.694, 21.290) |
| KAHM(query→MB corpus) | 0.785 (0.774, 0.796) | 0.497 (0.485, 0.509) | 0.393 (0.379, 0.407) | 0.381 (0.367, 0.394) | 0.217 (0.210, 0.224) | 27.260 (26.383, 28.152) |
| Full-KAHM (query→KAHM corpus) | 0.749 (0.737, 0.760) | 0.426 (0.414, 0.437) | 0.321 (0.308, 0.334) | 0.242 (0.230, 0.253) | 0.110 (0.106, 0.114) | 11.606 (11.182, 12.040) |

### Comparative analyses

**Table 2.** Paired deltas for KAHM(query→MB corpus) vs IDF–SVD (95% bootstrap CI).

| Metric | Δ (KAHM − IDF) | 95% CI excludes 0? | Superiority (lower CI > 0)? |
| --- | --- | --- | --- |
| Hit@100 | +0.062 (+0.051, +0.074) | Yes | Yes |
| MRR@100 (unique laws) | +0.113 (+0.102, +0.123) | Yes | Yes |
| Top-1 accuracy | +0.109 (+0.095, +0.123) | Yes | Yes |
| Majority-vote accuracy (predominance ≥ 0.10) | +0.108 (+0.096, +0.120) | Yes | Yes |
| Mean consensus fraction | +0.041 (+0.036, +0.045) | Yes | Yes |
| Mean lift vs prior | +6.771 (+5.989, +7.546) | Yes | Yes |

Interpretation: the superiority column reflects the one-sided criterion used in the evaluation (paired 95% CI lower bound > 0).

**Table 3.** Paired deltas for KAHM(query→MB corpus) vs Mixedbread (95% bootstrap CI).

| Metric | Δ (KAHM − Mixedbread) | 95% CI excludes 0? |
| --- | --- | --- |
| Hit@100 | -0.012 (-0.020, -0.003) | Yes |
| MRR@100 (unique laws) | +0.022 (+0.014, +0.030) | Yes |
| Top-1 accuracy | +0.031 (+0.020, +0.042) | Yes |
| Majority-vote accuracy (predominance ≥ 0.10) | +0.032 (+0.022, +0.041) | Yes |
| Mean consensus fraction | +0.021 (+0.018, +0.024) | Yes |
| Mean lift vs prior | +2.987 (+2.506, +3.470) | Yes |

### Majority-vote behavior

**Table 4.** Vote-distribution diagnostics over the top-k neighborhood (k=100; 95% bootstrap CI).

| Method | Top-law fraction | Vote margin | Vote entropy | #Unique laws | P(all 100 one law) |
| --- | --- | --- | --- | --- | --- |
| Mixedbread (true) | 0.338 (0.333, 0.344) | 0.191 (0.185, 0.197) | 2.239 (2.222, 2.257) | 20.002 (19.815, 20.192) | 0.000 (0.000, 0.000) |
| IDF–SVD | 0.357 (0.351, 0.363) | 0.209 (0.202, 0.216) | 2.171 (2.152, 2.191) | 19.440 (19.241, 19.638) | 0.002 (0.001, 0.004) |
| KAHM(query→MB corpus) | 0.356 (0.351, 0.362) | 0.207 (0.201, 0.214) | 2.163 (2.144, 2.182) | 19.073 (18.873, 19.271) | 0.000 (0.000, 0.000) |
| Full-KAHM (query→KAHM corpus) | 0.234 (0.230, 0.237) | 0.107 (0.103, 0.111) | 2.711 (2.699, 2.723) | 26.897 (26.733, 27.061) | 0.000 (0.000, 0.000) |

### Vote-based routing

**Table 5.** Routing sweep comparing KAHM(query→MB corpus) and Mixedbread at different predominance thresholds τ (95% bootstrap CI).

| τ | Coverage (KAHM) | Accuracy among covered (KAHM) | Routing accuracy (KAHM) | Coverage (Mixedbread) | Accuracy among covered (Mixedbread) | Routing accuracy (Mixedbread) |
| --- | --- | --- | --- | --- | --- | --- |
| 0.10 | 0.997 (0.996, 0.999) | 0.382 (0.368, 0.396) | 0.381 (0.367, 0.394) | 0.996 (0.995, 0.998) | 0.350 (0.337, 0.364) | 0.349 (0.335, 0.362) |
| 0.20 | 0.774 (0.762, 0.786) | 0.452 (0.437, 0.469) | 0.350 (0.337, 0.364) | 0.752 (0.740, 0.763) | 0.418 (0.403, 0.434) | 0.314 (0.301, 0.327) |
| 0.30 | 0.484 (0.471, 0.498) | 0.558 (0.538, 0.577) | 0.270 (0.257, 0.282) | 0.461 (0.447, 0.474) | 0.518 (0.498, 0.539) | 0.238 (0.227, 0.250) |
| 0.40 | 0.307 (0.295, 0.320) | 0.644 (0.620, 0.668) | 0.198 (0.187, 0.209) | 0.291 (0.278, 0.303) | 0.608 (0.584, 0.634) | 0.177 (0.166, 0.187) |
| 0.50 | 0.216 (0.205, 0.227) | 0.713 (0.685, 0.739) | 0.154 (0.144, 0.164) | 0.190 (0.179, 0.200) | 0.666 (0.635, 0.696) | 0.126 (0.117, 0.136) |
| 0.60 | 0.149 (0.139, 0.159) | 0.762 (0.731, 0.792) | 0.113 (0.105, 0.122) | 0.123 (0.114, 0.132) | 0.766 (0.732, 0.800) | 0.094 (0.086, 0.103) |
| 0.70 | 0.102 (0.094, 0.111) | 0.834 (0.801, 0.865) | 0.085 (0.078, 0.093) | 0.074 (0.066, 0.081) | 0.818 (0.778, 0.856) | 0.060 (0.054, 0.067) |
| 0.80 | 0.064 (0.057, 0.071) | 0.906 (0.873, 0.938) | 0.058 (0.052, 0.065) | 0.041 (0.036, 0.046) | 0.859 (0.810, 0.904) | 0.035 (0.030, 0.041) |

**Table 6.** Paired deltas vs Mixedbread for routing (KAHM(query→MB corpus) − Mixedbread).

| τ | ΔCoverage | ΔRouting accuracy |
| --- | --- | --- |
| 0.10 | +0.001 (-0.001, +0.003) | +0.032 (+0.022, +0.041) |
| 0.20 | +0.022 (+0.011, +0.034) | +0.036 (+0.027, +0.045) |
| 0.30 | +0.024 (+0.012, +0.036) | +0.032 (+0.024, +0.040) |
| 0.40 | +0.017 (+0.006, +0.027) | +0.021 (+0.014, +0.028) |
| 0.50 | +0.026 (+0.017, +0.035) | +0.027 (+0.021, +0.034) |
| 0.60 | +0.026 (+0.019, +0.033) | +0.019 (+0.014, +0.025) |
| 0.70 | +0.029 (+0.023, +0.035) | +0.025 (+0.020, +0.031) |
| 0.80 | +0.023 (+0.018, +0.029) | +0.023 (+0.018, +0.028) |

**Table 7.** Decomposition of ΔRouting accuracy into coverage and precision effects.

Point estimates:
| τ | ΔRouting accuracy | Coverage contribution | Precision contribution |
| --- | --- | --- | --- |
| 0.10 | +0.032 | +0.000 | +0.031 |
| 0.20 | +0.036 | +0.010 | +0.026 |
| 0.30 | +0.032 | +0.013 | +0.019 |
| 0.40 | +0.021 | +0.010 | +0.011 |
| 0.50 | +0.027 | +0.018 | +0.009 |
| 0.60 | +0.019 | +0.020 | -0.001 |
| 0.70 | +0.025 | +0.024 | +0.001 |
| 0.80 | +0.023 | +0.020 | +0.003 |

With paired-bootstrap CIs:
| τ | ΔRouting accuracy | Coverage contribution | Precision contribution |
| --- | --- | --- | --- |
| 0.10 | +0.032 (+0.022, +0.041) | +0.000 (-0.000, +0.001) | +0.031 (+0.022, +0.041) |
| 0.20 | +0.036 (+0.027, +0.045) | +0.010 (+0.004, +0.015) | +0.026 (+0.017, +0.035) |
| 0.30 | +0.032 (+0.024, +0.039) | +0.013 (+0.006, +0.019) | +0.019 (+0.012, +0.026) |
| 0.40 | +0.021 (+0.014, +0.028) | +0.010 (+0.004, +0.017) | +0.011 (+0.005, +0.016) |
| 0.50 | +0.027 (+0.021, +0.034) | +0.018 (+0.012, +0.024) | +0.009 (+0.005, +0.014) |
| 0.60 | +0.019 (+0.013, +0.025) | +0.020 (+0.014, +0.025) | -0.001 (-0.004, +0.003) |
| 0.70 | +0.025 (+0.020, +0.030) | +0.024 (+0.019, +0.029) | +0.001 (-0.001, +0.004) |
| 0.80 | +0.023 (+0.018, +0.028) | +0.020 (+0.016, +0.025) | +0.003 (+0.001, +0.004) |

**Table 8.** Suggested routing thresholds (coverage constraint and objectives).

Coverage constraint: **coverage ≥ 0.50**
Threshold search grid: **0.00–1.00 (step 0.01)**


| Method | τ* | Coverage | Accuracy among covered | Routing accuracy | Objective |
| --- | --- | --- | --- | --- | --- |
| Mixedbread (true) | 0.09 | 0.999 | 0.350 | 0.349 | Maximize precision |
| KAHM(query→MB corpus) | 0.11 | 0.991 | 0.384 | 0.381 | Maximize precision |
| Full-KAHM (query→KAHM corpus) | 0.00 | 1.000 | 0.243 | 0.243 | Maximize precision |
| IDF–SVD | 0.11 | 0.994 | 0.274 | 0.273 | Maximize precision |
| Mixedbread (true) | 0.09 | 0.999 | 0.350 | 0.349 | Maximize routing accuracy |
| KAHM(query→MB corpus) | 0.11 | 0.991 | 0.384 | 0.381 | Maximize routing accuracy |
| Full-KAHM (query→KAHM corpus) | 0.00 | 1.000 | 0.243 | 0.243 | Maximize routing accuracy |
| IDF–SVD | 0.11 | 0.994 | 0.274 | 0.273 | Maximize routing accuracy |

### Embedding-space alignment and neighborhood overlap

**Table 9.** Alignment and overlap statistics for Full-KAHM vs Mixedbread (95% bootstrap CI).

| Quantity | Estimate (95% CI) |
| --- | --- |
| Cosine alignment (corpus) | 0.9940 (0.9937, 0.9942) |
| Cosine alignment (queries) | 0.9547 (0.9538, 0.9556) |
| Sentence Jaccard@100 | 0.074 (0.072, 0.076) |
| Sentence overlap fraction@100 | 0.130 (0.127, 0.133) |
| Law-set Jaccard@100 | 0.447 (0.444, 0.450) |
| Δ sentence Jaccard (Full − IDF) | -0.045 (-0.047, -0.043) |
| Δ law-set Jaccard (Full − IDF) | +0.094 (+0.090, +0.098) |

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

Across 5000 queries (k=100), KAHM(query→MB corpus) achieved Hit@100=0.785 (0.774, 0.796) and MRR@100=0.497 (0.485, 0.509). Paired-bootstrap deltas supported KAHM(query→MB corpus) superiority over IDF–SVD on Hit@100, MRR@100, Top-1 accuracy (Table 2). Compared to Mixedbread, paired deltas indicated differences for Hit@100, MRR@100, Top-1 accuracy with 95% CIs excluding 0 (Table 3). Majority-vote behavior differed depending on the routing threshold τ (Tables 5–8). Majority-vote accuracy was higher by +0.032 (+0.022, +0.041) relative to Mixedbread. Full-KAHM embeddings showed high cosine alignment with Mixedbread in embedding space (mean corpus cosine 0.9940) and recovered similar law-level neighborhoods (law-set Jaccard@100=0.447 (0.444, 0.450); Δ vs IDF=+0.094 (+0.090, +0.098); Table 9), while sentence-level neighbor identity remained modest (sentence Jaccard@100=0.074 (0.072, 0.076); Δ vs IDF=-0.045 (-0.047, -0.043); Table 9).
