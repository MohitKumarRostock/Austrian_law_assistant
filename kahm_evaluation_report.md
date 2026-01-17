# KAHM embeddings: retrieval evaluation on Austrian laws

**Generated (UTC):** 2026-01-17T21:09:19Z  
**Source script:** `evaluate_three_embeddings.py` (version `2026-01-17-pubreport-v7`)  

## Abstract

We study whether KAHM can replace transformer-based query embedding at retrieval time by learning a lightweight mapping from an IDF–SVD representation into Mixedbread embedding space, enabling search against a fixed Mixedbread corpus index. On 200 human-labeled queries over 71,069 aligned sentences (k=10), KAHM(query→MB corpus) achieved Hit@10=0.895 (0.850, 0.935) and MRR@10=0.729 (0.679, 0.779). Compared with IDF–SVD, KAHM changed Hit@10 by +0.190 (+0.130, +0.250) and MRR@10 by +0.241 (+0.182, +0.301) under paired bootstrap. Against Mixedbread, paired deltas for Hit@10, MRR@10, Top-1 accuracy had 95% CIs that included 0 (differences not resolved under this bootstrap; not a formal equivalence claim). Mean lift vs prior changed by +2.727 (-0.966, +6.568).  Majority-vote accuracy was numerically higher by +0.055 (+0.000, +0.110) versus Mixedbread (CI touches 0).Finally, Full-KAHM embeddings showed high cosine alignment with Mixedbread geometry (corpus cosine=0.9405 (0.9403, 0.9407), query cosine=0.9365 (0.9322, 0.9405)) and recovered similar law-level neighborhoods (law-set Jaccard@10=0.545 (0.505, 0.585); Δ vs IDF=+0.201 (+0.156, +0.244)), while sentence-level neighbor identity remained modest (sentence Jaccard@10=0.115 (0.099, 0.132); Δ vs IDF=+0.063 (+0.046, +0.081)).

## Experimental configuration

The following settings were recorded in the evaluation run:

- Queries: **200**
- Corpus sentences (aligned): **71069**
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

Each embedding index provides sentence identifiers that are aligned against a shared sentence-level corpus. All reported retrieval metrics are computed over the intersection of sentence identifiers present in each embedding index, yielding 71,069 common sentences.

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
| Mixedbread (true) | 0.900 (0.855, 0.940) | 0.720 (0.670, 0.770) | 0.605 (0.540, 0.670) | 0.560 (0.490, 0.630) | 0.533 (0.485, 0.582) | 36.923 (30.426, 44.028) |
| IDF–SVD | 0.705 (0.640, 0.770) | 0.488 (0.428, 0.548) | 0.365 (0.295, 0.430) | 0.345 (0.280, 0.410) | 0.347 (0.301, 0.395) | 17.321 (14.188, 20.794) |
| KAHM(query→MB corpus) | 0.895 (0.850, 0.935) | 0.729 (0.679, 0.779) | 0.605 (0.535, 0.675) | 0.615 (0.550, 0.680) | 0.554 (0.506, 0.602) | 39.650 (32.216, 47.840) |
| Full-KAHM (query→KAHM corpus) | 0.885 (0.840, 0.930) | 0.696 (0.641, 0.749) | 0.580 (0.510, 0.650) | 0.595 (0.525, 0.665) | 0.557 (0.509, 0.606) | 39.949 (31.543, 49.614) |

### Comparative analyses

**Table 2.** Paired deltas for KAHM(query→MB corpus) vs IDF–SVD (95% bootstrap CI).

| Metric | Δ (KAHM − IDF) | 95% CI excludes 0? | Superiority (lower CI > 0)? |
| --- | --- | --- | --- |
| Hit@10 | +0.190 (+0.130, +0.250) | Yes | Yes |
| MRR@10 (unique laws) | +0.241 (+0.182, +0.301) | Yes | Yes |
| Top-1 accuracy | +0.240 (+0.160, +0.320) | Yes | Yes |
| Majority-vote accuracy (predominance ≥ 0.50) | +0.270 (+0.200, +0.335) | Yes | Yes |
| Mean consensus fraction | +0.207 (+0.164, +0.249) | Yes | Yes |
| Mean lift vs prior | +22.329 (+15.889, +29.655) | Yes | Yes |

Interpretation: the superiority column reflects the one-sided criterion used in the evaluation (paired 95% CI lower bound > 0).

**Table 3.** Paired deltas for KAHM(query→MB corpus) vs Mixedbread (95% bootstrap CI).

| Metric | Δ (KAHM − Mixedbread) | 95% CI excludes 0? |
| --- | --- | --- |
| Hit@10 | -0.005 (-0.040, +0.030) | No |
| MRR@10 (unique laws) | +0.009 (-0.032, +0.049) | No |
| Top-1 accuracy | +0.000 (-0.060, +0.065) | No |
| Majority-vote accuracy (predominance ≥ 0.50) | +0.055 (+0.000, +0.110) | No |
| Mean consensus fraction | +0.021 (-0.007, +0.048) | No |
| Mean lift vs prior | +2.727 (-0.966, +6.568) | No |

### Majority-vote behavior

**Table 4.** Vote-distribution diagnostics over the top-k neighborhood (k=10; 95% bootstrap CI).

| Method | Top-law fraction | Vote margin | Vote entropy | #Unique laws | P(all 10 one law) |
| --- | --- | --- | --- | --- | --- |
| Mixedbread (true) | 0.682 (0.650, 0.715) | 0.511 (0.465, 0.558) | 0.794 (0.714, 0.874) | 3.205 (2.955, 3.460) | 0.180 (0.130, 0.235) |
| IDF–SVD | 0.547 (0.514, 0.583) | 0.365 (0.322, 0.409) | 1.145 (1.061, 1.228) | 4.395 (4.105, 4.680) | 0.100 (0.060, 0.145) |
| KAHM(query→MB corpus) | 0.679 (0.647, 0.713) | 0.509 (0.465, 0.556) | 0.785 (0.710, 0.865) | 3.110 (2.875, 3.360) | 0.195 (0.140, 0.250) |
| Full-KAHM (query→KAHM corpus) | 0.677 (0.642, 0.714) | 0.521 (0.474, 0.570) | 0.811 (0.725, 0.896) | 3.310 (3.030, 3.585) | 0.215 (0.160, 0.270) |

### Vote-based routing

**Table 5.** Routing sweep comparing KAHM(query→MB corpus) and Mixedbread at different predominance thresholds τ (95% bootstrap CI).

| τ | Coverage (KAHM) | Accuracy among covered (KAHM) | Routing accuracy (KAHM) | Coverage (Mixedbread) | Accuracy among covered (Mixedbread) | Routing accuracy (Mixedbread) |
| --- | --- | --- | --- | --- | --- | --- |
| 0.50 | 0.780 (0.720, 0.835) | 0.788 (0.721, 0.849) | 0.615 (0.545, 0.680) | 0.800 (0.740, 0.855) | 0.700 (0.630, 0.768) | 0.560 (0.490, 0.625) |
| 0.60 | 0.690 (0.625, 0.755) | 0.797 (0.729, 0.863) | 0.550 (0.480, 0.620) | 0.680 (0.615, 0.745) | 0.757 (0.687, 0.826) | 0.515 (0.445, 0.585) |
| 0.70 | 0.550 (0.480, 0.620) | 0.809 (0.731, 0.881) | 0.445 (0.375, 0.515) | 0.540 (0.470, 0.605) | 0.787 (0.705, 0.862) | 0.425 (0.355, 0.495) |
| 0.80 | 0.390 (0.325, 0.460) | 0.872 (0.793, 0.940) | 0.340 (0.275, 0.405) | 0.430 (0.365, 0.500) | 0.837 (0.756, 0.910) | 0.360 (0.295, 0.425) |

**Table 6.** Paired deltas vs Mixedbread for routing (KAHM(query→MB corpus) − Mixedbread).

| τ | ΔCoverage | ΔRouting accuracy |
| --- | --- | --- |
| 0.50 | -0.020 (-0.080, +0.035) | +0.055 (+0.000, +0.110) |
| 0.60 | +0.010 (-0.050, +0.070) | +0.035 (-0.015, +0.090) |
| 0.70 | +0.010 (-0.055, +0.075) | +0.020 (-0.035, +0.070) |
| 0.80 | -0.040 (-0.100, +0.020) | -0.020 (-0.070, +0.030) |

**Table 7.** Decomposition of ΔRouting accuracy into coverage and precision effects.

Point estimates:
| τ | ΔRouting accuracy | Coverage contribution | Precision contribution |
| --- | --- | --- | --- |
| 0.50 | +0.055 | -0.015 | +0.070 |
| 0.60 | +0.035 | +0.008 | +0.027 |
| 0.70 | +0.020 | +0.008 | +0.012 |
| 0.80 | -0.020 | -0.034 | +0.014 |

With paired-bootstrap CIs:
| τ | ΔRouting accuracy | Coverage contribution | Precision contribution |
| --- | --- | --- | --- |
| 0.50 | +0.055 (+0.000, +0.110) | -0.015 (-0.057, +0.029) | +0.070 (+0.021, +0.118) |
| 0.60 | +0.035 (-0.020, +0.085) | +0.008 (-0.038, +0.053) | +0.027 (-0.014, +0.068) |
| 0.70 | +0.020 (-0.030, +0.075) | +0.008 (-0.043, +0.060) | +0.012 (-0.025, +0.049) |
| 0.80 | -0.020 (-0.070, +0.030) | -0.034 (-0.085, +0.020) | +0.014 (-0.018, +0.046) |

**Table 8.** Suggested routing thresholds (coverage constraint and objectives).

Coverage constraint: **coverage ≥ 0.50**

| Method | τ* | Coverage | Accuracy among covered | Routing accuracy | Objective |
| --- | --- | --- | --- | --- | --- |
| Mixedbread (true) | 0.70 | 0.540 | 0.787 | 0.425 | Maximize precision |
| KAHM(query→MB corpus) | 0.70 | 0.550 | 0.809 | 0.445 | Maximize precision |
| Full-KAHM (query→KAHM corpus) | 0.70 | 0.545 | 0.872 | 0.475 | Maximize precision |
| IDF–SVD | 0.50 | 0.575 | 0.600 | 0.345 | Maximize precision |
| Mixedbread (true) | 0.50 | 0.800 | 0.700 | 0.560 | Maximize routing accuracy |
| KAHM(query→MB corpus) | 0.50 | 0.780 | 0.788 | 0.615 | Maximize routing accuracy |
| Full-KAHM (query→KAHM corpus) | 0.50 | 0.765 | 0.778 | 0.595 | Maximize routing accuracy |
| IDF–SVD | 0.50 | 0.575 | 0.600 | 0.345 | Maximize routing accuracy |

### Embedding-space alignment and neighborhood overlap

**Table 9.** Alignment and overlap statistics for Full-KAHM vs Mixedbread (95% bootstrap CI).

| Quantity | Estimate (95% CI) |
| --- | --- |
| Cosine alignment (corpus) | 0.9405 (0.9403, 0.9407) |
| Cosine alignment (queries) | 0.9365 (0.9322, 0.9405) |
| Sentence Jaccard@10 | 0.115 (0.099, 0.132) |
| Sentence overlap fraction@10 | 0.187 (0.163, 0.212) |
| Law-set Jaccard@10 | 0.545 (0.505, 0.585) |
| Δ sentence Jaccard (Full − IDF) | +0.063 (+0.046, +0.081) |
| Δ law-set Jaccard (Full − IDF) | +0.201 (+0.156, +0.244) |

## Reproducibility checklist

- Query set: `query_set.TEST_QUERY_SET`
- Corpus parquet: `ris_sentences.parquet`
- Mixedbread model: `mixedbread-ai/deepset-mxbai-embed-de-large-v1`
- Query prefix: `query: `
- IDF–SVD model: `idf_svd_model.joblib`
- KAHM model: `kahm_regressor_idf_to_mixedbread.joblib` (mode `soft`)
- Indices: MB `embedding_index.npz`, IDF `embedding_index_idf_svd.npz`, KAHM `embedding_index_kahm_mixedbread_approx.npz`
- Device: `cpu`
- Threads cap: 1
- Bootstrap: samples=5000, seed=0

---
## Summary paragraph

Across 200 queries (k=10), KAHM(query→MB corpus) achieved Hit@10=0.895 (0.850, 0.935) and MRR@10=0.729 (0.679, 0.779). Paired-bootstrap deltas supported KAHM(query→MB corpus) superiority over IDF–SVD on Hit@10, MRR@10, Top-1 accuracy (Table 2). Compared to Mixedbread, paired deltas for Hit@10, MRR@10, Top-1 accuracy were small with 95% CIs that included 0 (Table 3). Majority-vote behavior differed depending on the routing threshold τ (Tables 5–8). Majority-vote accuracy was numerically higher by +0.055 (+0.000, +0.110) relative to Mixedbread (CI touches 0). Full-KAHM embeddings showed high cosine alignment with Mixedbread in embedding space (mean corpus cosine 0.9405) and recovered similar law-level neighborhoods (law-set Jaccard@10=0.545 (0.505, 0.585); Δ vs IDF=+0.201 (+0.156, +0.244); Table 9), while sentence-level neighbor identity remained modest (sentence Jaccard@10=0.115 (0.099, 0.132); Δ vs IDF=+0.063 (+0.046, +0.081); Table 9).
