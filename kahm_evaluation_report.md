# KAHM embeddings: retrieval evaluation on Austrian laws

**Generated (UTC):** 2026-01-10T18:08:55Z  
**Source script:** `evaluate_three_embeddings.py` (version `2026-01-10-pubreport-v6-final3`)  

## Abstract

We study whether KAHM can replace transformer-based query embedding at retrieval time by learning a lightweight mapping from an IDF–SVD representation into Mixedbread embedding space, enabling search against a fixed Mixedbread corpus index. On 200 human-labeled queries over 71,069 aligned sentences (k=10), KAHM(query→MB corpus) achieved Hit@10=0.890 (0.845, 0.930) and MRR@10=0.738 (0.687, 0.788). Compared with IDF–SVD, KAHM improved Hit@10 by +0.215 (+0.155, +0.280) and MRR@10 by +0.274 (+0.220, +0.331) under paired bootstrap. Against Mixedbread, paired deltas for Hit@10, MRR@10, and Top-1 accuracy have 95% CIs including 0 (differences not resolved at n=200 under this bootstrap; not a formal equivalence claim), while mean lift vs prior increased by +3.407 (+0.306, +6.944). Majority-vote accuracy was numerically higher by +0.055 (+0.000, +0.110) versus Mixedbread (CI touches 0). Finally, Full-KAHM embeddings showed high cosine alignment with Mixedbread geometry (corpus cosine=0.9068 (0.9064, 0.9072), query cosine=0.9381 (0.9336, 0.9420)) and recovered similar law-level neighborhoods (law-set Jaccard@10=0.501 (0.464, 0.540); Δ vs IDF=+0.114 (+0.074, +0.157)), while sentence-level neighbor identity remained modest (sentence Jaccard@10=0.063 (0.054, 0.073); Δ vs IDF=+0.008 (-0.005, +0.020)).

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
| IDF–SVD | 0.675 (0.610, 0.740) | 0.464 (0.406, 0.524) | 0.350 (0.285, 0.420) | 0.375 (0.305, 0.445) | 0.360 (0.311, 0.412) | 17.696 (14.388, 21.367) |
| KAHM(query→MB corpus) | 0.890 (0.845, 0.930) | 0.738 (0.687, 0.788) | 0.625 (0.555, 0.695) | 0.615 (0.545, 0.680) | 0.550 (0.502, 0.598) | 40.329 (32.819, 48.380) |
| Full-KAHM (query→KAHM corpus) | 0.895 (0.850, 0.935) | 0.650 (0.595, 0.706) | 0.520 (0.450, 0.585) | 0.555 (0.485, 0.620) | 0.504 (0.456, 0.548) | 40.775 (31.067, 51.449) |

### Comparative analyses

**Table 2.** Paired deltas for KAHM(query→MB corpus) vs IDF–SVD (95% bootstrap CI).

| Metric | Δ (KAHM − IDF) | 95% CI excludes 0? | Superiority (lower CI > 0)? |
| --- | --- | --- | --- |
| Hit@10 | +0.215 (+0.155, +0.280) | Yes | Yes |
| MRR@10 (unique laws) | +0.274 (+0.220, +0.331) | Yes | Yes |
| Top-1 accuracy | +0.275 (+0.200, +0.350) | Yes | Yes |
| Majority-vote accuracy (predominance ≥ 0.50) | +0.240 (+0.175, +0.305) | Yes | Yes |
| Mean consensus fraction | +0.190 (+0.150, +0.231) | Yes | Yes |
| Mean lift vs prior | +22.633 (+15.966, +30.273) | Yes | Yes |

Interpretation: the superiority column reflects the one-sided criterion used in the evaluation (paired 95% CI lower bound > 0).

**Table 3.** Paired deltas for KAHM(query→MB corpus) vs Mixedbread (95% bootstrap CI).

| Metric | Δ (KAHM − Mixedbread) | 95% CI excludes 0? |
| --- | --- | --- |
| Hit@10 | -0.010 (-0.050, +0.025) | No |
| MRR@10 (unique laws) | +0.018 (-0.022, +0.055) | No |
| Top-1 accuracy | +0.020 (-0.040, +0.080) | No |
| Majority-vote accuracy (predominance ≥ 0.50) | +0.055 (+0.000, +0.110) | No |
| Mean consensus fraction | +0.017 (-0.010, +0.043) | No |
| Mean lift vs prior | +3.407 (+0.306, +6.944) | Yes |

### Majority-vote behavior

**Table 4.** Vote-distribution diagnostics over the top-k neighborhood (k=10; 95% bootstrap CI).

| Method | Top-law fraction | Vote margin | Vote entropy | #Unique laws | P(all 10 one law) |
| --- | --- | --- | --- | --- | --- |
| Mixedbread (true) | 0.682 (0.650, 0.715) | 0.511 (0.465, 0.558) | 0.794 (0.714, 0.874) | 3.205 (2.955, 3.460) | 0.180 (0.130, 0.235) |
| IDF–SVD | 0.559 (0.524, 0.595) | 0.376 (0.331, 0.425) | 1.088 (1.003, 1.176) | 4.170 (3.880, 4.465) | 0.135 (0.090, 0.185) |
| KAHM(query→MB corpus) | 0.671 (0.638, 0.703) | 0.494 (0.449, 0.543) | 0.788 (0.711, 0.868) | 3.110 (2.865, 3.370) | 0.215 (0.160, 0.275) |
| Full-KAHM (query→KAHM corpus) | 0.606 (0.569, 0.643) | 0.432 (0.384, 0.481) | 0.980 (0.890, 1.072) | 3.835 (3.530, 4.140) | 0.175 (0.125, 0.230) |

### Vote-based routing

**Table 5.** Routing sweep comparing KAHM(query→MB corpus) and Mixedbread at different predominance thresholds τ (95% bootstrap CI).

| τ | Coverage (KAHM) | Accuracy among covered (KAHM) | Routing accuracy (KAHM) | Coverage (Mixedbread) | Accuracy among covered (Mixedbread) | Routing accuracy (Mixedbread) |
| --- | --- | --- | --- | --- | --- | --- |
| 0.50 | 0.780 (0.720, 0.835) | 0.788 (0.721, 0.851) | 0.615 (0.545, 0.680) | 0.800 (0.740, 0.855) | 0.700 (0.630, 0.768) | 0.560 (0.490, 0.625) |
| 0.60 | 0.645 (0.575, 0.710) | 0.814 (0.746, 0.879) | 0.525 (0.455, 0.595) | 0.680 (0.615, 0.745) | 0.757 (0.687, 0.826) | 0.515 (0.445, 0.585) |
| 0.70 | 0.515 (0.445, 0.585) | 0.845 (0.772, 0.910) | 0.435 (0.365, 0.505) | 0.540 (0.470, 0.605) | 0.787 (0.705, 0.862) | 0.425 (0.355, 0.495) |
| 0.80 | 0.380 (0.315, 0.450) | 0.895 (0.821, 0.959) | 0.340 (0.275, 0.405) | 0.430 (0.365, 0.500) | 0.837 (0.756, 0.910) | 0.360 (0.295, 0.425) |

**Table 6.** Paired deltas vs Mixedbread for routing (KAHM(query→MB corpus) − Mixedbread).

| τ | ΔCoverage | ΔRouting accuracy |
| --- | --- | --- |
| 0.50 | -0.020 (-0.075, +0.035) | +0.055 (+0.000, +0.115) |
| 0.60 | -0.035 (-0.095, +0.025) | +0.010 (-0.045, +0.065) |
| 0.70 | -0.025 (-0.085, +0.035) | +0.010 (-0.040, +0.060) |
| 0.80 | -0.050 (-0.105, +0.000) | -0.020 (-0.070, +0.025) |

**Table 7.** Decomposition of ΔRouting accuracy into coverage and precision effects.

Point estimates:
| τ | ΔRouting accuracy | Coverage contribution | Precision contribution |
| --- | --- | --- | --- |
| 0.50 | +0.055 | -0.015 | +0.070 |
| 0.60 | +0.010 | -0.027 | +0.037 |
| 0.70 | +0.010 | -0.020 | +0.030 |
| 0.80 | -0.020 | -0.043 | +0.023 |

With paired-bootstrap CIs:
| τ | ΔRouting accuracy | Coverage contribution | Precision contribution |
| --- | --- | --- | --- |
| 0.50 | +0.055 (+0.000, +0.115) | -0.015 (-0.058, +0.028) | +0.070 (+0.025, +0.117) |
| 0.60 | +0.010 (-0.050, +0.065) | -0.027 (-0.077, +0.020) | +0.037 (+0.005, +0.071) |
| 0.70 | +0.010 (-0.040, +0.060) | -0.020 (-0.068, +0.028) | +0.030 (+0.002, +0.061) |
| 0.80 | -0.020 (-0.065, +0.030) | -0.043 (-0.089, +0.004) | +0.023 (+0.000, +0.047) |

**Table 8.** Suggested routing thresholds (coverage constraint and objectives).

Coverage constraint: **coverage ≥ 0.50**

| Method | τ* | Coverage | Accuracy among covered | Routing accuracy | Objective |
| --- | --- | --- | --- | --- | --- |
| Mixedbread (true) | 0.70 | 0.540 | 0.787 | 0.425 | Maximize precision |
| KAHM(query→MB corpus) | 0.70 | 0.515 | 0.845 | 0.435 | Maximize precision |
| Full-KAHM (query→KAHM corpus) | 0.60 | 0.505 | 0.851 | 0.430 | Maximize precision |
| IDF–SVD | 0.50 | 0.610 | 0.615 | 0.375 | Maximize precision |
| Mixedbread (true) | 0.50 | 0.800 | 0.700 | 0.560 | Maximize routing accuracy |
| KAHM(query→MB corpus) | 0.50 | 0.780 | 0.788 | 0.615 | Maximize routing accuracy |
| Full-KAHM (query→KAHM corpus) | 0.50 | 0.675 | 0.822 | 0.555 | Maximize routing accuracy |
| IDF–SVD | 0.50 | 0.610 | 0.615 | 0.375 | Maximize routing accuracy |

### Embedding-space alignment and neighborhood overlap

**Table 9.** Alignment and overlap statistics for Full-KAHM vs Mixedbread (95% bootstrap CI).

| Quantity | Estimate (95% CI) |
| --- | --- |
| Cosine alignment (corpus) | 0.9068 (0.9064, 0.9072) |
| Cosine alignment (queries) | 0.9381 (0.9336, 0.9420) |
| Sentence Jaccard@10 | 0.063 (0.054, 0.073) |
| Sentence overlap fraction@10 | 0.112 (0.097, 0.128) |
| Law-set Jaccard@10 | 0.501 (0.464, 0.540) |
| Δ sentence Jaccard (Full − IDF) | +0.008 (-0.005, +0.020) |
| Δ law-set Jaccard (Full − IDF) | +0.114 (+0.074, +0.157) |

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

Across 200 queries (k=10), KAHM(query→MB corpus) achieved Hit@10=0.890 (0.845, 0.930) and MRR@10=0.738 (0.687, 0.788). Paired-bootstrap deltas favored KAHM(query→MB corpus) over IDF–SVD under the evaluation’s superiority criterion (Table 2). Compared to Mixedbread, paired deltas for Hit@10, MRR@10, and Top-1 accuracy were small with 95% CIs that included 0 (Table 3), while majority-vote behavior differed depending on the routing threshold τ (Tables 5–8). Majority-vote accuracy was numerically higher by +0.055 (+0.000, +0.110) relative to Mixedbread (CI touches 0). Full-KAHM embeddings showed high cosine alignment with Mixedbread in embedding space (mean corpus cosine 0.9068) and recovered similar law-level neighborhoods (law-set Jaccard@10=0.501 (0.464, 0.540); Δ vs IDF=+0.114 (+0.074, +0.157); Table 9), while sentence-level neighbor identity remained modest (sentence Jaccard@10=0.063 (0.054, 0.073); Δ vs IDF includes 0; Table 9).
