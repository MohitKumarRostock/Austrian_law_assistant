# KAHM embeddings: retrieval evaluation on Austrian laws

**Generated (UTC):** 2026-01-06T10:19:38Z  
**Source script:** `evaluate_three_embeddings.py` (version `2026-01-05-storylines-v4-majority-v3-pubreport`)  

## Abstract

We evaluate KAHM embeddings for sentence-level retrieval in Austrian legal texts by comparing (i) a strong low-cost baseline (IDF–SVD), (ii) a Mixedbread SentenceTransformers baseline, and (iii) KAHM query embeddings obtained by regressing IDF–SVD representations into Mixedbread space. Across 200 queries (k=10), KAHM(query→MB corpus) achieved Hit@10=0.920 and MRR@10=0.729, substantially improving over IDF–SVD (Hit@10=0.675, MRR@10=0.464) and remaining competitive with Mixedbread (Hit@10=0.900, MRR@10=0.720). Majority-vote accuracy increased by +0.070 [+0.020, +0.125]. Full-KAHM embeddings exhibited strong cosine alignment with Mixedbread in embedding space (mean cosine=0.9539) and their retrieval neighborhoods were closer to Mixedbread than those of IDF–SVD by overlap-based measures.

## Experimental configuration

The following settings were recorded in the evaluation run:

- Queries: **200**
- Corpus sentences (aligned): **71069**
- Embedding dimension: **1024**
- Retrieval depth: **k=10**
- Bootstrap: **paired nonparametric**, 5000 samples, seed=0
- Majority-vote routing thresholds: 0.5,0.6,0.7,0.8
- Predominance fraction (per-query majority accuracy): **0.50**

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

Let the reference label for a query be its *consensus law* (a single law identifier). Metrics are computed per query and then averaged:

- **Hit@k:** 1 if the consensus law appears at least once among the top-k retrieved laws; else 0.
- **Top-1 accuracy:** 1 if the top-ranked retrieved law equals the consensus law; else 0.
- **MRR@k (unique laws):** form the ordered list of *unique* laws appearing in the top-k list; if the consensus law occurs at rank r in this list, score 1/r; otherwise 0.
- **Majority accuracy (predominance ≥ 0.50):** let the majority law be the most frequent law in the top-k list and let f be its fraction. Score 1 if (i) the majority law equals the consensus law and (ii) f ≥ 0.50; otherwise 0.
- **Mean consensus fraction:** fraction of the top-k items belonging to the consensus law (count/k).
- **Mean lift vs prior:** for each query, divide the consensus fraction by the marginal probability of the consensus law in the aligned corpus; report the mean across queries.

### Majority-vote diagnostics and routing

We summarize the law vote distribution over each query’s top-k neighborhood using:
- **Top-law fraction:** fraction of the top-k belonging to the most frequent law.
- **Vote margin:** (top-law fraction) minus (runner-up law fraction).
- **Vote entropy:** Shannon entropy of the empirical law distribution in the top-k list.
- **#Unique laws:** number of distinct laws appearing in the top-k list.
For routing, we evaluate a threshold rule that accepts the majority-law prediction when the top-law fraction ≥ τ. We report (i) **coverage** P(covered), (ii) **accuracy among covered** P(correct | covered), and (iii) **majority accuracy** P(correct ∩ covered).

### Statistical analysis

We estimate 95% confidence intervals (CIs) using a paired, nonparametric bootstrap over queries. For paired deltas, the statistic is computed on per-query differences; a delta is treated as statistically different from 0 when its 95% CI excludes 0.

## Results

### Main retrieval performance

**Table 1.** Main retrieval metrics (mean over queries; 95% bootstrap CI).

| Method | Hit@10 | MRR@10 (unique laws) | Top-1 accuracy | Majority accuracy | Mean consensus fraction | Mean lift vs prior |
| --- | --- | --- | --- | --- | --- | --- |
| Mixedbread (true) | 0.900 (0.855, 0.940) | 0.720 (0.670, 0.770) | 0.605 (0.540, 0.670) | 0.560 (0.490, 0.630) | 0.533 (0.485, 0.582) | 36.923 (30.426, 44.028) |
| IDF–SVD | 0.675 (0.610, 0.740) | 0.464 (0.406, 0.524) | 0.350 (0.285, 0.420) | 0.375 (0.305, 0.445) | 0.360 (0.311, 0.412) | 17.696 (14.388, 21.367) |
| KAHM(query→MB corpus) | 0.920 (0.880, 0.955) | 0.729 (0.679, 0.777) | 0.595 (0.520, 0.665) | 0.630 (0.560, 0.695) | 0.557 (0.510, 0.604) | 39.589 (32.361, 47.312) |
| Full-KAHM (query→KAHM corpus) | 0.855 (0.805, 0.905) | 0.673 (0.617, 0.726) | 0.550 (0.480, 0.620) | 0.650 (0.580, 0.715) | 0.582 (0.530, 0.630) | 45.800 (35.158, 57.590) |

### Comparative analyses

**Table 2.** Paired deltas for KAHM(query→MB corpus) vs IDF–SVD (95% bootstrap CI).

| Metric | Δ (KAHM − IDF) | 95% CI excludes 0? | Superiority (lower CI > 0)? |
| --- | --- | --- | --- |
| hit@k | +0.245 (+0.185, +0.310) | Yes | Yes |
| MRR@k (unique laws) | +0.265 (+0.210, +0.320) | Yes | Yes |
| top1-accuracy | +0.245 (+0.175, +0.320) | Yes | Yes |
| majority-accuracy | +0.255 (+0.190, +0.320) | Yes | Yes |
| mean consensus fraction | +0.196 (+0.157, +0.238) | Yes | Yes |
| mean lift (prior) | +21.893 (+15.590, +29.178) | Yes | Yes |

Interpretation: the superiority column reflects the one-sided criterion used in the evaluation (paired 95% CI lower bound > 0).

**Table 3.** Paired deltas for KAHM(query→MB corpus) vs Mixedbread (95% bootstrap CI).

| Metric | Δ (KAHM − Mixedbread) | 95% CI excludes 0? |
| --- | --- | --- |
| hit@k | +0.020 (-0.005, +0.050) | No |
| MRR@k (unique laws) | +0.009 (-0.028, +0.044) | No |
| top1-accuracy | -0.010 (-0.065, +0.045) | No |
| majority-accuracy | +0.070 (+0.020, +0.125) | Yes |
| mean consensus fraction | +0.023 (-0.001, +0.048) | No |
| mean lift (prior) | +2.666 (-0.276, +5.787) | No |

### Majority-vote behavior

**Table 4.** Vote-distribution diagnostics over the top-k neighborhood (k=10; 95% bootstrap CI).

| Method | Top-law fraction | Vote margin | Vote entropy | #Unique laws | P(all 10 one law) |
| --- | --- | --- | --- | --- | --- |
| Mixedbread (true) | 0.682 (0.650, 0.715) | 0.511 (0.465, 0.558) | 0.794 (0.714, 0.874) | 3.205 (2.955, 3.460) | 0.180 (0.130, 0.235) |
| IDF–SVD | 0.559 (0.524, 0.595) | 0.376 (0.331, 0.425) | 1.088 (1.003, 1.176) | 4.170 (3.880, 4.465) | 0.135 (0.090, 0.185) |
| KAHM(query→MB corpus) | 0.678 (0.645, 0.711) | 0.509 (0.465, 0.556) | 0.786 (0.710, 0.867) | 3.115 (2.875, 3.375) | 0.210 (0.155, 0.265) |
| Full-KAHM (query→KAHM corpus) | 0.701 (0.663, 0.738) | 0.565 (0.517, 0.611) | 0.775 (0.688, 0.865) | 3.265 (2.985, 3.555) | 0.215 (0.160, 0.275) |

### Vote-based routing

**Table 5.** Routing sweep comparing KAHM(query→MB corpus) and Mixedbread at different predominance thresholds τ (95% bootstrap CI).

| τ | Coverage (KAHM) | Accuracy among covered (KAHM) | Majority accuracy (KAHM) | Coverage (Mixedbread) | Accuracy among covered (Mixedbread) | Majority accuracy (Mixedbread) |
| --- | --- | --- | --- | --- | --- | --- |
| 0.50 | 0.805 (0.750, 0.860) | 0.783 (0.716, 0.842) | 0.630 (0.560, 0.695) | 0.800 (0.740, 0.855) | 0.700 (0.630, 0.768) | 0.560 (0.490, 0.625) |
| 0.60 | 0.660 (0.595, 0.725) | 0.818 (0.750, 0.884) | 0.540 (0.470, 0.610) | 0.680 (0.615, 0.745) | 0.757 (0.687, 0.826) | 0.515 (0.445, 0.585) |
| 0.70 | 0.535 (0.465, 0.605) | 0.822 (0.745, 0.892) | 0.440 (0.370, 0.510) | 0.540 (0.470, 0.605) | 0.787 (0.705, 0.862) | 0.425 (0.355, 0.495) |
| 0.80 | 0.390 (0.325, 0.460) | 0.885 (0.806, 0.949) | 0.345 (0.280, 0.410) | 0.430 (0.365, 0.500) | 0.837 (0.756, 0.910) | 0.360 (0.295, 0.425) |

**Table 6.** Paired deltas vs Mixedbread for routing (KAHM(query→MB corpus) − Mixedbread).

| τ | ΔCoverage | ΔMajority accuracy |
| --- | --- | --- |
| 0.50 | +0.005 (-0.045, +0.055) | +0.070 (+0.015, +0.125) |
| 0.60 | -0.020 (-0.085, +0.040) | +0.025 (-0.030, +0.080) |
| 0.70 | -0.005 (-0.065, +0.055) | +0.015 (-0.030, +0.065) |
| 0.80 | -0.040 (-0.100, +0.020) | -0.015 (-0.070, +0.035) |

**Table 7.** Decomposition of ΔMajority accuracy into coverage and precision effects.

Point estimates:
| τ | ΔMajority accuracy | Coverage contribution | Precision contribution |
| --- | --- | --- | --- |
| 0.50 | +0.070 | +0.004 | +0.066 |
| 0.60 | +0.025 | -0.016 | +0.041 |
| 0.70 | +0.015 | -0.004 | +0.019 |
| 0.80 | -0.015 | -0.034 | +0.019 |

With paired-bootstrap CIs:
| τ | ΔMajority accuracy | Coverage contribution | Precision contribution |
| --- | --- | --- | --- |
| 0.50 | +0.070 (+0.020, +0.125) | +0.004 (-0.035, +0.044) | +0.066 (+0.023, +0.111) |
| 0.60 | +0.025 (-0.030, +0.080) | -0.016 (-0.064, +0.032) | +0.041 (+0.007, +0.076) |
| 0.70 | +0.015 (-0.035, +0.065) | -0.004 (-0.049, +0.043) | +0.019 (-0.008, +0.048) |
| 0.80 | -0.015 (-0.065, +0.040) | -0.034 (-0.086, +0.017) | +0.019 (-0.006, +0.046) |

**Table 8.** Suggested routing thresholds (coverage constraint and objectives).

Coverage constraint: **coverage ≥ 0.50**

| Method | τ* | Coverage | Accuracy among covered | Majority accuracy | Objective |
| --- | --- | --- | --- | --- | --- |
| Mixedbread (true) | 0.70 | 0.540 | 0.787 | 0.425 | Maximize precision |
| KAHM(query→MB corpus) | 0.70 | 0.535 | 0.822 | 0.440 | Maximize precision |
| Full-KAHM (query→KAHM corpus) | 0.80 | 0.540 | 0.889 | 0.480 | Maximize precision |
| IDF–SVD | 0.50 | 0.610 | 0.615 | 0.375 | Maximize precision |
| Mixedbread (true) | 0.50 | 0.800 | 0.700 | 0.560 | Maximize majority accuracy |
| KAHM(query→MB corpus) | 0.50 | 0.805 | 0.783 | 0.630 | Maximize majority accuracy |
| Full-KAHM (query→KAHM corpus) | 0.50 | 0.790 | 0.823 | 0.650 | Maximize majority accuracy |
| IDF–SVD | 0.50 | 0.610 | 0.615 | 0.375 | Maximize majority accuracy |

### Embedding-space alignment and neighborhood overlap

**Table 9.** Alignment and overlap statistics for Full-KAHM vs Mixedbread (95% bootstrap CI).

| Quantity | Estimate (95% CI) |
| --- | --- |
| Cosine alignment (corpus) | 0.9539 (0.9537, 0.9541) |
| Cosine alignment (queries) | 0.9383 (0.9341, 0.9420) |
| Sentence Jaccard@10 | 0.127 (0.110, 0.145) |
| Sentence overlap fraction@10 | 0.206 (0.180, 0.231) |
| Law-set Jaccard@10 | 0.540 (0.501, 0.581) |
| Δ sentence Jaccard (Full − IDF) | +0.072 (+0.055, +0.090) |
| Δ law-set Jaccard (Full − IDF) | +0.154 (+0.110, +0.201) |

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
## Copy-ready summary paragraph

Across 200 queries (k=10), KAHM(query→MB corpus) achieved Hit@10=0.920 (0.880, 0.955) and MRR@10=0.729 (0.679, 0.777). Paired-bootstrap deltas favored KAHM(query→MB corpus) over IDF–SVD under the evaluation’s superiority criterion (Table 2). Compared to Mixedbread, paired deltas for Hit@10, MRR@10, and Top-1 accuracy were small with 95% CIs that included 0 (Table 3), while majority-vote behavior differed depending on the routing threshold τ (Tables 5–8). Majority accuracy differed by +0.070 [+0.020, +0.125] relative to Mixedbread. Full-KAHM embeddings showed strong cosine alignment with Mixedbread in embedding space (mean corpus cosine 0.9539) and higher neighborhood overlap to Mixedbread than IDF–SVD by sentence- and law-level Jaccard measures (Table 9).
