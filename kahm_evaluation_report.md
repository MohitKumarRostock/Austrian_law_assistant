# KAHM embeddings: retrieval evaluation on Austrian laws

**Generated (UTC):** 2026-02-07T14:47:52Z  
**Source script:** `evaluate_three_embeddings.py` (version `2026-01-25-combined-distance-gate-v1`)  

**KAHM query embedding strategy:** `query_model`  
**KAHM combine tie-break:** `query`  

## Abstract

We study whether KAHM can replace transformer-based query embedding at retrieval time by learning a lightweight mapping from an IDF–SVD representation into Mixedbread embedding space, enabling search against a fixed Mixedbread corpus index. On 5000 human-labeled queries over 14,487 aligned sentences (k=10), KAHM(query→MB corpus) achieved Hit@10=0.571 (0.557, 0.585) and MRR@10=0.425 (0.412, 0.437). Compared with IDF–SVD, KAHM changed Hit@10 by +0.076 (+0.063, +0.090) and MRR@10 by +0.067 (+0.056, +0.078) under paired bootstrap. Against Mixedbread, paired deltas for Hit@10, MRR@10, Top-1 accuracy had 95% CIs excluding 0. Mean lift vs prior changed by -3.418 (-5.315, -1.596).  Majority-vote accuracy was lower by -0.016 (-0.027, -0.006) versus Mixedbread.Finally, Full-KAHM embeddings showed high cosine alignment with Mixedbread geometry (corpus cosine=0.9940 (0.9937, 0.9942), query cosine=0.9557 (0.9550, 0.9563)) and recovered similar law-level neighborhoods (law-set Jaccard@10=0.331 (0.325, 0.337); Δ vs IDF=+0.047 (+0.039, +0.054)), while sentence-level neighbor identity remained modest (sentence Jaccard@10=0.038 (0.036, 0.040); Δ vs IDF=-0.033 (-0.036, -0.031)).

## Experimental configuration

The following settings were recorded in the evaluation run:

- Queries: **5000**
- Corpus sentences (aligned): **14487**
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

Each embedding index provides sentence identifiers that are aligned against a shared sentence-level corpus. All reported retrieval metrics are computed over the intersection of sentence identifiers present in each embedding index, yielding 14,487 common sentences.

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
| Mixedbread (true) | 0.628 (0.614, 0.641) | 0.466 (0.453, 0.478) | 0.369 (0.356, 0.382) | 0.318 (0.306, 0.331) | 0.315 (0.306, 0.325) | 52.011 (49.398, 54.759) |
| IDF–SVD | 0.495 (0.481, 0.509) | 0.358 (0.346, 0.370) | 0.281 (0.268, 0.293) | 0.257 (0.245, 0.269) | 0.260 (0.250, 0.270) | 38.860 (36.485, 41.363) |
| KAHM(query→MB corpus) | 0.571 (0.557, 0.585) | 0.425 (0.412, 0.437) | 0.340 (0.327, 0.353) | 0.302 (0.289, 0.315) | 0.297 (0.287, 0.306) | 48.593 (45.988, 51.213) |
| Full-KAHM (query→KAHM corpus) | 0.507 (0.494, 0.521) | 0.361 (0.349, 0.372) | 0.278 (0.266, 0.290) | 0.183 (0.172, 0.193) | 0.205 (0.197, 0.213) | 25.268 (24.016, 26.565) |

### Comparative analyses

**Table 2.** Paired deltas for KAHM(query→MB corpus) vs IDF–SVD (95% bootstrap CI).

| Metric | Δ (KAHM − IDF) | 95% CI excludes 0? | Superiority (lower CI > 0)? |
| --- | --- | --- | --- |
| Hit@10 | +0.076 (+0.063, +0.090) | Yes | Yes |
| MRR@10 (unique laws) | +0.067 (+0.056, +0.078) | Yes | Yes |
| Top-1 accuracy | +0.059 (+0.045, +0.072) | Yes | Yes |
| Majority-vote accuracy (predominance ≥ 0.50) | +0.045 (+0.034, +0.056) | Yes | Yes |
| Mean consensus fraction | +0.036 (+0.029, +0.044) | Yes | Yes |
| Mean lift vs prior | +9.733 (+7.088, +12.383) | Yes | Yes |

Interpretation: the superiority column reflects the one-sided criterion used in the evaluation (paired 95% CI lower bound > 0).

**Table 3.** Paired deltas for KAHM(query→MB corpus) vs Mixedbread (95% bootstrap CI).

| Metric | Δ (KAHM − Mixedbread) | 95% CI excludes 0? |
| --- | --- | --- |
| Hit@10 | -0.056 (-0.067, -0.045) | Yes |
| MRR@10 (unique laws) | -0.041 (-0.050, -0.032) | Yes |
| Top-1 accuracy | -0.029 (-0.041, -0.018) | Yes |
| Majority-vote accuracy (predominance ≥ 0.50) | -0.016 (-0.027, -0.006) | Yes |
| Mean consensus fraction | -0.018 (-0.024, -0.012) | Yes |
| Mean lift vs prior | -3.418 (-5.315, -1.596) | Yes |

### Majority-vote behavior

**Table 4.** Vote-distribution diagnostics over the top-k neighborhood (k=10; 95% bootstrap CI).

| Method | Top-law fraction | Vote margin | Vote entropy | #Unique laws | P(all 10 one law) |
| --- | --- | --- | --- | --- | --- |
| Mixedbread (true) | 0.532 (0.525, 0.539) | 0.342 (0.334, 0.351) | 1.171 (1.155, 1.187) | 4.449 (4.393, 4.507) | 0.076 (0.069, 0.083) |
| IDF–SVD | 0.549 (0.542, 0.556) | 0.363 (0.354, 0.372) | 1.117 (1.099, 1.134) | 4.250 (4.193, 4.306) | 0.116 (0.107, 0.125) |
| KAHM(query→MB corpus) | 0.521 (0.514, 0.528) | 0.329 (0.321, 0.337) | 1.197 (1.181, 1.214) | 4.557 (4.499, 4.615) | 0.079 (0.072, 0.087) |
| Full-KAHM (query→KAHM corpus) | 0.443 (0.437, 0.449) | 0.249 (0.243, 0.257) | 1.412 (1.398, 1.426) | 5.349 (5.292, 5.407) | 0.029 (0.025, 0.034) |

### Vote-based routing

**Table 5.** Routing sweep comparing KAHM(query→MB corpus) and Mixedbread at different predominance thresholds τ (95% bootstrap CI).

| τ | Coverage (KAHM) | Accuracy among covered (KAHM) | Routing accuracy (KAHM) | Coverage (Mixedbread) | Accuracy among covered (Mixedbread) | Routing accuracy (Mixedbread) |
| --- | --- | --- | --- | --- | --- | --- |
| 0.50 | 0.541 (0.527, 0.555) | 0.558 (0.539, 0.576) | 0.302 (0.289, 0.315) | 0.551 (0.538, 0.565) | 0.577 (0.558, 0.595) | 0.318 (0.305, 0.331) |
| 0.60 | 0.382 (0.369, 0.396) | 0.668 (0.646, 0.688) | 0.255 (0.243, 0.268) | 0.404 (0.391, 0.418) | 0.657 (0.637, 0.678) | 0.266 (0.253, 0.278) |
| 0.70 | 0.278 (0.266, 0.290) | 0.753 (0.730, 0.775) | 0.209 (0.198, 0.221) | 0.296 (0.283, 0.309) | 0.745 (0.723, 0.767) | 0.221 (0.209, 0.232) |
| 0.80 | 0.202 (0.191, 0.213) | 0.839 (0.817, 0.861) | 0.169 (0.159, 0.180) | 0.214 (0.203, 0.226) | 0.801 (0.776, 0.825) | 0.171 (0.161, 0.182) |

**Table 6.** Paired deltas vs Mixedbread for routing (KAHM(query→MB corpus) − Mixedbread).

| τ | ΔCoverage | ΔRouting accuracy |
| --- | --- | --- |
| 0.50 | -0.010 (-0.024, +0.004) | -0.016 (-0.026, -0.006) |
| 0.60 | -0.022 (-0.035, -0.009) | -0.010 (-0.020, -0.001) |
| 0.70 | -0.019 (-0.030, -0.007) | -0.012 (-0.020, -0.003) |
| 0.80 | -0.012 (-0.022, -0.002) | -0.002 (-0.010, +0.006) |

**Table 7.** Decomposition of ΔRouting accuracy into coverage and precision effects.

Point estimates:
| τ | ΔRouting accuracy | Coverage contribution | Precision contribution |
| --- | --- | --- | --- |
| 0.50 | -0.016 | -0.006 | -0.011 |
| 0.60 | -0.010 | -0.014 | +0.004 |
| 0.70 | -0.012 | -0.014 | +0.002 |
| 0.80 | -0.002 | -0.010 | +0.008 |

With paired-bootstrap CIs:
| τ | ΔRouting accuracy | Coverage contribution | Precision contribution |
| --- | --- | --- | --- |
| 0.50 | -0.016 (-0.026, -0.007) | -0.006 (-0.014, +0.003) | -0.011 (-0.020, -0.002) |
| 0.60 | -0.010 (-0.020, -0.001) | -0.014 (-0.023, -0.006) | +0.004 (-0.003, +0.012) |
| 0.70 | -0.012 (-0.020, -0.003) | -0.014 (-0.022, -0.005) | +0.002 (-0.004, +0.008) |
| 0.80 | -0.002 (-0.010, +0.006) | -0.010 (-0.018, -0.002) | +0.008 (+0.003, +0.013) |

**Table 8.** Suggested routing thresholds (coverage constraint and objectives).

Coverage constraint: **coverage ≥ 0.50**
Threshold search grid: **0.00–1.00 (step 0.01)**


| Method | τ* | Coverage | Accuracy among covered | Routing accuracy | Objective |
| --- | --- | --- | --- | --- | --- |
| Mixedbread (true) | 0.00 | 1.000 | 0.407 | 0.407 | Maximize precision |
| KAHM(query→MB corpus) | 0.11 | 0.996 | 0.377 | 0.375 | Maximize precision |
| Full-KAHM (query→KAHM corpus) | 0.00 | 1.000 | 0.280 | 0.280 | Maximize precision |
| IDF–SVD | 0.11 | 0.998 | 0.309 | 0.308 | Maximize precision |
| Mixedbread (true) | 0.00 | 1.000 | 0.407 | 0.407 | Maximize routing accuracy |
| KAHM(query→MB corpus) | 0.11 | 0.996 | 0.377 | 0.375 | Maximize routing accuracy |
| Full-KAHM (query→KAHM corpus) | 0.00 | 1.000 | 0.280 | 0.280 | Maximize routing accuracy |
| IDF–SVD | 0.11 | 0.998 | 0.309 | 0.308 | Maximize routing accuracy |

### Embedding-space alignment and neighborhood overlap

**Table 9.** Alignment and overlap statistics for Full-KAHM vs Mixedbread (95% bootstrap CI).

| Quantity | Estimate (95% CI) |
| --- | --- |
| Cosine alignment (corpus) | 0.9940 (0.9937, 0.9942) |
| Cosine alignment (queries) | 0.9557 (0.9550, 0.9563) |
| Sentence Jaccard@10 | 0.038 (0.036, 0.040) |
| Sentence overlap fraction@10 | 0.066 (0.063, 0.069) |
| Law-set Jaccard@10 | 0.331 (0.325, 0.337) |
| Δ sentence Jaccard (Full − IDF) | -0.033 (-0.036, -0.031) |
| Δ law-set Jaccard (Full − IDF) | +0.047 (+0.039, +0.054) |

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

Across 5000 queries (k=10), KAHM(query→MB corpus) achieved Hit@10=0.571 (0.557, 0.585) and MRR@10=0.425 (0.412, 0.437). Paired-bootstrap deltas supported KAHM(query→MB corpus) superiority over IDF–SVD on Hit@10, MRR@10, Top-1 accuracy (Table 2). Compared to Mixedbread, paired deltas indicated differences for Hit@10, MRR@10, Top-1 accuracy with 95% CIs excluding 0 (Table 3). Majority-vote behavior differed depending on the routing threshold τ (Tables 5–8). Majority-vote accuracy was lower by -0.016 (-0.027, -0.006) relative to Mixedbread. Full-KAHM embeddings showed high cosine alignment with Mixedbread in embedding space (mean corpus cosine 0.9940) and recovered similar law-level neighborhoods (law-set Jaccard@10=0.331 (0.325, 0.337); Δ vs IDF=+0.047 (+0.039, +0.054); Table 9), while sentence-level neighbor identity remained modest (sentence Jaccard@10=0.038 (0.036, 0.040); Δ vs IDF=-0.033 (-0.036, -0.031); Table 9).
