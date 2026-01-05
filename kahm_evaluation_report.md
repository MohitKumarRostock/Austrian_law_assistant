# KAHM embeddings: retrieval evaluation on Austrian laws

**Generated (UTC):** 2026-01-05T19:27:49Z  
**Script:** `evaluate_three_embeddings.py` (version `2026-01-05-storylines-v4-majority-v3-pubreport`)  

## Experimental setup

- Queries: **200**  
- Corpus sentences (aligned): **71069**  
- Embedding dimension: **1024**  
- Retrieval depth: **k=10**  
- Bootstrap: **paired nonparametric**, 5000 samples, seed=0  
- Majority-vote routing thresholds: 0.5,0.6,0.7,0.8  
- Predominance fraction (per-query metric): 0.50

## Main retrieval results

| Method | Hit@10 | MRR@10 (unique laws) | Top-1 accuracy | Majority accuracy | Mean consensus fraction | Mean lift vs prior |
| --- | --- | --- | --- | --- | --- | --- |
| Mixedbread (true) | 0.900 (0.855, 0.940) | 0.720 (0.670, 0.770) | 0.605 (0.540, 0.670) | 0.560 (0.490, 0.630) | 0.533 (0.485, 0.582) | 36.923 (30.426, 44.028) |
| IDF–SVD | 0.675 (0.610, 0.740) | 0.464 (0.406, 0.524) | 0.350 (0.285, 0.420) | 0.375 (0.305, 0.445) | 0.360 (0.311, 0.412) | 17.696 (14.388, 21.367) |
| KAHM(query→MB corpus) | 0.920 (0.880, 0.955) | 0.729 (0.679, 0.777) | 0.595 (0.520, 0.665) | 0.630 (0.560, 0.695) | 0.557 (0.510, 0.604) | 39.589 (32.361, 47.312) |
| Full-KAHM (query→KAHM corpus) | 0.855 (0.805, 0.905) | 0.673 (0.617, 0.726) | 0.550 (0.480, 0.620) | 0.650 (0.580, 0.715) | 0.582 (0.530, 0.630) | 45.800 (35.158, 57.590) |

Notes: Metrics are averaged across queries. Values are point estimates with 95% bootstrap confidence intervals.

## Storyline A: Superiority vs low-cost baseline (KAHM(q→MB) vs IDF–SVD)

| Metric | Δ (KAHM − IDF) | Decision |
| --- | --- | --- |
| hit@k | +0.245 (+0.185, +0.310) | PASS |
| MRR@k (unique laws) | +0.265 (+0.210, +0.320) | PASS |
| top1-accuracy | +0.245 (+0.175, +0.320) | PASS |
| majority-accuracy | +0.255 (+0.190, +0.320) | PASS |
| mean consensus fraction | +0.196 (+0.157, +0.238) | PASS |
| mean lift (prior) | +21.893 (+15.590, +29.178) | PASS |

**Verdict:** Supported

## Storyline B: Competitiveness vs Mixedbread (KAHM(q→MB) vs MB)

| Metric | Δ (KAHM − MB) | CI excludes 0? |
| --- | --- | --- |
| hit@k | +0.020 (-0.005, +0.050) | No |
| MRR@k (unique laws) | +0.009 (-0.028, +0.044) | No |
| top1-accuracy | -0.010 (-0.065, +0.045) | No |
| majority-accuracy | +0.070 (+0.020, +0.125) | Yes |
| mean consensus fraction | +0.023 (-0.001, +0.048) | No |
| mean lift (prior) | +2.666 (-0.276, +5.787) | No |

## Majority-vote behavior and vote-based routing

We evaluate *top-k law voting* statistics that characterize neighborhood purity and support a simple routing heuristic based on the top-law fraction (τ).

| Method | Top-law fraction | Vote margin | Vote entropy | #Unique laws | P(all 10 one law) |
| --- | --- | --- | --- | --- | --- |
| Mixedbread (true) | 0.682 (0.650, 0.715) | 0.511 (0.465, 0.558) | 0.794 (0.714, 0.874) | 3.205 (2.955, 3.460) | 0.180 (0.130, 0.235) |
| IDF–SVD | 0.559 (0.524, 0.595) | 0.376 (0.331, 0.425) | 1.088 (1.003, 1.176) | 4.170 (3.880, 4.465) | 0.135 (0.090, 0.185) |
| KAHM(query→MB corpus) | 0.678 (0.645, 0.711) | 0.509 (0.465, 0.556) | 0.786 (0.710, 0.867) | 3.115 (2.875, 3.375) | 0.210 (0.155, 0.265) |
| Full-KAHM (query→KAHM corpus) | 0.701 (0.663, 0.738) | 0.565 (0.517, 0.611) | 0.775 (0.688, 0.865) | 3.265 (2.985, 3.555) | 0.215 (0.160, 0.275) |

### Routing threshold sweep (KAHM(q→MB) vs MB)

| τ | Coverage (KAHM) | Acc|Covered (KAHM) | Maj-Acc (KAHM) | Coverage (MB) | Acc|Covered (MB) | Maj-Acc (MB) |
| --- | --- | --- | --- | --- | --- | --- |
| 0.50 | 0.805 (0.750, 0.860) | 0.783 (0.716, 0.842) | 0.630 (0.560, 0.695) | 0.800 (0.740, 0.855) | 0.700 (0.630, 0.768) | 0.560 (0.490, 0.625) |
| 0.60 | 0.660 (0.595, 0.725) | 0.818 (0.750, 0.884) | 0.540 (0.470, 0.610) | 0.680 (0.615, 0.745) | 0.757 (0.687, 0.826) | 0.515 (0.445, 0.585) |
| 0.70 | 0.535 (0.465, 0.605) | 0.822 (0.745, 0.892) | 0.440 (0.370, 0.510) | 0.540 (0.470, 0.605) | 0.787 (0.705, 0.862) | 0.425 (0.355, 0.495) |
| 0.80 | 0.390 (0.325, 0.460) | 0.885 (0.806, 0.949) | 0.345 (0.280, 0.410) | 0.430 (0.365, 0.500) | 0.837 (0.756, 0.910) | 0.360 (0.295, 0.425) |

### Paired deltas vs Mixedbread for routing (KAHM(q→MB) − MB)

| τ | ΔCoverage | ΔMajority-Acc |
| --- | --- | --- |
| 0.50 | +0.005 (-0.045, +0.055) | +0.070 (+0.015, +0.125) |
| 0.60 | -0.020 (-0.085, +0.040) | +0.025 (-0.030, +0.080) |
| 0.70 | -0.005 (-0.065, +0.055) | +0.015 (-0.030, +0.065) |
| 0.80 | -0.040 (-0.100, +0.020) | -0.015 (-0.070, +0.035) |

### Decomposition of ΔMajority-Acc into coverage vs precision effects

Point-estimate decomposition:
| τ | ΔMaj-Acc | ΔCoverage part | ΔPrecision part |
| --- | --- | --- | --- |
| 0.50 | +0.070 | +0.004 | +0.066 |
| 0.60 | +0.025 | -0.016 | +0.041 |
| 0.70 | +0.015 | -0.004 | +0.019 |
| 0.80 | -0.015 | -0.034 | +0.019 |

With paired-bootstrap CIs:
| τ | ΔMaj-Acc | ΔCoverage part | ΔPrecision part |
| --- | --- | --- | --- |
| 0.50 | +0.070 (+0.020, +0.125) | +0.004 (-0.035, +0.044) | +0.066 (+0.023, +0.111) |
| 0.60 | +0.025 (-0.030, +0.080) | -0.016 (-0.064, +0.032) | +0.041 (+0.007, +0.076) |
| 0.70 | +0.015 (-0.035, +0.065) | -0.004 (-0.049, +0.043) | +0.019 (-0.008, +0.048) |
| 0.80 | -0.015 (-0.065, +0.040) | -0.034 (-0.086, +0.017) | +0.019 (-0.006, +0.046) |

### Suggested routing thresholds

Coverage constraint: **coverage ≥ 0.50**

| Method | τ* | Coverage | Acc|Covered | Majority-Acc | Objective |
| --- | --- | --- | --- | --- | --- |
| Mixedbread (true) | 0.70 | 0.540 | 0.787 | 0.425 | Max precision |
| KAHM(query→MB corpus) | 0.70 | 0.535 | 0.822 | 0.440 | Max precision |
| Full-KAHM (query→KAHM corpus) | 0.80 | 0.540 | 0.889 | 0.480 | Max precision |
| IDF–SVD | 0.50 | 0.610 | 0.615 | 0.375 | Max precision |
| Mixedbread (true) | 0.50 | 0.800 | 0.700 | 0.560 | Max majority-acc |
| KAHM(query→MB corpus) | 0.50 | 0.805 | 0.783 | 0.630 | Max majority-acc |
| Full-KAHM (query→KAHM corpus) | 0.50 | 0.790 | 0.823 | 0.650 | Max majority-acc |
| IDF–SVD | 0.50 | 0.610 | 0.615 | 0.375 | Max majority-acc |

## Storyline C: Embedding-space alignment and neighborhood overlap (Full-KAHM vs MB)

| Quantity | Estimate (95% CI) |
| --- | --- |
| Cosine alignment (corpus) | 0.9539 (0.9537, 0.9541) |
| Cosine alignment (queries) | 0.9383 (0.9341, 0.9420) |
| Sentence Jaccard@10 | 0.127 (0.110, 0.145) |
| Sentence overlap fraction@10 | 0.206 (0.180, 0.231) |
| Law-set Jaccard@10 | 0.540 (0.501, 0.581) |
| Δ sentence Jaccard (Full−IDF) | +0.072 (+0.055, +0.090) |
| Δ law-set Jaccard (Full−IDF) | +0.154 (+0.110, +0.201) |

---
### Copy-ready summary paragraph

Across 200 queries (k=10), KAHM(query→MB) achieves Hit@10=0.920 and MRR@10=0.729 and substantially outperforms the low-cost IDF–SVD baseline (Hit@10=0.675, MRR@10=0.464). Relative to Mixedbread (Hit@10=0.900, MRR@10=0.720), KAHM(query→MB) is competitive at top-k and exhibits improved majority-vote behavior for low τ thresholds, with gains primarily attributable to higher precision among covered cases in the vote-based routing decomposition. Full-KAHM embeddings are strongly aligned with MB in cosine space, and their retrieval neighborhoods are statistically closer to MB than IDF–SVD neighborhoods, supporting the geometric alignment storyline.
