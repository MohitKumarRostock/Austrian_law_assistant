# KAHM embeddings: retrieval evaluation on Austrian laws

**Generated (UTC):** 2026-02-10T13:25:00Z  
**Source script:** `evaluate_three_embeddings_storylines.py` (version `2026-02-10-storylines-report-v1`)  

## Narrative focus

KAHM is evaluated here as a **compute-efficient, gradient-free alternative to transformer query encoders**: we keep a strong transformer corpus index fixed (Mixedbread) and replace online query encoding with a lightweight adapter that maps **IDF–SVD query features into the Mixedbread embedding space**. This follows a broader line of work on geometry-/kernel-inspired learning beyond gradient descent and operator-theoretic, gradient-free training over fixed embeddings.

**Key system idea:** offline transformer corpus embeddings; online gradient-free query adapter (IDF–SVD → KAHM → MB space).

## Experimental setup

- Queries: **5000**
- Corpus (aligned sentences): **14487**
- Embedding dimension (MB space): **1024**
- Evaluated cutoffs: **k = 10, 20, 50, 100**
- Majority-vote predominance threshold for majority-accuracy: **τ = 0.10**
- Mean lift (prior): consensus fraction divided by corpus prior for the consensus law.

## Methods compared

- **IDF–SVD:** low-cost lexical baseline (sparse/linear).
- **KAHM(query→MB corpus):** gradient-free query adapter: IDF–SVD query features mapped into Mixedbread space; retrieval against a frozen Mixedbread corpus index.
- **Mixedbread (true) (reference):** transformer query embedding + transformer corpus embeddings (reported for context; not the main claim).

## Performance measures

All reported quantities are computed **per query** at a cutoff *k* and then averaged across queries. Unless stated otherwise, higher is better. Confidence intervals are **paired bootstrap 95% CIs** (nonparametric).

### Retrieval quality

- **Hit@k**: indicator that the *consensus law* appears at least once in the top-*k* retrieved sentences. This is a law-level recall diagnostic (it ignores rank within the top-*k*).
- **Top-1 accuracy**: indicator that the very first retrieved sentence belongs to the consensus law. This is the strictest top-of-ranking measure; it is invariant to the choice of *k* (it only depends on rank 1).
- **MRR@k (unique laws)**: reciprocal rank of the consensus law in the **deduplicated** top-*k* list, where we scan the ranked list and keep only the first occurrence of each law. This evaluates law-level ranking while being robust to multiple sentences per law. Formally, if the consensus law is the *r*-th distinct law encountered, MRR = 1/r; if absent, 0.

### Consensus and routing diagnostics

- **Majority-accuracy (τ)**: indicator that the *majority-vote law* in the top-*k* list equals the consensus law **and** the majority share is at least τ (here τ=0.10). This measures how often the system would be correct when it chooses to make a **confident, vote-based** decision.
- **Mean consensus fraction**: for each query, the fraction of top-*k* retrieved sentences that belong to the consensus law. This is a continuous notion of neighborhood purity around the correct label.
- **Mean lift (prior)**: consensus fraction divided by the corpus prior of that law. Lift > 1 indicates enrichment above chance; it helps separate genuine semantic concentration from frequency effects (common laws have higher priors and thus require more concentration to achieve the same lift).

## Abstract

On 5000 human-labeled queries over 14487 aligned sentences, KAHM(query→MB corpus) achieves **hit@100 = 0.785 [0.774, 0.797]**, **MRR@100 (unique laws) = 0.497 [0.485, 0.508]**, **Top-1 accuracy = 0.393 [0.379, 0.406]**, **majority-accuracy (τ≥0.10) = 0.381 [0.367, 0.394]**, **mean consensus fraction = 0.217 [0.210, 0.224]**, and **mean lift (prior) = 27.260 [26.396, 28.154]**. Versus IDF–SVD, KAHM improves **MRR@100** by **+0.113 [+0.102, +0.124]**, **Top-1** by **+0.109 [+0.095, +0.123]**, **majority-accuracy** by **+0.108 [+0.096, +0.120]**, **mean consensus fraction** by **+0.041 [+0.036, +0.045]**, and **mean lift (prior)** by **+6.771 [+5.977, +7.543]** (paired bootstrap). Operationally, this supports KAHM as a query-time substitute that preserves a strong transformer index while removing transformer inference from the online path. Transformer-query reference numbers are provided in the Appendix for context.

## Results

### Top-of-ranking quality across k

The main story is captured by **MRR@k over unique laws** and **Top-1 accuracy**, complemented by **majority-accuracy**, **mean consensus fraction**, and **mean lift (prior)** as routing- and consensus-sensitive diagnostics.

**MRR@k (unique laws)** (mean with 95% CI)

| k | IDF–SVD | KAHM(query→MB corpus) |
| --- | --- | --- |
| 10 | 0.358 [0.346, 0.370] | 0.480 [0.468, 0.493] |
| 20 | 0.370 [0.358, 0.381] | 0.489 [0.476, 0.501] |
| 50 | 0.380 [0.368, 0.391] | 0.494 [0.482, 0.506] |
| 100 | 0.384 [0.373, 0.396] | 0.497 [0.485, 0.508] |

**Top-1 accuracy** (mean with 95% CI)

| k | IDF–SVD | KAHM(query→MB corpus) |
| --- | --- | --- |
| 10 | 0.284 [0.272, 0.297] | 0.393 [0.379, 0.406] |
| 20 | 0.284 [0.272, 0.297] | 0.393 [0.379, 0.407] |
| 50 | 0.284 [0.271, 0.296] | 0.393 [0.380, 0.406] |
| 100 | 0.284 [0.271, 0.296] | 0.393 [0.379, 0.406] |

**Hit@k** (mean with 95% CI)

| k | IDF–SVD | KAHM(query→MB corpus) |
| --- | --- | --- |
| 10 | 0.488 [0.474, 0.502] | 0.624 [0.610, 0.637] |
| 20 | 0.560 [0.546, 0.574] | 0.678 [0.665, 0.690] |
| 50 | 0.650 [0.637, 0.663] | 0.741 [0.729, 0.754] |
| 100 | 0.723 [0.710, 0.735] | 0.785 [0.774, 0.797] |

**Majority-accuracy** (mean with 95% CI; majority vote counted only when top-law fraction ≥ τ=0.10)

| k | IDF–SVD | KAHM(query→MB corpus) |
| --- | --- | --- |
| 10 | 0.306 [0.293, 0.319] | 0.425 [0.411, 0.439] |
| 20 | 0.301 [0.288, 0.313] | 0.427 [0.414, 0.441] |
| 50 | 0.291 [0.278, 0.303] | 0.404 [0.391, 0.418] |
| 100 | 0.273 [0.261, 0.285] | 0.381 [0.367, 0.394] |

**Mean consensus fraction** (mean with 95% CI)

| k | IDF–SVD | KAHM(query→MB corpus) |
| --- | --- | --- |
| 10 | 0.257 [0.248, 0.267] | 0.339 [0.329, 0.349] |
| 20 | 0.241 [0.232, 0.251] | 0.311 [0.301, 0.320] |
| 50 | 0.209 [0.201, 0.218] | 0.260 [0.252, 0.268] |
| 100 | 0.176 [0.169, 0.183] | 0.217 [0.210, 0.224] |

**Mean lift (prior)** (mean with 95% CI; consensus fraction divided by corpus prior)

| k | IDF–SVD | KAHM(query→MB corpus) |
| --- | --- | --- |
| 10 | 36.435 [34.416, 38.584] | 56.332 [53.434, 59.247] |
| 20 | 33.303 [31.546, 35.108] | 48.550 [46.346, 50.786] |
| 50 | 26.537 [25.369, 27.775] | 36.308 [34.900, 37.692] |
| 100 | 20.489 [19.700, 21.290] | 27.260 [26.396, 28.154] |

### Paired deltas for KAHM (query adapter)

Paired bootstrap deltas (**KAHM(query→MB corpus) − IDF–SVD**) emphasize what changes when the online transformer query encoder is replaced.

| k | Δhit@k | ΔMRR@k (unique laws) | ΔTop-1 | ΔMajority-acc | ΔMean cons frac | ΔMean lift (prior) |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | +0.135 [+0.122, +0.149] | +0.122 [+0.111, +0.134] | +0.109 [+0.095, +0.123] | +0.119 [+0.107, +0.132] | +0.082 [+0.074, +0.090] | +19.896 [+17.352, +22.657] |
| 20 | +0.118 [+0.104, +0.131] | +0.119 [+0.107, +0.130] | +0.109 [+0.095, +0.123] | +0.126 [+0.114, +0.139] | +0.069 [+0.063, +0.077] | +15.248 [+13.354, +17.204] |
| 50 | +0.091 [+0.078, +0.104] | +0.115 [+0.104, +0.126] | +0.109 [+0.095, +0.123] | +0.113 [+0.101, +0.126] | +0.051 [+0.045, +0.057] | +9.771 [+8.554, +10.984] |
| 100 | +0.062 [+0.051, +0.074] | +0.113 [+0.102, +0.124] | +0.109 [+0.095, +0.123] | +0.108 [+0.096, +0.120] | +0.041 [+0.036, +0.045] | +6.771 [+5.977, +7.543] |

## Interpretation and storylines

### What changes when you replace transformer query inference

The central question is what happens when we keep the **Mixedbread corpus index fixed** and replace the transformer query encoder with a **gradient-free adapter** (IDF–SVD → KAHM → MB space). At **k=100** the paired deltas (KAHM(query→MB corpus) − IDF–SVD) are:

- **Hit@k**: +0.062 [+0.051, +0.074] (6.2 pp); improves (CI > 0).
- **MRR@k (unique laws)**: +0.113 [+0.102, +0.124] (11.3 pp); improves (CI > 0).
- **Top-1 accuracy**: +0.109 [+0.095, +0.123] (10.9 pp); improves (CI > 0).
- **Majority-accuracy (τ)**: +0.108 [+0.096, +0.120] (10.8 pp); improves (CI > 0).
- **Mean consensus fraction**: +0.041 [+0.036, +0.045]; improves (CI > 0).
- **Mean lift (prior)**: +6.771 [+5.977, +7.543]; improves (CI > 0).

Across all evaluated cutoffs, **every** reported measure improves versus IDF–SVD with paired bootstrap CIs excluding 0. This is strong evidence that the KAHM adapter is not merely matching IDF–SVD, but systematically moving query representations into a neighborhood structure that supports the correct law labels.

### Expected trends as k increases

Some metrics change systematically with k. For KAHM(query→MB corpus), Hit@k rises from **0.624 [0.610, 0.637]** at k=10 to **0.785 [0.774, 0.797]** at k=100, because larger cutoffs make it easier to include at least one sentence from the correct law.
Conversely, purity-style metrics typically **decrease** with k because the top-*k* set gets broader: mean consensus fraction changes from **0.339 [0.329, 0.349]** (k=10) to **0.217 [0.210, 0.224]** (k=100), and majority-accuracy changes from **0.425 [0.411, 0.439]** to **0.381 [0.367, 0.394]**.
Top-1 accuracy is invariant to k by definition (it only depends on rank 1), which provides a built-in sanity check for the evaluation.

### Storyline A (k=100): superiority over a strong low-cost baseline

This storyline formalizes the claim that KAHM(query→MB) **beats a strong low-cost baseline** (IDF–SVD). The test is one-sided: PASS means the paired 95% bootstrap CI lower bound is > 0.

| Measure | Δ (KAHM(q→MB) − IDF–SVD) | Superiority |
| --- | --- | --- |
| hit@k | +0.062 [+0.051, +0.074] | PASS |
| MRR@k (unique laws) | +0.113 [+0.102, +0.123] | PASS |
| top1-accuracy | +0.109 [+0.095, +0.123] | PASS |
| majority-accuracy | +0.108 [+0.096, +0.120] | PASS |
| mean consensus fraction | +0.041 [+0.036, +0.045] | PASS |
| mean lift (prior) | +6.771 [+5.989, +7.546] | PASS |

**Verdict:** Supported.

### Storyline B (k=100): competitiveness vs Mixedbread at top-k

This storyline asks how close the gradient-free adapter is to a transformer query encoder on the *same* transformer corpus index. Deltas are paired (KAHM − Mixedbread). CI overlap with 0 indicates statistical indistinguishability at this sample size.

| Measure | Δ (KAHM(q→MB) − MB) | CI excludes 0? |
| --- | --- | --- |
| hit@k | -0.012 [-0.020, -0.003] | Yes |
| MRR@k (unique laws) | +0.022 [+0.014, +0.030] | Yes |
| top1-accuracy | +0.031 [+0.020, +0.042] | Yes |
| majority-accuracy | +0.032 [+0.022, +0.041] | Yes |
| mean consensus fraction | +0.021 [+0.018, +0.024] | Yes |
| mean lift (prior) | +2.987 [+2.506, +3.470] | Yes |

At k=100, KAHM vs Mixedbread shows ΔMRR=+0.022 [+0.014, +0.030], ΔTop-1=+0.031 [+0.020, +0.042], ΔMajority-acc=+0.032 [+0.023, +0.041], and ΔHit@k=-0.012 [-0.020, -0.003]. In practice, this pattern corresponds to being highly competitive on top-of-ranking law quality while potentially trading off a small amount of deep recall at large k.

### Storyline C (k=100): alignment evidence (geometry + neighborhood overlap)

Storyline C complements retrieval metrics with direct evidence that Full-KAHM embeddings preserve Mixedbread geometry. High cosine alignment supports global geometric agreement; neighborhood overlaps support local consistency.

| Alignment measure | Estimate (95% CI) |
| --- | --- |
| cosine(KAHM, MB) on corpus | 0.9972 [0.9970, 0.9974] |
| cosine(KAHM, MB) on queries | 0.9547 [0.9538, 0.9556] |
| sentence Jaccard@100 (Full-KAHM vs MB) | 0.123 [0.120, 0.126] |
| sentence overlap frac@100 (Full-KAHM vs MB) | 0.203 [0.198, 0.207] |
| law-set Jaccard@100 (Full-KAHM vs MB) | 0.499 [0.496, 0.503] |
| Δ sentence Jaccard (Full-KAHM − IDF) | 0.004 [0.002, 0.007] |
| Δ law-set Jaccard (Full-KAHM − IDF) | 0.146 [0.142, 0.150] |

### Majority-vote routing story (why consensus metrics matter)

Many downstream systems only act when retrieval is sufficiently concentrated (e.g., to auto-route a query to a law). The following diagnostics describe how pure the retrieved neighborhoods are under top-k voting.

| Method | mean top-law fraction | mean vote margin | mean vote entropy | mean #unique laws | P(all from one law) |
| --- | --- | --- | --- | --- | --- |
| Mixedbread (true) | 0.338 [0.333, 0.344] | 0.191 [0.185, 0.197] | 2.239 [2.222, 2.257] | 20.002 [19.815, 20.192] | 0.000 [0.000, 0.000] |
| KAHM(query→MB corpus) | 0.356 [0.351, 0.362] | 0.207 [0.201, 0.214] | 2.163 [2.144, 2.182] | 19.073 [18.873, 19.271] | 0.000 [0.000, 0.000] |
| IDF–SVD | 0.357 [0.351, 0.363] | 0.209 [0.202, 0.216] | 2.171 [2.152, 2.191] | 19.440 [19.241, 19.638] | 0.002 [0.001, 0.004] |

### Routing decomposition vs Mixedbread

To explain differences in majority-accuracy, we decompose Δmajority-accuracy into a **coverage** component (how often a query is confident enough to route) and a **precision** component (how accurate routing is when confident).

| τ | Δmaj-acc | Δcov-part | Δprec-part |
| --- | --- | --- | --- |
| 0.10 | +0.032 [+0.022, +0.041] | +0.000 [-0.000, +0.001] | +0.031 [+0.022, +0.041] |
| 0.20 | +0.036 [+0.027, +0.045] | +0.010 [+0.004, +0.015] | +0.026 [+0.017, +0.035] |
| 0.30 | +0.032 [+0.024, +0.039] | +0.013 [+0.006, +0.019] | +0.019 [+0.012, +0.026] |
| 0.40 | +0.021 [+0.014, +0.028] | +0.010 [+0.004, +0.017] | +0.011 [+0.005, +0.016] |
| 0.50 | +0.027 [+0.021, +0.034] | +0.018 [+0.012, +0.024] | +0.009 [+0.005, +0.014] |
| 0.60 | +0.019 [+0.013, +0.025] | +0.020 [+0.014, +0.025] | -0.001 [-0.004, +0.003] |
| 0.70 | +0.025 [+0.020, +0.030] | +0.024 [+0.019, +0.029] | +0.001 [-0.001, +0.004] |
| 0.80 | +0.023 [+0.018, +0.028] | +0.020 [+0.016, +0.025] | +0.003 [+0.001, +0.004] |

### Suggested routing thresholds

Recommendations are computed subject to **coverage ≥ 0.50**.

**τ* maximizing precision (acc|covered)**

| Method | τ* | coverage | acc|covered | majority-acc |
| --- | --- | --- | --- | --- |
| Mixedbread (true) | 0.09 | 0.999 | 0.350 | 0.349 |
| KAHM(query→MB corpus) | 0.11 | 0.991 | 0.384 | 0.381 |
| Full-KAHM (query→KAHM corpus) | 0.10 | 0.999 | 0.264 | 0.263 |
| IDF–SVD | 0.11 | 0.994 | 0.274 | 0.273 |

**τ* maximizing majority-accuracy**

| Method | τ* | coverage | acc|covered | majority-acc |
| --- | --- | --- | --- | --- |
| Mixedbread (true) | 0.09 | 0.999 | 0.350 | 0.349 |
| KAHM(query→MB corpus) | 0.11 | 0.991 | 0.384 | 0.381 |
| Full-KAHM (query→KAHM corpus) | 0.10 | 0.999 | 0.264 | 0.263 |
| IDF–SVD | 0.11 | 0.994 | 0.274 | 0.273 |

## Appendix: transformer-query reference (context)

For completeness, we also report the transformer-query baseline (**Mixedbread queries → Mixedbread corpus**) as a *contextual* reference. These numbers are not the main claim (the compute benefit comes from removing transformer inference from the query path), but they help interpret how close the gradient-free adapter is to a transformer query encoder on the same index.

**MRR@k (unique laws)** (mean with 95% CI)

| k | Mixedbread (true) | KAHM(query→MB corpus) | IDF–SVD |
| --- | --- | --- | --- |
| 10 | 0.456 [0.444, 0.468] | 0.480 [0.468, 0.493] | 0.358 [0.346, 0.370] |
| 20 | 0.466 [0.454, 0.477] | 0.489 [0.476, 0.501] | 0.370 [0.358, 0.381] |
| 50 | 0.473 [0.461, 0.485] | 0.494 [0.482, 0.506] | 0.380 [0.368, 0.391] |
| 100 | 0.475 [0.464, 0.487] | 0.497 [0.485, 0.508] | 0.384 [0.373, 0.396] |

**Top-1 accuracy** (mean with 95% CI)

| k | Mixedbread (true) | KAHM(query→MB corpus) | IDF–SVD |
| --- | --- | --- | --- |
| 10 | 0.362 [0.349, 0.375] | 0.393 [0.379, 0.406] | 0.284 [0.272, 0.297] |
| 20 | 0.362 [0.349, 0.376] | 0.393 [0.379, 0.407] | 0.284 [0.272, 0.297] |
| 50 | 0.362 [0.348, 0.375] | 0.393 [0.380, 0.406] | 0.284 [0.271, 0.296] |
| 100 | 0.362 [0.349, 0.375] | 0.393 [0.379, 0.406] | 0.284 [0.271, 0.296] |

**Hit@k** (mean with 95% CI)

| k | Mixedbread (true) | KAHM(query→MB corpus) | IDF–SVD |
| --- | --- | --- | --- |
| 10 | 0.616 [0.603, 0.630] | 0.624 [0.610, 0.637] | 0.488 [0.474, 0.502] |
| 20 | 0.678 [0.665, 0.691] | 0.678 [0.665, 0.690] | 0.560 [0.546, 0.574] |
| 50 | 0.752 [0.741, 0.765] | 0.741 [0.729, 0.754] | 0.650 [0.637, 0.663] |
| 100 | 0.797 [0.785, 0.808] | 0.785 [0.774, 0.797] | 0.723 [0.710, 0.735] |

**Majority-accuracy** (mean with 95% CI; majority vote counted only when top-law fraction ≥ τ=0.10)

| k | Mixedbread (true) | KAHM(query→MB corpus) | IDF–SVD |
| --- | --- | --- | --- |
| 10 | 0.393 [0.380, 0.407] | 0.425 [0.411, 0.439] | 0.306 [0.293, 0.319] |
| 20 | 0.391 [0.378, 0.405] | 0.427 [0.414, 0.441] | 0.301 [0.288, 0.313] |
| 50 | 0.372 [0.359, 0.385] | 0.404 [0.391, 0.418] | 0.291 [0.278, 0.303] |
| 100 | 0.349 [0.335, 0.362] | 0.381 [0.367, 0.394] | 0.273 [0.261, 0.285] |

**Mean consensus fraction** (mean with 95% CI)

| k | Mixedbread (true) | KAHM(query→MB corpus) | IDF–SVD |
| --- | --- | --- | --- |
| 10 | 0.308 [0.298, 0.317] | 0.339 [0.329, 0.349] | 0.257 [0.248, 0.267] |
| 20 | 0.279 [0.270, 0.288] | 0.311 [0.301, 0.320] | 0.241 [0.232, 0.251] |
| 50 | 0.235 [0.228, 0.242] | 0.260 [0.252, 0.268] | 0.209 [0.201, 0.218] |
| 100 | 0.196 [0.189, 0.202] | 0.217 [0.210, 0.224] | 0.176 [0.169, 0.183] |

**Mean lift (prior)** (mean with 95% CI; consensus fraction divided by corpus prior)

| k | Mixedbread (true) | KAHM(query→MB corpus) | IDF–SVD |
| --- | --- | --- | --- |
| 10 | 50.771 [48.024, 53.732] | 56.332 [53.434, 59.247] | 36.435 [34.416, 38.584] |
| 20 | 42.969 [40.932, 45.090] | 48.550 [46.346, 50.786] | 33.303 [31.546, 35.108] |
| 50 | 32.088 [30.846, 33.324] | 36.308 [34.900, 37.692] | 26.537 [25.369, 27.775] |
| 100 | 24.273 [23.455, 25.054] | 27.260 [26.396, 28.154] | 20.489 [19.700, 21.290] |

### Paired deltas vs transformer-query baseline (context)

| k | Δhit@k (KAHM − Mixedbread) | ΔMRR@k (KAHM − Mixedbread) | ΔTop-1 (KAHM − Mixedbread) | ΔMajority-acc | ΔMean cons frac | ΔMean lift (prior) |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | +0.007 [-0.003, +0.017] | +0.024 [+0.015, +0.032] | +0.031 [+0.020, +0.042] | +0.032 [+0.022, +0.043] | +0.031 [+0.026, +0.037] | +5.561 [+3.645, +7.288] |
| 20 | -0.000 [-0.010, +0.010] | +0.023 [+0.014, +0.031] | +0.031 [+0.020, +0.042] | +0.036 [+0.026, +0.046] | +0.032 [+0.027, +0.036] | +5.581 [+4.291, +6.836] |
| 50 | -0.011 [-0.020, -0.002] | +0.022 [+0.013, +0.030] | +0.031 [+0.020, +0.042] | +0.032 [+0.022, +0.042] | +0.025 [+0.022, +0.029] | +4.220 [+3.429, +4.987] |
| 100 | -0.012 [-0.020, -0.003] | +0.022 [+0.014, +0.030] | +0.031 [+0.020, +0.042] | +0.032 [+0.023, +0.041] | +0.021 [+0.018, +0.024] | +2.987 [+2.504, +3.476] |

## Operational implication

If the corpus is already indexed with transformer embeddings, KAHM provides a practical route to **remove transformer inference from the query path** while retaining transformer-level semantics via the shared embedding space. This is especially attractive in high-QPS settings where online query encoding dominates compute.

## References

- JAIR 16821: https://jair.org/index.php/jair/article/view/16821
- JAIR 15071: https://jair.org/index.php/jair/article/view/15071
- arXiv 2512.01025: https://arxiv.org/abs/2512.01025
