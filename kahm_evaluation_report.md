# KAHM embeddings: retrieval evaluation on Austrian laws

**Generated (UTC):** 2026-02-09T21:11:43Z  
**Source script:** `evaluate_three_embeddings.py` (version `2026-02-09-focused-multi-k-v4`)  

## Narrative focus

KAHM is evaluated here as a **compute-efficient, gradient-free alternative to transformer query encoders**: we keep a strong transformer corpus index fixed (Mixedbread) and replace online query encoding with a lightweight adapter that maps **IDF–SVD query features into the Mixedbread embedding space**. This follows a broader line of work on geometry-/kernel-inspired learning beyond gradient descent and operator-theoretic, gradient-free training over fixed embeddings.

**Key system idea:** offline transformer corpus embeddings; online gradient-free query adapter (IDF–SVD → KAHM → MB space).

## Experimental setup

- Queries: **5000**
- Corpus (aligned sentences): **14487**
- Embedding dimension (MB space): **1024**
- Evaluated cutoffs: **k = 10, 20, 50, 100**

## Methods compared

- **IDF–SVD:** low-cost lexical baseline (sparse/linear).
- **KAHM(query→MB corpus):** gradient-free query adapter: IDF–SVD query features mapped into Mixedbread space; retrieval against a frozen Mixedbread corpus index.
- **Mixedbread (true) (reference):** transformer query embedding + transformer corpus embeddings (reported for context; not the main claim).

## Abstract

On 5000 human-labeled queries over 14487 aligned sentences, KAHM(query→MB corpus) achieves **MRR@100 (unique laws) = 0.497 [0.485, 0.509]** and **Top-1 accuracy = 0.393 [0.379, 0.406]**. Versus IDF–SVD, KAHM improves **MRR@100** by **+0.113 [+0.101, +0.124]** and **Top-1** by **+0.109 [+0.095, +0.123]** (paired bootstrap). Operationally, this supports KAHM as a query-time substitute that preserves a strong transformer index while removing transformer inference from the online path. Transformer-query reference numbers are provided in the Appendix for context.

## Results

### Top-of-ranking quality across k

The main story is captured by **MRR@k over unique laws** and **Top-1 accuracy** at multiple retrieval cutoffs.

**MRR@k (unique laws)** (mean with 95% CI)

| k | IDF–SVD | KAHM(query→MB corpus) |
| --- | --- | --- |
| 10 | 0.358 [0.346, 0.370] | 0.480 [0.467, 0.493] |
| 20 | 0.370 [0.358, 0.382] | 0.489 [0.477, 0.501] |
| 50 | 0.380 [0.368, 0.391] | 0.494 [0.483, 0.506] |
| 100 | 0.384 [0.373, 0.396] | 0.497 [0.485, 0.509] |

**Top-1 accuracy** (mean with 95% CI)

| k | IDF–SVD | KAHM(query→MB corpus) |
| --- | --- | --- |
| 10 | 0.284 [0.271, 0.296] | 0.393 [0.379, 0.406] |
| 20 | 0.284 [0.271, 0.296] | 0.393 [0.379, 0.406] |
| 50 | 0.284 [0.272, 0.296] | 0.393 [0.380, 0.406] |
| 100 | 0.284 [0.271, 0.296] | 0.393 [0.379, 0.406] |

### Paired deltas for KAHM (query adapter)

Paired bootstrap deltas (**KAHM(query→MB corpus) − IDF–SVD**) emphasize what changes when the online transformer query encoder is replaced.

| k | ΔMRR@k (unique laws) | ΔTop-1 |
| --- | --- | --- |
| 10 | +0.122 [+0.111, +0.134] | +0.109 [+0.095, +0.123] |
| 20 | +0.119 [+0.108, +0.130] | +0.109 [+0.095, +0.123] |
| 50 | +0.115 [+0.104, +0.126] | +0.109 [+0.095, +0.123] |
| 100 | +0.113 [+0.101, +0.124] | +0.109 [+0.095, +0.123] |

## Appendix: transformer-query reference (context)

For completeness, we also report the transformer-query baseline (**Mixedbread queries → Mixedbread corpus**) as a *contextual* reference. These numbers are not the main claim (the compute benefit comes from removing transformer inference from the query path), but they help interpret how close the gradient-free adapter is to a transformer query encoder on the same index.

**MRR@k (unique laws)** (mean with 95% CI)

| k | Mixedbread (true) | KAHM(query→MB corpus) | IDF–SVD |
| --- | --- | --- | --- |
| 10 | 0.456 [0.444, 0.468] | 0.480 [0.467, 0.493] | 0.358 [0.346, 0.370] |
| 20 | 0.466 [0.454, 0.478] | 0.489 [0.477, 0.501] | 0.370 [0.358, 0.382] |
| 50 | 0.473 [0.461, 0.485] | 0.494 [0.483, 0.506] | 0.380 [0.368, 0.391] |
| 100 | 0.475 [0.464, 0.487] | 0.497 [0.485, 0.509] | 0.384 [0.373, 0.396] |

**Top-1 accuracy** (mean with 95% CI)

| k | Mixedbread (true) | KAHM(query→MB corpus) | IDF–SVD |
| --- | --- | --- | --- |
| 10 | 0.362 [0.349, 0.375] | 0.393 [0.379, 0.406] | 0.284 [0.271, 0.296] |
| 20 | 0.362 [0.349, 0.375] | 0.393 [0.379, 0.406] | 0.284 [0.271, 0.296] |
| 50 | 0.362 [0.349, 0.375] | 0.393 [0.380, 0.406] | 0.284 [0.272, 0.296] |
| 100 | 0.362 [0.349, 0.376] | 0.393 [0.379, 0.406] | 0.284 [0.271, 0.296] |

### Paired deltas vs transformer-query baseline (context)

| k | ΔMRR@k (KAHM − Mixedbread) | ΔTop-1 (KAHM − Mixedbread) |
| --- | --- | --- |
| 10 | +0.024 [+0.015, +0.032] | +0.031 [+0.020, +0.042] |
| 20 | +0.023 [+0.015, +0.032] | +0.031 [+0.020, +0.042] |
| 50 | +0.022 [+0.014, +0.030] | +0.031 [+0.020, +0.042] |
| 100 | +0.022 [+0.014, +0.030] | +0.031 [+0.020, +0.042] |

## Operational implication

If the corpus is already indexed with transformer embeddings, KAHM provides a practical route to **remove transformer inference from the query path** while retaining transformer-level semantics via the shared embedding space. This is especially attractive in high-QPS settings where online query encoding dominates compute.

## References

- JAIR 16821: https://jair.org/index.php/jair/article/view/16821
- JAIR 15071: https://jair.org/index.php/jair/article/view/15071
- arXiv 2512.01025: https://arxiv.org/abs/2512.01025
