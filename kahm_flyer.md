# KAHM: Compute-efficient opportunity to build a first-of-its-kind in-house language model

**Validated on German-language Austrian laws downloaded from the RIS website.**  
This flyer combines the executive brief and a compact evidence snapshot from the retrieval evaluation.

## Decision request
Schedule a **30-minute** management briefing and approve a scoped **12-16 week** pilot to assess feasibility and ROI of **KAHM-driven language model components**.

## Why this matters
- Training and iterating on large language models via gradient descent is capital-intensive and slows experimentation and time-to-value.
- KAHM offers a mathematically grounded alternative that can reduce reliance on large-scale training compute, enabling faster internal innovation.
- A compute-efficient, in-house approach supports technology sovereignty, IP defensibility, and governance in regulated domains.

## What is KAHM (in one paragraph)
KAHM learns a lightweight transformation that maps a deterministic sparse representation (IDF-SVD) into a high-quality semantic embedding space (Mixedbread). Instead of training large neural models end-to-end, KAHM models topic geometry in embedding space and learns structured mappings that can approximate transformer-quality query embeddings at retrieval time, using modest compute.

## Validated use case: Austrian legal retrieval (German)
- Data: German-language Austrian laws corpus downloaded from RIS; aligned sentence-level corpus intersection of **71,069 sentences**.
- Evaluation: **200** human-labeled queries, retrieval depth **k=10**; cosine similarity via FAISS over L2-normalized embeddings; paired bootstrap (5,000 samples) for **95% CIs**.
- Result headline: **KAHM(query->MB corpus)** achieves **Hit@10=0.890** and **MRR@10=0.738**, strongly outperforming the classical IDF-SVD baseline and performing close to the Mixedbread transformer baseline.

## Strategic opportunity for the company
- Build an in-house language model direction that is **first of its kind in computational efficiency**: research KAHM-driven model components that do not require large-scale gradient descent training.
- Leverage demonstrated domain traction (Austrian law) as a credible foundation; extend toward controlled generation, structured inference, and domain QA.
- Potential benefits: lower compute footprint, faster iteration, clearer control levers, and stronger compliance posture.

## Reference
arXiv:2512.01025 (https://arxiv.org/pdf/2512.01025)

---

# Evidence snapshot: retrieval evaluation on Austrian laws (RIS)

## Methods compared
- **Mixedbread**: transformer embedding baseline (`mixedbread-ai/deepset-mxbai-embed-de-large-v1`) for query and corpus.
- **IDF-SVD**: deterministic TF-IDF -> SVD (LSA) embedding baseline, low compute and auditable.
- **KAHM(query->MB corpus)**: regresses IDF-SVD queries into Mixedbread space, enabling search against a fixed Mixedbread corpus index without transformer query embedding at retrieval time.

## Main metrics (mean over queries; 95% CI)

| Method | Hit@10 | MRR@10 | Top-1 | Maj-vote (>=0.50) |
| --- | --- | --- | --- | --- |
| Mixedbread | 0.900 [0.855, 0.940] | 0.720 [0.670, 0.770] | 0.605 [0.540, 0.670] | 0.560 [0.490, 0.630] |
| IDF-SVD | 0.675 [0.610, 0.740] | 0.464 [0.406, 0.524] | 0.350 [0.285, 0.420] | 0.375 [0.305, 0.445] |
| KAHM(query->MB corpus) | 0.890 [0.845, 0.930] | 0.738 [0.687, 0.788] | 0.625 [0.555, 0.695] | 0.615 [0.545, 0.680] |

## Key deltas and interpretation
- **KAHM vs IDF-SVD (paired deltas):** Hit@10 +0.215 [0.155, 0.280]; MRR@10 +0.274 [0.220, 0.331]; Top-1 +0.275 [0.200, 0.350]; Maj-vote +0.240 [0.175, 0.305].
- **KAHM vs Mixedbread:** differences in Hit@10, MRR@10, and Top-1 are small and not resolved at n=200 (95% CIs include 0). This is **not** a formal equivalence claim.

## Proposed pilot (12-16 weeks)
- Objective: extend KAHM from retrieval embeddings toward language-model components (controlled generation / structured inference) while maintaining a materially lower compute footprint than gradient-descent training.
- Work packages: (1) modeling and routing experiments; (2) task benchmarks (domain QA, summarization, RAG); (3) robustness and governance checks; (4) internal demo and decision package.
- Deliverables: pilot report with KPI outcomes, reproducible evaluation harness, and demonstrator integrated into the existing dashboard.

## Risks and mitigations
- Retrieval validation does not guarantee generation quality -> define pilot tasks and baselines upfront.
- n=200 yields uncertainty vs Mixedbread -> expand labeled queries and robustness checks.
- Domain transfer beyond RIS -> include at least one additional internal dataset if available.

**Owner:** [Mohit Kumar] ([AISEC]) - [mohit.kumar@scch.at]
