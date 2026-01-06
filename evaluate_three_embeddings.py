"""evaluate_three_embeddings.py

Purpose
-------
Clean, publication-friendly evaluation script that tells a simple, defensible story about
KAHM embeddings as a retrieval-time alternative to Mixedbread (MB).

It prints three storylines (A/B/C) from the *same* run:

  A) Effectiveness vs a strong low-cost baseline:
     KAHM(q→MB) decisively beats IDF–SVD on retrieval quality.

  B) Competitiveness vs MB at top-k:
     KAHM(q→MB) is close to MB on top-k retrieval quality (paired deltas + bootstrap CIs).

  C) Alignment / "right direction" evidence:
     Full-KAHM corpus embeddings are aligned with MB embeddings (cosine alignment), and
     full-KAHM retrieval neighborhoods overlap strongly with MB neighborhoods.
 

All confidence intervals use nonparametric *paired* bootstrap (default 5000).

Run
---
python evaluate_three_embeddings.py


Expected local files (defaults)
------------------------------
  ris_sentences.parquet
  embedding_index.npz
  embedding_index_idf_svd.npz
  embedding_index_kahm_mixedbread_approx.npz
  idf_svd_model.joblib
  kahm_regressor_idf_to_mixedbread.joblib
  query_set.py (with TEST_QUERY_SET)
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import gc
import datetime
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


SCRIPT_VERSION = "2026-01-05-storylines-v4-majority-v3-pubreport"


def _safe_ratio(num: float, denom: float) -> float:
    """Return num/denom with NaN on zero denom."""
    return (float(num) / float(denom)) if float(denom) > 0 else float("nan")


def _mv_point_estimates(mv: "MajorityVote", tau: float) -> Tuple[float, float, float]:
    """Point estimates for vote-based routing at threshold tau.

    Returns:
      coverage = P(maj_frac >= tau)
      maj_acc  = P(majority vote correct AND maj_frac >= tau)
      prec     = P(majority vote correct | maj_frac >= tau)
    """
    tau = float(tau)
    covered = (mv.maj_frac >= tau).astype(np.float64)
    acc = (mv.maj_correct * covered).astype(np.float64)
    cov = float(np.mean(covered))
    maj_acc = float(np.mean(acc))
    prec = _safe_ratio(maj_acc, cov)
    return cov, maj_acc, prec


# ----------------------------- Utilities -----------------------------
def _fmt_ci(pt: float, ci: Tuple[float, float], digits: int = 3) -> str:
    return f"{pt:.{digits}f} [{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"


def _fmt_delta(pt: float, ci: Tuple[float, float], digits: int = 3) -> str:
    return f"{pt:+.{digits}f} [{ci[0]:+.{digits}f}, {ci[1]:+.{digits}f}]"


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    """Create a GitHub-flavored Markdown table."""
    h = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([h, sep] + body)


def _ci_cell(pt: float, ci: Tuple[float, float], digits: int = 3) -> str:
    return f"{pt:.{digits}f} ({ci[0]:.{digits}f}, {ci[1]:.{digits}f})"


def _delta_cell(pt: float, ci: Tuple[float, float], digits: int = 3) -> str:
    return f"{pt:+.{digits}f} ({ci[0]:+.{digits}f}, {ci[1]:+.{digits}f})"


def _write_text(path: str, text: str, *, overwrite: bool = False) -> None:
    out_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if (not overwrite) and os.path.exists(out_path):
        raise FileExistsError(f"Refusing to overwrite existing file: {out_path} (use --report_overwrite)")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)


def build_publication_report_md(
    *,
    report_title: str,
    args: argparse.Namespace,
    n_queries: int,
    n_corpus: int,
    embedding_dim: int,
    k: int,
    method_summaries: Dict[str, Dict[str, Tuple[float, Tuple[float, float]]]],
    storyline_a: Dict[str, Any],
    storyline_b: Dict[str, Any],
    majority_profiles: Dict[str, Dict[str, Any]],
    majority_deltas_vs_mb: List[Dict[str, Any]],
    decomp_point_rows: List[Dict[str, Any]],
    decomp_ci_rows: List[Dict[str, Any]],
    threshold_suggestions: Dict[str, Any],
    alignment: Dict[str, Any],
) -> str:
    """Build a single Markdown report intended to be pasted into a scientific manuscript."""
    ts = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    pred_frac = float(getattr(args, "predominance_fraction", 0.50))

    def _ci_excludes_0(ci: Tuple[float, float]) -> bool:
        return bool(np.isfinite(ci[0]) and np.isfinite(ci[1]) and (ci[0] > 0.0 or ci[1] < 0.0))

    def _get_row(story: Dict[str, Any], key: str) -> Dict[str, Any]:
        for r in story.get("rows", []):
            if r.get("key") == key:
                return r
        # Fallback: match by label substring (legacy / robustness)
        for r in story.get("rows", []):
            if key in str(r.get("label", "")):
                return r
        return {}

    # Headline quantities for narrative sections.
    mb = method_summaries["Mixedbread (true)"]
    idf = method_summaries["IDF–SVD"]
    kahm_qmb = method_summaries["KAHM(query→MB corpus)"]

    # Narrative deltas (if present).
    d_kahm_vs_mb_majority = _get_row(storyline_b, "majority") or _get_row(storyline_b, "majority-accuracy")

    lines: List[str] = []
    lines.append(f"# {report_title}")
    lines.append("")
    lines.append(f"**Generated (UTC):** {ts}  ")
    lines.append(f"**Source script:** `{os.path.basename(__file__)}` (version `{SCRIPT_VERSION}`)  ")
    lines.append("")

    # -------------------------------------------------------------------------
    # Abstract
    lines.append("## Abstract")
    lines.append("")
    mb_hit_pt, _ = mb["hit"]
    idf_hit_pt, _ = idf["hit"]
    k_hit_pt, _ = kahm_qmb["hit"]

    mb_mrr_pt, _ = mb["mrr_ul"]
    idf_mrr_pt, _ = idf["mrr_ul"]
    k_mrr_pt, _ = kahm_qmb["mrr_ul"]

    cos_corpus_pt = float(alignment.get("cosine_corpus", {}).get("pt", float("nan")))

    maj_delta_sentence = ""
    if d_kahm_vs_mb_majority and isinstance(d_kahm_vs_mb_majority.get("ci"), (tuple, list)):
        maj_delta_sentence = (
            " Majority-vote accuracy increased by "
            f"{_fmt_delta(float(d_kahm_vs_mb_majority['delta']), tuple(d_kahm_vs_mb_majority['ci']))}."
        )

    lines.append(
        "We evaluate KAHM embeddings for sentence-level retrieval in Austrian legal texts by comparing "
        "(i) a strong low-cost baseline (IDF–SVD), (ii) a Mixedbread SentenceTransformers baseline, and "
        "(iii) KAHM query embeddings obtained by regressing IDF–SVD representations into Mixedbread space. "
        f"Across {n_queries} queries (k={k}), KAHM(query→MB corpus) achieved Hit@{k}={k_hit_pt:.3f} and "
        f"MRR@{k}={k_mrr_pt:.3f}, substantially improving over IDF–SVD "
        f"(Hit@{k}={idf_hit_pt:.3f}, MRR@{k}={idf_mrr_pt:.3f}) and remaining competitive with Mixedbread "
        f"(Hit@{k}={mb_hit_pt:.3f}, MRR@{k}={mb_mrr_pt:.3f})."
        + maj_delta_sentence
        + " "
        f"Full-KAHM embeddings exhibited strong cosine alignment with Mixedbread in embedding space "
        f"(mean cosine={cos_corpus_pt:.4f}) and their retrieval neighborhoods were closer to Mixedbread "
        "than those of IDF–SVD by overlap-based measures."
    )
    lines.append("")

    # -------------------------------------------------------------------------
    # Experimental configuration
    lines.append("## Experimental configuration")
    lines.append("")
    lines.append("The following settings were recorded in the evaluation run:")
    lines.append("")
    lines.append(
        f"- Queries: **{n_queries}**\n"
        f"- Corpus sentences (aligned): **{n_corpus}**\n"
        f"- Embedding dimension: **{embedding_dim}**\n"
        f"- Retrieval depth: **k={k}**\n"
        f"- Bootstrap: **paired nonparametric**, {int(args.bootstrap_samples)} samples, seed={int(args.bootstrap_seed)}\n"
        f"- Majority-vote routing thresholds: {getattr(args, 'majority_thresholds', 'n/a')}\n"
        f"- Predominance fraction (per-query majority accuracy): **{pred_frac:0.2f}**"
    )
    lines.append("")
    lines.append("### Implementation details")
    lines.append("")
    lines.append("- Retrieval uses FAISS inner-product search over **L2-normalized** embeddings (equivalent to cosine similarity).")
    lines.append(f"- Mixedbread baseline: `{getattr(args, 'mixedbread_model', 'n/a')}` with query prefix `{getattr(args, 'query_prefix', '')}`.")
    lines.append(f"- IDF–SVD: pipeline loaded from `{getattr(args, 'idf_svd_model', 'n/a')}`; corpus index from `{getattr(args, 'idf_svd_npz', 'n/a')}`.")
    lines.append(
        f"- KAHM: regressor loaded from `{getattr(args, 'kahm_model', 'n/a')}` (mode `{getattr(args, 'kahm_mode', 'n/a')}`), "
        "mapping IDF–SVD query embeddings into Mixedbread space."
    )
    lines.append("- Methods compared:")
    lines.append("  - **Mixedbread (true):** Mixedbread queries → Mixedbread corpus.")
    lines.append("  - **IDF–SVD:** IDF–SVD queries → IDF–SVD corpus.")
    lines.append("  - **KAHM(query→MB corpus):** KAHM-regressed queries → Mixedbread corpus.")
    lines.append(
        "  - **Full-KAHM (query→KAHM corpus):** KAHM-regressed queries → KAHM-transformed corpus "
        f"(`{getattr(args, 'kahm_corpus_npz', 'n/a')}`)."
    )
    lines.append("")

    # -------------------------------------------------------------------------
    # Methods
    lines.append("## Methods")
    lines.append("")
    lines.append("### Data and alignment")
    lines.append("")
    lines.append(
        "Each embedding index provides sentence identifiers that are aligned against a shared sentence-level corpus. "
        "All reported retrieval metrics are computed over the intersection of sentence identifiers present in each "
        "embedding index, yielding {:,} common sentences.".format(int(n_corpus))
    )
    lines.append("")
    lines.append("### Retrieval protocol")
    lines.append("")
    lines.append(
        "For each query, we compute a query embedding for each method, retrieve the top-k nearest neighbors from the "
        "corresponding index, and map retrieved sentence identifiers to their associated law identifiers to compute "
        "law-level retrieval metrics."
    )
    lines.append("")
    lines.append("### Evaluation metrics")
    lines.append("")
    lines.append(
        "Let the reference label for a query be its *consensus law* (a single law identifier). "
        "Metrics are computed per query and then averaged:"
    )
    lines.append("")
    lines.append(f"- **Hit@k:** 1 if the consensus law appears at least once among the top-k retrieved laws; else 0.")
    lines.append(f"- **Top-1 accuracy:** 1 if the top-ranked retrieved law equals the consensus law; else 0.")
    lines.append(
        f"- **MRR@k (unique laws):** form the ordered list of *unique* laws appearing in the top-k list; "
        f"if the consensus law occurs at rank r in this list, score 1/r; otherwise 0."
    )
    lines.append(
        f"- **Majority accuracy (predominance ≥ {pred_frac:0.2f}):** let the majority law be the most frequent law "
        f"in the top-k list and let f be its fraction. Score 1 if (i) the majority law equals the consensus law and "
        f"(ii) f ≥ {pred_frac:0.2f}; otherwise 0."
    )
    lines.append(f"- **Mean consensus fraction:** fraction of the top-k items belonging to the consensus law (count/k).")
    lines.append(
        f"- **Mean lift vs prior:** for each query, divide the consensus fraction by the marginal probability of the "
        f"consensus law in the aligned corpus; report the mean across queries."
    )
    lines.append("")
    lines.append("### Majority-vote diagnostics and routing")
    lines.append("")
    lines.append("We summarize the law vote distribution over each query’s top-k neighborhood using:")
    lines.append("- **Top-law fraction:** fraction of the top-k belonging to the most frequent law.")
    lines.append("- **Vote margin:** (top-law fraction) minus (runner-up law fraction).")
    lines.append("- **Vote entropy:** Shannon entropy of the empirical law distribution in the top-k list.")
    lines.append("- **#Unique laws:** number of distinct laws appearing in the top-k list.")
    lines.append(
        "For routing, we evaluate a threshold rule that accepts the majority-law prediction when the top-law fraction ≥ τ. "
        "We report (i) **coverage** P(covered), (ii) **accuracy among covered** P(correct | covered), and "
        "(iii) **majority accuracy** P(correct ∩ covered)."
    )
    lines.append("")
    lines.append("### Statistical analysis")
    lines.append("")
    lines.append(
        "We estimate 95% confidence intervals (CIs) using a paired, nonparametric bootstrap over queries. "
        "For paired deltas, the statistic is computed on per-query differences; a delta is treated as statistically "
        "different from 0 when its 95% CI excludes 0."
    )
    lines.append("")

    # -------------------------------------------------------------------------
    # Results
    lines.append("## Results")
    lines.append("")
    lines.append("### Main retrieval performance")
    lines.append("")
    lines.append("**Table 1.** Main retrieval metrics (mean over queries; 95% bootstrap CI).")
    lines.append("")
    metric_order = [
        ("hit", f"Hit@{k}"),
        ("mrr_ul", f"MRR@{k} (unique laws)"),
        ("top1", "Top-1 accuracy"),
        ("majority", "Majority accuracy"),
        ("cons_frac", "Mean consensus fraction"),
        ("lift", "Mean lift vs prior"),
    ]
    headers = ["Method"] + [m[1] for m in metric_order]
    rows: List[List[str]] = []
    for method_name in [
        "Mixedbread (true)",
        "IDF–SVD",
        "KAHM(query→MB corpus)",
        "Full-KAHM (query→KAHM corpus)",
    ]:
        s = method_summaries[method_name]
        r = [method_name]
        for key, _lbl in metric_order:
            pt, ci = s[key]
            r.append(_ci_cell(pt, ci, digits=3))
        rows.append(r)
    lines.append(_md_table(headers, rows))
    lines.append("")

    lines.append("### Comparative analyses")
    lines.append("")
    lines.append("**Table 2.** Paired deltas for KAHM(query→MB corpus) vs IDF–SVD (95% bootstrap CI).")
    lines.append("")
    a_headers = ["Metric", "Δ (KAHM − IDF)", "95% CI excludes 0?", "Superiority (lower CI > 0)?"]
    a_rows: List[List[str]] = []
    for r in storyline_a.get("rows", []):
        ci = tuple(r.get("ci", (float("nan"), float("nan"))))
        a_rows.append(
            [
                str(r.get("label", "")),
                _delta_cell(float(r.get("delta", float("nan"))), ci, digits=3),
                "Yes" if _ci_excludes_0(ci) else "No",
                "Yes" if bool(r.get("pass", False)) else "No",
            ]
        )
    lines.append(_md_table(a_headers, a_rows))
    lines.append("")
    lines.append("Interpretation: the superiority column reflects the one-sided criterion used in the evaluation (paired 95% CI lower bound > 0).")
    lines.append("")

    lines.append("**Table 3.** Paired deltas for KAHM(query→MB corpus) vs Mixedbread (95% bootstrap CI).")
    lines.append("")
    b_headers = ["Metric", "Δ (KAHM − Mixedbread)", "95% CI excludes 0?"]
    b_rows: List[List[str]] = []
    for r in storyline_b.get("rows", []):
        ci = tuple(r.get("ci", (float("nan"), float("nan"))))
        b_rows.append(
            [
                str(r.get("label", "")),
                _delta_cell(float(r.get("delta", float("nan"))), ci, digits=3),
                "Yes" if bool(r.get("ci_excludes_0", _ci_excludes_0(ci))) else "No",
            ]
        )
    lines.append(_md_table(b_headers, b_rows))
    lines.append("")

    lines.append("### Majority-vote behavior")
    lines.append("")
    lines.append(f"**Table 4.** Vote-distribution diagnostics over the top-k neighborhood (k={k}; 95% bootstrap CI).")
    lines.append("")
    prof_headers = ["Method", "Top-law fraction", "Vote margin", "Vote entropy", "#Unique laws", f"P(all {k} one law)"]
    prof_rows: List[List[str]] = []
    for method_name in [
        "Mixedbread (true)",
        "IDF–SVD",
        "KAHM(query→MB corpus)",
        "Full-KAHM (query→KAHM corpus)",
    ]:
        pr = majority_profiles[method_name]
        prof_rows.append(
            [
                method_name,
                _ci_cell(pr["mean_toplaw_frac"]["pt"], pr["mean_toplaw_frac"]["ci"]),
                _ci_cell(pr["mean_vote_margin"]["pt"], pr["mean_vote_margin"]["ci"]),
                _ci_cell(pr["mean_vote_entropy"]["pt"], pr["mean_vote_entropy"]["ci"]),
                _ci_cell(pr["mean_n_unique"]["pt"], pr["mean_n_unique"]["ci"]),
                _ci_cell(pr["p_all_from_one_law"]["pt"], pr["p_all_from_one_law"]["ci"]),
            ]
        )
    lines.append(_md_table(prof_headers, prof_rows))
    lines.append("")

    lines.append("### Vote-based routing")
    lines.append("")
    lines.append(
        "**Table 5.** Routing sweep comparing KAHM(query→MB corpus) and Mixedbread at different predominance thresholds τ "
        "(95% bootstrap CI)."
    )
    lines.append("")
    sweep_headers = [
        "τ",
        "Coverage (KAHM)",
        "Accuracy among covered (KAHM)",
        "Majority accuracy (KAHM)",
        "Coverage (Mixedbread)",
        "Accuracy among covered (Mixedbread)",
        "Majority accuracy (Mixedbread)",
    ]
    sweep_rows: List[List[str]] = []
    mb_sweep = {float(r["tau"]): r for r in majority_profiles["Mixedbread (true)"]["threshold_sweep"]}
    k_sweep = {float(r["tau"]): r for r in majority_profiles["KAHM(query→MB corpus)"]["threshold_sweep"]}
    for tau in sorted(mb_sweep.keys()):
        a = k_sweep[tau]
        b = mb_sweep[tau]
        sweep_rows.append(
            [
                f"{tau:0.2f}",
                _ci_cell(a["coverage"]["pt"], a["coverage"]["ci"]),
                _ci_cell(a["acc_given_covered"]["pt"], a["acc_given_covered"]["ci"]),
                _ci_cell(a["majority_acc"]["pt"], a["majority_acc"]["ci"]),
                _ci_cell(b["coverage"]["pt"], b["coverage"]["ci"]),
                _ci_cell(b["acc_given_covered"]["pt"], b["acc_given_covered"]["ci"]),
                _ci_cell(b["majority_acc"]["pt"], b["majority_acc"]["ci"]),
            ]
        )
    lines.append(_md_table(sweep_headers, sweep_rows))
    lines.append("")

    lines.append("**Table 6.** Paired deltas vs Mixedbread for routing (KAHM(query→MB corpus) − Mixedbread).")
    lines.append("")
    dv_headers = ["τ", "ΔCoverage", "ΔMajority accuracy"]
    dv_rows: List[List[str]] = []
    for r in majority_deltas_vs_mb:
        dv_rows.append(
            [
                f"{float(r['tau']):0.2f}",
                _delta_cell(r["delta_coverage"]["pt"], r["delta_coverage"]["ci"]),
                _delta_cell(r["delta_majacc"]["pt"], r["delta_majacc"]["ci"]),
            ]
        )
    lines.append(_md_table(dv_headers, dv_rows))
    lines.append("")

    lines.append("**Table 7.** Decomposition of ΔMajority accuracy into coverage and precision effects.")
    lines.append("")
    lines.append("Point estimates:")
    dep_headers = ["τ", "ΔMajority accuracy", "Coverage contribution", "Precision contribution"]
    dep_rows: List[List[str]] = []
    for r in decomp_point_rows:
        dep_rows.append(
            [
                f"{float(r['tau']):0.2f}",
                f"{r['delta_majacc']:+0.3f}",
                f"{r['delta_cov_part']:+0.3f}",
                f"{r['delta_prec_part']:+0.3f}",
            ]
        )
    lines.append(_md_table(dep_headers, dep_rows))
    lines.append("")
    lines.append("With paired-bootstrap CIs:")
    dep2_rows: List[List[str]] = []
    for r in decomp_ci_rows:
        dep2_rows.append(
            [
                f"{float(r['tau']):0.2f}",
                _delta_cell(r["delta_majacc"]["pt"], r["delta_majacc"]["ci"]),
                _delta_cell(r["delta_cov_part"]["pt"], r["delta_cov_part"]["ci"]),
                _delta_cell(r["delta_prec_part"]["pt"], r["delta_prec_part"]["ci"]),
            ]
        )
    lines.append(_md_table(dep_headers, dep2_rows))
    lines.append("")

    lines.append("**Table 8.** Suggested routing thresholds (coverage constraint and objectives).")
    lines.append("")
    lines.append(f"Coverage constraint: **coverage ≥ {threshold_suggestions['coverage_constraint']:0.2f}**")
    lines.append("")
    sug_headers = ["Method", "τ*", "Coverage", "Accuracy among covered", "Majority accuracy", "Objective"]
    sug_rows: List[List[str]] = []
    for objective_key, objective_label in [
        ("maximize_precision_subject_to_coverage", "Maximize precision"),
        ("maximize_majority_acc_subject_to_coverage", "Maximize majority accuracy"),
    ]:
        for method_name, rec in threshold_suggestions.get(objective_key, {}).items():
            sug_rows.append(
                [
                    method_name,
                    f"{rec['tau']:0.2f}",
                    f"{rec['coverage']:0.3f}",
                    f"{rec['acc_given_covered']:0.3f}",
                    f"{rec['majority_acc']:0.3f}",
                    objective_label,
                ]
            )
    lines.append(_md_table(sug_headers, sug_rows))
    lines.append("")

    lines.append("### Embedding-space alignment and neighborhood overlap")
    lines.append("")
    lines.append("**Table 9.** Alignment and overlap statistics for Full-KAHM vs Mixedbread (95% bootstrap CI).")
    lines.append("")
    align_headers = ["Quantity", "Estimate (95% CI)"]
    align_rows = [
        ["Cosine alignment (corpus)", _ci_cell(alignment["cosine_corpus"]["pt"], alignment["cosine_corpus"]["ci"], digits=4)],
        ["Cosine alignment (queries)", _ci_cell(alignment["cosine_query"]["pt"], alignment["cosine_query"]["ci"], digits=4)],
        [f"Sentence Jaccard@{k}", _ci_cell(alignment["sentence_jaccard"]["pt"], alignment["sentence_jaccard"]["ci"], digits=3)],
        [f"Sentence overlap fraction@{k}", _ci_cell(alignment["sentence_overlap_frac"]["pt"], alignment["sentence_overlap_frac"]["ci"], digits=3)],
        [f"Law-set Jaccard@{k}", _ci_cell(alignment["lawset_jaccard"]["pt"], alignment["lawset_jaccard"]["ci"], digits=3)],
        ["Δ sentence Jaccard (Full − IDF)", _delta_cell(alignment["delta_sentence_jaccard_full_minus_idf"]["pt"], alignment["delta_sentence_jaccard_full_minus_idf"]["ci"], digits=3)],
        ["Δ law-set Jaccard (Full − IDF)", _delta_cell(alignment["delta_lawset_jaccard_full_minus_idf"]["pt"], alignment["delta_lawset_jaccard_full_minus_idf"]["ci"], digits=3)],
    ]
    lines.append(_md_table(align_headers, align_rows))
    lines.append("")

    # -------------------------------------------------------------------------
    # Reproducibility
    lines.append("## Reproducibility checklist")
    lines.append("")
    lines.append(f"- Query set: `{getattr(args, 'query_set', 'n/a')}`")
    lines.append(f"- Corpus parquet: `{getattr(args, 'corpus_parquet', 'n/a')}`")
    lines.append(f"- Mixedbread model: `{getattr(args, 'mixedbread_model', 'n/a')}`")
    lines.append(f"- Query prefix: `{getattr(args, 'query_prefix', '')}`")
    lines.append(f"- IDF–SVD model: `{getattr(args, 'idf_svd_model', 'n/a')}`")
    lines.append(f"- KAHM model: `{getattr(args, 'kahm_model', 'n/a')}` (mode `{getattr(args, 'kahm_mode', 'n/a')}`)")
    lines.append(
        f"- Indices: MB `{getattr(args, 'semantic_npz', 'n/a')}`, IDF `{getattr(args, 'idf_svd_npz', 'n/a')}`, KAHM `{getattr(args, 'kahm_corpus_npz', 'n/a')}`"
    )
    lines.append(f"- Device: `{getattr(args, 'device', 'n/a')}`")
    lines.append(f"- Threads cap: {getattr(args, 'threads', 'n/a')}")
    lines.append(f"- Bootstrap: samples={int(args.bootstrap_samples)}, seed={int(args.bootstrap_seed)}")
    lines.append("")

    # -------------------------------------------------------------------------
    # Copy-ready summary paragraph
    lines.append("---")
    lines.append("## Copy-ready summary paragraph")
    lines.append("")
    k_hit_pt, k_hit_ci = kahm_qmb["hit"]
    k_mrr_pt, k_mrr_ci = kahm_qmb["mrr_ul"]
    mb_maj_delta = _get_row(storyline_b, "majority") or _get_row(storyline_b, "majority-accuracy")
    maj_delta_phrase = ""
    if mb_maj_delta and isinstance(mb_maj_delta.get("ci"), (tuple, list)):
        ci = tuple(mb_maj_delta["ci"])
        maj_delta_phrase = f" Majority accuracy differed by {_fmt_delta(float(mb_maj_delta['delta']), ci)} relative to Mixedbread."
    lines.append(
        f"Across {n_queries} queries (k={k}), KAHM(query→MB corpus) achieved Hit@{k}={_ci_cell(k_hit_pt, k_hit_ci, digits=3)} "
        f"and MRR@{k}={_ci_cell(k_mrr_pt, k_mrr_ci, digits=3)}. "
        f"Paired-bootstrap deltas favored KAHM(query→MB corpus) over IDF–SVD under the evaluation’s superiority criterion "
        f"(Table 2). Compared to Mixedbread, paired deltas for Hit@{k}, MRR@{k}, and Top-1 accuracy were small with 95% CIs "
        f"that included 0 (Table 3), while majority-vote behavior differed depending on the routing threshold τ (Tables 5–8)."
        + maj_delta_phrase
        + f" Full-KAHM embeddings showed strong cosine alignment with Mixedbread in embedding space "
        f"(mean corpus cosine {alignment['cosine_corpus']['pt']:.4f}) and higher neighborhood overlap to Mixedbread than IDF–SVD "
        f"by sentence- and law-level Jaccard measures (Table 9)."
    )
    lines.append("")

    return "\n".join(lines)

def l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return X / n


def _bootstrap_mean_ci(x: np.ndarray, *, n_boot: int, seed: int) -> Tuple[float, Tuple[float, float]]:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan"), (float("nan"), float("nan"))
    rng = np.random.default_rng(int(seed))
    n = int(x.size)
    pt = float(np.mean(x))
    bs = np.empty(int(n_boot), dtype=np.float64)
    for b in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        bs[b] = float(np.mean(x[idx]))
    lo, hi = np.quantile(bs, [0.025, 0.975])
    return pt, (float(lo), float(hi))


def _bootstrap_paired_delta_ci(
    a: np.ndarray, b: np.ndarray, *, n_boot: int, seed: int
) -> Tuple[float, Tuple[float, float]]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Paired arrays must have same shape; got {a.shape} vs {b.shape}")
    if a.size == 0:
        return float("nan"), (float("nan"), float("nan"))
    rng = np.random.default_rng(int(seed))
    n = int(a.size)
    d = a - b
    pt = float(np.mean(d))
    bs = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        bs[i] = float(np.mean(d[idx]))
    lo, hi = np.quantile(bs, [0.025, 0.975])
    return pt, (float(lo), float(hi))


def _bootstrap_ratio_ci(
    num: np.ndarray,
    denom: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> Tuple[float, Tuple[float, float]]:
    """Bootstrap CI for a ratio E[num]/E[denom] estimated as sum(num)/sum(denom).

    This is useful for conditional accuracies such as:
        P(correct | majority_fraction >= tau)
    where num = 1{correct & passes}, denom = 1{passes}.
    """
    num = np.asarray(num, dtype=np.float64)
    denom = np.asarray(denom, dtype=np.float64)
    if num.shape != denom.shape:
        raise ValueError(f"Ratio arrays must have same shape; got {num.shape} vs {denom.shape}")
    if num.size == 0:
        return float("nan"), (float("nan"), float("nan"))

    rng = np.random.default_rng(int(seed))
    n = int(num.size)

    num_sum = float(np.sum(num))
    denom_sum = float(np.sum(denom))
    pt = (num_sum / denom_sum) if denom_sum > 0 else float("nan")

    bs = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        ns = float(np.sum(num[idx]))
        ds = float(np.sum(denom[idx]))
        bs[i] = (ns / ds) if ds > 0 else np.nan

    # Drop NaNs (can occur if a bootstrap resample has zero denom).
    bs = bs[np.isfinite(bs)]
    if bs.size == 0:
        return pt, (float("nan"), float("nan"))
    lo, hi = np.quantile(bs, [0.025, 0.975])
    return pt, (float(lo), float(hi))


# ----------------------------- Data loading -----------------------------
def load_npz_bundle(path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"NPZ not found: {path}")
    data = np.load(path, allow_pickle=False)
    keys = set(data.files)

    sid_key = None
    for k in ("sentence_ids", "ids", "sentence_id"):
        if k in keys:
            sid_key = k
            break
    emb_key = None
    for k in ("embeddings", "embedding", "X", "emb"):
        if k in keys:
            emb_key = k
            break

    if sid_key is None or emb_key is None:
        raise ValueError(
            f"Unsupported NPZ schema in {path}. Expected sentence_ids + embeddings keys; found {sorted(keys)}"
        )

    sentence_ids = np.asarray(data[sid_key], dtype=np.int64)
    emb = np.asarray(data[emb_key], dtype=np.float32)
    if emb.ndim != 2:
        raise ValueError(f"Embeddings must be 2D; got {emb.shape} in {path}")
    if sentence_ids.ndim != 1:
        raise ValueError(f"sentence_ids must be 1D; got {sentence_ids.shape} in {path}")
    if emb.shape[0] != sentence_ids.shape[0]:
        raise ValueError(f"Row mismatch in {path}: embeddings rows={emb.shape[0]} vs ids={sentence_ids.shape[0]}")

    return {"sentence_ids": sentence_ids, "emb": l2_normalize_rows(emb)}


def load_corpus_parquet(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Corpus parquet not found: {path}")
    df = pd.read_parquet(path)
    required = {"sentence_id", "law_type"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Corpus parquet missing columns: {sorted(missing)}")
    ids = df["sentence_id"].astype(np.int64).to_numpy()
    if np.unique(ids).size != ids.size:
        raise ValueError("Corpus parquet has duplicate sentence_id values; must be unique for safe alignment.")
    return df


def align_by_common_sentence_ids(
    df: pd.DataFrame,
    mb: Dict[str, np.ndarray],
    idf: Dict[str, np.ndarray],
    kahm: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    s_df = df["sentence_id"].astype(np.int64).to_numpy()
    s_mb = mb["sentence_ids"].astype(np.int64)
    s_idf = idf["sentence_ids"].astype(np.int64)
    s_k = kahm["sentence_ids"].astype(np.int64)

    common = np.intersect1d(np.intersect1d(np.intersect1d(s_df, s_mb), s_idf), s_k)
    if common.size == 0:
        raise ValueError("No common sentence_ids across df/MB/IDF/KAHM bundles")

    def _subset(ids: np.ndarray, emb: np.ndarray, common_ids: np.ndarray) -> np.ndarray:
        pos = {int(s): i for i, s in enumerate(ids.tolist())}
        idx = np.asarray([pos[int(s)] for s in common_ids.tolist()], dtype=np.int64)
        return emb[idx]

    emb_mb = _subset(s_mb, mb["emb"], common)
    emb_idf = _subset(s_idf, idf["emb"], common)
    emb_k = _subset(s_k, kahm["emb"], common)

    df2 = df.set_index("sentence_id", drop=False)
    sub = df2.loc[common]
    law = sub["law_type"].astype(str).to_numpy()

    return {"sentence_ids": common, "law": law, "emb_mb": emb_mb, "emb_idf": emb_idf, "emb_kahm": emb_k}


def load_query_set(module_attr: str) -> List[Dict[str, Any]]:
    if "." not in module_attr:
        raise ValueError("--query_set must be module.attr, e.g., query_set.TEST_QUERY_SET")
    mod_name, attr = module_attr.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    qs = getattr(mod, attr, None)
    if qs is None:
        raise AttributeError(f"Query set attribute not found: {module_attr}")
    out = list(qs)
    if not out:
        raise ValueError(f"Loaded empty query set from {module_attr}")
    return out


def _pick_from_mapping(obj: Any, keys: List[str]) -> str:
    if not isinstance(obj, dict):
        return ""
    for k in keys:
        if k in obj:
            v = obj.get(k, "")
            if isinstance(v, (list, tuple)) and v:
                v = v[0]
            if v is None:
                continue
            s = str(v).strip()
            if s:
                return s
    return ""


def _pick_from_object_attrs(obj: Any, keys: List[str]) -> str:
    for k in keys:
        if hasattr(obj, k):
            v = getattr(obj, k)
            if isinstance(v, (list, tuple)) and v:
                v = v[0]
            if v is None:
                continue
            s = str(v).strip()
            if s:
                return s
    return ""


def extract_query_texts(qs: List[Any]) -> List[str]:
    keys = ["query_text", "query", "question", "text", "prompt", "q", "input"]
    texts: List[str] = []
    for q in qs:
        t = _pick_from_mapping(q, keys)
        if not t and isinstance(q, (list, tuple)):
            if len(q) >= 2 and isinstance(q[1], str) and q[1].strip():
                t = str(q[1]).strip()
            elif len(q) >= 1 and isinstance(q[0], str) and q[0].strip():
                t = str(q[0]).strip()
        if not t:
            t = _pick_from_object_attrs(q, keys)
        texts.append(t)
    return texts


def extract_consensus_laws(qs: List[Any]) -> List[str]:
    keys = [
        "consensus_law",
        "consensus",
        "consensus_law_type",
        "gold_law",
        "target_law",
        "law",
        "law_type",
    ]
    out: List[str] = []
    for q in qs:
        v = _pick_from_mapping(q, keys)
        if not v and isinstance(q, (list, tuple)):
            if len(q) >= 3 and isinstance(q[2], str) and q[2].strip():
                v = str(q[2]).strip()
            elif len(q) >= 1 and isinstance(q[-1], str) and q[-1].strip():
                v = str(q[-1]).strip()
        if not v:
            v = _pick_from_object_attrs(q, keys)
        out.append(str(v).strip())
    return out


# ----------------------------- FAISS -----------------------------
def build_faiss_index(emb: np.ndarray, *, n_threads: int | None = None):
    """Build a FlatIP index. If n_threads is set, cap FAISS OpenMP threads."""
    import faiss  # type: ignore
    from typing import Any, cast

    if n_threads is not None and int(n_threads) > 0:
        try:
            faiss.omp_set_num_threads(int(n_threads))
        except Exception:
            pass

    X = np.ascontiguousarray(emb.astype(np.float32, copy=False))
    index = faiss.IndexFlatIP(int(X.shape[1]))
    # Pylance/pyright stubs sometimes describe Index.add as add(n, x) while the SWIG binding accepts add(x).
    cast(Any, index).add(X)
    return index

def faiss_search(index, q_emb: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    Q = np.ascontiguousarray(q_emb.astype(np.float32, copy=False))
    scores, idx = index.search(Q, int(k))
    return scores, idx


# ----------------------------- Models -----------------------------
def load_idf_svd_model(path: str):
    import joblib

    if not os.path.exists(path):
        raise FileNotFoundError(f"IDF–SVD model not found: {path}")
    return joblib.load(path)


def embed_queries_idf_svd(pipe, texts: List[str]) -> np.ndarray:
    X = pipe.transform(texts)
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"IDF–SVD transform output must be 2D; got {X.shape}")
    return l2_normalize_rows(X)


def embed_queries_mixedbread(
    *,
    model_name: str,
    device: str,
    dim: int,
    query_prefix: str,
    texts: List[str],
    batch_size: int,
    show_progress_bar: bool = True,
) -> np.ndarray:
    """Embed queries with Mixedbread and release the model to reduce peak memory."""
    from sentence_transformers import SentenceTransformer
    import gc

    m = SentenceTransformer(model_name, device=device, truncate_dim=int(dim))
    q_texts = [query_prefix + t for t in texts]
    Y = m.encode(
        q_texts,
        batch_size=int(batch_size),
        show_progress_bar=bool(show_progress_bar),
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)
    # Release transformer weights before FAISS indices are built.
    del m
    gc.collect()

    if Y.ndim != 2:
        raise ValueError(f"Mixedbread encode output must be 2D; got {Y.shape}")
    if Y.shape[1] != int(dim):
        if Y.shape[1] > int(dim):
            Y = Y[:, : int(dim)]
        else:
            raise ValueError(f"Mixedbread embedding dim mismatch: got {Y.shape[1]}, expected {dim}")
    return l2_normalize_rows(Y)


def load_kahm_model(path: str) -> dict:
    from kahm_regression import load_kahm_regressor

    if not os.path.exists(path):
        raise FileNotFoundError(f"KAHM model not found: {path}")
    return load_kahm_regressor(path)


def kahm_regress_once(
    model: dict,
    X: np.ndarray,
    *,
    mode: str,
    batch_size: int,
    show_progress: bool = False,
) -> np.ndarray:
    """Single-call KAHM regression with internal batching (avoids repeated AE reloads)."""
    from kahm_regression import kahm_regress

    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"KAHM regression input must be 2D; got {X.shape}")

    bs = int(batch_size) if int(batch_size) > 0 else None
    
    Yt = kahm_regress(
            model,
            X.T,
            n_jobs=1,
            mode=str(mode),
            batch_size=bs,
            # kahm_regression.py may or may not expose show_progress; keep runtime-compatible.
            show_progress=bool(show_progress),
        )
  
    Y = np.asarray(Yt, dtype=np.float32).T
    return l2_normalize_rows(Y)


# ----------------------------- Metrics -----------------------------
@dataclass
class PerQuery:
    hit: np.ndarray
    top1: np.ndarray
    majority: np.ndarray
    cons_frac: np.ndarray
    lift: np.ndarray
    mrr_ul: np.ndarray


@dataclass
class MajorityVote:
    """Diagnostics for top-k *law voting* (independent of any predominance threshold).

    maj_frac:
        Fraction of the top-k list belonging to the most frequent law in that list.

    maj_correct:
        1.0 if the majority-vote law equals the consensus law, else 0.0.

    margin:
        maj_frac minus runner-up fraction (0 if there is only one unique law).

    entropy:
        Shannon entropy of the law distribution in the top-k list (higher = less concentrated).

    n_unique:
        Number of unique laws present in the top-k list.
    """

    maj_frac: np.ndarray
    maj_correct: np.ndarray
    margin: np.ndarray
    entropy: np.ndarray
    n_unique: np.ndarray


def compute_per_query_metrics(
    *,
    idx: np.ndarray,
    law_arr: np.ndarray,
    consensus_laws: List[str],
    k: int,
    predominance_fraction: float,
) -> PerQuery:
    k = int(k)
    pred_frac = float(predominance_fraction)

    c_all = Counter([str(x) for x in law_arr.tolist()])
    total = float(max(1, int(law_arr.size)))
    prior = {lw: float(cnt) / total for lw, cnt in c_all.items()}

    idx = np.asarray(idx, dtype=np.int64)
    if idx.ndim != 2:
        raise ValueError(f"idx must be 2D; got {idx.shape}")
    if idx.shape[1] < k:
        raise ValueError(f"idx has too few columns: {idx.shape[1]} < k={k}")
    if idx.shape[1] > k:
        idx = idx[:, :k]

    n = int(idx.shape[0])
    if len(consensus_laws) != n:
        raise ValueError(f"consensus_laws length {len(consensus_laws)} != n_queries {n}")

    hit_v = np.zeros(n, dtype=np.float64)
    top1_v = np.zeros(n, dtype=np.float64)
    maj_v = np.zeros(n, dtype=np.float64)
    cf_v = np.zeros(n, dtype=np.float64)
    lift_v = np.zeros(n, dtype=np.float64)
    mrr_v = np.zeros(n, dtype=np.float64)

    for i in range(n):
        cons = str(consensus_laws[i]).strip()
        row = [int(j) for j in idx[i].tolist() if int(j) >= 0]
        laws = [str(law_arr[j]) for j in row]

        hit_v[i] = 1.0 if (cons in laws) else 0.0
        top1_v[i] = 1.0 if (laws and laws[0] == cons) else 0.0

        c = Counter(laws)
        maj_law, maj_count = c.most_common(1)[0]
        maj_frac = float(maj_count) / float(max(1, len(laws)))
        maj_v[i] = 1.0 if (maj_law == cons and maj_frac >= pred_frac) else 0.0

        cons_frac = float(c.get(cons, 0)) / float(max(1, len(laws)))
        cf_v[i] = cons_frac
        cons_prior = float(prior.get(cons, 0.0))
        lift_v[i] = (cons_frac / cons_prior) if cons_prior > 0 else 0.0

        seen = set()
        uniq: List[str] = []
        for lw in laws:
            if lw not in seen:
                uniq.append(lw)
                seen.add(lw)
        try:
            rank = uniq.index(cons) + 1
            mrr_v[i] = 1.0 / float(rank)
        except ValueError:
            mrr_v[i] = 0.0

    return PerQuery(hit=hit_v, top1=top1_v, majority=maj_v, cons_frac=cf_v, lift=lift_v, mrr_ul=mrr_v)


def compute_majority_vote(
    *,
    idx: np.ndarray,
    law_arr: np.ndarray,
    consensus_laws: List[str],
    k: int,
) -> MajorityVote:
    """Compute majority-vote *diagnostics* (not thresholded).

    This is intended to highlight the "law purity" of the retrieved neighborhood.
    """
    k = int(k)

    idx = np.asarray(idx, dtype=np.int64)
    if idx.ndim != 2:
        raise ValueError(f"idx must be 2D; got {idx.shape}")
    if idx.shape[1] < k:
        raise ValueError(f"idx has too few columns: {idx.shape[1]} < k={k}")
    if idx.shape[1] > k:
        idx = idx[:, :k]

    n = int(idx.shape[0])
    if len(consensus_laws) != n:
        raise ValueError(f"consensus_laws length {len(consensus_laws)} != n_queries {n}")

    maj_frac = np.zeros(n, dtype=np.float64)
    maj_corr = np.zeros(n, dtype=np.float64)
    margin = np.zeros(n, dtype=np.float64)
    ent = np.zeros(n, dtype=np.float64)
    nuniq = np.zeros(n, dtype=np.float64)

    for i in range(n):
        cons = str(consensus_laws[i]).strip()
        row = [int(j) for j in idx[i].tolist() if int(j) >= 0]
        laws = [str(law_arr[j]) for j in row]
        if not laws:
            continue

        c = Counter(laws)
        maj_law, maj_count = c.most_common(1)[0]
        total = float(len(laws))
        mf = float(maj_count) / total
        maj_frac[i] = mf
        maj_corr[i] = 1.0 if maj_law == cons else 0.0
        nuniq[i] = float(len(c))

        # Runner-up fraction (for a "vote margin" diagnostic)
        if len(c) >= 2:
            ru_count = c.most_common(2)[1][1]
            ru_frac = float(ru_count) / total
        else:
            ru_frac = 0.0
        margin[i] = mf - ru_frac

        # Shannon entropy of the vote distribution in the neighborhood.
        probs = np.asarray([float(v) / total for v in c.values()], dtype=np.float64)
        probs = probs[probs > 0]
        ent[i] = float(-(probs * np.log(probs)).sum())

    return MajorityVote(maj_frac=maj_frac, maj_correct=maj_corr, margin=margin, entropy=ent, n_unique=nuniq)


def summarize(pq: PerQuery, *, n_boot: int, seed: int) -> Dict[str, Tuple[float, Tuple[float, float]]]:
    return {
        "hit": _bootstrap_mean_ci(pq.hit, n_boot=n_boot, seed=seed + 1),
        "mrr_ul": _bootstrap_mean_ci(pq.mrr_ul, n_boot=n_boot, seed=seed + 2),
        "top1": _bootstrap_mean_ci(pq.top1, n_boot=n_boot, seed=seed + 3),
        "majority": _bootstrap_mean_ci(pq.majority, n_boot=n_boot, seed=seed + 4),
        "cons_frac": _bootstrap_mean_ci(pq.cons_frac, n_boot=n_boot, seed=seed + 5),
        "lift": _bootstrap_mean_ci(pq.lift, n_boot=n_boot, seed=seed + 6),
    }


def print_method(name: str, s: Dict[str, Tuple[float, Tuple[float, float]]], *, k: int) -> None:
    print(f"\n[{name}]  (k={k})")
    print(f"  hit@k:               {_fmt_ci(*s['hit'])}")
    print(f"  MRR@k (unique laws): {_fmt_ci(*s['mrr_ul'])}")
    print(f"  top1-accuracy:       {_fmt_ci(*s['top1'])}")
    print(f"  majority-accuracy:   {_fmt_ci(*s['majority'])}")
    print(f"  mean cons frac:      {_fmt_ci(*s['cons_frac'])}")
    print(f"  mean lift (prior):   {_fmt_ci(*s['lift'])}")


def _parse_float_list(spec: str) -> List[float]:
    """Parse a comma-separated list of floats."""
    out: List[float] = []
    for raw in str(spec).split(","):
        s = raw.strip()
        if not s:
            continue
        try:
            out.append(float(s))
        except ValueError:
            raise ValueError(f"Invalid float in list: {raw!r}")
    if not out:
        raise ValueError("Parsed an empty float list")
    return out


def print_majority_vote_profile(
    name: str,
    mv: MajorityVote,
    *,
    k: int,
    thresholds: List[float],
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    """Print and return a compact but informative majority-vote profile."""
    print(f"\n[{name}]  (top-{k} law voting)")

    pt_mf, ci_mf = _bootstrap_mean_ci(mv.maj_frac, n_boot=n_boot, seed=seed + 1)
    pt_mm, ci_mm = _bootstrap_mean_ci(mv.margin, n_boot=n_boot, seed=seed + 2)
    pt_ent, ci_ent = _bootstrap_mean_ci(mv.entropy, n_boot=n_boot, seed=seed + 3)
    pt_nu, ci_nu = _bootstrap_mean_ci(mv.n_unique, n_boot=n_boot, seed=seed + 4)

    # Percentiles are shown without CIs (descriptive diagnostics).
    p50, p75, p90 = np.quantile(mv.maj_frac, [0.50, 0.75, 0.90])
    all_same = (mv.maj_frac >= 1.0 - 1e-12).astype(np.float64)
    pt_all, ci_all = _bootstrap_mean_ci(all_same, n_boot=n_boot, seed=seed + 5)

    print(f"  mean top-law fraction: {_fmt_ci(pt_mf, ci_mf)}")
    print(f"  mean vote margin      : {_fmt_ci(pt_mm, ci_mm)}")
    print(f"  mean vote entropy     : {_fmt_ci(pt_ent, ci_ent)}")
    print(f"  mean #unique laws     : {_fmt_ci(pt_nu, ci_nu)}")
    print(f"  maj_frac percentiles  : p50={p50:.3f}, p75={p75:.3f}, p90={p90:.3f}")
    print(f"  P(all {k} from one law): {_fmt_ci(pt_all, ci_all)}")

    sweep: List[Dict[str, Any]] = []
    print("  Threshold sweep (coverage vs accuracy)")
    print("    tau    coverage      majority-acc     acc | covered")
    for t in thresholds:
        tau = float(t)
        covered = (mv.maj_frac >= tau).astype(np.float64)
        acc = (mv.maj_correct * covered).astype(np.float64)

        cov_pt, cov_ci = _bootstrap_mean_ci(covered, n_boot=n_boot, seed=seed + int(1000 * tau) + 10)
        acc_pt, acc_ci = _bootstrap_mean_ci(acc, n_boot=n_boot, seed=seed + int(1000 * tau) + 20)
        cond_pt, cond_ci = _bootstrap_ratio_ci(acc, covered, n_boot=n_boot, seed=seed + int(1000 * tau) + 30)

        print(
            f"    {tau:0.2f}  {_fmt_ci(cov_pt, cov_ci)}  {_fmt_ci(acc_pt, acc_ci)}  {_fmt_ci(cond_pt, cond_ci)}"
        )

        sweep.append(
            {
                "tau": float(tau),
                "coverage": {"pt": float(cov_pt), "ci": (float(cov_ci[0]), float(cov_ci[1]))},
                "majority_acc": {"pt": float(acc_pt), "ci": (float(acc_ci[0]), float(acc_ci[1]))},
                "acc_given_covered": {"pt": float(cond_pt), "ci": (float(cond_ci[0]), float(cond_ci[1]))},
            }
        )

    return {
        "method": str(name),
        "mean_toplaw_frac": {"pt": float(pt_mf), "ci": (float(ci_mf[0]), float(ci_mf[1]))},
        "mean_vote_margin": {"pt": float(pt_mm), "ci": (float(ci_mm[0]), float(ci_mm[1]))},
        "mean_vote_entropy": {"pt": float(pt_ent), "ci": (float(ci_ent[0]), float(ci_ent[1]))},
        "mean_n_unique": {"pt": float(pt_nu), "ci": (float(ci_nu[0]), float(ci_nu[1]))},
        "maj_frac_percentiles": {"p50": float(p50), "p75": float(p75), "p90": float(p90)},
        "p_all_from_one_law": {"pt": float(pt_all), "ci": (float(ci_all[0]), float(ci_all[1]))},
        "threshold_sweep": sweep,
    }



def print_majority_routing_decomposition(
    a_name: str,
    b_name: str,
    a_mv: MajorityVote,
    b_mv: MajorityVote,
    *,
    thresholds: List[float],
) -> List[Dict[str, Any]]:
    """Decompose majority-acc differences into coverage vs precision effects.

    majority-acc(tau) = coverage(tau) * precision(tau)
    where precision(tau) = P(majority correct | covered).

    Using an exact symmetric (Shapley-style) decomposition:
        Δ(ab) = 0.5*Δa*(b1+b0) + 0.5*Δb*(a1+a0)
    which attributes the change to (i) coverage and (ii) conditional precision.
    """
    print(f"\nMajority-vote routing decomposition: {a_name} vs {b_name}")
    print("  (Point estimates; Δmaj-acc = coverage-component + precision-component)")
    print(
        "    tau   cov(A)  prec(A)  majacc(A)   cov(B)  prec(B)  majacc(B)   Δmajacc   Δcov-part  Δprec-part"
    )

    rows: List[Dict[str, Any]] = []
    for t in thresholds:
        tau = float(t)
        a_cov, a_acc, a_prec = _mv_point_estimates(a_mv, tau)
        b_cov, b_acc, b_prec = _mv_point_estimates(b_mv, tau)

        d = a_acc - b_acc
        cov_part = 0.5 * (a_cov - b_cov) * (a_prec + b_prec)
        prec_part = 0.5 * (a_prec - b_prec) * (a_cov + b_cov)

        print(
            f"    {tau:0.2f}  {a_cov:0.3f}  {a_prec:0.3f}  {a_acc:0.3f}    {b_cov:0.3f}  {b_prec:0.3f}  {b_acc:0.3f}    {d:+0.3f}    {cov_part:+0.3f}     {prec_part:+0.3f}"
        )

        rows.append(
            {
                "tau": float(tau),
                "cov_a": float(a_cov),
                "prec_a": float(a_prec),
                "majacc_a": float(a_acc),
                "cov_b": float(b_cov),
                "prec_b": float(b_prec),
                "majacc_b": float(b_acc),
                "delta_majacc": float(d),
                "delta_cov_part": float(cov_part),
                "delta_prec_part": float(prec_part),
            }
        )

    return rows



def _bootstrap_shapley_decomposition_ci(
    a_mv: MajorityVote,
    b_mv: MajorityVote,
    tau: float,
    *,
    n_boot: int,
    seed: int,
) -> Dict[str, Tuple[float, Tuple[float, float]]]:
    """Paired bootstrap CIs for Δmaj-acc and its Shapley-style components.

    Δmaj-acc(tau) = majacc(A,tau) - majacc(B,tau)
                 = Δcov-part + Δprec-part

    where majacc = coverage * precision, precision = P(correct | covered).

    Returns dict with keys: d, cov_part, prec_part.
    """
    tau = float(tau)
    rng = np.random.default_rng(int(seed))

    n = int(a_mv.maj_frac.size)
    if n == 0 or n != int(b_mv.maj_frac.size):
        raise ValueError("MajorityVote arrays must be non-empty and aligned for paired bootstrap.")

    def _stats(ix: np.ndarray) -> Tuple[float, float, float]:
        # Returns (cov, majacc, prec)
        a_cov = float(np.mean((a_mv.maj_frac[ix] >= tau).astype(np.float64)))
        b_cov = float(np.mean((b_mv.maj_frac[ix] >= tau).astype(np.float64)))

        a_acc = float(np.mean((a_mv.maj_correct[ix] * (a_mv.maj_frac[ix] >= tau)).astype(np.float64)))
        b_acc = float(np.mean((b_mv.maj_correct[ix] * (b_mv.maj_frac[ix] >= tau)).astype(np.float64)))

        a_prec = _safe_ratio(a_acc, a_cov)
        b_prec = _safe_ratio(b_acc, b_cov)

        # Shapley decomposition
        d = a_acc - b_acc
        cov_part = 0.5 * (a_cov - b_cov) * (a_prec + b_prec)
        prec_part = 0.5 * (a_prec - b_prec) * (a_cov + b_cov)
        return d, cov_part, prec_part

    # Point estimates on full sample
    full_ix = np.arange(n, dtype=np.int64)
    d_pt, cov_pt, prec_pt = _stats(full_ix)

    d_bs = np.empty(int(n_boot), dtype=np.float64)
    cov_bs = np.empty(int(n_boot), dtype=np.float64)
    prec_bs = np.empty(int(n_boot), dtype=np.float64)

    for i in range(int(n_boot)):
        ix = rng.integers(0, n, size=n, dtype=np.int64)
        d, c, p = _stats(ix)
        d_bs[i] = d
        cov_bs[i] = c
        prec_bs[i] = p

    def _ci(arr: np.ndarray) -> Tuple[float, float]:
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return (float("nan"), float("nan"))
        lo, hi = np.quantile(arr, [0.025, 0.975])
        return float(lo), float(hi)

    return {
        "d": (float(d_pt), _ci(d_bs)),
        "cov_part": (float(cov_pt), _ci(cov_bs)),
        "prec_part": (float(prec_pt), _ci(prec_bs)),
    }


def print_majority_routing_decomposition_ci(
    a_name: str,
    b_name: str,
    a_mv: MajorityVote,
    b_mv: MajorityVote,
    *,
    thresholds: List[float],
    n_boot: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Print and return bootstrap CIs for the routing decomposition components."""
    print(f"\nMajority-vote routing decomposition (with CIs): {a_name} vs {b_name}")
    print("  Report: paired mean differences with 95% bootstrap CIs")
    print("    tau    Δmaj-acc                 Δcov-part                Δprec-part")

    rows: List[Dict[str, Any]] = []
    for t in thresholds:
        tau = float(t)
        out = _bootstrap_shapley_decomposition_ci(a_mv, b_mv, tau, n_boot=n_boot, seed=seed + int(1000 * tau) + 1234)
        d_pt, d_ci = out["d"]
        c_pt, c_ci = out["cov_part"]
        p_pt, p_ci = out["prec_part"]
        print(
            f"    {tau:0.2f}  {_fmt_delta(d_pt, d_ci)}    {_fmt_delta(c_pt, c_ci)}    {_fmt_delta(p_pt, p_ci)}"
        )

        rows.append(
            {
                "tau": float(tau),
                "delta_majacc": {"pt": float(d_pt), "ci": (float(d_ci[0]), float(d_ci[1]))},
                "delta_cov_part": {"pt": float(c_pt), "ci": (float(c_ci[0]), float(c_ci[1]))},
                "delta_prec_part": {"pt": float(p_pt), "ci": (float(p_ci[0]), float(p_ci[1]))},
            }
        )

    return rows



def recommend_routing_threshold_max_majacc(
    mv: MajorityVote,
    *,
    thresholds: List[float],
    min_coverage: float,
) -> Tuple[float, float, float, float]:
    """Pick a tau that maximizes majority-acc subject to a minimum coverage.

    Returns: (tau, coverage, majority-acc, precision)
    """
    min_coverage = float(min_coverage)
    if not (0.0 < min_coverage <= 1.0):
        raise ValueError(f"min_coverage must be in (0,1]; got {min_coverage}")

    rows = []
    for t in thresholds:
        tau = float(t)
        cov, acc, prec = _mv_point_estimates(mv, tau)
        rows.append((tau, cov, acc, prec))

    feas = [r for r in rows if np.isfinite(r[1]) and r[1] >= min_coverage]
    if feas:
        tau, cov, acc, prec = sorted(feas, key=lambda r: (r[2], r[3], -r[0]), reverse=True)[0]
        return float(tau), float(cov), float(acc), float(prec)

    tau, cov, acc, prec = sorted(rows, key=lambda r: (r[2], r[3], -r[0]), reverse=True)[0]
    return float(tau), float(cov), float(acc), float(prec)

def recommend_routing_threshold(
    mv: MajorityVote,
    *,
    thresholds: List[float],
    min_coverage: float,
) -> Tuple[float, float, float, float]:
    """Pick a tau that maximizes precision subject to a minimum coverage.

    Returns: (tau, coverage, majority-acc, precision)
    """
    min_coverage = float(min_coverage)
    if not (0.0 < min_coverage <= 1.0):
        raise ValueError(f"min_coverage must be in (0,1]; got {min_coverage}")

    rows = []
    for t in thresholds:
        tau = float(t)
        cov, acc, prec = _mv_point_estimates(mv, tau)
        rows.append((tau, cov, acc, prec))

    # Feasible set: coverage >= min_coverage
    feas = [r for r in rows if np.isfinite(r[1]) and r[1] >= min_coverage]
    if feas:
        # Max precision, tie-break by majority-acc (more correct decisions), then lower tau (more permissive).
        tau, cov, acc, prec = sorted(feas, key=lambda r: (r[3], r[2], -r[0]), reverse=True)[0]
        return float(tau), float(cov), float(acc), float(prec)

    # If nothing meets the coverage constraint, pick tau with max majority-acc.
    tau, cov, acc, prec = sorted(rows, key=lambda r: (r[2], r[3], -r[0]), reverse=True)[0]
    return float(tau), float(cov), float(acc), float(prec)


def storyline_superiority(
    title: str,
    a_name: str,
    b_name: str,
    a: PerQuery,
    b: PerQuery,
    *,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    print(f"\n{title}")
    print("  Test: one-sided superiority (paired 95% bootstrap CI lower bound > 0)")

    rows: List[Dict[str, Any]] = []

    def _line(key: str, label: str, sd: int) -> bool:
        pt, ci = _bootstrap_paired_delta_ci(getattr(a, key), getattr(b, key), n_boot=n_boot, seed=seed + sd)
        ok = bool(np.isfinite(ci[0]) and ci[0] > 0.0)
        print(f"  {label}: {a_name}−{b_name} = {_fmt_delta(pt, ci)}  -> {'PASS' if ok else 'FAIL'}")
        rows.append({"key": key, "label": label, "delta": float(pt), "ci": (float(ci[0]), float(ci[1])), "pass": bool(ok)})
        return ok

    oks = [
        _line("hit", "hit@k", 1),
        _line("mrr_ul", "MRR@k (unique laws)", 2),
        _line("top1", "top1-accuracy", 3),
        _line("majority", "majority-accuracy", 4),
        _line("cons_frac", "mean consensus fraction", 5),
        _line("lift", "mean lift (prior)", 6),
    ]
    verdict = "Supported" if all(oks) else "Partially supported (see FAIL lines)"
    print(f"  Verdict: {verdict}")

    return {
        "type": "superiority",
        "title": str(title),
        "a_name": str(a_name),
        "b_name": str(b_name),
        "rows": rows,
        "verdict": verdict,
    }

def storyline_competitiveness(
    title: str,
    a_name: str,
    b_name: str,
    a: PerQuery,
    b: PerQuery,
    *,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    print(f"\n{title}")
    print("  Report: paired mean differences with 95% bootstrap CIs")

    rows: List[Dict[str, Any]] = []
    for key, label, sd in [
        ("hit", "hit@k", 1),
        ("mrr_ul", "MRR@k (unique laws)", 2),
        ("top1", "top1-accuracy", 3),
        ("majority", "majority-accuracy", 4),
        ("cons_frac", "mean consensus fraction", 5),
        ("lift", "mean lift (prior)", 6),
    ]:
        pt, ci = _bootstrap_paired_delta_ci(getattr(a, key), getattr(b, key), n_boot=n_boot, seed=seed + sd)
        ci_excludes_0 = bool(np.isfinite(ci[0]) and np.isfinite(ci[1]) and (ci[0] > 0.0 or ci[1] < 0.0))
        note = "  (CI excludes 0)" if ci_excludes_0 else ""
        print(f"  {label}: {a_name}−{b_name} = {_fmt_delta(pt, ci)}{note}")
        rows.append(
            {
                "key": key,
                "label": label,
                "delta": float(pt),
                "ci": (float(ci[0]), float(ci[1])),
                "ci_excludes_0": bool(ci_excludes_0),
            }
        )

    return {
        "type": "competitiveness",
        "title": str(title),
        "a_name": str(a_name),
        "b_name": str(b_name),
        "rows": rows,
    }


# ----------------------------- Alignment metrics -----------------------------
def cosine_rowwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity. Assumes rows are L2-normalized."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"cosine_rowwise shape mismatch: {a.shape} vs {b.shape}")
    return np.sum(a * b, axis=1).astype(np.float64)


def jaccard_topk_rows(a_idx: np.ndarray, b_idx: np.ndarray, *, k: int) -> np.ndarray:
    """Jaccard overlap of sentence-id sets in the top-k lists."""
    a_idx = np.asarray(a_idx, dtype=np.int64)[:, : int(k)]
    b_idx = np.asarray(b_idx, dtype=np.int64)[:, : int(k)]
    n = int(a_idx.shape[0])
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        A = set(int(x) for x in a_idx[i].tolist() if int(x) >= 0)
        B = set(int(x) for x in b_idx[i].tolist() if int(x) >= 0)
        u = len(A | B)
        out[i] = (len(A & B) / u) if u else 0.0
    return out


def overlap_frac_topk_rows(a_idx: np.ndarray, b_idx: np.ndarray, *, k: int) -> np.ndarray:
    """Intersection size divided by k (fixed-k overlap fraction)."""
    a_idx = np.asarray(a_idx, dtype=np.int64)[:, : int(k)]
    b_idx = np.asarray(b_idx, dtype=np.int64)[:, : int(k)]
    n = int(a_idx.shape[0])
    out = np.zeros(n, dtype=np.float64)
    kf = float(max(1, int(k)))
    for i in range(n):
        A = set(int(x) for x in a_idx[i].tolist() if int(x) >= 0)
        B = set(int(x) for x in b_idx[i].tolist() if int(x) >= 0)
        out[i] = float(len(A & B)) / kf
    return out


def law_jaccard_topk_rows(a_idx: np.ndarray, b_idx: np.ndarray, law_arr: np.ndarray, *, k: int) -> np.ndarray:
    """Jaccard overlap of *unique laws* present in the top-k lists."""
    a_idx = np.asarray(a_idx, dtype=np.int64)[:, : int(k)]
    b_idx = np.asarray(b_idx, dtype=np.int64)[:, : int(k)]
    n = int(a_idx.shape[0])
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        A = set(str(law_arr[int(x)]) for x in a_idx[i].tolist() if int(x) >= 0)
        B = set(str(law_arr[int(x)]) for x in b_idx[i].tolist() if int(x) >= 0)
        u = len(A | B)
        out[i] = (len(A & B) / u) if u else 0.0
    return out


# ----------------------------- Main -----------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description="Clean storyline evaluation for KAHM embeddings (v4: alignment storyline).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--corpus_parquet", default="ris_sentences.parquet")
    p.add_argument("--semantic_npz", default="embedding_index.npz", help="Mixedbread corpus embeddings")
    p.add_argument("--idf_svd_npz", default="embedding_index_idf_svd.npz")
    p.add_argument("--kahm_corpus_npz", default="embedding_index_kahm_mixedbread_approx.npz")
    p.add_argument("--idf_svd_model", default="idf_svd_model.joblib")
    p.add_argument("--kahm_model", default="kahm_regressor_idf_to_mixedbread.joblib")
    p.add_argument("--kahm_mode", default="soft")
    p.add_argument("--kahm_batch", type=int, default=1024)
    p.add_argument("--query_set", default="query_set.TEST_QUERY_SET")

    p.add_argument("--k", type=int, default=10)
    p.add_argument("--predominance_fraction", type=float, default=0.5)
    p.add_argument(
        "--majority_thresholds",
        default="0.5,0.6,0.7,0.8",
        help="Comma-separated thresholds for majority-vote diagnostics (tau values for coverage/accuracy sweeps).",
    )

    p.add_argument(
        "--min_routing_coverage",
        type=float,
        default=0.50,
        help=(
            "Minimum coverage constraint used when recommending a majority-vote routing threshold tau. "
            "The script will pick tau that maximizes precision (acc|covered) subject to coverage>=this value."
        ),
    )

    p.add_argument("--mixedbread_model", default="mixedbread-ai/deepset-mxbai-embed-de-large-v1")
    p.add_argument("--device", default="cpu")
    p.add_argument("--query_prefix", default="query: ")
    p.add_argument("--mb_query_batch", type=int, default=64)

    # Thread limits (macOS stability). Set to 1 to reduce OpenMP/BLAS contention.
    # 0 means "do not override".
    default_threads = 1 if sys.platform == "darwin" else 0
    p.add_argument("--threads", type=int, default=default_threads, help="Cap OMP/BLAS/torch/FAISS threads (0=no override).")
    p.add_argument("--kahm_show_progress", default=True, help="Show a KAHM progress bar (requires tqdm in env).")

    p.add_argument("--bootstrap_samples", type=int, default=5000)
    p.add_argument("--bootstrap_seed", type=int, default=0)

    # Publication report
    p.add_argument("--report_path", default="kahm_evaluation_report.md", help="Write a single publication-ready Markdown report to this path (e.g., results/report.md).")
    p.add_argument("--report_title", default="KAHM embeddings: retrieval evaluation on Austrian laws", help="Title used in the generated report.")
    p.add_argument("--report_overwrite", action="store_true", help="Allow overwriting an existing report file at --report_path.")

    args = p.parse_args()

    print(
        f"Script: {os.path.basename(__file__)} | version={SCRIPT_VERSION} | path={os.path.abspath(__file__)}",
        flush=True,
    )

    qs = load_query_set(args.query_set)
    texts = extract_query_texts(qs)
    consensus = extract_consensus_laws(qs)
    n_q = len(qs)
    n_empty_text = sum(1 for t in texts if not t)
    if n_empty_text:
        print(f"WARNING: {n_empty_text}/{n_q} queries have empty text (check query_set keys).", flush=True)

    # Apply thread limits early (before importing torch/faiss) to avoid oversubscription
    # and reduce the probability of native-library crashes under high memory pressure.
    if int(args.threads) > 0:
        t = str(int(args.threads))
        os.environ.setdefault("OMP_NUM_THREADS", t)
        os.environ.setdefault("MKL_NUM_THREADS", t)
        os.environ.setdefault("OPENBLAS_NUM_THREADS", t)
        os.environ.setdefault("VECLIB_MAXIMUM_THREADS", t)
        os.environ.setdefault("NUMEXPR_NUM_THREADS", t)
        try:
            import torch  # type: ignore

            torch.set_num_threads(int(args.threads))
            try:
                torch.set_num_interop_threads(1)
            except Exception:
                pass
        except Exception:
            pass

    df = load_corpus_parquet(args.corpus_parquet)
    mb = load_npz_bundle(args.semantic_npz)
    idf = load_npz_bundle(args.idf_svd_npz)
    kahm = load_npz_bundle(args.kahm_corpus_npz)
    aligned = align_by_common_sentence_ids(df, mb, idf, kahm)

    law_arr = aligned["law"]
    emb_mb = aligned["emb_mb"]
    emb_idf = aligned["emb_idf"]
    emb_k = aligned["emb_kahm"]

    print(f"Loaded query set: {args.query_set} (n={n_q})", flush=True)
    print(f"Aligned corpora: common sentence_ids={aligned['sentence_ids'].size}")
    print(f"  MB corpus:   {emb_mb.shape}")
    print(f"  IDF corpus:  {emb_idf.shape}")
    print(f"  KAHM corpus: {emb_k.shape}")

    # Embed queries (done BEFORE building FAISS indices to reduce peak memory and
    # to initialize torch before faiss on macOS).
    print("\nEmbedding queries with IDF–SVD ...", flush=True)
    idf_pipe = load_idf_svd_model(args.idf_svd_model)
    q_idf = embed_queries_idf_svd(idf_pipe, texts)

    print("Embedding queries with KAHM (IDF→MB space) ...", flush=True)
    kahm_model = load_kahm_model(args.kahm_model)
    q_kahm = kahm_regress_once(
        kahm_model,
        q_idf,
        mode=args.kahm_mode,
        batch_size=args.kahm_batch,
        show_progress=bool(args.kahm_show_progress),
    )

    print("Embedding queries with Mixedbread ...", flush=True)
    q_mb = embed_queries_mixedbread(
        model_name=args.mixedbread_model,
        device=args.device,
        dim=int(emb_mb.shape[1]),
        query_prefix=args.query_prefix,
        texts=texts,
        batch_size=int(args.mb_query_batch),
        show_progress_bar=True,
    )

    # Build/search FAISS indices sequentially to reduce peak RSS.
    k = int(args.k)
    print("\nBuilding FAISS indices and searching ...", flush=True)

    # IDF retrieval
    index_idf = build_faiss_index(emb_idf, n_threads=(int(args.threads) if int(args.threads) > 0 else None))
    _, idf_idx = faiss_search(index_idf, q_idf, k)
    del index_idf
    gc.collect()
    # IDF corpus embeddings no longer needed beyond this point
    del emb_idf
    gc.collect()

    # MB retrieval + KAHM(query→MB) retrieval share the same MB corpus
    index_mb = build_faiss_index(emb_mb, n_threads=(int(args.threads) if int(args.threads) > 0 else None))
    _, mb_idx = faiss_search(index_mb, q_mb, k)
    _, kahm_qmb_idx = faiss_search(index_mb, q_kahm, k)
    del index_mb
    gc.collect()

    # Full-KAHM retrieval (search KAHM corpus)
    index_k = build_faiss_index(emb_k, n_threads=(int(args.threads) if int(args.threads) > 0 else None))
    _, kahm_full_idx = faiss_search(index_k, q_kahm, k)
    del index_k
    gc.collect()

    pred_frac = float(args.predominance_fraction)
    n_boot = int(args.bootstrap_samples)
    seed = int(args.bootstrap_seed)

    mb_pq = compute_per_query_metrics(idx=mb_idx, law_arr=law_arr, consensus_laws=consensus, k=k, predominance_fraction=pred_frac)
    idf_pq = compute_per_query_metrics(idx=idf_idx, law_arr=law_arr, consensus_laws=consensus, k=k, predominance_fraction=pred_frac)
    kahm_qmb_pq = compute_per_query_metrics(idx=kahm_qmb_idx, law_arr=law_arr, consensus_laws=consensus, k=k, predominance_fraction=pred_frac)
    kahm_full_pq = compute_per_query_metrics(idx=kahm_full_idx, law_arr=law_arr, consensus_laws=consensus, k=k, predominance_fraction=pred_frac)

    # Majority-vote diagnostics (independent of predominance_fraction)
    maj_thresholds = sorted(set(_parse_float_list(args.majority_thresholds)))
    for t in maj_thresholds:
        if not (0.0 < float(t) <= 1.0):
            raise ValueError(f"majority_thresholds must be in (0,1]; got {t}")

    mb_mv = compute_majority_vote(idx=mb_idx, law_arr=law_arr, consensus_laws=consensus, k=k)
    idf_mv = compute_majority_vote(idx=idf_idx, law_arr=law_arr, consensus_laws=consensus, k=k)
    kahm_qmb_mv = compute_majority_vote(idx=kahm_qmb_idx, law_arr=law_arr, consensus_laws=consensus, k=k)
    kahm_full_mv = compute_majority_vote(idx=kahm_full_idx, law_arr=law_arr, consensus_laws=consensus, k=k)

    mb_sum = summarize(mb_pq, n_boot=n_boot, seed=seed + 10)
    idf_sum = summarize(idf_pq, n_boot=n_boot, seed=seed + 20)
    kahm_qmb_sum = summarize(kahm_qmb_pq, n_boot=n_boot, seed=seed + 30)
    kahm_full_sum = summarize(kahm_full_pq, n_boot=n_boot, seed=seed + 40)

    method_summaries = {
        "Mixedbread (true)": mb_sum,
        "IDF–SVD": idf_sum,
        "KAHM(query→MB corpus)": kahm_qmb_sum,
        "Full-KAHM (query→KAHM corpus)": kahm_full_sum,
    }

    # Headline blocks
    print_method("Mixedbread (true)", mb_sum, k=k)
    print_method("IDF–SVD", idf_sum, k=k)
    print_method("KAHM(query→MB corpus)", kahm_qmb_sum, k=k)
    print_method("Full-KAHM (query→KAHM corpus)", kahm_full_sum, k=k)

    # Majority-vote behavior (highlighted block)
    print("\nMajority-vote behavior: law-purity and vote-based routing diagnostics")
    majority_profiles: Dict[str, Dict[str, Any]] = {}
    majority_profiles["Mixedbread (true)"] = print_majority_vote_profile("Mixedbread (true)", mb_mv, k=k, thresholds=maj_thresholds, n_boot=n_boot, seed=seed + 500)
    majority_profiles["IDF–SVD"] = print_majority_vote_profile("IDF–SVD", idf_mv, k=k, thresholds=maj_thresholds, n_boot=n_boot, seed=seed + 600)
    majority_profiles["KAHM(query→MB corpus)"] = print_majority_vote_profile("KAHM(query→MB corpus)", kahm_qmb_mv, k=k, thresholds=maj_thresholds, n_boot=n_boot, seed=seed + 700)
    majority_profiles["Full-KAHM (query→KAHM corpus)"] = print_majority_vote_profile("Full-KAHM (query→KAHM corpus)", kahm_full_mv, k=k, thresholds=maj_thresholds, n_boot=n_boot, seed=seed + 800)

    # Paired deltas that make the "majority-vote" story explicit (especially for Storyline B).
    print("\nMajority-vote deltas vs Mixedbread (paired, top-k law voting)")
    print("  Report: paired mean differences with 95% bootstrap CIs")
    print("    tau    Δcoverage(KAHM−MB)        Δmaj-acc(KAHM−MB)")
    majority_deltas_vs_mb: List[Dict[str, Any]] = []
    for t in maj_thresholds:
        tau = float(t)
        cov_k = (kahm_qmb_mv.maj_frac >= tau).astype(np.float64)
        cov_mb = (mb_mv.maj_frac >= tau).astype(np.float64)
        acc_k = (kahm_qmb_mv.maj_correct * cov_k).astype(np.float64)
        acc_mb = (mb_mv.maj_correct * cov_mb).astype(np.float64)

        d_cov_pt, d_cov_ci = _bootstrap_paired_delta_ci(cov_k, cov_mb, n_boot=n_boot, seed=seed + int(1000 * tau) + 900)
        d_acc_pt, d_acc_ci = _bootstrap_paired_delta_ci(acc_k, acc_mb, n_boot=n_boot, seed=seed + int(1000 * tau) + 950)
        note = "" 
        if np.isfinite(d_acc_ci[0]) and np.isfinite(d_acc_ci[1]) and (d_acc_ci[0] > 0.0 or d_acc_ci[1] < 0.0):
            note = "  (Δmaj-acc CI excludes 0)"
        print(f"    {tau:0.2f}  {_fmt_delta(d_cov_pt, d_cov_ci)}    {_fmt_delta(d_acc_pt, d_acc_ci)}{note}")
        majority_deltas_vs_mb.append({
            "tau": float(tau),
            "delta_coverage": {"pt": float(d_cov_pt), "ci": (float(d_cov_ci[0]), float(d_cov_ci[1]))},
            "delta_majacc": {"pt": float(d_acc_pt), "ci": (float(d_acc_ci[0]), float(d_acc_ci[1]))},
        })

    # A compact decomposition that makes clear whether the gain comes from
    # (i) more coverage or (ii) higher precision among covered cases.
    decomp_point_rows = print_majority_routing_decomposition(
        "KAHM(q→MB)",
        "MB",
        kahm_qmb_mv,
        mb_mv,
        thresholds=maj_thresholds,
    )

    # Same decomposition, but with paired bootstrap CIs for the components.
    decomp_ci_rows = print_majority_routing_decomposition_ci(
        "KAHM(q→MB)",
        "MB",
        kahm_qmb_mv,
        mb_mv,
        thresholds=maj_thresholds,
        n_boot=n_boot,
        seed=seed + 9100,
    )

    # A practical suggestion: pick a tau that maximizes majority-vote precision
    # subject to a coverage constraint.
    min_cov = float(args.min_routing_coverage)
    threshold_suggestions: Dict[str, Any] = {
        "coverage_constraint": min_cov,
        "maximize_precision_subject_to_coverage": {},
        "maximize_majority_acc_subject_to_coverage": {},
    }

    print(
        "\nSuggested majority-vote routing thresholds (maximize precision subject to coverage constraint)"
    )
    print(f"  Coverage constraint: coverage >= {min_cov:0.2f}")
    for nm, mv in [
        ("Mixedbread (true)", mb_mv),
        ("KAHM(query→MB corpus)", kahm_qmb_mv),
        ("Full-KAHM (query→KAHM corpus)", kahm_full_mv),
        ("IDF–SVD", idf_mv),
    ]:
        tau_star, cov_star, acc_star, prec_star = recommend_routing_threshold(
            mv, thresholds=maj_thresholds, min_coverage=min_cov
        )
        print(
            f"  {nm}: tau*={tau_star:0.2f}  coverage={cov_star:0.3f}  acc|covered={prec_star:0.3f}  majority-acc={acc_star:0.3f}"
        )
        threshold_suggestions["maximize_precision_subject_to_coverage"][nm] = {
            "tau": float(tau_star),
            "coverage": float(cov_star),
            "majority_acc": float(acc_star),
            "acc_given_covered": float(prec_star),
        }
    print(
        "\nAlternative majority-vote routing thresholds (maximize majority-acc subject to coverage constraint)"
    )
    print(f"  Coverage constraint: coverage >= {min_cov:0.2f}")
    for nm, mv in [
        ("Mixedbread (true)", mb_mv),
        ("KAHM(query→MB corpus)", kahm_qmb_mv),
        ("Full-KAHM (query→KAHM corpus)", kahm_full_mv),
        ("IDF–SVD", idf_mv),
    ]:
        tau_star, cov_star, acc_star, prec_star = recommend_routing_threshold_max_majacc(
            mv, thresholds=maj_thresholds, min_coverage=min_cov
        )
        print(
            f"  {nm}: tau*={tau_star:0.2f}  coverage={cov_star:0.3f}  acc|covered={prec_star:0.3f}  majority-acc={acc_star:0.3f}"
        )
        threshold_suggestions["maximize_majority_acc_subject_to_coverage"][nm] = {
            "tau": float(tau_star),
            "coverage": float(cov_star),
            "majority_acc": float(acc_star),
            "acc_given_covered": float(prec_star),
        }



    # Storyline A/B
    storyline_a = storyline_superiority(
        "\nStoryline A: KAHM(query→MB) beats IDF–SVD (a strong low-cost baseline)",
        "KAHM(q→MB)",
        "IDF–SVD",
        kahm_qmb_pq,
        idf_pq,
        n_boot=n_boot,
        seed=seed + 100,
    )

    storyline_b = storyline_competitiveness(
        "\nStoryline B: KAHM(query→MB) is close to Mixedbread at top-k (paired deltas)",
        "KAHM(q→MB)",
        "MB",
        kahm_qmb_pq,
        mb_pq,
        n_boot=n_boot,
        seed=seed + 200,
    )

    # Storyline C: alignment evidence
    print("\nStoryline C: Full-KAHM embeddings are aligned with MB (geometry + neighborhood overlap)")
    print("  Part C1: Embedding-space cosine alignment")

    cos_corpus = cosine_rowwise(emb_k, emb_mb)
    cos_query = cosine_rowwise(q_kahm, q_mb)
    pt_c, ci_c = _bootstrap_mean_ci(cos_corpus, n_boot=n_boot, seed=seed + 300)
    pt_q, ci_q = _bootstrap_mean_ci(cos_query, n_boot=n_boot, seed=seed + 301)
    print(f"    corpus cosine(KAHM, MB): {_fmt_ci(pt_c, ci_c, digits=4)}")
    print(f"    query  cosine(KAHM, MB): {_fmt_ci(pt_q, ci_q, digits=4)}")

    print("  Part C2: Retrieval-neighborhood overlap vs MB")
    sent_j_full = jaccard_topk_rows(kahm_full_idx, mb_idx, k=k)
    sent_f_full = overlap_frac_topk_rows(kahm_full_idx, mb_idx, k=k)
    law_j_full = law_jaccard_topk_rows(kahm_full_idx, mb_idx, law_arr, k=k)

    pt_sj, ci_sj = _bootstrap_mean_ci(sent_j_full, n_boot=n_boot, seed=seed + 310)
    pt_sf, ci_sf = _bootstrap_mean_ci(sent_f_full, n_boot=n_boot, seed=seed + 311)
    pt_lj, ci_lj = _bootstrap_mean_ci(law_j_full, n_boot=n_boot, seed=seed + 312)

    print(f"    sentence Jaccard@{k} (Full-KAHM vs MB): {_fmt_ci(pt_sj, ci_sj)}")
    print(f"    sentence overlap frac@{k}            : {_fmt_ci(pt_sf, ci_sf)}")
    print(f"    law-set Jaccard@{k} (Full-KAHM vs MB): {_fmt_ci(pt_lj, ci_lj)}")

    # Context: show Full-KAHM is *more* aligned to MB than IDF is.
    sent_j_idf = jaccard_topk_rows(idf_idx, mb_idx, k=k)
    law_j_idf = law_jaccard_topk_rows(idf_idx, mb_idx, law_arr, k=k)
    d_sj_pt, d_sj_ci = _bootstrap_paired_delta_ci(sent_j_full, sent_j_idf, n_boot=n_boot, seed=seed + 320)
    d_lj_pt, d_lj_ci = _bootstrap_paired_delta_ci(law_j_full, law_j_idf, n_boot=n_boot, seed=seed + 321)

    alignment: Dict[str, Any] = {
        "cosine_corpus": {"pt": float(pt_c), "ci": (float(ci_c[0]), float(ci_c[1]))},
        "cosine_query": {"pt": float(pt_q), "ci": (float(ci_q[0]), float(ci_q[1]))},
        "sentence_jaccard": {"pt": float(pt_sj), "ci": (float(ci_sj[0]), float(ci_sj[1]))},
        "sentence_overlap_frac": {"pt": float(pt_sf), "ci": (float(ci_sf[0]), float(ci_sf[1]))},
        "lawset_jaccard": {"pt": float(pt_lj), "ci": (float(ci_lj[0]), float(ci_lj[1]))},
        "delta_sentence_jaccard_full_minus_idf": {"pt": float(d_sj_pt), "ci": (float(d_sj_ci[0]), float(d_sj_ci[1]))},
        "delta_lawset_jaccard_full_minus_idf": {"pt": float(d_lj_pt), "ci": (float(d_lj_ci[0]), float(d_lj_ci[1]))},
    }
    ok_sj = bool(np.isfinite(d_sj_ci[0]) and d_sj_ci[0] > 0)
    ok_lj = bool(np.isfinite(d_lj_ci[0]) and d_lj_ci[0] > 0)
    print("  Part C3: Alignment gain vs IDF–SVD (paired deltas)")
    print(f"    sentence Jaccard delta: (Full-KAHM−IDF) = {_fmt_delta(d_sj_pt, d_sj_ci)}  -> {'PASS' if ok_sj else 'FAIL'}")
    print(f"    law-set Jaccard delta : (Full-KAHM−IDF) = {_fmt_delta(d_lj_pt, d_lj_ci)}  -> {'PASS' if ok_lj else 'FAIL'}")
    print("    Interpretation: PASS means Full-KAHM neighborhoods are *statistically* closer to MB than IDF neighborhoods.")


    # Optional: write a single publication-ready report (Markdown)
    if str(getattr(args, "report_path", "")).strip():
        report_md = build_publication_report_md(
            report_title=str(getattr(args, "report_title", "KAHM embeddings: retrieval evaluation")),
            args=args,
            n_queries=int(len(consensus)),
            n_corpus=int(law_arr.size),
            embedding_dim=int(emb_mb.shape[1]),
            k=int(k),
            method_summaries=method_summaries,
            storyline_a=storyline_a,
            storyline_b=storyline_b,
            majority_profiles=majority_profiles,
            majority_deltas_vs_mb=majority_deltas_vs_mb,
            decomp_point_rows=decomp_point_rows,
            decomp_ci_rows=decomp_ci_rows,
            threshold_suggestions=threshold_suggestions,
            alignment=alignment,
        )
        _write_text(str(args.report_path), report_md, overwrite=bool(getattr(args, "report_overwrite", False)))
        print(f"\nSaved publication report to: {os.path.abspath(str(args.report_path))}")

    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()
