"""
kahm_dashboard_gradio.py

Professional dashboard for KAHM embedding evaluation and interactive retrieval.

What it does
------------
1) Reads the publication-ready markdown report produced by evaluate_three_embeddings.py
   and visualizes key results via tables and charts.
2) Loads the 200 human-labeled test queries and supports:
     - selecting a test query and retrieving Top-K sentences using:
         (a) KAHM(query→MB corpus)  : KAHM-regressed query embedding searched on MB corpus index
         (b) Full-KAHM              : KAHM-regressed query embedding searched on KAHM-transformed corpus index
     - entering an arbitrary Austrian-law-related query and retrieving Top-K results.
   The dashboard emphasizes *sentence-level evidence* (possibly spanning multiple law topics),
   not single-label classification.
3) Provides a short "Science behind KAHM" explanation with citations (see References tab).

How to run
----------
pip install gradio faiss-cpu pandas numpy plotly joblib pyarrow
python kahm_dashboard_gradio.py

If you run from your project root, default relative paths should work.

Notes
-----
- The dashboard intentionally reuses the same loading + KAHM regression helpers from
  evaluate_three_embeddings.py to ensure consistency.
- If you changed filenames/paths, set them in the sidebar (left).
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import gradio as gr
import plotly.graph_objects as go

# Optional dependency: faiss
try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover
    faiss = None  # type: ignore


# ----------------------------- Project imports -----------------------------
# Ensure we can import evaluate_three_embeddings.py when running from other working dirs.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

try:
    import evaluate_three_embeddings as eval3  # noqa: E402
except Exception as e:  # pragma: no cover
    eval3 = None  # type: ignore
    _EVAL3_IMPORT_ERROR = e
else:
    _EVAL3_IMPORT_ERROR = None


# ----------------------------- Parsing helpers -----------------------------
_NUM_CI_RE = re.compile(
    r"^\s*([+-]?\d+(?:\.\d+)?)\s*\(\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\)\s*$"
)


@dataclass(frozen=True)
class MeanCI:
    mean: float
    lo: float
    hi: float

    @staticmethod
    def parse(cell: str) -> Optional["MeanCI"]:
        m = _NUM_CI_RE.match(str(cell))
        if not m:
            return None
        return MeanCI(float(m.group(1)), float(m.group(2)), float(m.group(3)))


def _extract_md_table(md: str, header_prefix: str) -> Optional[List[str]]:
    """
    Extract a markdown table (as list of lines) whose header line begins with `header_prefix`.
    Returns None if not found.
    """
    lines = md.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith(header_prefix):
            start = i
            break
    if start is None:
        return None

    # Collect consecutive table lines.
    out: List[str] = []
    for j in range(start, len(lines)):
        line = lines[j]
        if not line.strip().startswith("|"):
            break
        out.append(line.rstrip())
    return out if len(out) >= 3 else None


def _md_table_to_df(table_lines: Sequence[str]) -> pd.DataFrame:
    """
    Convert markdown table lines into a DataFrame (string cells preserved).
    """
    # Header + rows; skip the separator row (| --- | --- |)
    header = [c.strip() for c in table_lines[0].strip().strip("|").split("|")]
    rows: List[List[str]] = []
    for line in table_lines[2:]:
        parts = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(parts) != len(header):
            # tolerate ragged rows by padding/truncation
            if len(parts) < len(header):
                parts = parts + [""] * (len(header) - len(parts))
            else:
                parts = parts[: len(header)]
        rows.append(parts)
    return pd.DataFrame(rows, columns=header)


def parse_report_tables(report_path: str) -> Dict[str, pd.DataFrame]:
    """
    Parse key tables from the markdown report produced by evaluate_three_embeddings.py.
    Returns a dict of DataFrames keyed by table id.
    """
    p = Path(report_path)
    if not p.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")
    md = p.read_text(encoding="utf-8", errors="replace")

    tables: Dict[str, pd.DataFrame] = {}

    # Table 1 (main metrics)
    t1 = _extract_md_table(md, "| Method | Hit@")
    if t1:
        tables["table1"] = _md_table_to_df(t1)

    # Table 2 (KAHM vs IDF)
    t2 = _extract_md_table(md, "| Metric | Δ (KAHM")
    # There are two delta tables: vs IDF and vs Mixedbread. We'll capture both by scanning.
    # We'll pick by title proximity if possible; otherwise parse sequentially.
    if t2:
        tables["table2"] = _md_table_to_df(t2)

    # Table 3 (KAHM vs Mixedbread) - find the *next* delta table after Table 2.
    if t2:
        # locate index in md and search after
        anchor = "\n".join(t2)
        pos = md.find(anchor)
        if pos >= 0:
            md_after = md[pos + len(anchor) :]
            t3 = _extract_md_table(md_after, "| Metric | Δ (KAHM")
            if t3:
                tables["table3"] = _md_table_to_df(t3)

    # Table 5 (routing sweep)
    t5 = _extract_md_table(md, "| τ | Coverage (KAHM)")
    if t5:
        tables["table5"] = _md_table_to_df(t5)

    # Table 9 (alignment)
    t9 = _extract_md_table(md, "| Quantity | Estimate")
    if t9:
        tables["table9"] = _md_table_to_df(t9)

    return tables


def _metric_table_long(df_table1: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    """
    Convert Table 1 into long format with mean/ci for plotting.
    metrics: column names in df_table1
    """
    out_rows: List[Dict[str, Any]] = []
    for _, row in df_table1.iterrows():
        method = str(row.get("Method", "")).strip()
        for m in metrics:
            cell = str(row.get(m, "")).strip()
            parsed = MeanCI.parse(cell)
            if parsed is None:
                # fallback: try to parse the first float
                try:
                    mean = float(re.findall(r"[+-]?\d+(?:\.\d+)?", cell)[0])
                except Exception:
                    continue
                parsed = MeanCI(mean=mean, lo=np.nan, hi=np.nan)
            out_rows.append(
                {"Method": method, "Metric": m, "Mean": parsed.mean, "CI_lo": parsed.lo, "CI_hi": parsed.hi}
            )
    return pd.DataFrame(out_rows)


def fig_main_metrics(df_table1: pd.DataFrame) -> go.Figure:
    metrics = [
        "Hit@10",
        "MRR@10 (unique laws)",
        "Top-1 accuracy",
        "Majority-vote accuracy (predominance ≥ 0.50)",
    ]
    long = _metric_table_long(df_table1, metrics)
    fig = go.Figure()
    for method in long["Method"].unique():
        sub = long[long["Method"] == method]
        fig.add_trace(
            go.Bar(
                name=method,
                x=sub["Metric"],
                y=sub["Mean"],
                error_y=dict(
                    type="data",
                    array=(sub["CI_hi"] - sub["Mean"]).to_numpy(),
                    arrayminus=(sub["Mean"] - sub["CI_lo"]).to_numpy(),
                    visible=True,
                ),
            )
        )
    fig.update_layout(
        barmode="group",
        height=420,
        margin=dict(l=30, r=30, t=50, b=30),
        title="Retrieval quality metrics (mean ± 95% bootstrap CI)",
        yaxis_title="Score",
        legend_title="Method",
    )
    return fig


def _bar_metric(df_table1: pd.DataFrame, metric: str, *, title: str, yaxis_title: str) -> go.Figure:
    long = _metric_table_long(df_table1, [metric])
    fig = go.Figure()
    for method in long["Method"].unique():
        sub = long[long["Method"] == method]
        fig.add_trace(
            go.Bar(
                name=method,
                x=[metric],
                y=sub["Mean"].to_numpy(),
                error_y=dict(
                    type="data",
                    array=(sub["CI_hi"] - sub["Mean"]).to_numpy(),
                    arrayminus=(sub["Mean"] - sub["CI_lo"]).to_numpy(),
                    visible=True,
                ),
            )
        )
    fig.update_layout(
        barmode="group",
        height=380,
        margin=dict(l=30, r=30, t=50, b=30),
        title=title,
        yaxis_title=yaxis_title,
        legend_title="Method",
    )
    return fig


def fig_mean_consensus_fraction(df_table1: pd.DataFrame) -> go.Figure:
    return _bar_metric(
        df_table1,
        "Mean consensus fraction",
        title="Mean consensus fraction (mean ± 95% bootstrap CI)",
        yaxis_title="Fraction",
    )


def fig_mean_lift(df_table1: pd.DataFrame) -> go.Figure:
    return _bar_metric(
        df_table1,
        "Mean lift vs prior",
        title="Mean lift vs prior (mean ± 95% bootstrap CI)",
        yaxis_title="Lift",
    )


def fig_routing(df_table5: pd.DataFrame) -> go.Figure:
    def _parse_col(col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # values like "0.780 (0.720, 0.835)"
        parsed = [MeanCI.parse(v) for v in df_table5[col].tolist()]
        means = np.array([p.mean if p else np.nan for p in parsed], dtype=float)
        lo = np.array([p.lo if p else np.nan for p in parsed], dtype=float)
        hi = np.array([p.hi if p else np.nan for p in parsed], dtype=float)
        return means, lo, hi

    tau = df_table5["τ"].astype(float).to_numpy()
    cov_k, cov_k_lo, cov_k_hi = _parse_col("Coverage (KAHM)")
    acc_k, acc_k_lo, acc_k_hi = _parse_col("Accuracy among covered (KAHM)")
    cov_m, cov_m_lo, cov_m_hi = _parse_col("Coverage (Mixedbread)")
    acc_m, acc_m_lo, acc_m_hi = _parse_col("Accuracy among covered (Mixedbread)")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tau, y=acc_m, mode="lines+markers", name="Mixedbread: acc|covered"))
    fig.add_trace(go.Scatter(x=tau, y=acc_k, mode="lines+markers", name="KAHM: acc|covered"))
    fig.add_trace(go.Scatter(x=tau, y=cov_m, mode="lines+markers", name="Mixedbread: coverage", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=tau, y=cov_k, mode="lines+markers", name="KAHM: coverage", line=dict(dash="dash")))

    fig.update_layout(
        height=420,
        margin=dict(l=30, r=30, t=50, b=30),
        title="Vote-based routing tradeoff (coverage vs accuracy among covered)",
        xaxis_title="Predominance threshold τ",
        yaxis_title="Probability",
        legend_title="Curve",
    )
    return fig


# ----------------------------- Retrieval engine -----------------------------
@dataclass
class CorpusIndex:
    name: str
    sentence_ids: np.ndarray  # shape (n,)
    embeddings: np.ndarray  # shape (n, d)
    faiss_index: Any  # faiss.Index


def _pick_first_existing(cols: Sequence[str], available: Sequence[str]) -> Optional[str]:
    aset = set(available)
    for c in cols:
        if c in aset:
            return c
    return None


@lru_cache(maxsize=4)
def _load_dataframe(corpus_parquet: str) -> pd.DataFrame:
    if eval3 is None:
        raise ImportError(f"Could not import evaluate_three_embeddings.py: {_EVAL3_IMPORT_ERROR}")
    df = eval3.load_corpus_parquet(corpus_parquet)
    # ensure sentence_id exists and is int64
    if "sentence_id" not in df.columns:
        raise KeyError("Corpus parquet must contain column 'sentence_id'")
    df["sentence_id"] = df["sentence_id"].astype(np.int64)
    return df


@lru_cache(maxsize=4)
def _load_bundle(npz_path: str) -> Dict[str, np.ndarray]:
    if eval3 is None:
        raise ImportError(f"Could not import evaluate_three_embeddings.py: {_EVAL3_IMPORT_ERROR}")
    return eval3.load_npz_bundle(npz_path)


def _subset_to_metadata(bundle: Dict[str, np.ndarray], df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    ids = bundle["sentence_ids"].astype(np.int64)
    emb = bundle["emb"].astype(np.float32)
    meta_ids = df["sentence_id"].astype(np.int64).to_numpy()
    meta_set = set(meta_ids.tolist())
    keep = np.array([int(s) in meta_set for s in ids.tolist()], dtype=bool)
    ids2 = ids[keep]
    emb2 = emb[keep]
    return ids2, emb2


def _build_faiss_index(embeddings: np.ndarray) -> Any:
    if faiss is None:
        raise ImportError("faiss is not installed. Install faiss-cpu (or faiss-gpu).")
    d = int(embeddings.shape[1])
    index: Any = faiss.IndexFlatIP(d)
    # Ensure contiguous float32
    X = np.ascontiguousarray(embeddings.astype(np.float32))
    index.add(X)
    return index


@lru_cache(maxsize=2)
def _load_idf_svd_model(path: str):
    if eval3 is None:
        raise ImportError(f"Could not import evaluate_three_embeddings.py: {_EVAL3_IMPORT_ERROR}")
    return eval3.load_idf_svd_model(path)


@lru_cache(maxsize=2)
def _load_kahm_model(path: str) -> dict:
    if eval3 is None:
        raise ImportError(f"Could not import evaluate_three_embeddings.py: {_EVAL3_IMPORT_ERROR}")
    return eval3.load_kahm_model(path)


def _embed_query_kahm(
    query: str,
    *,
    idf_svd_model_path: str,
    kahm_model_path: str,
    kahm_mode: str,
    kahm_batch: int,
) -> np.ndarray:
    if eval3 is None:
        raise ImportError(f"Could not import evaluate_three_embeddings.py: {_EVAL3_IMPORT_ERROR}")
    pipe = _load_idf_svd_model(idf_svd_model_path)
    X = eval3.embed_queries_idf_svd(pipe, [query])  # shape (1, d)
    model = _load_kahm_model(kahm_model_path)
    Y = eval3.kahm_regress_once(model, X, mode=kahm_mode, batch_size=kahm_batch, show_progress=False)  # (1, 1024)
    return Y[0]


def _search_topk(corpus: CorpusIndex, q_emb: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    q = np.ascontiguousarray(q_emb.astype(np.float32).reshape(1, -1))
    D, I = corpus.faiss_index.search(q, int(top_k))
    scores = D[0]
    idxs = I[0]
    sids = corpus.sentence_ids[idxs]
    return scores, sids


def _format_hits(
    df_meta: pd.DataFrame,
    scores: np.ndarray,
    sentence_ids: np.ndarray,
    *,
    top_k: int,
) -> Tuple[pd.DataFrame, go.Figure]:
    # Prepare metadata mapping
    meta = df_meta.set_index("sentence_id", drop=False)

    available_cols = list(df_meta.columns)
    text_col = _pick_first_existing(
        ["sentence", "sentence_text", "text", "content", "passage", "segment"],
        available_cols,
    )
    page_col = _pick_first_existing(
        ["page_no", "page", "pageno", "page_number", "pdf_page"],
        available_cols,
    )
    law_col = _pick_first_existing(["law_type", "law", "law_id", "law_name"], available_cols) or "law_type"

    rows: List[Dict[str, Any]] = []
    law_types: List[str] = []
    for r, (sid, sc) in enumerate(zip(sentence_ids.tolist(), scores.tolist()), start=1):
        rec: Dict[str, Any] = {"rank": r, "score": float(round(sc, 4))}
        if int(sid) in meta.index:
            row = meta.loc[int(sid)]
            rec["law_type"] = str(row.get(law_col, ""))
            if page_col:
                rec["page_no"] = row.get(page_col, "")
            else:
                rec["page_no"] = ""
            if text_col:
                rec["sentence"] = str(row.get(text_col, ""))
            else:
                rec["sentence"] = ""
        else:
            rec["law_type"] = ""
            rec["page_no"] = ""
            rec["sentence"] = ""
        rows.append(rec)
        law_types.append(str(rec["law_type"]))

    out_df = pd.DataFrame(rows)
    # Law-type distribution
    ser = pd.Series(law_types).replace("", np.nan).dropna()
    counts = ser.value_counts().head(12)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=counts.index.tolist(), y=counts.values.tolist(), name="count"))
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=30, b=20),
        title="Top law-types in retrieved Top-K (informative only)",
        xaxis_title="law_type",
        yaxis_title="count",
    )
    return out_df, fig


# ----------------------------- UI callables -----------------------------
def ui_load_report(report_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, go.Figure, go.Figure, go.Figure, go.Figure]:
    tables = parse_report_tables(report_path)
    if "table1" not in tables:
        raise ValueError("Could not parse Table 1 from report. Ensure it contains the Table 1 markdown.")
    df1 = tables["table1"]
    df2 = tables.get("table2", pd.DataFrame())
    df3 = tables.get("table3", pd.DataFrame())
    df5 = tables.get("table5", pd.DataFrame())

    fig1 = fig_main_metrics(df1)
    fig2 = fig_mean_consensus_fraction(df1)
    fig_lift = fig_mean_lift(df1)
    fig3 = fig_routing(df5) if not df5.empty else go.Figure()

    return df1, df2, df3, fig1, fig2, fig_lift, fig3


@lru_cache(maxsize=2)
def _load_corpora(
    base_dir: str,
    corpus_parquet: str,
    mb_npz: str,
    kahm_npz: str,
) -> Tuple[pd.DataFrame, CorpusIndex, CorpusIndex]:
    df = _load_dataframe(str(Path(base_dir) / corpus_parquet))
    mb_bundle = _load_bundle(str(Path(base_dir) / mb_npz))
    k_bundle = _load_bundle(str(Path(base_dir) / kahm_npz))

    mb_ids, mb_emb = _subset_to_metadata(mb_bundle, df)
    k_ids, k_emb = _subset_to_metadata(k_bundle, df)

    # Normalize (safe even if already normalized)
    if eval3 is None:
        raise ImportError(f"Could not import evaluate_three_embeddings.py: {_EVAL3_IMPORT_ERROR}")
    mb_emb = eval3.l2_normalize_rows(mb_emb)
    k_emb = eval3.l2_normalize_rows(k_emb)

    mb_index = _build_faiss_index(mb_emb)
    k_index = _build_faiss_index(k_emb)

    return (
        df,
        CorpusIndex(name="MB corpus", sentence_ids=mb_ids, embeddings=mb_emb, faiss_index=mb_index),
        CorpusIndex(name="KAHM corpus", sentence_ids=k_ids, embeddings=k_emb, faiss_index=k_index),
    )


@lru_cache(maxsize=1)
def _load_test_queries(module_attr: str) -> Tuple[List[str], List[str]]:
    if eval3 is None:
        raise ImportError(f"Could not import evaluate_three_embeddings.py: {_EVAL3_IMPORT_ERROR}")
    qs = eval3.load_query_set(module_attr)
    texts = eval3.extract_query_texts(qs)
    labels = eval3.extract_consensus_laws(qs)
    return texts, labels


def _build_query_choices(texts: List[str], labels: List[str]) -> List[str]:
    out = []
    for i, (t, lab) in enumerate(zip(texts, labels)):
        t0 = (t or "").replace("\n", " ").strip()
        if len(t0) > 110:
            t0 = t0[:110] + "…"
        lab0 = (lab or "").strip()
        suffix = f" [{lab0}]" if lab0 else ""
        out.append(f"{i:03d}: {t0}{suffix}")
    return out


def ui_retrieve_for_selection(
    selection: str,
    *,
    base_dir: str,
    corpus_parquet: str,
    mb_npz: str,
    kahm_npz: str,
    idf_svd_model: str,
    kahm_model: str,
    kahm_mode: str,
    kahm_batch: int,
    top_k: int,
    query_set: str,
) -> Tuple[str, pd.DataFrame, go.Figure, pd.DataFrame, go.Figure]:
    texts, labels = _load_test_queries(query_set)
    # Parse index from selection string
    m = re.match(r"^\s*(\d+)\s*:", selection)
    if not m:
        raise ValueError("Could not parse selected query index.")
    idx = int(m.group(1))
    if idx < 0 or idx >= len(texts):
        raise IndexError("Selected query index out of range.")
    query_text = texts[idx]
    label = labels[idx] if idx < len(labels) else ""
    header = f"Selected query #{idx:03d}  |  Gold (human) label: {label}" if label else f"Selected query #{idx:03d}"

    df_meta, mb_corpus, k_corpus = _load_corpora(base_dir, corpus_parquet, mb_npz, kahm_npz)
    q_emb = _embed_query_kahm(
        query_text,
        idf_svd_model_path=str(Path(base_dir) / idf_svd_model),
        kahm_model_path=str(Path(base_dir) / kahm_model),
        kahm_mode=kahm_mode,
        kahm_batch=int(kahm_batch),
    )

    # KAHM(q->MB)
    s1, ids1 = _search_topk(mb_corpus, q_emb, int(top_k))
    out1, fig1 = _format_hits(df_meta, s1, ids1, top_k=int(top_k))

    # Full-KAHM (q->KAHM corpus)
    s2, ids2 = _search_topk(k_corpus, q_emb, int(top_k))
    out2, fig2 = _format_hits(df_meta, s2, ids2, top_k=int(top_k))

    return header, out1, fig1, out2, fig2


def ui_retrieve_custom(
    query_text: str,
    *,
    base_dir: str,
    corpus_parquet: str,
    mb_npz: str,
    kahm_npz: str,
    idf_svd_model: str,
    kahm_model: str,
    kahm_mode: str,
    kahm_batch: int,
    top_k: int,
) -> Tuple[pd.DataFrame, go.Figure, pd.DataFrame, go.Figure]:
    if not query_text or not query_text.strip():
        raise ValueError("Please enter a query.")
    df_meta, mb_corpus, k_corpus = _load_corpora(base_dir, corpus_parquet, mb_npz, kahm_npz)
    q_emb = _embed_query_kahm(
        query_text.strip(),
        idf_svd_model_path=str(Path(base_dir) / idf_svd_model),
        kahm_model_path=str(Path(base_dir) / kahm_model),
        kahm_mode=kahm_mode,
        kahm_batch=int(kahm_batch),
    )
    s1, ids1 = _search_topk(mb_corpus, q_emb, int(top_k))
    out1, fig1 = _format_hits(df_meta, s1, ids1, top_k=int(top_k))
    s2, ids2 = _search_topk(k_corpus, q_emb, int(top_k))
    out2, fig2 = _format_hits(df_meta, s2, ids2, top_k=int(top_k))
    return out1, fig1, out2, fig2


# ----------------------------- KAHM science narrative (dashboard tab) -----------------------------
SCIENCE_MD = r"""
# Science behind KAHM (high level)

KAHM (Kernel Affine Hull Machine) is a *geometrically inspired* kernel method that represents a class (or task) by the **affine hull** of its examples in a reproducing kernel Hilbert space (RKHS). This yields kernelized distance measures to these hulls and supports learning and inference using *closed-form, convex* computations rather than multi-epoch stochastic gradient descent.

In the cited line of work, KAHM-style models are used as **gradient-free alternatives** (or complements) to deep models in settings such as collaborative / federated learning and learning under constraints (e.g., differential privacy).

## Why KAHM is suitable for "embedding approximation"

In this project, KAHM is used as a lightweight mapping that learns to approximate a target embedding space (Mixedbread `deepset-mxbai-embed-de-large-v1`) *for a specific domain* (Austrian laws):

- **Input features:** a strong, low-cost IDF–SVD representation.
- **Target space:** the 1024-D Mixedbread embedding space.
- **Learned mapping:** a KAHM regressor (kernel machine) trained to map IDF–SVD vectors into the Mixedbread space.

Because KAHM training/inference is kernel-based and does not require backpropagation through a transformer, it can be trained and executed on standard CPU-class hardware (e.g., Apple M1 with 8 GB RAM), subject to corpus size and your chosen training setup.

## References (as requested)

a) JAIR article view: https://jair.org/index.php/jair/article/view/16821  
b) JAIR article view: https://jair.org/index.php/jair/article/view/15071  
c) arXiv: https://arxiv.org/abs/2512.01025

(If your environment cannot open the JAIR view pages programmatically, use the PDF links from your browser; the dashboard itself does not depend on programmatic access to those pages.)
"""


# ----------------------------- Build UI -----------------------------
CSS = r""".gradio-container {font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";}
#title {margin-bottom: 0.2rem;}
.small-note {color: #666; font-size: 0.9rem;}
/* Layout tuning */
.config-col {max-width: 360px;}
.results-col {min-width: 720px;}
/* Make retrieval tables feel less cramped */
.kahm-table .gr-dataframe {min-height: 260px;}
/* Widen retrieval table key columns */
.retrieval-table table th:nth-child(1),
.retrieval-table table td:nth-child(1) {min-width: 56px; width: 56px;}
.retrieval-table table th:nth-child(2),
.retrieval-table table td:nth-child(2) {min-width: 96px; width: 96px;}
/* Give the sentence column more room */
.retrieval-table table th:nth-child(5),
.retrieval-table table td:nth-child(5) {min-width: 560px;}
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="KAHM Embeddings Dashboard") as demo:
        gr.Markdown("# KAHM embeddings dashboard", elem_id="title")
        gr.Markdown(
            "Dashboard for (1) visualizing the evaluation report and (2) interactively retrieving evidence sentences "
            "with **KAHM(query→MB corpus)** and **Full-KAHM**.",
            elem_classes=["small-note"],
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=320, elem_classes=['config-col']):
                gr.Markdown("## Configuration")
                with gr.Accordion('Configuration settings', open=True):
                    base_dir = gr.Textbox(value=".", label="Project/data directory (base_dir)")
                    report_path = gr.Textbox(value="kahm_evaluation_report.md", label="Report markdown path (relative to base_dir allowed)")
                    corpus_parquet = gr.Textbox(value="ris_sentences.parquet", label="Corpus parquet (sentence metadata)")
                    mb_npz = gr.Textbox(value="embedding_index.npz", label="Mixedbread corpus embeddings (.npz)")
                    kahm_npz = gr.Textbox(value="embedding_index_kahm_mixedbread_approx.npz", label="Full-KAHM corpus embeddings (.npz)")
                    idf_svd_model = gr.Textbox(value="idf_svd_model.joblib", label="IDF–SVD model (joblib)")
                    kahm_model = gr.Textbox(value="kahm_regressor_idf_to_mixedbread.joblib", label="KAHM regressor (joblib)")
                    kahm_mode = gr.Dropdown(choices=["soft", "hard"], value="soft", label="KAHM mode")
                    kahm_batch = gr.Number(value=512, label="KAHM batch size", precision=0)
                    top_k = gr.Slider(minimum=1, maximum=50, step=1, value=10, label="Top-K for retrieval")
                    query_set = gr.Textbox(value="query_set.TEST_QUERY_SET", label="Query set (module.attr)")
                    btn_load_report = gr.Button("Load/refresh report visuals", variant="primary")

            with gr.Column(scale=4, min_width=780, elem_classes=['results-col']):
                with gr.Tabs():
                    with gr.Tab("Results (from report)"):
                        gr.Markdown("## Report highlights")
                        table1 = gr.Dataframe(label="Table 1: Main retrieval metrics", interactive=False, wrap=True)
                        with gr.Row():
                            plot_main = gr.Plot(label="Main metrics")
                        with gr.Row():
                            plot_cons = gr.Plot(label="Mean consensus fraction")
                            plot_lift = gr.Plot(label="Mean lift vs prior")
                        plot_route = gr.Plot(label="Routing tradeoffs")
                        with gr.Accordion("Delta tables (from report)", open=False):
                            table2 = gr.Dataframe(label="Table 2: Paired deltas (KAHM − IDF)", interactive=False, wrap=True)
                            table3 = gr.Dataframe(label="Table 3: Paired deltas (KAHM − Mixedbread)", interactive=False, wrap=True)

                    with gr.Tab("Explore retrieval"):
                        gr.Markdown(
                            "## Interactive retrieval\n"
                            "This view retrieves **Top-K sentences** for a selected test query or an arbitrary query.\n\n"
                            "**Interpretation note:** the goal is to surface *relevant evidence sentences*, which may span multiple law topics; "
                            "this is not a single-label classification task."
                        )

                        with gr.Accordion("Select from the 200 test queries", open=True):
                            # We'll populate choices on load
                            test_query_dropdown = gr.Dropdown(choices=[], value=None, label="Pick a test query")
                            btn_retrieve_sel = gr.Button("Retrieve for selected test query", variant="secondary")
                            selected_header = gr.Markdown()

                            gr.Markdown("### KAHM(query→MB corpus)")
                            out_kahm_mb = gr.Dataframe(interactive=False, wrap=True, elem_classes=['kahm-table','retrieval-table'])
                            plot_kahm_mb = gr.Plot()

                            gr.Markdown("### Full-KAHM (query→KAHM corpus)")
                            out_full = gr.Dataframe(interactive=False, wrap=True, elem_classes=['kahm-table','retrieval-table'])
                            plot_full = gr.Plot()

                        with gr.Accordion("Search with a custom query", open=False):
                            custom_query = gr.Textbox(lines=2, label="Custom query (German or English)")
                            btn_retrieve_custom = gr.Button("Search", variant="primary")
                            gr.Markdown("### KAHM(query→MB corpus)")
                            out_kahm_mb2 = gr.Dataframe(interactive=False, wrap=True, elem_classes=['kahm-table','retrieval-table'])
                            plot_kahm_mb2 = gr.Plot()

                            gr.Markdown("### Full-KAHM (query→KAHM corpus)")
                            out_full2 = gr.Dataframe(interactive=False, wrap=True, elem_classes=['kahm-table','retrieval-table'])
                            plot_full2 = gr.Plot()

                    with gr.Tab("KAHM science"):
                        gr.Markdown(SCIENCE_MD)

        # ---------- Wiring ----------
        def _resolve_path(base: str, p: str) -> str:
            pth = Path(p)
            if pth.is_absolute():
                return str(pth)
            return str(Path(base) / p)

        def _load_report_wrapper(
            base: str,
            rp: str,
        ):
            rp2 = _resolve_path(base, rp)
            return ui_load_report(rp2)

        btn_load_report.click(
            fn=_load_report_wrapper,
            inputs=[base_dir, report_path],
            outputs=[table1, table2, table3, plot_main, plot_cons, plot_lift, plot_route],
        )

        # Populate test query dropdown on load
        def _init_queries(qset: str):
            texts, labels = _load_test_queries(qset)
            choices = _build_query_choices(texts, labels)
            default = choices[0] if choices else None
            return gr.Dropdown(choices=choices, value=default)

        demo.load(fn=_init_queries, inputs=[query_set], outputs=[test_query_dropdown])

        # Retrieve for selected test query
        def _retrieve_sel(
            sel: str,
            base: str,
            corpus_pq: str,
            mb: str,
            kahm_npz_p: str,
            idf_model: str,
            kahm_model_p: str,
            mode: str,
            batch: float,
            k: int,
            qset: str,
        ):
            return ui_retrieve_for_selection(
                sel,
                base_dir=base,
                corpus_parquet=corpus_pq,
                mb_npz=mb,
                kahm_npz=kahm_npz_p,
                idf_svd_model=idf_model,
                kahm_model=kahm_model_p,
                kahm_mode=mode,
                kahm_batch=int(batch),
                top_k=int(k),
                query_set=qset,
            )

        btn_retrieve_sel.click(
            fn=_retrieve_sel,
            inputs=[
                test_query_dropdown,
                base_dir,
                corpus_parquet,
                mb_npz,
                kahm_npz,
                idf_svd_model,
                kahm_model,
                kahm_mode,
                kahm_batch,
                top_k,
                query_set,
            ],
            outputs=[selected_header, out_kahm_mb, plot_kahm_mb, out_full, plot_full],
        )

        # Retrieve for custom query
        def _retrieve_custom(
            q: str,
            base: str,
            corpus_pq: str,
            mb: str,
            kahm_npz_p: str,
            idf_model: str,
            kahm_model_p: str,
            mode: str,
            batch: float,
            k: int,
        ):
            return ui_retrieve_custom(
                q,
                base_dir=base,
                corpus_parquet=corpus_pq,
                mb_npz=mb,
                kahm_npz=kahm_npz_p,
                idf_svd_model=idf_model,
                kahm_model=kahm_model_p,
                kahm_mode=mode,
                kahm_batch=int(batch),
                top_k=int(k),
            )

        btn_retrieve_custom.click(
            fn=_retrieve_custom,
            inputs=[
                custom_query,
                base_dir,
                corpus_parquet,
                mb_npz,
                kahm_npz,
                idf_svd_model,
                kahm_model,
                kahm_mode,
                kahm_batch,
                top_k,
            ],
            outputs=[out_kahm_mb2, plot_kahm_mb2, out_full2, plot_full2],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    # By default, gradio listens on localhost:7860
    try:
        app.launch(css=CSS)
    except TypeError:
        app.launch()