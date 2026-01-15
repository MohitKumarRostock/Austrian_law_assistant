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
3) Provides a short "Science behind KAHM" explanation with citations (see KAHM science tab).
4) Presents a management-ready funding pitch (see Funding pitch tab).

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




# ----------------------------- Private document (BYO) helpers -----------------------------
# The dashboard can optionally accept a user-uploaded document (txt/md/docx/pdf), extract sentences,
# and use a selected sentence as a query for the same retrieval methods.
try:  # python-docx
    import docx  # type: ignore
except Exception:  # pragma: no cover
    docx = None  # type: ignore


def _read_text_from_path(path: str) -> str:
    """
    Read text from a local file path (txt/md/docx/pdf).
    Returns extracted text as a single string.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    ext = p.suffix.lower()

    if ext in {".txt", ".md", ".csv", ".log"}:
        data = p.read_bytes()
        for enc in ("utf-8", "utf-16", "cp1252", "latin-1"):
            try:
                return data.decode(enc)
            except Exception:
                continue
        # last resort
        return data.decode("utf-8", errors="replace")

    if ext == ".docx":
        if docx is None:
            raise ImportError("python-docx is not installed; cannot read .docx files.")
        d = docx.Document(str(p))
        parts = []
        for para in d.paragraphs:
            t = (para.text or "").strip()
            if t:
                parts.append(t)
        # also include table text (common in policies/contracts)
        for tbl in d.tables:
            for row in tbl.rows:
                for cell in row.cells:
                    t = (cell.text or "").strip()
                    if t:
                        parts.append(t)
        return "\n".join(parts)

    if ext == ".pdf":
        # Prefer PyMuPDF (fitz); fallback to pypdf if needed.
        try:
            import fitz  # type: ignore
            doc = fitz.open(str(p))
            pages = []
            for i in range(len(doc)):
                pages.append(doc.load_page(i).get_text("text"))
            doc.close()
            return "\n".join(pages)
        except Exception:
            try:
                from pypdf import PdfReader  # type: ignore
                reader = PdfReader(str(p))
                pages = []
                for page in reader.pages:
                    pages.append(page.extract_text() or "")
                return "\n".join(pages)
            except Exception as e:
                raise ImportError(f"Could not read PDF. Install PyMuPDF (fitz) or pypdf. Details: {e}")

    raise ValueError(f"Unsupported document type: {ext}. Please upload a .txt, .md, .docx, or .pdf file.")


_ABBREV_RE = re.compile(
    r"(?:\b(?:z\.B|u\.a|u\.U|bzw|vgl|ggf|usw|etc|e\.g|i\.e|Dr|Prof|Abs|Art|Nr|No|Stk|Zl)\.)$",
    re.IGNORECASE,
)


def _split_sentences(text: str, *, max_sentences: int = 5000) -> List[str]:
    """
    Heuristic sentence splitter (German/English friendly).
    Returns a list of sentences; caps output to max_sentences for UI stability.
    """
    if not text:
        return []
    # normalize whitespace
    t = re.sub(r"\s+", " ", text.replace("\u00ad", " ")).strip()
    if not t:
        return []

    # Candidate split on punctuation followed by whitespace + a likely sentence start.
    parts = re.split(r'(?<=[\.\!\?])["\')\]]*\s+(?=(?:[A-ZÄÖÜ0-9„"\'\(\[]))', t)

    # Post-process: merge obvious false splits (abbreviations, tiny fragments)
    merged: List[str] = []
    for seg in parts:
        seg = seg.strip()
        if not seg:
            continue
        if merged:
            prev = merged[-1]
            if _ABBREV_RE.search(prev) or len(prev) < 25:
                merged[-1] = (prev + " " + seg).strip()
                continue
        merged.append(seg)

    # Fallback: if we failed to split, try line-ish splitting
    if len(merged) <= 1:
        merged = [s.strip() for s in re.split(r"(?:\s*[\n\r]\s*|;\s+)", text) if s.strip()]

    # Cap for UI
    if len(merged) > int(max_sentences):
        merged = merged[: int(max_sentences)]
    return merged


def _build_sentence_choices(sentences: List[str]) -> List[str]:
    out: List[str] = []
    for i, s in enumerate(sentences):
        s0 = (s or "").replace("\n", " ").strip()
        if len(s0) > 120:
            s0 = s0[:120] + "…"
        out.append(f"{i:04d}: {s0}")
    return out


def _empty_retrieval_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["rank", "score", "law_type", "page_no", "sentence"])


def ui_parse_private_document(
    file_path: Optional[str],
) -> Tuple[Dict[str, Any], Any, str, str, str, pd.DataFrame, go.Figure, pd.DataFrame, go.Figure]:
    """
    Parse an uploaded private document, extract sentences, and prepare UI elements.
    Returns:
      - state dict (filename, sentences)
      - dropdown update (choices + default)
      - info markdown
      - initial selected sentence preview
      - cleared header markdown
      - empty result table + plot (KAHM->MB)
      - empty result table + plot (Full-KAHM)
    """
    empty_df = _empty_retrieval_df()
    empty_fig = go.Figure()

    if not file_path:
        info = "Upload a document to begin. Supported: **.txt, .md, .docx, .pdf**."
        return {"filename": "", "sentences": []}, gr.update(choices=[], value=None), info, "", "", empty_df, empty_fig, empty_df, empty_fig

    try:
        text = _read_text_from_path(str(file_path))
        sentences = _split_sentences(text, max_sentences=5000)
        fname = Path(str(file_path)).name
        if not sentences:
            info = f"Loaded **{fname}**, but could not extract any sentences. Try a different file format or a text-based PDF."
            return {"filename": fname, "sentences": []}, gr.update(choices=[], value=None), info, "", "", empty_df, empty_fig, empty_df, empty_fig

        choices = _build_sentence_choices(sentences)
        default = choices[0] if choices else None
        preview = sentences[0] if sentences else ""
        info = (
            f"**Loaded:** {fname}\n\n"
            f"- Extracted characters: {len(text):,}\n"
            f"- Extracted sentences: {len(sentences):,} (UI cap: 5,000)\n\n"
            "Select a sentence below and run retrieval."
        )
        state = {"filename": fname, "sentences": sentences}
        return state, gr.update(choices=choices, value=default), info, preview, "", empty_df, empty_fig, empty_df, empty_fig
    except Exception as e:
        info = f"Could not read the uploaded document. Details: {type(e).__name__}: {e}"
        return {"filename": "", "sentences": []}, gr.update(choices=[], value=None), info, "", "", empty_df, empty_fig, empty_df, empty_fig


def ui_private_sentence_preview(selection: str, state: Dict[str, Any]) -> str:
    if not selection or not state or not state.get("sentences"):
        return ""
    m = re.match(r"^\s*(\d+)\s*:", str(selection))
    if not m:
        return ""
    idx = int(m.group(1))
    sents = state.get("sentences", [])
    if idx < 0 or idx >= len(sents):
        return ""
    return str(sents[idx])


def ui_retrieve_private_sentence(
    selection: str,
    state: Dict[str, Any],
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
) -> Tuple[str, pd.DataFrame, go.Figure, pd.DataFrame, go.Figure]:
    if not state or not state.get("sentences"):
        raise ValueError("Please upload a document and select a sentence first.")
    m = re.match(r"^\s*(\d+)\s*:", str(selection))
    if not m:
        raise ValueError("Please select a sentence from the dropdown.")
    idx = int(m.group(1))
    sents: List[str] = state.get("sentences", [])
    if idx < 0 or idx >= len(sents):
        raise IndexError("Selected sentence index out of range.")
    query_text = str(sents[idx]).strip()
    if not query_text:
        raise ValueError("Selected sentence is empty.")

    header = (
        f"**Document:** {state.get('filename', '')}  \n"
        f"**Selected sentence #{idx:04d}:** {query_text}"
    )

    out1, fig1, out2, fig2 = ui_retrieve_custom(
        query_text,
        base_dir=base_dir,
        corpus_parquet=corpus_parquet,
        mb_npz=mb_npz,
        kahm_npz=kahm_npz,
        idf_svd_model=idf_svd_model,
        kahm_model=kahm_model,
        kahm_mode=kahm_mode,
        kahm_batch=int(kahm_batch),
        top_k=int(top_k),
    )
    return header, out1, fig1, out2, fig2



# ----------------------------- KAHM science narrative (dashboard tab) -----------------------------

SCIENCE_MD = r"""
# Embeddings and methods used in this dashboard

This dashboard compares and operationalizes three representations of Austrian-law sentences:

1. **MB embeddings (Mixedbread baseline)** — a high-quality *neural* sentence embedding model used as a strong retrieval baseline and as a “teacher space”.
2. **IDF–SVD embeddings** — a lightweight, *non-neural* dense representation derived from TF‑IDF and latent semantic analysis (LSA).
3. **KAHM embeddings (MB-approx)** — a KAHM regressor that maps IDF–SVD vectors into the MB embedding space via *geometric topic modeling*.

The evaluation report focuses on **sentence-level evidence retrieval** (Top‑K), not single-label classification.

---

## 1) MB embeddings (Mixedbread)

**What they are**  
The “MB” index is built with a SentenceTransformer model:

- **Model:** `mixedbread-ai/deepset-mxbai-embed-de-large-v1` (default in `build_embedding_index_npz.py`)
- **Granularity:** sentence-level (each RIS sentence becomes one vector)
- **Role in this project:** *reference embedding space* for retrieval quality; also the **regression target** for KAHM training.

**Why this model / why MB at all**  
For Austrian laws, the corpus is predominantly German and includes legal phrasing, citations, and domain-specific vocabulary. A strong German-capable semantic embedding model is an appropriate *upper baseline* for retrieval quality and a useful “teacher space” when we want to approximate neural embeddings with cheaper features.

**Where it is produced**  
`build_embedding_index_npz.py` builds `embedding_index.npz` from `ris_sentences.parquet` using SentenceTransformer encoding and stores metadata such as `model_name`, `created_at_utc`, and a dataset fingerprint alongside the embedding matrix.

---

## 2) IDF–SVD embeddings (TF‑IDF → SVD / LSA)

**What they are**  
“IDF–SVD” is a classic, compute-efficient embedding pipeline:

- **TF‑IDF features** (inverse document frequency weighted term features)
  - word n‑grams: **(1, 3)**
  - optional character n‑grams (“char_wb”): **(3, 6)**
  - large feature budgets (defaults): **600k word features**, **300k char features**
- **TruncatedSVD** (randomized) to obtain a dense vector (default **dim=1024**)
- optional **L2 normalization** (disabled by default)

This is implemented in `build_embedding_index_idf_svd_npz.py` as a scikit‑learn `Pipeline`:
`PreprocessTransformer → FeatureUnion(TF‑IDF word + TF‑IDF char) → AdaptiveTruncatedSVD → (optional) Normalizer`.

**Why it was considered**  
IDF–SVD is attractive as a “no-neural-training” baseline and as an input representation for KAHM because it is:

- **cheap to compute** (CPU-friendly; no GPU required),
- **deterministic and auditable** (important in legal/regulated settings),
- **strong on lexical signals** (citations, terminology, statute references),
- a solid bridge between purely lexical retrieval and fully neural embeddings.

**Where it is produced**  
`build_embedding_index_idf_svd_npz.py` writes both:
- `embedding_index_idf_svd.npz` (dense vectors keyed by `sentence_id`), and
- `idf_svd_model.joblib` (portable pipeline for generating query vectors the same way).

---

## 3) KAHM embeddings (geometric topic modeling in embedding space)

**What they are**  
“KAHM embeddings” in this repo are **MB-approx embeddings**: vectors predicted by a KAHM regressor that maps
IDF–SVD inputs into the MB target space.

Training is described in `train_kahm_embedding_regressor.py`:

- **Inputs (X):** L2-normalized IDF–SVD vectors
- **Targets (Y):** L2-normalized MB vectors
- **Objective:** approximate Y from X with a KAHM-style, geometry-driven model.

**Geometrical modeling of topics**  
KAHM operationalizes “topics” as **regions in the target embedding space**:

1. **Cluster the target embeddings (Y)** into `n_clusters` (KMeans / MiniBatchKMeans).  
   Each cluster corresponds to a coherent region of semantic space (a “topic region” in practice).
2. **Learn a geometric descriptor per region** by training one autoencoder per cluster on the *input* vectors that map to that cluster.  
   At inference time, each autoencoder yields a **reconstruction-distance** for the query/input vector, i.e., “how well does this input fit region c?”.
3. **Convert distances into routing weights** (hard or soft routing) and **reconstruct an output embedding** as a weighted combination of cluster centers in the MB space.

This yields an MB-like embedding without running a large neural encoder at query time.

**Two retrieval modes in the dashboard**  

- **KAHM(query→MB corpus):** regress the query into MB space and search the **MB corpus index**.  
  This isolates the benefit of the *query-side* KAHM mapping while keeping the corpus in the original MB space.
- **Full‑KAHM (query→KAHM corpus):** regress the query and search a **KAHM-transformed corpus index**.  
  This evaluates the fully “compressed” pipeline where neither queries nor corpus require MB encoding at runtime.

---

## Practical interpretation

- If **MB > KAHM(query→MB corpus)**, the gap is mostly due to *approximation quality of the query mapping*.
- If **KAHM(query→MB corpus) ≈ Full‑KAHM**, then the corpus transformation is not the limiting factor.
- If **IDF–SVD** is competitive on a subset of queries, those are likely **lexically dominated** information needs (citations, exact phrases).

---

## References

- arXiv: 2512.01025 (advantages motivating KAHM-driven language models)
- See the project report tab for empirical retrieval results on Austrian-law queries.
"""



EMBEDDINGS_QUICKREF_MD = r"""
### Embeddings in this dashboard (quick reference)

- **MB (Mixedbread) embeddings:** SentenceTransformer model `mixedbread-ai/deepset-mxbai-embed-de-large-v1` (strong neural baseline; also the **target** space for KAHM).
- **IDF–SVD embeddings:** TF‑IDF (word 1–3 grams + optional char 3–6 grams) → TruncatedSVD/LSA (**dim=1024** by default). Fast, deterministic, audit-friendly baseline and KAHM input.
- **KAHM embeddings:** KAHM regressor that maps IDF–SVD → MB space via **output-space clustering (“topic regions”)** and **distance-based routing** (hard/soft) to reconstruct MB-like vectors.
"""



# ----------------------------- Management pitch (dashboard tab) -----------------------------
PITCH_MD = r"""
# Management pitch: funding KAHM-driven language models

## Executive summary
KAHM has already demonstrated strong practical value in our Austrian‑law retrieval prototype. In our latest evaluation (see **Results** tab; 200 human‑labeled test queries, k=10), **KAHM(query→MB corpus)** achieves high Top‑K evidence retrieval quality (Hit@10 and MRR@10 typically in the ~0.89–0.92 and ~0.73–0.75 range across recent runs). This is achieved with a lightweight, mathematically grounded kernel mapping that runs efficiently on CPU‑class hardware.

**Proposal:** fund a focused, time‑boxed R&D program to extend KAHM from *embedding approximation* into **KAHM‑driven language‑model (LM) components**—specifically: gradient‑free heads, routing policies, and decoding‑adjacent modules that reduce reliance on expensive end‑to‑end training and large‑model inference while preserving practical quality.

## Strategic rationale for the company
- **Differentiation and IP:** KAHM is a mathematically principled alternative to gradient‑descent‑heavy learning. A successful KAHM‑driven LM stack would be highly differentiating and likely patentable (routing, gradient‑free heads, and deployment patterns).
- **Compute and cost discipline:** large‑model training and high‑QPS inference are capital intensive. KAHM emphasizes closed‑form / convex computations and domain‑specific mappings that are compatible with constrained compute and energy budgets.
- **Governance‑friendly engineering:** kernel‑based components are comparatively easier to analyze, bound, and audit than end‑to‑end deep models—beneficial for regulated or high‑stakes domains.

## Evidence of traction: Austrian‑law evidence retrieval
The current dashboard and evaluation demonstrate that KAHM can approximate transformer query embeddings at retrieval time while maintaining strong quality:
- **Performance:** see Table 1 in the report (Results tab) for Hit@10, MRR@10, Top‑1 accuracy, and vote‑based routing metrics with bootstrap CIs.
- **Clear lift over the low‑cost baseline:** KAHM consistently outperforms IDF–SVD under paired evaluation.
- **Operational simplicity:** queries are mapped into a fixed corpus embedding space, enabling reuse of an existing index and minimizing production complexity.

## How this extends to LM components (beyond retrieval)
KAHM should not be framed as a standalone autoregressive decoder. Instead, it is a **latent‑space adapter and geometry‑based scoring head** that can become a practical LM component in a modern generation stack:

- **Reranker / head (decoding‑adjacent):** the base LLM proposes a small Top‑M candidate set (tokens, actions, templates). KAHM scores and reranks only those candidates using a lightweight geometry‑based criterion. This improves control and reliability without requiring KAHM to score the full vocabulary.
- **Constrained decoding controller:** when the output set is naturally small (tool/action selection, template choice, controlled language), KAHM can act as the primary decision head—high leverage, low risk, production‑friendly.
- **Hierarchical / token‑cluster decoding (research extension):** KAHM predicts a coarse bucket (semantic token cluster) conditioned on the LLM state; the LLM then generates within that bucket or KAHM reranks within the bucket. This offers a credible path to broader decoding influence while keeping class‑scaling tractable.

This roadmap turns today’s “embedding approximation” capability into reusable LM infrastructure: **routing, control, and reliability components** that sit alongside a frozen or lightly tuned LLM.

## Why it makes sense to invest now
Recent KAHM work (arXiv:2512.01025) supports the broader thesis that KAHM‑style models can underpin **gradient‑free learning systems** with advantages that map directly to company constraints:
- reduced communication via low‑dimensional “space folding” summaries,
- privacy/security‑friendly structure (differential privacy mechanisms and secure inference compatibility),
- operator‑theoretic and complexity‑based analysis that strengthens the theoretical and auditability story.

Our retrieval results provide an internal “green light”: the core adapter mechanism works in a real domain and is already integrated into an interactive dashboard.

## Proposed funded pilot (12–16 weeks) and deliverables
1) **LM component Phase 1 — KAHM reranker/head (Weeks 1–6):**
   - Implement KAHM as a lightweight scoring head over Top‑M candidates produced by a frozen LLM (or over action/template candidates in an agent).
   - Evaluate quality, controllability, and reliability gains at fixed latency/cost (ablation against baseline decoding/prompt routing).

2) **LM component Phase 2 — constrained decoding controller (Weeks 4–10):**
   - Deploy KAHM for small‑alphabet generation tasks (tool routing, template selection, controlled outputs).
   - Validate measurable reductions in expensive model calls and improved “abstain/escalate” behavior.

3) **LM component Phase 3 — hierarchical/token‑cluster decoding (Weeks 8–16):**
   - Research prototype: cluster vocabulary into semantic buckets; use KAHM to select buckets, then decode within‑bucket.
   - Deliver a feasibility assessment with clear go/no‑go criteria (quality impact vs complexity).

4) **IP package and product roadmap (Weeks 10–16):**
   - Identify patentable components and produce a productization roadmap (data, evaluation, governance, deployment assumptions).

## Lean resource request
- **Staffing:** 1 research engineer (FT) + 0.2–0.4 legal/domain SME (PT) + minimal security/compliance review.
- **Compute:** CPU‑first for KAHM; small GPU budget only for baseline LLM comparisons and reporting.

## Decision criteria (what “success” looks like)
- A demonstrable LM‑component prototype (reranker/router/controller) with competitive task performance at materially lower incremental compute/training requirements than gradient‑descent‑heavy alternatives.
- A quantified business case (cost, latency, compliance) plus a clear go/no‑go recommendation for productization.
- Defensible technical assets: evaluation suite, benchmarks, and prototype code that can be leveraged as IP.
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

                    with gr.Accordion("Embeddings & methods (overview)", open=False):
                        gr.Markdown(EMBEDDINGS_QUICKREF_MD)

            with gr.Column(scale=4, min_width=780, elem_classes=['results-col']):
                with gr.Tabs():
                    with gr.Tab("Results (from report)"):
                        gr.Markdown("## Report highlights")
                        gr.Markdown(EMBEDDINGS_QUICKREF_MD, elem_classes=["small-note"])

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
                        gr.Markdown(EMBEDDINGS_QUICKREF_MD, elem_classes=["small-note"])


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


                    with gr.Tab("Private document"):
                        gr.Markdown(
                            "## Private document retrieval\n"
                            "Upload an internal document, select a sentence, and retrieve related Austrian‑law evidence sentences "
                            "using the same methods as the other retrieval views."
                        )
                        gr.Markdown(
                            "**Privacy note:** the uploaded file is processed by this dashboard instance (your local runtime) and is not "
                            "sent to external services by this code.",
                            elem_classes=["small-note"],
                        )

                        private_doc_state = gr.State({"filename": "", "sentences": []})

                        with gr.Row():
                            with gr.Column(scale=1, min_width=360):
                                private_file = gr.File(
                                    label="1) Upload your document",
                                    file_types=[".txt", ".md", ".docx", ".pdf"],
                                    type="filepath",
                                )
                                private_info = gr.Markdown(elem_classes=["small-note"])
                                private_sentence_dropdown = gr.Dropdown(
                                    choices=[],
                                    value=None,
                                    label="2) Select a sentence from your document",
                                )
                                private_sentence_preview = gr.Textbox(
                                    label="Selected sentence (used as query)",
                                    lines=3,
                                    interactive=False,
                                )
                                private_btn_retrieve = gr.Button(
                                    "3) Retrieve related Austrian‑law sentences",
                                    variant="primary",
                                )

                            with gr.Column(scale=2, min_width=560):
                                private_header = gr.Markdown()

                                gr.Markdown("### KAHM(query→MB corpus)")
                                private_out_kahm = gr.Dataframe(
                                    interactive=False,
                                    wrap=True,
                                    elem_classes=["kahm-table", "retrieval-table"],
                                )
                                private_plot_kahm = gr.Plot()

                                gr.Markdown("### Full-KAHM (query→KAHM corpus)")
                                private_out_full_doc = gr.Dataframe(
                                    interactive=False,
                                    wrap=True,
                                    elem_classes=["kahm-table", "retrieval-table"],
                                )
                                private_plot_full_doc = gr.Plot()

                    with gr.Tab("Funding pitch"):
                        gr.Markdown(PITCH_MD)

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

        # ---------- Private document tab ----------
        def _parse_private_doc(fp: str):
            return ui_parse_private_document(fp)

        private_file.change(
            fn=_parse_private_doc,
            inputs=[private_file],
            outputs=[
                private_doc_state,
                private_sentence_dropdown,
                private_info,
                private_sentence_preview,
                private_header,
                private_out_kahm,
                private_plot_kahm,
                private_out_full_doc,
                private_plot_full_doc,
            ],
        )

        private_sentence_dropdown.change(
            fn=ui_private_sentence_preview,
            inputs=[private_sentence_dropdown, private_doc_state],
            outputs=[private_sentence_preview],
        )

        def _retrieve_private(
            sel: str,
            st: Dict[str, Any],
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
            return ui_retrieve_private_sentence(
                sel,
                st,
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

        private_btn_retrieve.click(
            fn=_retrieve_private,
            inputs=[
                private_sentence_dropdown,
                private_doc_state,
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
            outputs=[private_header, private_out_kahm, private_plot_kahm, private_out_full_doc, private_plot_full_doc],
        )



    return demo


if __name__ == "__main__":
    app = build_app()
    # By default, gradio listens on localhost:7860
    try:
        app.launch(css=CSS)
    except TypeError:
        app.launch()
