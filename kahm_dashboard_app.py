#!/usr/bin/env python3
"""
kahm_dashboard_app.py

Customer-facing dashboard for presenting KAHM retrieval results (Austrian laws) and pitching
a compute-efficient research program around KAHM-style adapters.

Run:
  pip install -r requirements_dashboard.txt
  streamlit run kahm_dashboard_app.py

Inputs (expected, same naming as evaluation scripts):
  - kahm_evaluation_report.md (this repo)
  - ris_sentences.parquet (corpus; must include sentence_id, law_type; ideally includes a text column)
  - embedding_index.npz (Mixedbread corpus embeddings; keys: sentence_ids + embeddings)
  - idf_svd_model.joblib
  - kahm_query_regressors_by_law/ (directory of *.joblib KAHM query models)

Optional:
  - train.jsonl / test.jsonl next to query_set.py, or set environment variables from query_set.py.

Notes:
  - Live demo uses KahmQueryEmbedder from kahm_inference_embedder.py. That module depends on
    kahm_regression and combine_kahm_regressors_generalized_fast (or non-fast fallback).
"""

from __future__ import annotations

import os
import re
import math
import json
import html
import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go




# ----------------------------- Streamlit compatibility -----------------------------
def do_rerun() -> None:
    """Compatibility wrapper: prefer st.rerun(); fall back to st.experimental_rerun() on older Streamlit."""
    fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if callable(fn):
        fn()

# ----------------------------- Page config -----------------------------
st.set_page_config(
    page_title="KAHM Dashboard — Compute‑Efficient Legal Retrieval",
    page_icon="⚡",
    layout="wide",
)

# Lightweight CSS to get a more "product" feel without external assets.
st.markdown(
    """
    <style>
      .kahm-hero {
        padding: 18px 18px 14px 18px;
        border-radius: 18px;
        border: 1px solid rgba(200,200,200,0.35);
        background: linear-gradient(135deg, rgba(250,250,255,0.9), rgba(245,250,245,0.9));
      }
      .kahm-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        border: 1px solid rgba(0,0,0,0.12);
        background: rgba(255,255,255,0.8);
        margin-right: 8px;
        font-size: 12px;
      }
      .kahm-card {
        padding: 14px 14px 10px 14px;
        border-radius: 16px;
        border: 1px solid rgba(200,200,200,0.35);
        background: rgba(255,255,255,0.65);
      }
      .kahm-badge-pass {
        display:inline-block; padding: 4px 10px; border-radius: 999px;
        background: rgba(0, 155, 60, 0.13); border: 1px solid rgba(0, 155, 60, 0.35);
        font-size: 12px;
      }
      .kahm-badge-warn {
        display:inline-block; padding: 4px 10px; border-radius: 999px;
        background: rgba(220, 160, 0, 0.13); border: 1px solid rgba(220, 160, 0, 0.35);
        font-size: 12px;
      }
      .kahm-badge-fail {
        display:inline-block; padding: 4px 10px; border-radius: 999px;
        background: rgba(200, 40, 40, 0.13); border: 1px solid rgba(200, 40, 40, 0.35);
        font-size: 12px;
      }
      .small-muted { color: rgba(0,0,0,0.55); font-size: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------- Parsing helpers -----------------------------
@dataclass(frozen=True)
class CIValue:
    mean: float
    lo: float
    hi: float

    @property
    def err_lo(self) -> float:
        return float(self.mean - self.lo)

    @property
    def err_hi(self) -> float:
        return float(self.hi - self.mean)


_CI_RE = re.compile(
    r"""
    (?P<mean>[+-]?\d+(?:\.\d+)?)
    \s*\[
        \s*(?P<lo>[+-]?\d+(?:\.\d+)?)\s*,\s*
        (?P<hi>[+-]?\d+(?:\.\d+)?)\s*
    \]
    """,
    re.VERBOSE,
)

def parse_ci_cell(s: str) -> Optional[CIValue]:
    s = str(s).strip()
    m = _CI_RE.search(s)
    if not m:
        return None
    return CIValue(float(m.group("mean")), float(m.group("lo")), float(m.group("hi")))

def _make_unique_columns(cols: List[str]) -> List[str]:
    """Ensure DataFrame columns are unique (Streamlit/Arrow requires this)."""
    seen: Dict[str, int] = {}
    out: List[str] = []
    for i, c in enumerate(cols, start=1):
        name = str(c).strip()
        if not name:
            name = f"col{i}"
        base = name
        if base in seen:
            seen[base] += 1
            name = f"{base}_{seen[base]}"
        else:
            seen[base] = 1
        out.append(name)
    return out

def extract_md_table(md: str, heading_fragment: str) -> Optional[pd.DataFrame]:
    """
    Find the first GitHub-style markdown table that appears at/after the line containing `heading_fragment`.
    Robust to `heading_fragment` occurring inside the table header row.
    """
    idx = md.find(heading_fragment)
    if idx < 0:
        return None

    # Start scanning from the beginning of the line where the fragment occurs.
    line_start = md.rfind("\n", 0, idx)
    line_start = 0 if line_start < 0 else (line_start + 1)
    lines = md[line_start:].splitlines()

    table_lines: List[str] = []
    started = False

    for ln in lines:
        s = ln.lstrip()
        if s.startswith("|"):
            started = True
            table_lines.append(s.rstrip())
        else:
            if started:
                break

    if len(table_lines) < 2:
        return None

    def split_row(row: str) -> List[str]:
        row = row.strip().strip("|")
        return [c.strip() for c in row.split("|")]

    def _is_sep_cell(c: str) -> bool:
        # markdown separator cells: ---  :---:  ---:
        return bool(re.fullmatch(r":?-{3,}:?", c.strip()))

    header = split_row(table_lines[0])

    # If we accidentally captured a separator row as header, synthesize generic headers.
    if header and all(_is_sep_cell(c) for c in header):
        # Use the next non-separator row to determine column count.
        for probe in table_lines[1:]:
            cells = split_row(probe)
            if cells and not all(_is_sep_cell(c) for c in cells):
                header = [f"col{i+1}" for i in range(len(cells))]
                break

    # Determine where data starts: skip one separator row if present.
    data_start = 1
    if len(table_lines) >= 2:
        maybe_sep = split_row(table_lines[1])
        if maybe_sep and len(maybe_sep) == len(header) and all(_is_sep_cell(c) for c in maybe_sep):
            data_start = 2

    rows: List[List[str]] = []
    for r in table_lines[data_start:]:
        parts = split_row(r)
        if len(parts) != len(header):
            continue
        if parts and all(_is_sep_cell(c) for c in parts):
            continue
        rows.append(parts)

    df = pd.DataFrame(rows, columns=_make_unique_columns(header))
    return df

def df_long_from_k_table(df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    """Convert a k-row table (first col 'k') into long format with numeric mean/CI."""
    out_rows = []
    k_col = df.columns[0]
    for _, row in df.iterrows():
        k = int(str(row[k_col]).strip())
        for col in df.columns[1:]:
            v = parse_ci_cell(row[col])
            if v is None:
                continue
            out_rows.append(
                {
                    "metric": metric_name,
                    "k": k,
                    "method": col,
                    "mean": v.mean,
                    "lo": v.lo,
                    "hi": v.hi,
                    "err_lo": v.err_lo,
                    "err_hi": v.err_hi,
                }
            )
    return pd.DataFrame(out_rows)

def df_long_from_single_table(df: pd.DataFrame, name: str, key_col: str) -> pd.DataFrame:
    """For small 2-col tables (e.g., alignment evidence)"""
    out = []
    for _, r in df.iterrows():
        key = str(r[key_col]).strip()
        val = parse_ci_cell(r[df.columns[1]])
        if val is None:
            continue
        out.append({"group": name, "name": key, "mean": val.mean, "lo": val.lo, "hi": val.hi, "err_lo": val.err_lo, "err_hi": val.err_hi})
    return pd.DataFrame(out)

def safe_read_text(path: str) -> Optional[str]:
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception:
        return None


# ----------------------------- Data loading (cached) -----------------------------
@st.cache_data(show_spinner=False)
def load_report_md(report_path: str) -> Optional[str]:
    return safe_read_text(report_path)

@st.cache_data(show_spinner=False)
def parse_report_tables(md: str) -> Dict[str, pd.DataFrame]:
    """
    Parse the known tables from the generated report.
    Returns a dict of dataframes keyed by a stable name.
    """
    tables_opt: Dict[str, Optional[pd.DataFrame]] = {}


    # Primary metrics
    tables_opt["mrr"] = extract_md_table(md, "MRR@k (unique laws)")
    tables_opt["top1"] = extract_md_table(md, "Top-1 accuracy")
    tables_opt["hit"] = extract_md_table(md, "Hit@k")
    tables_opt["maj"] = extract_md_table(md, "Majority-accuracy")
    tables_opt["cons_frac"] = extract_md_table(md, "Mean consensus fraction")
    tables_opt["lift"] = extract_md_table(md, "Mean lift (prior)")

    # Deltas vs IDF baseline
    tables_opt["deltas"] = extract_md_table(md, "Paired bootstrap deltas (**KAHM(query→MB corpus) − IDF–SVD**)")

    # Routing decomposition table
    tables_opt["routing_decomp"] = extract_md_table(md, "Routing decomposition vs Mixedbread")

    # Alignment evidence
    tables_opt["alignment"] = extract_md_table(md, "Alignment measure")

    # Suggested routing thresholds
    tables_opt["tau_precision"] = extract_md_table(md, "τ* maximizing precision")
    tables_opt["tau_majority"] = extract_md_table(md, "τ* maximizing majority-accuracy")

    # Clean missing (and satisfy static type checkers)
    tables: Dict[str, pd.DataFrame] = {k: v for k, v in tables_opt.items() if v is not None and not v.empty}
    return tables

@st.cache_data(show_spinner=False)
def load_npz_bundle(path: str) -> Dict[str, np.ndarray]:
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
        raise ValueError(f"Unsupported NPZ schema in {path}. Need sentence_ids + embeddings; got keys={sorted(keys)}")

    sentence_ids = np.asarray(data[sid_key], dtype=np.int64)
    emb = np.asarray(data[emb_key], dtype=np.float32)
    if emb.ndim != 2 or sentence_ids.ndim != 1 or emb.shape[0] != sentence_ids.shape[0]:
        raise ValueError(f"Bad shapes in {path}: ids={sentence_ids.shape}, emb={emb.shape}")

    # L2 normalize
    n = np.linalg.norm(emb, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    emb = emb / n
    return {"sentence_ids": sentence_ids, "emb": emb}

@st.cache_data(show_spinner=False)
def load_corpus_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"sentence_id", "law_type"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Corpus parquet missing columns: {sorted(missing)}")

    # Choose best available text column.
    text_col = None
    for c in ["text", "sentence", "sentence_text", "content", "aligned_sentence", "ris_text"]:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        # Create a placeholder column so UI still works.
        df = df.copy()
        df["text"] = ""
        text_col = "text"

    df = df[["sentence_id", "law_type", text_col]].rename(columns={text_col: "text"})
    df["sentence_id"] = df["sentence_id"].astype(np.int64)
    df["law_type"] = df["law_type"].astype(str)
    df["text"] = df["text"].astype(str)
    df = df.drop_duplicates(subset=["sentence_id"], keep="first").set_index("sentence_id", drop=False)
    return df

@st.cache_resource(show_spinner=False)
def load_kahm_embedder(idf_svd_model_path: str, kahm_query_model_dir: str, kahm_mode: str = "soft") -> Any:
    """
    Lazy-load KahmQueryEmbedder. This relies on kahm_inference_embedder.py plus its dependencies.
    """
    from kahm_inference_embedder import KahmQueryEmbedder
    return KahmQueryEmbedder(
        idf_svd_model_path=idf_svd_model_path,
        kahm_query_model_dir=kahm_query_model_dir,
        kahm_mode=kahm_mode,
        batch_size=2048,
        materialize_classifier=True,
        cache_cluster_centers=True,
        tie_break="first",
        show_progress=False,
    )

def retrieve_topk(
    query_emb: np.ndarray,
    corpus_emb: np.ndarray,
    corpus_ids: np.ndarray,
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cosine similarity retrieval assuming embeddings are L2-normalized.
    Returns: (top_ids, top_scores) sorted by score desc.
    """
    q = np.asarray(query_emb, dtype=np.float32).reshape(1, -1)
    if q.shape[1] != corpus_emb.shape[1]:
        raise ValueError(f"Dim mismatch: query={q.shape} corpus={corpus_emb.shape}")

    scores = corpus_emb @ q.T  # (N,1)
    scores = scores.reshape(-1)

    k = int(max(1, min(top_k, scores.size)))
    idx = np.argpartition(-scores, kth=k-1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return corpus_ids[idx], scores[idx]


# ----------------------------- Query suggestion helpers -----------------------------
def simple_tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-zäöüß0-9\- ]+", " ", text, flags=re.IGNORECASE)
    parts = [p.strip() for p in text.split() if p.strip()]
    return parts

GERMAN_STOP = {
    "der","die","das","und","oder","nicht","ist","sind","ein","eine","einer","eines","im","in","am","an","auf","zu","zur","zum",
    "mit","für","von","den","dem","des","ich","wir","sie","er","es","man","wie","was","welche","welcher","welches",
    "bei","als","auch","noch","nur","mehr","weniger","kann","können","darf","dürfen","muss","müssen","soll","sollen",
    "austria","österreich","gesetz","gesetze",
}

def keyword_candidates(queries: List[str], top_n: int = 18) -> List[str]:
    from collections import Counter
    c = Counter()
    for q in queries:
        for t in simple_tokenize(q):
            if len(t) <= 2:
                continue
            if t in GERMAN_STOP:
                continue
            c[t] += 1
    return [w for w,_ in c.most_common(top_n)]

def fuzzy_suggestions(prefix: str, options: List[str], limit: int = 8) -> List[str]:
    """Return up to `limit` suggestions from `options` given the current input text.

    Must always return a list (never None) to satisfy Streamlit + type checkers.
    Uses cheap heuristics first (substring / token overlap), then falls back to difflib.
    """
    options = options or []
    prefix = (prefix or "").strip().lower()
    if not options:
        return []
    if not prefix:
        return options[:limit]

    # Score by substring presence + token overlap (fast, dependency-free)
    scored: List[Tuple[int, int, str]] = []
    p_tokens = set(simple_tokenize(prefix))
    for s in options:
        sl = (s or "").lower()
        if not sl:
            continue
        if prefix in sl:
            scored.append((0, len(sl), s))
        else:
            s_tokens = set(simple_tokenize(sl))
            overlap = len(p_tokens & s_tokens)
            if overlap > 0:
                scored.append((1, -overlap, s))

    if scored:
        scored.sort()
        out = [t[2] for t in scored][:limit]
        return out

    # Fallback: approximate string similarity (still stdlib)
    return difflib.get_close_matches(prefix, options, n=limit, cutoff=0.2)


# ----------------------------- Retrieval table rendering -----------------------------
def render_retrieval_results_table(res: pd.DataFrame, max_height_px: int = 520) -> None:
    """Render retrieval results with controlled column widths and full text visibility (wrapping, no truncation)."""
    if res is None or res.empty:
        st.info("No retrieval results to display yet.")
        return

    # Defensive: ensure expected columns exist
    cols = ["rank", "score", "law_type", "sentence_id", "text"]
    for c in cols:
        if c not in res.columns:
            res[c] = "" if c == "text" else np.nan

    # Build HTML table (Streamlit dataframe truncates long text; HTML allows wrapping)
    def _esc(x: Any) -> str:
        s = "" if x is None else str(x)
        s = s.replace("\n", "<br/>")
        return html.escape(s, quote=True).replace("&lt;br/&gt;", "<br/>")

    # Format score nicely
    view = res[cols].copy()
    try:
        view["rank"] = view["rank"].astype(int)
    except Exception:
        pass
    try:
        view["score"] = pd.to_numeric(view["score"], errors="coerce")
    except Exception:
        pass

    rows_html = []
    for _, r in view.iterrows():
        sc = r["score"]
        sc_s = "" if pd.isna(sc) else f"{float(sc):.4f}"
        rows_html.append(
            f"""<tr>
                <td class='col-rank'>{_esc(r['rank'])}</td>
                <td class='col-score'>{html.escape(sc_s)}</td>
                <td class='col-law'>{_esc(r['law_type'])}</td>
                <td class='col-sid'>{_esc(r['sentence_id'])}</td>
                <td class='col-text'>{_esc(r['text'])}</td>
            </tr>"""
        )

    st.markdown(
        f"""
        <style>
          .kahm-results-wrap {{
            max-height: {int(max_height_px)}px;
            overflow-y: auto;
            border: 1px solid rgba(0,0,0,0.10);
            border-radius: 14px;
            padding: 6px 10px;
            background: rgba(255,255,255,0.65);
          }}
          table.kahm-results {{
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
          }}
          table.kahm-results th {{
            text-align: left;
            font-size: 12px;
            color: rgba(0,0,0,0.65);
            border-bottom: 1px solid rgba(0,0,0,0.12);
            padding: 8px 8px;
            position: sticky;
            top: 0;
            background: rgba(255,255,255,0.92);
            z-index: 1;
          }}
          table.kahm-results td {{
            vertical-align: top;
            border-bottom: 1px solid rgba(0,0,0,0.06);
            padding: 8px 8px;
            font-size: 13px;
            line-height: 1.25rem;
          }}
          table.kahm-results td.col-text {{
            white-space: pre-wrap;
            word-break: break-word;
          }}
          table.kahm-results td.col-rank {{ width: 54px; }}
          table.kahm-results td.col-score {{ width: 84px; }}
          table.kahm-results td.col-law {{ width: 140px; }}
          table.kahm-results td.col-sid {{ width: 110px; }}
          table.kahm-results col.col-rank {{ width: 54px; }}
          table.kahm-results col.col-score {{ width: 84px; }}
          table.kahm-results col.col-law {{ width: 140px; }}
          table.kahm-results col.col-sid {{ width: 110px; }}
          table.kahm-results col.col-text {{ width: auto; }}
        </style>
        <div class="kahm-results-wrap">
          <table class="kahm-results">
            <colgroup>
              <col class="col-rank"/>
              <col class="col-score"/>
              <col class="col-law"/>
              <col class="col-sid"/>
              <col class="col-text"/>
            </colgroup>
            <thead>
              <tr>
                <th>rank</th>
                <th>score</th>
                <th>law_type</th>
                <th>sentence_id</th>
                <th>text</th>
              </tr>
            </thead>
            <tbody>
              {''.join(rows_html)}
            </tbody>
          </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------- Sidebar (inputs) -----------------------------
st.sidebar.header("Data & demo configuration")

default_report = str(Path(__file__).with_name("kahm_evaluation_report.md"))
report_path = st.sidebar.text_input("Report markdown path", value=default_report)

with st.sidebar.expander("Live demo assets (paths)", expanded=False):
    corpus_parquet = st.text_input("Corpus parquet", value="ris_sentences.parquet")
    mb_npz = st.text_input("Mixedbread corpus embeddings (NPZ)", value="embedding_index.npz")
    idf_svd_model = st.text_input("IDF–SVD model (joblib)", value="idf_svd_model.joblib")
    kahm_query_models = st.text_input("KAHM query model dir", value="kahm_query_regressors_by_law")
    kahm_mode = st.selectbox("KAHM mode", options=["soft", "hard"], index=0)
    top_k = st.slider("Retrieve top‑k sentences", min_value=5, max_value=100, value=20, step=5)

with st.sidebar.expander("Cost model (bring your own numbers)", expanded=False):
    st.caption("Use this to quantify expected savings for your deployment.")
    base_ms = st.number_input("Baseline transformer query‑encoder latency (ms)", min_value=0.0, value=18.0, step=1.0)
    kahm_ms = st.number_input("KAHM query adapter latency estimate (ms)", min_value=0.0, value=1.5, step=0.1)
    qps = st.number_input("Queries per second", min_value=1.0, value=200.0, step=10.0)

    if base_ms > 0:
        speedup = base_ms / max(1e-9, kahm_ms)
    else:
        speedup = float("nan")
    saved_core_ms_per_s = (base_ms - kahm_ms) * qps
    st.metric("Estimated encoder speedup", f"{speedup:.1f}×" if math.isfinite(speedup) else "—")
    st.metric("Compute saved per second", f"{saved_core_ms_per_s:,.0f} ms/s")

    st.caption("Interpretation: saved ms/s is proportional to CPU/GPU time freed at the query‑encoder stage.")


# ----------------------------- Load report & parse -----------------------------
md = load_report_md(report_path)
if md is None:
    st.error("Could not read the report markdown. Check the path in the sidebar.")
    st.stop()

tables = parse_report_tables(md)

# Build long data for charts (if available)
metric_long_parts = []
if "mrr" in tables: metric_long_parts.append(df_long_from_k_table(tables["mrr"], "MRR@k (unique laws)"))
if "top1" in tables: metric_long_parts.append(df_long_from_k_table(tables["top1"], "Top‑1 accuracy"))
if "hit" in tables: metric_long_parts.append(df_long_from_k_table(tables["hit"], "Hit@k"))
if "maj" in tables: metric_long_parts.append(df_long_from_k_table(tables["maj"], "Majority‑accuracy (τ=0.10)"))
if "cons_frac" in tables: metric_long_parts.append(df_long_from_k_table(tables["cons_frac"], "Mean consensus fraction"))
if "lift" in tables: metric_long_parts.append(df_long_from_k_table(tables["lift"], "Mean lift (prior)"))

metrics_long = pd.concat(metric_long_parts, ignore_index=True) if metric_long_parts else pd.DataFrame()


# ----------------------------- UI: Top navigation -----------------------------
tab_overview, tab_evidence, tab_demo, tab_project, tab_appendix = st.tabs(
    ["Executive pitch", "Evidence & graphics", "Live demo", "Research project proposal", "Appendix"]
)

# ----------------------------- Executive pitch -----------------------------
with tab_overview:
    st.markdown(
        """
        <div class="kahm-hero">
          <div style="font-size: 26px; font-weight: 700; margin-bottom: 6px;">KAHM: Transformer‑free query encoding on a transformer index</div>
          <div class="small-muted" style="margin-bottom: 10px;">
            Replace online transformer query inference with a lightweight, gradient‑free adapter (IDF–SVD → KAHM → Mixedbread space),
            while keeping the strong transformer corpus index fixed.
          </div>
          <span class="kahm-pill">Compute‑efficient</span>
          <span class="kahm-pill">No gradient descent at query time</span>
          <span class="kahm-pill">Plugs into existing embedding indices</span>
          <span class="kahm-pill">Routing‑ready (majority vote)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    # Pull headline k=100 for KAHM from report tables if present
    def headline(metric_key: str, method_name: str, k: int = 100) -> Optional[CIValue]:
        t = tables.get(metric_key)
        if t is None:
            return None
        # first col is k
        kcol = t.columns[0]
        sub = t[t[kcol].astype(str).str.strip() == str(k)]
        if sub.empty or method_name not in t.columns:
            return None
        return parse_ci_cell(sub.iloc[0][method_name])

    kahm_method = "KAHM(query→MB corpus)"
    idf_method = "IDF–SVD"

    h_top1 = headline("top1", kahm_method, 100)
    h_mrr = headline("mrr", kahm_method, 100)
    h_hit = headline("hit", kahm_method, 100)
    h_maj = headline("maj", kahm_method, 100)
    h_cf = headline("cons_frac", kahm_method, 100)
    h_lift = headline("lift", kahm_method, 100)

    if h_top1: c1.metric("Top‑1", f"{h_top1.mean:.3f}", f"[{h_top1.lo:.3f}, {h_top1.hi:.3f}]")
    if h_mrr: c2.metric("MRR@100", f"{h_mrr.mean:.3f}", f"[{h_mrr.lo:.3f}, {h_mrr.hi:.3f}]")
    if h_hit: c3.metric("Hit@100", f"{h_hit.mean:.3f}", f"[{h_hit.lo:.3f}, {h_hit.hi:.3f}]")
    if h_maj: c4.metric("Majority‑acc", f"{h_maj.mean:.3f}", f"[{h_maj.lo:.3f}, {h_maj.hi:.3f}]")
    if h_cf: c5.metric("Consensus frac", f"{h_cf.mean:.3f}", f"[{h_cf.lo:.3f}, {h_cf.hi:.3f}]")
    if h_lift: c6.metric("Lift (prior)", f"{h_lift.mean:.1f}", f"[{h_lift.lo:.1f}, {h_lift.hi:.1f}]")

    st.write("")
    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.subheader("Why this matters (customer story)")
        st.markdown(
            """
            - **Operational win:** keep your best corpus index (transformer embeddings), but remove transformer inference from the **online query path**.
            - **Scientific claim supported by the evaluation:** KAHM(query→MB) *systematically* improves over a strong low‑cost baseline (IDF–SVD) across
              rank‑sensitive and routing‑sensitive measures, with paired bootstrap CIs excluding 0.
            - **Product angle:** enables **high‑QPS** semantic retrieval / routing where latency and cost are dominated by query embedding.
            """
        )

        st.subheader("Core system idea")
        st.markdown(
            """
            **Offline:** transformer corpus embeddings (Mixedbread) → vector index  
            **Online:** IDF–SVD query features → **KAHM adapter** → Mixedbread space → vector search  

            This isolates transformer compute to offline indexing and lets you scale query handling cheaply.
            """
        )

    with right:
        st.subheader("Storylines at a glance")
        st.markdown(
            """
            <div class="kahm-card">
              <div style="font-weight:700; margin-bottom:6px;">A) Beats low‑cost baseline</div>
              <div class="small-muted">KAHM(query→MB) − IDF–SVD: improvements across all measures, CI lower bounds &gt; 0.</div>
              <div style="margin-top:8px;"><span class="kahm-badge-pass">Supported</span></div>
            </div>
            <div style="height:10px;"></div>
            <div class="kahm-card">
              <div style="font-weight:700; margin-bottom:6px;">B) Competitive on top‑of‑ranking</div>
              <div class="small-muted">Comparable to transformer‑query baseline on top‑k law quality, while removing query transformer inference.</div>
              <div style="margin-top:8px;"><span class="kahm-badge-pass">Supported (context)</span></div>
            </div>
            <div style="height:10px;"></div>
            <div class="kahm-card">
              <div style="font-weight:700; margin-bottom:6px;">C) Geometry alignment evidence</div>
              <div class="small-muted">High cosine alignment in embedding space and strong law‑set neighborhood overlap.</div>
              <div style="margin-top:8px;"><span class="kahm-badge-pass">Supported</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ----------------------------- Evidence & graphics -----------------------------
with tab_evidence:
    st.subheader("Metrics across cutoffs (k)")
    if metrics_long.empty:
        st.warning("Could not parse metric tables from the report. Check that the report matches the expected format.")
    else:
        metric_choice = st.selectbox(
            "Metric",
            options=sorted(metrics_long["metric"].unique().tolist()),
            index=0,
        )
        dfm = metrics_long[metrics_long["metric"] == metric_choice].copy()

        # Prefer showing KAHM + IDF; include Mixedbread if present in table.
        method_order = [m for m in [kahm_method, idf_method, "Mixedbread (true)", "Full-KAHM (query→KAHM corpus)"] if m in dfm["method"].unique()]
        # Add any remaining
        for m in dfm["method"].unique():
            if m not in method_order:
                method_order.append(m)
        dfm["method"] = pd.Categorical(dfm["method"], categories=method_order, ordered=True)
        dfm = dfm.sort_values(["k", "method"])

        fig = go.Figure()
        for method in dfm["method"].cat.categories:
            sub = dfm[dfm["method"] == method]
            if sub.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=sub["k"],
                    y=sub["mean"],
                    mode="lines+markers",
                    name=str(method),
                    error_y=dict(
                        type="data",
                        symmetric=False,
                        array=sub["err_hi"],
                        arrayminus=sub["err_lo"],
                        thickness=1.2,
                        width=0,
                    ),
                )
            )
        fig.update_layout(
            height=420,
            xaxis_title="k (retrieval cutoff)",
            yaxis_title=metric_choice,
            legend_title="Method",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("Headline improvement vs IDF–SVD (paired deltas)")
    d = tables.get("deltas")
    if d is None:
        st.info("Delta table not found in report.")
    else:
        # Parse deltas table; pick k=100 if available.
        kcol = d.columns[0]
        sub = d[d[kcol].astype(str).str.strip() == "100"]
        if sub.empty:
            sub = d
        # Build bar plot for k row.
        row = sub.iloc[0].to_dict()
        bars = []
        for col in d.columns[1:]:
            v = parse_ci_cell(row[col])
            if v is None:
                continue
            bars.append({"metric": col.replace("Δ", "").strip(), "mean": v.mean, "lo": v.lo, "hi": v.hi, "err_lo": v.err_lo, "err_hi": v.err_hi})
        bdf = pd.DataFrame(bars)
        if not bdf.empty:
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=bdf["metric"],
                        y=bdf["mean"],
                        error_y=dict(
                            type="data",
                            symmetric=False,
                            array=bdf["err_hi"],
                            arrayminus=bdf["err_lo"],
                            thickness=1.2,
                            width=0,
                        ),
                    )
                ]
            )
            fig.update_layout(height=360, xaxis_title="Measure", yaxis_title="Δ (KAHM − IDF–SVD)", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width="stretch")
        st.caption("Deltas and confidence intervals are parsed directly from the generated report.")

    st.divider()
    st.subheader("Routing behavior (vote threshold τ)")
    rd = tables.get("routing_decomp")
    if rd is None:
        st.info("Routing decomposition table not found in report.")
    else:
        # Columns: τ, Δmaj-acc, Δcov-part, Δprec-part
        tau_col = rd.columns[0]
        rows = []
        for _, r in rd.iterrows():
            tau = float(str(r[tau_col]).strip())
            dm = parse_ci_cell(r[rd.columns[1]])
            dc = parse_ci_cell(r[rd.columns[2]])
            dp = parse_ci_cell(r[rd.columns[3]])
            if dm and dc and dp:
                rows.append(
                    {"tau": tau, "delta_majority_acc": dm.mean, "maj_err_hi": dm.err_hi, "maj_err_lo": dm.err_lo,
                     "delta_cov": dc.mean, "cov_err_hi": dc.err_hi, "cov_err_lo": dc.err_lo,
                     "delta_prec": dp.mean, "prec_err_hi": dp.err_hi, "prec_err_lo": dp.err_lo}
                )
        rdf = pd.DataFrame(rows).sort_values("tau")
        if not rdf.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=rdf["tau"], y=rdf["delta_majority_acc"], mode="lines+markers", name="Δ majority‑acc"))
            fig.add_trace(go.Scatter(x=rdf["tau"], y=rdf["delta_cov"], mode="lines+markers", name="Δ coverage component"))
            fig.add_trace(go.Scatter(x=rdf["tau"], y=rdf["delta_prec"], mode="lines+markers", name="Δ precision component"))
            fig.update_layout(height=380, xaxis_title="τ (vote predominance threshold)", yaxis_title="Δ vs Mixedbread (context)", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width="stretch")
            st.caption("Use this to justify ‘confident routing’ thresholds and to communicate coverage vs precision trade‑offs.")

    st.divider()
    st.subheader("Geometry alignment evidence")
    al = tables.get("alignment")
    if al is None:
        st.info("Alignment evidence table not found in report.")
    else:
        # alignment table is 2 columns: measure | estimate
        if len(al.columns) >= 2:
            arows = []
            for _, r in al.iterrows():
                name = str(r[al.columns[0]]).strip()
                v = parse_ci_cell(r[al.columns[1]])
                if v is None:
                    continue
                arows.append({"Alignment measure": name, "mean": v.mean, "lo": v.lo, "hi": v.hi})
            adf = pd.DataFrame(arows)
            st.dataframe(adf, width="stretch", hide_index=True)



# ----------------------------- Live demo -----------------------------
with tab_demo:
    st.subheader("Live demo: retrieve Austrian-law sentences for a query")
    st.caption("This runs local retrieval against your corpus embeddings. Provide the files in the sidebar.")

    # Attempt to load query sets for suggestions
    query_examples: List[str] = []
    try:
        import query_set  # uses environment vars / local train/test files
        train = getattr(query_set, "TRAIN_QUERY_SET", [])
        test = getattr(query_set, "TEST_QUERY_SET", [])
        query_examples = [
            str(q.get("query_text", "")).strip()
            for q in (train[:400] + test[:400])
            if isinstance(q, dict) and str(q.get("query_text", "")).strip()
        ]
    except Exception:
        query_examples = []

    st.markdown("##### Query input")

    st.session_state.setdefault("demo_query", "")
    st.session_state.setdefault("demo_example_pick", "—")
    st.session_state.setdefault("demo_run_requested", False)

    st.session_state.setdefault("demo_query_pending", None)
    if st.session_state.get("demo_query_pending") is not None:
        st.session_state["demo_query"] = st.session_state["demo_query_pending"]
        st.session_state["demo_query_pending"] = None

    colA, colB = st.columns([0.62, 0.38], gap="large")

    with colA:
        st.text_area(
            "Your question / prompt",
            key="demo_query",
            height=100,
            placeholder="e.g., Wie hoch darf die Miete erhöht werden? (Mietrecht in Österreich)",
        )

        if query_examples:
            st.caption("Typeahead: pick an example query (train/test) and optionally apply it.")
            pick = st.selectbox(
                "Example queries",
                options=["—"] + query_examples,
                key="demo_example_pick",
                index=0,
            )
            apply_example = st.button(
                "Use selected example",
                key="apply_example_btn",
                    disabled=(pick == "—"),
            )
            if apply_example and pick != "—":
                st.session_state["demo_query_pending"] = pick
                do_rerun()
        else:
            st.caption("To enable example query suggestions, place train/test query files for query_set.py, or set AUSTLAW_QUERYSET_DIR.")

    with colB:
        if query_examples:
            kws = keyword_candidates(query_examples, top_n=18)
            st.caption("Suggested keywords (click to append)")
            kw_cols = st.columns(3)
            for i, w in enumerate(kws):
                if kw_cols[i % 3].button(w, key=f"kw_{i}", width="stretch"):
                    st.session_state["demo_query_pending"] = (str(st.session_state.get("demo_query","")).rstrip() + " " + w).strip()
                    do_rerun()

            st.caption("Fast suggestions based on your current text")
            sugg = fuzzy_suggestions(st.session_state["demo_query"], query_examples, limit=6)
            for j, s in enumerate(sugg):
                if st.button(f"↳ {s}", key=f"sugg_{j}", width="stretch"):
                    st.session_state["demo_query_pending"] = s
                    do_rerun()

    st.write("")
    # Use a latched flag so the click can't be lost to other widget-triggered reruns.
    if st.button("Run retrieval", key="run_retrieval_btn", type="primary", width="stretch"):
        st.session_state["demo_run_requested"] = True

    # Optional: clear cached demo artifacts (helpful if you change files on disk).
    cbtn1, cbtn2 = st.columns([0.6, 0.4])
    with cbtn2:
        if st.button("Reset demo cache", key="reset_demo_cache", width="stretch"):
            try:
                st.cache_data.clear()
                st.cache_resource.clear()
            except Exception:
                pass
            for k in ["demo_results", "demo_diag", "demo_last_query"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.session_state["demo_run_requested"] = False

    if st.session_state.get("demo_run_requested", False):
        q = (st.session_state.get("demo_query") or "").strip()
        st.session_state["demo_run_requested"] = False  # reset latch immediately
        if not q:
            st.warning("Please enter a query.")
        else:
            try:
                with st.spinner("Loading corpus & embeddings..."):
                    corp_df = load_corpus_parquet(corpus_parquet)
                    mb_bundle = load_npz_bundle(mb_npz)

                with st.spinner("Embedding query with KAHM..."):
                    try:
                        embedder = load_kahm_embedder(idf_svd_model, kahm_query_models, kahm_mode=kahm_mode)
                    except Exception:
                        # If the cached resource is in a bad state, clear and retry once.
                        try:
                            load_kahm_embedder.clear()
                        except Exception:
                            pass
                        embedder = load_kahm_embedder(idf_svd_model, kahm_query_models, kahm_mode=kahm_mode)

                    q_emb, chosen, min_dist, names = embedder.embed([q])
                    q_emb = np.asarray(q_emb[0], dtype=np.float32)

                    # Normalize query (retrieve_topk assumes cosine over L2-normalized vectors).
                    n = float(np.linalg.norm(q_emb))
                    if n > 0:
                        q_emb = q_emb / max(n, 1e-12)

                with st.spinner("Retrieving top‑k..."):
                    top_ids, top_scores = retrieve_topk(q_emb, mb_bundle["emb"], mb_bundle["sentence_ids"], int(top_k))

                rows = []
                for rank, (sid, sc) in enumerate(zip(top_ids.tolist(), top_scores.tolist()), start=1):
                    sid_i = int(sid)
                    if sid_i in corp_df.index:
                        law = str(corp_df.loc[sid_i, "law_type"])
                        text = str(corp_df.loc[sid_i, "text"])
                    else:
                        law = "UNKNOWN"
                        text = ""
                    rows.append(
                        {
                            "rank": int(rank),
                            "score": float(sc),
                            "law_type": law,
                            "sentence_id": sid_i,
                            "text": text,
                        }
                    )

                res = pd.DataFrame(rows)

                # Majority / consensus diagnostics for pitching “routing”
                law_counts = res["law_type"].value_counts()
                maj_law = str(law_counts.index[0]) if len(law_counts) else "—"
                maj_frac = float(law_counts.iloc[0] / len(res)) if len(law_counts) else 0.0

                diag = {
                    "maj_law": maj_law,
                    "maj_frac": maj_frac,
                    "chosen": str(int(chosen[0])) if hasattr(chosen, "__len__") and len(chosen) else "—",
                    "min_dist": float(min_dist[0]) if hasattr(min_dist, "__len__") and len(min_dist) else None,
                }

                st.session_state["demo_results"] = res
                st.session_state["demo_diag"] = diag
                st.session_state["demo_last_query"] = q

            except Exception as e:
                st.error("Live demo failed to run with the provided paths.")
                st.code(str(e))
                st.markdown(
                    """
                    **Checklist**
                    - `ris_sentences.parquet` exists and includes `sentence_id` and `law_type` (and ideally a sentence text column).
                    - `embedding_index.npz` exists and includes `sentence_ids` and `embeddings`.
                    - `idf_svd_model.joblib` exists.
                    - `kahm_query_regressors_by_law/` exists and contains `*.joblib` models.
                    - Dependencies for `kahm_inference_embedder.py` are installed (notably `kahm_regression` and the combiner modules).
                    """
                )

    # Always show last results (if any) so the user can iterate.
    if "demo_results" in st.session_state and isinstance(st.session_state["demo_results"], pd.DataFrame):
        res = st.session_state["demo_results"]
        diag = st.session_state.get("demo_diag", {}) or {}
        last_q = st.session_state.get("demo_last_query", "")

        st.success("Retrieval complete.")
        if last_q:
            st.caption(f"Query: {last_q}")

        info1, info2, info3, info4 = st.columns(4)
        info1.metric("Top law (majority vote)", str(diag.get("maj_law", "—")))
        info2.metric("Top‑law fraction", f"{float(diag.get('maj_frac', 0.0)):.2f}")
        info3.metric("Chosen KAHM sub‑model", str(diag.get("chosen", "—")))
        md = diag.get("min_dist", None)
        info4.metric("Min distance (gate score)", f"{float(md):.4f}" if md is not None else "—")

        st.markdown("##### Retrieved sentences")
        render_retrieval_results_table(res, max_height_px=560)

        csv_bytes = res.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results as CSV",
            data=csv_bytes,
            file_name="kahm_retrieval_results.csv",
            mime="text/csv",
        )

        with st.expander("Show law distribution in top‑k"):
            st.bar_chart(res["law_type"].value_counts())

# ----------------------------- Research project proposal -----------------------------
with tab_project:
    st.subheader("Proposal: joint R&D project — compute‑efficient LLM components via KAHM")
    st.markdown(
        """
        The dashboard evidence supports KAHM as a **query‑time substitute** for transformer encoders on retrieval/routing tasks.
        The research opportunity is bigger: develop **compute‑efficient components** that preserve the semantics of strong foundation
        embeddings while dramatically reducing online compute.

        Below is a concrete project structure you can show in a customer meeting.
        """
    )

    c1, c2 = st.columns([0.55, 0.45], gap="large")
    with c1:
        st.markdown("### Research hypotheses")
        st.markdown(
            """
            1) **Adapter hypothesis:** A gradient‑free (or near‑gradient‑free) operator learned over fixed embedding spaces can recover
               high‑quality neighborhood structure without online transformer inference.

            2) **Composable components hypothesis:** KAHM‑style adapters can be stacked or composed to form lightweight **semantic modules**
               (routing heads, retrieval heads, domain adapters) that are stable under distribution shift.

            3) **System hypothesis:** The biggest ROI comes from system designs that keep heavy models offline (indexing / distillation)
               and keep online components sparse, linear, or low‑rank.
            """
        )

        st.markdown("### Work packages (12–16 weeks)")
        st.markdown(
            """
            **WP1 — Reproducible benchmark & ablations (3–4 weeks)**  
            - Validate results on your corpora (Austrian laws + at least 1 customer domain).  
            - Ablate: adapter family, gating strategy, dimensionality, calibration (τ sweeps).

            **WP2 — Next‑gen KAHM components (5–6 weeks)**  
            - Train adapters for: domain routing, citation‑grade retrieval, and “fast semantic filters”.  
            - Explore *multi‑head* / *mixture* adapters with hard/soft gating.

            **WP3 — Productization prototype (4–6 weeks)**  
            - Integrate into your inference stack; define SLOs (latency, cost, accuracy).  
            - Add monitoring: drift, confidence, and “fallback to transformer” policy.
            """
        )

    with c2:
        st.markdown("### Deliverables")
        st.markdown(
            """
            - A customer‑ready **technical report**: accuracy + routing curves + confidence calibration.
            - A deployable **demo service**: transformer‑free query embedding + retrieval + routing.
            - A “go/no‑go” **business case**: cost model with your QPS and infrastructure assumptions.
            - Optional: a patentable **system design** around adapter‑on‑index + confidence‑gated fallback.
            """
        )

        st.markdown("### Decision points")
        st.markdown(
            """
            - **Week 4:** does KAHM maintain retrieval quality within an acceptable margin while reducing encoder cost?
            - **Week 10:** does confidence‑gated routing improve workflow metrics (precision@coverage) for at least one business process?
            - **Week 16:** do we have a stable production design (monitoring + fallback) with measurable cost savings?
            """
        )

        st.markdown("### Risk controls")
        st.markdown(
            """
            - Always keep a **fallback** path (transformer query encoder) for low‑confidence cases.
            - Use majority‑vote τ curves to pick operating points and communicate tradeoffs.
            - Track drift with embedding‑space statistics and retrieval‑purity measures.
            """
        )

# ----------------------------- Appendix -----------------------------
with tab_appendix:
    st.subheader("Full report (markdown)")
    st.caption("Rendered from the report file configured in the sidebar.")
    st.markdown(md)

    with st.expander("Raw parsed tables"):
        for k, v in tables.items():
            st.markdown(f"**{k}**")
            st.dataframe(v, width="stretch", hide_index=True)
