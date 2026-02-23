
# kahm_dashboard_app.py
# Streamlit dashboard for KAHM vs baselines + interactive Austrian law retrieval demo.
#
# Run:
#   pip install streamlit plotly pandas numpy faiss-cpu sentence-transformers joblib pyarrow
#   streamlit run kahm_dashboard_app.py
#
# Notes:
# - For KAHM(query→MB), you must have kahm_inference_embedder.py and the trained model dir available.
# - For IDF–SVD, you must provide the joblib pipeline used in evaluation.
# - For the corpus, provide a parquet with at least: sentence_id (int), law_type (str), and ideally a text column.
#
from __future__ import annotations

import os
import re
import json
import time
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Iterable

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# -----------------------------
# UI setup
# -----------------------------
st.set_page_config(
    page_title="KAHM Embeddings Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CUSTOM_CSS = """
<style>
/* Hide Streamlit default chrome for a more "product" look */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Typography */
html, body, [class*="css"]  {
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji";
}

/* KPI cards */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.75rem;
}
.kpi {
  border: 1px solid rgba(49, 51, 63, 0.15);
  border-radius: 16px;
  padding: 0.9rem 1rem;
  background: rgba(255,255,255,0.65);
  backdrop-filter: blur(6px);
}
.kpi .label {font-size: 0.85rem; opacity: 0.75; margin-bottom: 0.25rem;}
.kpi .value {font-size: 1.55rem; font-weight: 700; line-height: 1.2;}
.kpi .delta {font-size: 0.92rem; opacity: 0.85; margin-top: 0.2rem;}
.badge {
  display: inline-block;
  padding: 0.16rem 0.55rem;
  border-radius: 999px;
  border: 1px solid rgba(49, 51, 63, 0.15);
  font-size: 0.78rem;
  opacity: 0.85;
}
.small-note {font-size: 0.85rem; opacity: 0.75;}
hr {opacity: 0.4;}
</style>
"""
st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------
# Helpers
# -----------------------------
_METRIC_ALIASES = {
    "hit@k": "hit",
    "MRR@k (unique laws)": "mrr_ul",
    "MRR@k": "mrr_ul",
    "top1": "top1",
    "majority-acc": "majority",
    "consensus frac": "cons_frac",
    "lift (prior)": "lift",
}

_NUM_CI_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*\[\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\]\s*$")
_NUM_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*$")


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def _parse_cell(cell: str) -> Tuple[float, float, float]:
    """
    Parses a cell like '0.503 [0.488, 0.516]' into (mean, lo, hi).
    If no CI is present, returns (value, nan, nan).
    """
    cell = cell.strip()
    m = _NUM_CI_RE.match(cell)
    if m:
        return float(m.group(1)), float(m.group(2)), float(m.group(3))
    m2 = _NUM_RE.match(cell)
    if m2:
        v = float(m2.group(1))
        return v, float("nan"), float("nan")
    # fallback: try to extract first float
    m3 = re.search(r"[+-]?\d+(?:\.\d+)?", cell)
    if not m3:
        return float("nan"), float("nan"), float("nan")
    v = float(m3.group(0))
    return v, float("nan"), float("nan")


def _parse_markdown_table(lines: List[str], start_idx: int) -> Tuple[pd.DataFrame, int]:
    """
    Parse a markdown table starting at lines[start_idx] where that line begins with '|'.
    Returns (df, next_idx_after_table).
    """
    header = [c.strip() for c in lines[start_idx].strip().strip("|").split("|")]
    # alignment row at start_idx+1
    rows = []
    i = start_idx + 2
    while i < len(lines):
        ln = lines[i].rstrip("\n")
        if not ln.strip():
            break
        if not ln.lstrip().startswith("|"):
            break
        parts = [c.strip() for c in ln.strip().strip("|").split("|")]
        # pad if needed
        if len(parts) < len(header):
            parts += [""] * (len(header) - len(parts))
        rows.append(parts[: len(header)])
        i += 1
    df = pd.DataFrame(rows, columns=header)
    return df, i


@dataclass
class ParsedReport:
    title: str
    generated_line: str
    micro: Dict[int, pd.DataFrame]
    delta: Dict[int, pd.DataFrame]
    compute_init: Optional[pd.DataFrame]
    compute_paths: Optional[pd.DataFrame]
    compute_machine: Optional[pd.DataFrame]
    routing: Optional[pd.DataFrame]


@st.cache_data(show_spinner=False)
def load_report_md(path: str, mtime_ns: int, size_bytes: int) -> str:
    # mtime/size are included only to invalidate cache when the file changes
    _ = (mtime_ns, size_bytes)
    return Path(path).read_text(encoding="utf-8")


@st.cache_data(show_spinner=False)
def parse_report(md: str) -> ParsedReport:
    """Parse evaluation markdown reports.

    Supports two report formats:

    (A) Per-k micro tables (older):
        ### Micro-average (per query) at k=10
        | Method | hit@k | ... |

    (B) Metric-wise micro tables (publication report):
        ### Micro-averaged quality (mean ± 95% CI)
        **MRR@k (unique laws)**
        | k | IDF–SVD | KAHM(...) | Mixedbread(...) |
    """
    lines = md.splitlines()

    title = lines[0].lstrip("#").strip() if lines and lines[0].startswith("#") else ""
    generated = ""
    for ln in lines[:60]:
        if ln.startswith("Generated:") or ln.startswith("**Generated (UTC):**"):
            generated = ln.strip()
            break

    micro: Dict[int, pd.DataFrame] = {}
    delta: Dict[int, pd.DataFrame] = {}
    compute_init: Optional[pd.DataFrame] = None
    compute_paths: Optional[pd.DataFrame] = None
    compute_machine: Optional[pd.DataFrame] = None
    routing: Optional[pd.DataFrame] = None

    # --------- Pass 1: old format (per-k micro + per-k delta) ---------
    i = 0
    while i < len(lines):
        ln = lines[i].strip()

        m = re.match(r"^###\s+Micro-average.*at\s+k=(\d+)\s*$", ln)
        if m:
            k = int(m.group(1))
            j = i + 1
            while j < len(lines) and not lines[j].lstrip().startswith("| Method"):
                j += 1
            if j < len(lines):
                df, nxt = _parse_markdown_table(lines, j)
                micro[k] = df
                i = nxt
                continue

        m2 = re.match(r"^Δ\s+at\s+k=(\d+)\s*\(.*\)\s*$", ln)
        if m2:
            k = int(m2.group(1))
            j = i + 1
            while j < len(lines) and not lines[j].lstrip().startswith("| Comparison"):
                j += 1
            if j < len(lines):
                df, nxt = _parse_markdown_table(lines, j)
                delta[k] = df
                i = nxt
                continue

        # Compute tables (older reports)
        if ln.startswith("| Component | Wall time"):
            df, nxt = _parse_markdown_table(lines, i)
            compute_init = df
            i = nxt
            continue

        if (
            ln.startswith("| Path | Query source | Query embed / q")
            or ln.startswith("| Path | Query embed / q")
            or (ln.startswith("| Path |") and ("FAISS search / q" in ln or "Total online / q" in ln))
        ):
            df, nxt = _parse_markdown_table(lines, i)
            compute_paths = df
            i = nxt
            continue

        # Machine profile table (newer publication reports)
        if ln.startswith("| Field | Value"):
            window = "\n".join(lines[max(0, i-4): i+1])
            if ("Machine profile" in window) or ("auto-detected" in window):
                df, nxt = _parse_markdown_table(lines, i)
                compute_machine = df
                i = nxt
                continue

        # Routing (older report)
        if ln.startswith("| Method | tau* | coverage"):
            df, nxt = _parse_markdown_table(lines, i)
            routing = df
            i = nxt
            continue

        i += 1

    # --------- Pass 2: publication format (metric-wise micro + delta tables) ---------
    if not micro:
        metric_map: Dict[str, str] = {
            "MRR@k (unique laws)": "MRR@k (unique laws)",
            "Hit@k": "hit@k",
            "Top-1 accuracy": "top1",
            "Mean consensus fraction": "consensus frac",
            "Mean lift (prior)": "lift (prior)",
        }

        def _canon_metric(label: str) -> str:
            lab = label.strip()
            # Keep discriminating suffixes like '(unique laws)' and '(prior)'.
            if lab.startswith("Majority-accuracy"):
                return "majority-acc"
            if lab in metric_map:
                return metric_map[lab]
            # Gentle normalization for formatting variants (do not strip semantic suffixes).
            lab_norm = re.sub(r"\s+", " ", lab).strip()
            if lab_norm in metric_map:
                return metric_map[lab_norm]
            if lab_norm.lower() in {"mrr@k", "mrr"}:
                return "MRR@k (unique laws)"
            if lab_norm.lower() in {"mean lift", "lift", "lift over prior"}:
                return "lift (prior)"
            return lab

        micro_cells: Dict[int, Dict[str, Dict[str, str]]] = {}

        sec_idx: Optional[int] = None
        for t, ln in enumerate(lines):
            if re.match(r"^###\s+Micro-averaged\s+quality", ln.strip()):
                sec_idx = t
                break

        if sec_idx is not None:
            t = sec_idx + 1
            current_metric: Optional[str] = None
            while t < len(lines):
                ln = lines[t].strip()

                if ln.startswith("## ") and t > sec_idx:
                    break

                # metric label line (bold)
                m_bold = re.match(r"^\*\*(.+?)\*\*", ln)
                if m_bold:
                    current_metric = _canon_metric(m_bold.group(1))
                    t += 1
                    continue

                # parse table if present
                if current_metric and re.match(r"^\|\s*k\s*\|", ln):
                    df, nxt = _parse_markdown_table(lines, t)

                    # Ensure first column is named 'k'
                    cols = [c.strip() for c in df.columns.tolist()]
                    if cols:
                        cols[0] = "k"
                        df.columns = cols

                    for _, r in df.iterrows():
                        k_raw = str(r["k"]).strip()
                        try:
                            k_val = int(float(k_raw))
                        except Exception:
                            continue
                        micro_cells.setdefault(k_val, {})
                        for col in df.columns:
                            if col == "k":
                                continue
                            method = str(col).strip()
                            cell = str(r[col]).strip()
                            micro_cells[k_val].setdefault(method, {})
                            micro_cells[k_val][method][current_metric] = cell

                    t = nxt
                    current_metric = None
                    continue

                t += 1

        if micro_cells:
            metric_order = ["hit@k", "MRR@k (unique laws)", "top1", "majority-acc", "consensus frac", "lift (prior)"]
            for k_val, by_method in sorted(micro_cells.items(), key=lambda x: x[0]):
                rows: List[Dict[str, str]] = []
                for method, met_dict in by_method.items():
                    row: Dict[str, str] = {"Method": method}
                    for met in metric_order:
                        if met in met_dict:
                            row[met] = met_dict[met]
                    rows.append(row)
                micro[k_val] = pd.DataFrame(rows)

    if not delta:
        def _parse_delta_table(header_pat: str, comparison_label: str) -> Dict[int, Dict[str, str]]:
            out: Dict[int, Dict[str, str]] = {}
            idx: Optional[int] = None
            for t, ln in enumerate(lines):
                if re.match(header_pat, ln.strip()):
                    idx = t
                    break
            if idx is None:
                return out
            t = idx + 1
            while t < len(lines) and not re.match(r"^\|\s*k\s*\|", lines[t].lstrip()):
                t += 1
            if t >= len(lines):
                return out
            df, _ = _parse_markdown_table(lines, t)

            col_map = {
                "Δhit@k": "Δhit",
                "Δhit": "Δhit",
                "ΔMRR@k": "ΔMRR_ul",
                "ΔMRR_ul": "ΔMRR_ul",
                "ΔTop-1": "Δtop1",
                "ΔTop1": "Δtop1",
                "ΔMajority-acc": "Δmajority",
                "ΔMajority": "Δmajority",
                "ΔMean cons frac": "Δcons_frac",
                "ΔMean consensus frac": "Δcons_frac",
                "ΔMean lift": "Δlift",
                "Δlift": "Δlift",
            }
            df2 = df.copy()
            df2.columns = [col_map.get(str(c).strip(), str(c).strip()) for c in df2.columns]

            for _, r in df2.iterrows():
                k_raw = str(r.get("k", "")).strip()
                try:
                    k_val = int(float(k_raw))
                except Exception:
                    continue
                out[k_val] = {
                    "Comparison": comparison_label,
                    "Δhit": str(r.get("Δhit", "")).strip(),
                    "ΔMRR_ul": str(r.get("ΔMRR_ul", "")).strip(),
                    "Δtop1": str(r.get("Δtop1", "")).strip(),
                    "Δmajority": str(r.get("Δmajority", "")).strip(),
                    "Δcons_frac": str(r.get("Δcons_frac", "")).strip(),
                    "Δlift": str(r.get("Δlift", "")).strip(),
                }
            return out

        d_idf = _parse_delta_table(r"^###\s+Paired\s+deltas\s*\(KAHM\s+−\s+IDF", "KAHM − IDF")
        d_mb = _parse_delta_table(r"^###\s+Paired\s+deltas\s+vs\s+transformer-query\s+baseline", "KAHM − MB")

        for k_val in sorted(set(d_idf.keys()) | set(d_mb.keys())):
            rows: List[Dict[str, str]] = []
            if k_val in d_idf:
                rows.append(d_idf[k_val])
            if k_val in d_mb:
                rows.append(d_mb[k_val])
            if rows:
                delta[k_val] = pd.DataFrame(rows)

    if routing is None:
        for t, ln in enumerate(lines):
            s = ln.strip()
            if s.startswith("| Method") and ("Coverage" in s and ("Precision" in s or "acc|covered" in s)):
                df, _ = _parse_markdown_table(lines, t)
                routing = df
                break

    return ParsedReport(
        title=title,
        generated_line=generated,
        micro=micro,
        delta=delta,
        compute_init=compute_init,
        compute_paths=compute_paths,
        compute_machine=compute_machine,
        routing=routing,
    )

def micro_long_df(pr: ParsedReport, metric_col: str) -> pd.DataFrame:
    """
    Returns a long DataFrame with columns: k, method, mean, lo, hi for a given metric column label.
    """
    rows = []
    for k, df in sorted(pr.micro.items(), key=lambda t: t[0]):
        if metric_col not in df.columns:
            continue
        for _, r in df.iterrows():
            method = str(r["Method"])
            mean, lo, hi = _parse_cell(str(r[metric_col]))
            rows.append({"k": k, "method": method, "mean": mean, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def delta_long_df(pr: ParsedReport, metric_key: str) -> pd.DataFrame:
    """
    metric_key: one of {'Δhit','ΔMRR_ul','Δtop1','Δmajority','Δcons_frac','Δlift'} columns in delta tables.
    """
    rows = []
    for k, df in sorted(pr.delta.items(), key=lambda t: t[0]):
        if metric_key not in df.columns:
            continue
        for _, r in df.iterrows():
            comp = str(r["Comparison"])
            mean, lo, hi = _parse_cell(str(r[metric_key]).replace("+", ""))
            rows.append({"k": k, "comparison": comp, "mean": mean, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


# -----------------------------
# Retrieval stack
# -----------------------------
@st.cache_data(show_spinner=False)
def load_corpus(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    if "sentence_id" not in df.columns or "law_type" not in df.columns:
        raise ValueError("Corpus parquet must contain columns: sentence_id, law_type")
    return df


@st.cache_data(show_spinner=False)
def load_npz_bundle(npz_path: str) -> Dict[str, np.ndarray]:
    """Load an embedding bundle from .npz with flexible key detection.

    Expected content:
      - embeddings array of shape (N, D)
      - sentence id array of shape (N,)

    Different pipelines save different key names; we accept common variants and
    also fall back to shape-based inference (2D array = embeddings, 1D array = ids).
    """
    z = np.load(npz_path, allow_pickle=False)
    keys = list(z.files)

    def _pick(candidates: List[str]) -> Optional[str]:
        for k in candidates:
            if k in z.files:
                return k
        return None

    emb_key = _pick(["emb", "embeddings", "embedding", "E", "X", "vecs", "vectors"])
    ids_key = _pick(["sentence_ids", "sentence_id", "sent_ids", "ids", "idx", "indices"])

    # Fall back to positional keys if present (np.savez without names -> arr_0, arr_1, ...)
    if emb_key is None and "arr_0" in z.files:
        emb_key = "arr_0"
    if ids_key is None and "arr_1" in z.files:
        ids_key = "arr_1"

    # Shape-based inference if still missing
    if emb_key is None or ids_key is None:
        arrays: List[Tuple[str, np.ndarray]] = [(k, z[k]) for k in z.files]
        # pick embeddings as first 2D array
        if emb_key is None:
            for k, a in arrays:
                if np.asarray(a).ndim == 2:
                    emb_key = k
                    break
        # pick ids as 1D array matching embeddings length
        if ids_key is None and emb_key is not None:
            n = int(np.asarray(z[emb_key]).shape[0]) if np.asarray(z[emb_key]).ndim == 2 else -1
            for k, a in arrays:
                aa = np.asarray(a)
                if aa.ndim == 1 and n > 0 and int(aa.shape[0]) == n:
                    ids_key = k
                    break

    if emb_key is None or ids_key is None:
        raise ValueError(
            "Could not infer embedding/id arrays from NPZ. "
            f"Found keys={keys}. "
            "Expected embeddings under one of {emb, embeddings, embedding, E, X, vecs, vectors, arr_0} "
            "and ids under one of {sentence_ids, sentence_id, sent_ids, ids, idx, indices, arr_1}. "
            "If your NPZ uses custom keys, ensure it contains exactly one 2D array (embeddings) and one 1D array (ids) "
            "with matching length."
        )

    emb = np.asarray(z[emb_key], dtype=np.float32)
    sids = np.asarray(z[ids_key], dtype=np.int64).reshape(-1)

    if emb.ndim != 2:
        raise ValueError(f"Embeddings array must be 2D (N,D). Got key='{emb_key}' with shape {emb.shape}.")
    if sids.ndim != 1:
        raise ValueError(f"Sentence id array must be 1D (N,). Got key='{ids_key}' with shape {sids.shape}.")
    if emb.shape[0] != sids.shape[0]:
        raise ValueError(
            f"Embeddings and sentence_ids must align in length. Got emb.shape[0]={emb.shape[0]} vs ids={sids.shape[0]} "
            f"(emb_key='{emb_key}', ids_key='{ids_key}')."
        )

    return {"emb": emb, "sentence_ids": sids}


def _l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return (X / n).astype(np.float32)


@st.cache_data(show_spinner=False)
def align_corpus_and_embeddings(
    corpus_parquet: str,
    mb_npz: Optional[str],
    idf_npz: Optional[str],
) -> Dict[str, Any]:
    """
    Align df / mb embeddings / idf embeddings by common sentence_ids.
    Returns dict with:
      df_aligned, sentence_ids, mb_emb (or None), idf_emb (or None)
    """
    df = load_corpus(corpus_parquet)
    s_df = df["sentence_id"].astype(np.int64).to_numpy()

    mb_bundle = load_npz_bundle(mb_npz) if mb_npz else None
    idf_bundle = load_npz_bundle(idf_npz) if idf_npz else None

    sets = [set(s_df.tolist())]
    if mb_bundle is not None:
        sets.append(set(mb_bundle["sentence_ids"].tolist()))
    if idf_bundle is not None:
        sets.append(set(idf_bundle["sentence_ids"].tolist()))
    common = sorted(set.intersection(*sets))
    if not common:
        raise ValueError("No common sentence_ids across provided corpus/embeddings.")

    common_ids = np.asarray(common, dtype=np.int64)

    pos_df = {int(s): i for i, s in enumerate(s_df.tolist())}
    df_aligned = df.iloc[[pos_df[int(s)] for s in common_ids.tolist()]].reset_index(drop=True)

    def _subset(bundle: Dict[str, np.ndarray]) -> np.ndarray:
        pos = {int(s): i for i, s in enumerate(bundle["sentence_ids"].tolist())}
        idx = np.asarray([pos[int(s)] for s in common_ids.tolist()], dtype=np.int64)
        return bundle["emb"][idx]

    mb_emb = _subset(mb_bundle) if mb_bundle is not None else None
    idf_emb = _subset(idf_bundle) if idf_bundle is not None else None

    if mb_emb is not None:
        mb_emb = _l2_normalize_rows(mb_emb)
    if idf_emb is not None:
        idf_emb = _l2_normalize_rows(idf_emb)

    return {
        "df": df_aligned,
        "sentence_ids": common_ids,
        "mb_emb": mb_emb,
        "idf_emb": idf_emb,
    }


@st.cache_resource(show_spinner=False)
def build_faiss_index(emb: np.ndarray):
    # NOTE: FAISS has incomplete/unstable type stubs; Pylance may mis-type `Index.add`
    # and report "missing argument x" even though runtime is correct. We cast to `Any`
    # so static checking does not block usage.
    import faiss  # type: ignore
    from typing import Any, cast

    if emb.ndim != 2:
        raise ValueError(f"Embeddings must be 2D, got {emb.shape}")
    dim = int(emb.shape[1])

    idx = cast(Any, faiss.IndexFlatIP(dim))
    idx.add(emb.astype(np.float32))
    return idx


@st.cache_resource(show_spinner=False)
def load_idf_svd_pipeline(joblib_path: str):
    import joblib  # type: ignore

    return joblib.load(joblib_path)


@st.cache_resource(show_spinner=False)
def load_mixedbread_model(model_name: str, device: str, dim: int):
    from sentence_transformers import SentenceTransformer  # type: ignore

    return SentenceTransformer(model_name, device=device, truncate_dim=int(dim))


@st.cache_resource(show_spinner=False)
def load_kahm_embedder(
    idf_svd_model_path: str,
    kahm_query_model_dir: str,
    kahm_mode: str,
    batch_size: int,
    show_progress: bool,
):
    """
    Loads KahmQueryEmbedder from kahm_inference_embedder.py (preferred), with a file-path fallback.
    """
    try:
        from kahm_inference_embedder import KahmQueryEmbedder  # type: ignore
    except Exception:
        import importlib.util
        p = os.path.join(os.getcwd(), "kahm_inference_embedder.py")
        if not os.path.exists(p):
            raise ImportError(
                "Could not import KahmQueryEmbedder. Ensure kahm_inference_embedder.py is on PYTHONPATH "
                "or in the current working directory."
            )
        spec = importlib.util.spec_from_file_location("kahm_inference_embedder", p)
        if spec is None or spec.loader is None:
            raise ImportError("Could not load kahm_inference_embedder.py via importlib.")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        KahmQueryEmbedder = getattr(mod, "KahmQueryEmbedder")

    return KahmQueryEmbedder(
        idf_svd_model_path=str(idf_svd_model_path),
        kahm_query_model_dir=str(kahm_query_model_dir),
        kahm_mode=str(kahm_mode),
        batch_size=int(batch_size),
        materialize_classifier=True,
        cache_cluster_centers=True,
        tie_break="first",
        show_progress=bool(show_progress),
    )


def embed_query(
    query: str,
    method: str,
    *,
    query_prefix: str,
    mb_model=None,
    idf_pipe=None,
    kahm_embedder=None,
    target_dim: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Returns (1,dim) normalized embedding and diagnostics dict.
    """
    diag: Dict[str, Any] = {}
    t0 = time.perf_counter()

    if method == "Mixedbread (true)":
        if mb_model is None:
            raise ValueError("Mixedbread model not loaded.")
        vec = mb_model.encode(
            [query_prefix + query],
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        ).astype(np.float32)
        vec = _l2_normalize_rows(vec)
        diag["query_embed_ms"] = (time.perf_counter() - t0) * 1000.0

    elif method == "IDF–SVD":
        if idf_pipe is None:
            raise ValueError("IDF–SVD pipeline not loaded.")
        vec = idf_pipe.transform([query])
        vec = np.asarray(vec, dtype=np.float32)
        if vec.ndim != 2:
            raise ValueError(f"IDF–SVD transform output must be 2D, got {vec.shape}")
        vec = _l2_normalize_rows(vec)
        diag["query_embed_ms"] = (time.perf_counter() - t0) * 1000.0

    elif method == "KAHM(query→MB corpus)":
        if kahm_embedder is None:
            raise ValueError("KAHM embedder not loaded.")
        Y, chosen, score, names = kahm_embedder.embed([query])
        vec = np.asarray(Y, dtype=np.float32)
        vec = _l2_normalize_rows(vec)
        diag["query_embed_ms"] = (time.perf_counter() - t0) * 1000.0
        diag["chosen_submodel"] = str(chosen[0]) if isinstance(chosen, (list, tuple, np.ndarray)) else str(chosen)
        try:
            diag["submodel_score"] = float(score[0]) if hasattr(score, "__len__") else float(score)
        except Exception:
            pass

    else:
        raise ValueError(f"Unknown method: {method}")

    if target_dim is not None and vec.shape[1] != int(target_dim):
        # be permissive: truncate if larger
        if vec.shape[1] > int(target_dim):
            vec = vec[:, : int(target_dim)]
        else:
            raise ValueError(f"Query dim {vec.shape[1]} < target dim {target_dim}")

    return vec.astype(np.float32), diag


def faiss_search(index, q: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray, float]:
    import faiss  # type: ignore

    t0 = time.perf_counter()
    scores, ids = index.search(q.astype(np.float32), int(top_k))
    ms = (time.perf_counter() - t0) * 1000.0
    return ids[0], scores[0], ms


# -----------------------------
# Query autocomplete model
# -----------------------------
def _tokenize(s: str) -> List[str]:
    s = s.strip().lower()
    # Keep German diacritics; split on whitespace/punct.
    toks = re.findall(r"[a-zäöüß0-9]+", s, flags=re.IGNORECASE)
    return toks


@st.cache_data(show_spinner=False)
def load_queries_from_jsonl(path: str) -> List[str]:
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            obj = json.loads(ln)
            if isinstance(obj, dict):
                t = obj.get("query_text") or obj.get("query") or obj.get("text") or ""
                if t:
                    texts.append(str(t))
    return texts


@st.cache_data(show_spinner=False)
def load_queries_from_module(module_attr: str) -> List[str]:
    import importlib

    if "." not in module_attr:
        raise ValueError("module_attr must be module.ATTR, e.g. query_set.TRAIN_QUERY_SET")
    mod, attr = module_attr.rsplit(".", 1)
    m = importlib.import_module(mod)
    qs = getattr(m, attr)
    out = []
    for q in list(qs):
        if isinstance(q, dict):
            t = q.get("query_text") or q.get("query") or q.get("text") or ""
        elif isinstance(q, (list, tuple)):
            t = q[1] if len(q) > 1 else (q[0] if q else "")
        else:
            t = getattr(q, "query_text", "") or getattr(q, "query", "") or getattr(q, "text", "")
        if t:
            out.append(str(t))
    return out


@st.cache_data(show_spinner=False)
def build_ngram_model(texts: List[str], n: int = 3) -> Dict[Tuple[str, ...], Dict[str, int]]:
    """
    Builds a simple n-gram next-token model: context (n-1 tokens) -> next token counts.
    """
    ctx_to_counts: Dict[Tuple[str, ...], Dict[str, int]] = {}
    for t in texts:
        toks = _tokenize(t)
        if len(toks) < 2:
            continue
        # pad
        pad = ["<s>"] * (n - 1)
        toks2 = pad + toks + ["</s>"]
        for i in range(n - 1, len(toks2)):
            ctx = tuple(toks2[i - (n - 1) : i])
            nxt = toks2[i]
            d = ctx_to_counts.setdefault(ctx, {})
            d[nxt] = d.get(nxt, 0) + 1
    return ctx_to_counts


def suggest_next_words(model: Dict[Tuple[str, ...], Dict[str, int]], prefix: str, top: int = 8) -> List[str]:
    toks = _tokenize(prefix)
    ctx = tuple((["<s>", "<s>"] + toks)[-2:])  # trigram context
    cand = model.get(ctx)
    if not cand:
        # backoff: use last token as context with wildcard second token
        if len(toks) >= 1:
            ctx2 = ("<s>", toks[-1])
            cand = model.get(ctx2)
    if not cand:
        return []
    items = sorted(cand.items(), key=lambda kv: kv[1], reverse=True)
    out = []
    for w, _c in items:
        if w in ("</s>", "<s>"):
            continue
        out.append(w)
        if len(out) >= top:
            break
    return out

def suggest_token_completions(vocab: Iterable[str], prefix: str, top: int = 8) -> List[str]:
    """
    If the user is currently typing a partial token, suggest completions from the query vocabulary.
    Example: "... künd" -> ["kündigung", "kündigungsfrist", ...]
    """
    prefix = prefix.strip().lower()
    if not prefix or len(prefix) < 2:
        return []
    cand = [w for w in vocab if w.startswith(prefix) and w != prefix]
    cand = sorted(cand, key=lambda w: (len(w), w))  # short-first for typing UX
    return cand[:top]





# -----------------------------
# Streamlit state update helpers (callbacks)
# -----------------------------
def _set_session_key(key: str, value: str) -> None:
    """Set a session_state key from a widget callback."""
    st.session_state[key] = value


def _copy_session_key(src_key: str, dst_key: str) -> None:
    """Copy one session_state key to another (callback-safe)."""
    st.session_state[dst_key] = st.session_state.get(src_key, "")


def _apply_suggestion_to_widget(text_key: str, word: str, is_completion: bool) -> None:
    """Apply a suggestion token to a text widget key (append or replace last token)."""
    cur = (st.session_state.get(text_key) or "").rstrip()
    if is_completion:
        st.session_state[text_key] = re.sub(
            r"[A-Za-zÄÖÜäöüß0-9]+$",
            word,
            cur,
            flags=re.IGNORECASE,
        )
    else:
        st.session_state[text_key] = (cur + " " + word).strip()


def _apply_selectbox_pick_to_widget(select_key: str, text_key: str, completion_flag_key: str) -> None:
    """When a selectbox changes, apply its pick to a text widget key."""
    picked = st.session_state.get(select_key, "—")
    if not picked or picked == "—":
        return
    is_completion = bool(st.session_state.get(completion_flag_key, False))
    _apply_suggestion_to_widget(text_key, str(picked), is_completion)
    # reset for nicer UX (safe in callback)
    st.session_state[select_key] = "—"

# -----------------------------
# Sidebar configuration
# -----------------------------
st.title("KAHM Embeddings Dashboard")
st.caption("Interactive evidence + live Austrian-law retrieval demo (KAHM query adapter → Mixedbread corpus space).")

with st.sidebar:
    st.markdown("### Data sources")

    default_report = "kahm_evaluation_report.md" if Path("kahm_evaluation_report.md").exists() else ""
    report_path = st.text_input("Evaluation report (Markdown)", value=default_report, help="Path to kahm_evaluation_report.md")

    st.markdown("---")
    st.markdown("### Retrieval artifacts")

    corpus_parquet = st.text_input("Corpus parquet", value="ris_sentences.parquet")
    mb_npz = st.text_input("Mixedbread corpus embeddings (NPZ)", value="embedding_index.npz")
    idf_npz = st.text_input("IDF–SVD corpus embeddings (NPZ)", value="embedding_index_idf_svd.npz")

    st.markdown("---")
    st.markdown("### Models (for online query embedding)")

    query_prefix = st.text_input("Query prefix", value="query: ")
    mixedbread_model_name = st.text_input("Mixedbread model name", value="mixedbread-ai/deepset-mxbai-embed-de-large-v1")
    device = st.selectbox("Device", options=["cpu", "cuda"], index=0)

    idf_svd_model_path = st.text_input("IDF–SVD pipeline (joblib)", value="idf_svd_model.joblib")
    kahm_model_dir = st.text_input("KAHM query model dir", value="kahm_query_regressors_by_law")
    kahm_mode = st.selectbox("KAHM mode", options=["soft", "hard"], index=0)
    kahm_batch = st.number_input("KAHM batch size", min_value=1, max_value=4096, value=1024, step=1)

    st.markdown("---")
    st.markdown("### Query autocomplete (typeahead + next-word)")
    q_source = st.radio("Query source", options=["Python module", "JSONL files"], horizontal=True)

    # Initialize to avoid "possibly unbound" static-type warnings
    train_attr: str = ""
    test_attr: str = ""
    train_jsonl: str = ""
    test_jsonl: str = ""

    if q_source == "Python module":
        train_attr = st.text_input("Train query set", value="query_set.TRAIN_QUERY_SET")
        test_attr = st.text_input("Test query set", value="query_set.TEST_QUERY_SET")
        use_split = st.selectbox("Autocomplete corpus", options=["train+test", "train only", "test only"], index=0)
    else:
        train_jsonl = st.text_input("Train JSONL path", value="train.jsonl")
        test_jsonl = st.text_input("Test JSONL path", value="test.jsonl")
        use_split = st.selectbox("Autocomplete corpus", options=["train+test", "train only", "test only"], index=0)

    st.markdown("---")
    st.markdown("### Demo settings")
    default_topk = st.slider("Top-k to display", min_value=3, max_value=50, value=10, step=1)
    show_debug = st.checkbox("Show debug/provenance", value=False)


# -----------------------------
# Autocomplete corpus (shared across tabs)
# -----------------------------
autocomplete_texts: List[str] = []
autocomplete_vocab: List[str] = []
autocomplete_model: Dict[Tuple[str, ...], Dict[str, int]] = {}
_autocomplete_err = None

try:
    if q_source == "Python module":
        train_q = load_queries_from_module(train_attr) if train_attr else []
        test_q = load_queries_from_module(test_attr) if test_attr else []
    else:
        train_q = load_queries_from_jsonl(train_jsonl) if train_jsonl and Path(train_jsonl).exists() else []
        test_q = load_queries_from_jsonl(test_jsonl) if test_jsonl and Path(test_jsonl).exists() else []

    if use_split == "train+test":
        autocomplete_texts = train_q + test_q
    elif use_split == "train only":
        autocomplete_texts = train_q
    else:
        autocomplete_texts = test_q

    if autocomplete_texts:
        autocomplete_model = build_ngram_model(autocomplete_texts, n=3)
        # vocab for token-completion suggestions
        vocab_set = set()
        for q in autocomplete_texts[:20000]:
            vocab_set.update(_tokenize(q))
        autocomplete_vocab = sorted(vocab_set)
    else:
        autocomplete_model = {}
        autocomplete_vocab = []
except Exception as e:
    _autocomplete_err = str(e)

# Shared query state between tabs
if "retrieval_query" not in st.session_state:
    st.session_state["retrieval_query"] = ""


# -----------------------------
# Load report + show results
# -----------------------------
report_md = None
pr = None
if report_path and Path(report_path).exists():
    _rp = Path(report_path)
    _st = _rp.stat()
    report_md = load_report_md(
        report_path,
        int(getattr(_st, "st_mtime_ns", int(_st.st_mtime * 1e9))),
        int(_st.st_size),
    )
    pr = parse_report(report_md)

tabs = st.tabs(["📊 Results", "🔎 Retrieval demo", "🧠 Query assistant", "📄 Report (raw)"])


def render_kpis(pr: ParsedReport, k: int):
    df = pr.micro.get(k)
    if df is None:
        st.warning(f"No micro table found for k={k} in report.")
        return

    # Find KAHM row
    def _row(method: str) -> Optional[pd.Series]:
        for _, r in df.iterrows():
            if str(r["Method"]).strip() == method:
                return r
        return None

    r_kahm = _row("KAHM(query→MB corpus)")
    r_idf = _row("IDF–SVD")
    r_mb = _row("Mixedbread (true)")

    if r_kahm is None:
        st.warning("KAHM row not found in report.")
        return

    top1, *_ = _parse_cell(str(r_kahm.get("top1", "")))
    mrr, *_ = _parse_cell(str(r_kahm.get("MRR@k (unique laws)", r_kahm.get("MRR@k", ""))))
    hit, *_ = _parse_cell(str(r_kahm.get("hit@k", "")))
    lift, *_ = _parse_cell(str(r_kahm.get("lift (prior)", "")))

    # deltas if available
    d = pr.delta.get(k)
    d_kahm_idf = None
    d_kahm_mb = None
    if d is not None and "Comparison" in d.columns:
        for _, rr in d.iterrows():
            comp = str(rr["Comparison"])
            if "KAHM" in comp and "IDF" in comp:
                d_kahm_idf = rr
            if "KAHM" in comp and "MB" in comp:
                d_kahm_mb = rr

    def _delta_str(row, col):
        if row is None or col not in row:
            return "—"
        mean, lo, hi = _parse_cell(str(row[col]).replace("+", ""))
        if np.isnan(lo) or np.isnan(hi):
            return f"{mean:+.3f}"
        return f"{mean:+.3f} [{lo:+.3f}, {hi:+.3f}]"

    st.markdown(f"#### Snapshot at k={k}  <span class='badge'>from report</span>", unsafe_allow_html=True)
    st.markdown(
        f"""
<div class="kpi-grid">
  <div class="kpi">
    <div class="label">KAHM Top-1</div>
    <div class="value">{top1:.3f}</div>
    <div class="delta">Δ vs IDF: {_delta_str(d_kahm_idf, "Δtop1")}<br/>Δ vs MB: {_delta_str(d_kahm_mb, "Δtop1")}</div>
  </div>
  <div class="kpi">
    <div class="label">KAHM MRR@k (unique laws)</div>
    <div class="value">{mrr:.3f}</div>
    <div class="delta">Δ vs IDF: {_delta_str(d_kahm_idf, "ΔMRR_ul")}<br/>Δ vs MB: {_delta_str(d_kahm_mb, "ΔMRR_ul")}</div>
  </div>
  <div class="kpi">
    <div class="label">KAHM Hit@k</div>
    <div class="value">{hit:.3f}</div>
    <div class="delta">Δ vs IDF: {_delta_str(d_kahm_idf, "Δhit")}<br/>Δ vs MB: {_delta_str(d_kahm_mb, "Δhit")}</div>
  </div>
  <div class="kpi">
    <div class="label">KAHM Lift over prior</div>
    <div class="value">{lift:.1f}×</div>
    <div class="delta small-note">Mean lift (can be heavy-tailed for rare labels)</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # quick table
    st.markdown("##### Micro-average table (as reported)")
    st.dataframe(df, width='stretch', hide_index=True)


with tabs[0]:
    if pr is None:
        st.info("Provide a valid report path in the sidebar to render the results dashboard.")
    else:
        st.markdown(f"**{pr.title}**")
        if pr.generated_line:
            st.caption(pr.generated_line)

        ks = sorted(pr.micro.keys())
        if not ks:
            st.warning("No micro-average sections found in the report.")
        else:
            k_sel = st.selectbox("Select cutoff k", options=ks, index=min(ks.index(10) if 10 in ks else 0, len(ks)-1))
            render_kpis(pr, int(k_sel))

            st.markdown("---")
            st.markdown("#### Quality curves across k")
            metric_choice = st.selectbox(
                "Metric",
                options=["hit@k", "MRR@k (unique laws)", "top1", "majority-acc", "consensus frac", "lift (prior)"],
                index=0,
            )
            df_long = micro_long_df(pr, metric_choice)
            if df_long.empty:
                st.warning(f"Metric '{metric_choice}' not found in micro tables.")
            else:
                fig = px.line(df_long, x="k", y="mean", color="method", markers=True)
                fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, width="stretch")

            st.markdown("---")
            st.markdown("#### Compute profile (from report)")
            c1, c2 = st.columns(2)
            with c1:
                if pr.compute_paths is None:
                    st.info("No per-query compute table found in this report file (some report variants omit compute profiling tables).")
                else:
                    df_raw = pr.compute_paths.copy()
                    df_plot = df_raw.copy()

                    def _to_ms_num(s: object) -> float:
                        txtv = str(s).strip().lower()
                        if txtv in {"", "n/a", "na", "nan", "none", "—", "-"}:
                            return float("nan")
                        txtv = txtv.replace("milliseconds", "ms").replace("millisecond", "ms")
                        m = re.search(r"([+-]?\d+(?:\.\d+)?)", txtv)
                        if not m:
                            return float("nan")
                        val = float(m.group(1))
                        if " s" in f" {txtv}" and "ms" not in txtv:
                            return val * 1000.0
                        return val

                    candidate_cols = [
                        "Total online / q",
                        "Observed step sum / q",
                        "Query embed / q",
                        "FAISS search / q",
                    ]
                    numeric_cols: List[str] = []
                    for col in candidate_cols:
                        if col in df_plot.columns:
                            df_plot[col] = df_plot[col].map(_to_ms_num)
                            if pd.to_numeric(df_plot[col], errors="coerce").notna().any():
                                numeric_cols.append(col)

                    if numeric_cols and "Path" in df_plot.columns:
                        metric_col = st.selectbox(
                            "Compute metric (per query, ms)",
                            options=numeric_cols,
                            index=0,
                            key="compute_metric_sel",
                        )
                        fig = px.bar(df_plot, x="Path", y=metric_col, text=metric_col, color="Query source" if "Query source" in df_plot.columns else None)
                        fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
                        st.plotly_chart(fig, width="stretch")
                        if "Total online / q" in df_plot.columns and "Path" in df_plot.columns:
                            online = df_plot[["Path", "Total online / q"]].dropna()
                            if len(online) >= 2:
                                online = online.sort_values("Total online / q", ascending=True).reset_index(drop=True)
                                fastest = online.iloc[0]
                                slowest = online.iloc[-1]
                                if float(fastest["Total online / q"]) > 0:
                                    speedup = float(slowest["Total online / q"]) / float(fastest["Total online / q"])
                                    st.caption(f"Fastest reported online path in this run: {fastest['Path']} ({fastest['Total online / q']:.3f} ms/q). Slowest/fastest ratio: {speedup:.2f}×.")
                    else:
                        st.info("Per-query timing table found, but no numeric timing columns could be parsed.")

                    st.markdown("##### Per-query timing table")
                    st.dataframe(df_raw, width='stretch', hide_index=True)

                    if pr.compute_init is not None:
                        st.markdown("##### Measured components (wall-clock)")
                        st.dataframe(pr.compute_init, width='stretch', hide_index=True)

            with c2:
                if pr.compute_machine is not None:
                    st.markdown("##### Machine profile")
                    st.dataframe(pr.compute_machine, width='stretch', hide_index=True)
                    if {"Field", "Value"}.issubset(set(pr.compute_machine.columns)):
                        _mp = {str(r["Field"]): str(r["Value"]) for _, r in pr.compute_machine.iterrows()}
                        accelerator = _mp.get("Accelerator name") or _mp.get("Accelerator type")
                        cpu = _mp.get("CPU logical cores")
                        ram = _mp.get("RAM total")
                        pills = [p for p in [accelerator, (f"{cpu} logical cores" if cpu else None), ram] if p]
                        if pills:
                            st.caption(" | ".join(pills))

                if pr.routing is None:
                    st.info("No routing table found.")
                else:
                    st.markdown("##### Majority-vote routing recommendation")
                    st.dataframe(pr.routing, width='stretch', hide_index=True)
                    st.caption("If τ* = 0.00 for a method, it indicates the confidence signal didn't improve acc|covered under the chosen coverage constraint in that run.")

        if show_debug and report_path and Path(report_path).exists():
            st.markdown("---")
            st.markdown("#### Provenance")
            st.code(f"report_sha256={_sha256_file(report_path)}")


# -----------------------------
# Retrieval demo
# -----------------------------
with tabs[1]:
    st.markdown("### Live retrieval demo")
    st.write("Embed your query and retrieve the most relevant Austrian-law sentences (sentence_id, law_type, text).")

    # Attempt to load aligned corpus + embeddings
    retrieval_ready = False
    aligned: Optional[Dict[str, Any]] = None
    err: Optional[str] = None
    try:
        if corpus_parquet and Path(corpus_parquet).exists():
            mb_npz_ok = mb_npz and Path(mb_npz).exists()
            idf_npz_ok = idf_npz and Path(idf_npz).exists()
            if mb_npz_ok or idf_npz_ok:
                aligned = align_corpus_and_embeddings(
                    corpus_parquet=corpus_parquet,
                    mb_npz=mb_npz if mb_npz_ok else None,
                    idf_npz=idf_npz if idf_npz_ok else None,
                )
                retrieval_ready = True
            else:
                err = "Provide at least one embeddings NPZ (Mixedbread corpus or IDF–SVD corpus)."
        else:
            err = "Provide a valid corpus parquet path."
    except Exception as e:
        err = str(e)

    if (not retrieval_ready) or (aligned is None):
        st.warning(err or "Retrieval artifacts not loaded yet.")
    else:
        assert aligned is not None
        df_c = aligned["df"]
        mb_emb = aligned["mb_emb"]
        idf_emb = aligned["idf_emb"]

        method_options = []
        if mb_emb is not None:
            method_options.append("KAHM(query→MB corpus)")
            method_options.append("Mixedbread (true)")
        if idf_emb is not None:
            method_options.append("IDF–SVD")

        colA, colB = st.columns([2, 1])
        with colA:
            st.markdown("#### Query")
            query_text = st.text_area(
                "Enter a query (German)",
                key="retrieval_query",
                height=90,
                placeholder="z.B. Was sind die Voraussetzungen für eine Kündigung in Österreich?"
            )

            # Optional: professional autocomplete/prediction from TRAIN/TEST queries
            if _autocomplete_err:
                st.caption(f"Autocomplete unavailable: {_autocomplete_err}")
            elif autocomplete_model and autocomplete_vocab:
                st.markdown("**Suggestions**")
                last = (query_text or "").rstrip()

                # If the user is in the middle of typing a token (no trailing space),
                # offer token-completion; otherwise offer next-word suggestions.
                is_mid_token = (len(last) > 0) and (not (query_text or "").endswith(" "))
                m_tok = re.search(r"[A-Za-zÄÖÜäöüß0-9]+$", last)
                partial = m_tok.group(0).lower() if (m_tok and is_mid_token) else ""

                completions = suggest_token_completions(autocomplete_vocab, partial, top=8) if partial else []
                next_words = suggest_next_words(autocomplete_model, last, top=8)

                candidates = completions if completions else next_words
                if candidates:
                    cols = st.columns(min(8, len(candidates)))
                    for i, w in enumerate(candidates[:8]):
                        with cols[i]:
                            st.button(
                                w,
                                key=f"sugg_retr_{i}",
                                use_container_width=True,
                                on_click=_apply_suggestion_to_widget,
                                args=("retrieval_query", w, bool(completions)),
                            )
                else:
                    st.caption("No suggestions for the current context.")
        with colB:
            st.markdown("#### Retrieval settings")
            method = st.selectbox("Query embedding method", options=method_options, index=0)
            top_k = st.slider("Retrieve top-k", min_value=3, max_value=50, value=int(default_topk), step=1)
            run = st.button("Retrieve", type="primary", use_container_width=True)

        # Build indices lazily
        idx_mb = build_faiss_index(mb_emb) if mb_emb is not None else None
        idx_idf = build_faiss_index(idf_emb) if idf_emb is not None else None

        # Load models lazily
        dim_mb = int(mb_emb.shape[1]) if mb_emb is not None else None
        dim_idf = int(idf_emb.shape[1]) if idf_emb is not None else None

        mb_model = None
        idf_pipe = None
        kahm_embedder = None

        if run:
            if not query_text.strip():
                st.error("Please enter a query.")
            else:
                # Select index + model stack
                if method in ("KAHM(query→MB corpus)", "Mixedbread (true)"):
                    if idx_mb is None or dim_mb is None:
                        st.error("Mixedbread corpus embeddings are required for this method.")
                        st.stop()
                    target_dim = dim_mb
                    index = idx_mb
                else:
                    if idx_idf is None or dim_idf is None:
                        st.error("IDF–SVD corpus embeddings are required for this method.")
                        st.stop()
                    target_dim = dim_idf
                    index = idx_idf

                # load query encoders
                try:
                    if method == "Mixedbread (true)":
                        mb_model = load_mixedbread_model(mixedbread_model_name, device=device, dim=target_dim)
                    elif method == "IDF–SVD":
                        idf_pipe = load_idf_svd_pipeline(idf_svd_model_path)
                    elif method == "KAHM(query→MB corpus)":
                        kahm_embedder = load_kahm_embedder(
                            idf_svd_model_path=idf_svd_model_path,
                            kahm_query_model_dir=kahm_model_dir,
                            kahm_mode=kahm_mode,
                            batch_size=int(kahm_batch),
                            show_progress=False,
                        )
                    else:
                        raise ValueError("Unknown method")
                except Exception as e:
                    st.error(f"Failed to load models for {method}: {e}")
                    st.stop()

                # embed + search
                try:
                    q_vec, diag = embed_query(
                        query_text.strip(),
                        method,
                        query_prefix=query_prefix,
                        mb_model=mb_model,
                        idf_pipe=idf_pipe,
                        kahm_embedder=kahm_embedder,
                        target_dim=target_dim,
                    )
                    ids, scores, search_ms = faiss_search(index, q_vec, top_k=top_k)
                except Exception as e:
                    st.error(f"Embedding/retrieval failed: {e}")
                    st.stop()

                # assemble results
                rows = []
                for rank, (row_idx, sc) in enumerate(zip(ids.tolist(), scores.tolist()), start=1):
                    if row_idx < 0:
                        continue
                    rec = df_c.iloc[int(row_idx)]
                    # attempt to find a text field
                    txt = ""
                    for cand in ("text", "sentence", "content", "paragraph", "body"):
                        if cand in df_c.columns:
                            txt = str(rec.get(cand, ""))
                            break
                    rows.append(
                        {
                            "rank": rank,
                            "score": float(sc),
                            "sentence_id": int(rec["sentence_id"]),
                            "law_type": str(rec["law_type"]),
                            "text": (txt[:400] + "…") if len(txt) > 401 else txt,
                        }
                    )
                out_df = pd.DataFrame(rows)

                # headline + diagnostics
                st.markdown("#### Results")
                law_counts = out_df["law_type"].value_counts().rename_axis("law_type").reset_index(name="count")
                pred_law = str(law_counts["law_type"].iloc[0]) if len(law_counts) else "—"
                conf = float(law_counts["count"].iloc[0] / max(1, len(out_df))) if len(law_counts) else float("nan")

                h1, h2, h3, h4 = st.columns(4)
                h1.metric("Predicted law_type (majority)", pred_law)
                h2.metric("Predominance", f"{conf:.2f}")
                h3.metric("Query embed time", f"{diag.get('query_embed_ms', float('nan')):.1f} ms")
                h4.metric("FAISS search time", f"{search_ms:.1f} ms")

                if method == "KAHM(query→MB corpus)" and "chosen_submodel" in diag:
                    st.caption(f"KAHM chosen sub-model: `{diag['chosen_submodel']}` (score: {diag.get('submodel_score', '—')})")

                st.dataframe(out_df, width="stretch", hide_index=True)

                # law_type distribution
                st.markdown("##### law_type distribution in top-k")
                fig = px.bar(law_counts, x="law_type", y="count")
                fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, width="stretch")

                if show_debug:
                    st.markdown("##### Debug")
                    st.code({"method": method, "target_dim": target_dim, **diag})


# -----------------------------
# Query assistant (autocomplete)
# -----------------------------
with tabs[2]:
    st.markdown("### Query assistant")
    st.write("Type-ahead examples + next-word prediction derived from your TRAIN/TEST query corpora.")

    if _autocomplete_err:
        st.warning(f"Could not load queries for autocomplete: {_autocomplete_err}")
    elif not autocomplete_texts:
        st.info("No queries loaded yet. Provide TRAIN/TEST sources in the sidebar.")
    else:
        st.caption(f"Loaded {len(autocomplete_texts):,} queries for autocomplete ({use_split}).")

        model: Dict[Tuple[str, ...], Dict[str, int]] = autocomplete_model
        vocab: List[str] = autocomplete_vocab

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("#### Compose a query with suggestions")

            if "assistant_query" not in st.session_state:
                st.session_state["assistant_query"] = st.session_state.get("retrieval_query", "")

            st.text_input(
                "Draft query",
                key="assistant_query",
                placeholder="Start typing…",
            )

            draft = st.session_state.get("assistant_query", "")
            last = draft.rstrip()
            is_mid_token = (len(last) > 0) and (not draft.endswith(" "))
            m_tok = re.search(r"[A-Za-zÄÖÜäöüß0-9]+$", last)
            partial = m_tok.group(0).lower() if (m_tok and is_mid_token) else ""

            completions = suggest_token_completions(vocab, partial, top=10) if partial else []
            next_words = suggest_next_words(model, last, top=10)
            candidates = completions if completions else next_words

            st.markdown("**Suggestions**")
            if candidates:
                cols = st.columns(min(6, len(candidates)))
                for i, w in enumerate(candidates[:6]):
                    with cols[i]:
                        st.button(
                            w,
                            key=f"sugg_asst_{i}",
                            use_container_width=True,
                            on_click=_apply_suggestion_to_widget,
                            args=("assistant_query", w, bool(completions)),
                        )

                if len(candidates) > 6:
                    st.session_state["assistant_completion_mode"] = bool(completions)
                    if "assistant_more_pick" not in st.session_state:
                        st.session_state["assistant_more_pick"] = "—"

                    st.selectbox(
                        "More",
                        options=["—"] + candidates[6:],
                        index=0,
                        key="assistant_more_pick",
                        on_change=_apply_selectbox_pick_to_widget,
                        args=("assistant_more_pick", "assistant_query", "assistant_completion_mode"),
                    )
            else:
                st.caption("No suggestions for the current context (try another start).")

            cta1, cta2 = st.columns(2)
            with cta1:
                st.button(
                    "Copy to Retrieval tab",
                    use_container_width=True,
                    on_click=_copy_session_key,
                    args=("assistant_query", "retrieval_query"),
                )
            with cta2:
                st.button(
                    "Clear",
                    use_container_width=True,
                    on_click=_set_session_key,
                    args=("assistant_query", ""),
                )

        with col2:
            st.markdown("#### Type-ahead examples")

            max_examples = min(5000, len(autocomplete_texts))
            sample = st.selectbox(
                "Search & pick an example query",
                options=autocomplete_texts[:max_examples],
            )
            st.button(
                "Use example in Retrieval",
                use_container_width=True,
                on_click=_set_session_key,
                args=("retrieval_query", sample),
            )
            if st.session_state.get("retrieval_query", "") == sample:
                st.success("Copied to retrieval query.")

            st.markdown("---")
            st.markdown("#### Quick stats")
            tok_counts = [len(_tokenize(q)) for q in autocomplete_texts[:5000]]
            st.metric("Median tokens/query", int(np.median(tok_counts)))
            st.metric("P95 tokens/query", int(np.percentile(tok_counts, 95)))

            if show_debug and model is not None:
                st.code({"ngram_contexts": len(model), "vocab_size": len(vocab)})



# -----------------------------
# Raw report view
# -----------------------------
with tabs[3]:
    if report_md is None:
        st.info("Provide a valid report path to view it here.")
    else:
        st.markdown("### Raw report")
        st.download_button("Download report markdown", data=report_md.encode("utf-8"), file_name="kahm_evaluation_report.md")
        st.code(report_md, language="markdown")
