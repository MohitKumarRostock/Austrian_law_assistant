#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_sentences_from_German_pdfs.py  (RIS law PDF extraction; paragraph-robust)

What changed vs. the original
-----------------------------
This variant keeps the output schema identical but improves paragraph extraction robustness
for Austrian RIS PDFs by:

- Using layout-aware extraction (PyMuPDF "dict") to reconstruct *lines* in reading order.
- Removing repeating headers/footers using a frequency-based blacklist + existing regex drops.
- Segmenting paragraphs using a combination of:
    * legal markers (e.g., "§ 12", "Art. 5")
    * indentation patterns (common in RIS PDFs)
    * large vertical gaps (blank-line style spacing)
- Merging paragraphs that continue across page breaks (optional; enabled by default for paragraph/passage).
- Fixing dehyphenation so compound hyphens like "Privat-Rechte" are preserved.

Downstream compatibility
-----------------------
Output columns remain:
  sentence_id, law_type, page, sentence, source_file

The column name "sentence" is intentionally kept for compatibility even when
the extraction unit is a paragraph/passage.

You can still use:
  --unit sentence   (legacy sentence-ish splitting)
  --unit paragraph  (layout paragraphs; robust)
  --unit passage    (chunked passages for embeddings; default)

"""

from __future__ import annotations

import argparse
import importlib
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast, Set

import pandas as pd


# ----------------------------- Normalization & filtering -----------------------------

_ABBREV_PLACEHOLDER = "∯"  # unlikely to appear in RIS law texts

_DEFAULT_DROP_PATTERNS: List[str] = [
    r"^\s*Seite\s+\d+(\s+von\s+\d+)?\s*$",
    r"^\s*www\.ris\.bka\.gv\.at\s*$",
    r"^\s*RIS\s*$",
]

_NOISE_PREFIX_STRIP_REGEXES: List[re.Pattern[str]] = [
    re.compile(r"^\s*Bundesrecht\s+konsolidiert\s*:?\s*", flags=re.IGNORECASE),
    re.compile(r"^\s*Gesamte\s+Rechtsvorschrift(?:\s+für)?\s*:?\s*", flags=re.IGNORECASE),
    re.compile(r"^\s*RIS\s*-\s*Rechtsinformationssystem(?:\s+des\s+Bundes)?\s*:?\s*", flags=re.IGNORECASE),
    re.compile(r"^\s*Rechtsinformationssystem(?:\s+des\s+Bundes)?\s*:?\s*", flags=re.IGNORECASE),
    re.compile(r"^\s*CELEX-?Nr\.?\s*:?\s*", flags=re.IGNORECASE),
    re.compile(
        r"^\s*(?:Kundmachungsorgan|Gesetzesnummer|Dokumenttyp|Kurztitel|Langtitel|Abkürzung|Fassung\s+vom|"
        r"Inkrafttretensdatum|Zuletzt\s+geändert\s+durch|Zuletzt\s+aktualisiert|Norm|Anmerkung|Schlagworte|"
        r"Dokumentnummer|Stand)\s*:?\s*",
        flags=re.IGNORECASE,
    ),
]

_RIS_INLINE_FOOTER_RE = re.compile(
    r"^\s*www\.ris\.bka\.gv\.at\s+Seite\s+\d+\s+von\s+\d+\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)
_RIS_URL_ONLY_RE = re.compile(r"^\s*https?://\S+\s*$", flags=re.IGNORECASE | re.MULTILINE)

_ENUM_LINE_RE = re.compile(r"^\s*(\d{1,3})\.\s+", flags=re.MULTILINE)
_ENUM_PAREN_LINE_RE = re.compile(r"^\s*\((\d{1,3})\)\s+", flags=re.MULTILINE)
_ENUM_ALPHA_LINE_RE = re.compile(r"^\s*([A-Za-z])\.\s+", flags=re.MULTILINE)
_ENUM_ROMAN_LINE_RE = re.compile(r"^\s*([IVXLCDM]{1,8})\.\s+", flags=re.MULTILINE)
_ENUM_AFTER_COLON_SEMI_RE = re.compile(r"([:;])\s*(\d{1,3})\.\s+")

_LEGAL_REF_DOT_REGEXES: List[re.Pattern[str]] = [
    re.compile(
        r"§{1,2}\s*\d+[A-Za-z]?\.(?=\s*(?:Abs|Absatz|Z|Ziffer|Ziff|lit|Satz|Nr|iVm|i\.?V\.?m)\b)",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"(?:Art\.?|Artikel)\s*\d+[A-Za-z]?\.(?=\s*(?:Abs|Absatz|Z|Ziffer|Ziff|lit|Satz|Nr|iVm|i\.?V\.?m)\b)",
        flags=re.IGNORECASE,
    ),
]

_ABBREV_REGEXES: List[re.Pattern[str]] = [
    re.compile(r"\bAbs\.", flags=re.IGNORECASE),
    re.compile(r"\bArt\.", flags=re.IGNORECASE),
    re.compile(r"\bZ\.", flags=re.IGNORECASE),
    re.compile(r"\bNr\.", flags=re.IGNORECASE),
    re.compile(r"\bZl\.", flags=re.IGNORECASE),
    re.compile(r"\bBGBl\.", flags=re.IGNORECASE),
    re.compile(r"\biVm\b", flags=re.IGNORECASE),
    re.compile(r"\bi\.?V\.?m\.", flags=re.IGNORECASE),
    re.compile(r"\bzB\b", flags=re.IGNORECASE),
    re.compile(r"\bz\.?B\.", flags=re.IGNORECASE),
    re.compile(r"\bu\.?a\.", flags=re.IGNORECASE),
    re.compile(r"\bu\.?U\.", flags=re.IGNORECASE),
    re.compile(r"\bua\b", flags=re.IGNORECASE),
]

_DANGLING_REF_RE = re.compile(r"(?:§{1,2}\s*\d+[A-Za-z]?\.)\s*$", flags=re.IGNORECASE)
_CONTINUATION_START_RE = re.compile(
    r"^(?:Abs|Absatz|Z|Ziffer|Ziff|lit|Satz|Nr|iVm|i\.?V\.?m)\b",
    flags=re.IGNORECASE,
)

_WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß]+", flags=re.UNICODE)
_DIGIT_RE = re.compile(r"\d")
_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def _strip_noise_prefixes(line: str) -> str:
    for rx in _NOISE_PREFIX_STRIP_REGEXES:
        line = rx.sub("", line)
    return line


def _normalize_whitespace(raw: str, *, drop_patterns: Optional[List[str]] = None) -> str:
    """
    Recall-first normalization.
    - Keep line boundaries long enough to strip RIS prefixes safely.
    - Dehyphenate across line breaks, *without* destroying compound hyphens.
    - Normalize enumeration markers.
    - Collapse whitespace.
    """
    if not raw:
        return ""

    s = raw.replace("\u00ad", "")
    s = _RIS_INLINE_FOOTER_RE.sub("", s)
    s = _RIS_URL_ONLY_RE.sub("", s)

    lines: List[str] = []
    for ln in s.splitlines():
        ln = ln.rstrip("\n")
        if drop_patterns:
            if any(re.fullmatch(pat, ln.strip(), flags=re.IGNORECASE) for pat in drop_patterns):
                continue
        ln = _strip_noise_prefixes(ln)
        lines.append(ln)

    s = "\n".join(lines)

    # Dehyphenate across line breaks:
    # - If the next line continues a word (lowercase), remove the hyphen.
    # - If the next line starts with uppercase, keep the hyphen but remove the newline.
    s = re.sub(r"([A-Za-zÄÖÜäöüß])-\s*\n\s*([a-zäöüß])", r"\1\2", s)
    s = re.sub(r"([A-Za-zÄÖÜäöüß])-\s*\n\s*([A-ZÄÖÜ])", r"\1-\2", s)

    # Normalize enumeration markers at line starts
    s = _ENUM_ROMAN_LINE_RE.sub(r"\1) ", s)
    s = _ENUM_PAREN_LINE_RE.sub(r"\1) ", s)
    s = _ENUM_ALPHA_LINE_RE.sub(r"\1) ", s)
    s = _ENUM_LINE_RE.sub(r"\1) ", s)

    # Preserve enumerations after ':' or ';'
    s = _ENUM_AFTER_COLON_SEMI_RE.sub(r"\1 \2) ", s)

    # Newlines -> spaces
    s = re.sub(r"\s*\n\s*", " ", s)

    # Collapse whitespace
    s = re.sub(r"[ \t\r\f\v]+", " ", s).strip()
    return s


def _protect_abbreviations(s: str) -> str:
    if not s:
        return s

    for rx in _LEGAL_REF_DOT_REGEXES:
        def _repl(m: re.Match[str]) -> str:
            return m.group(0).replace(".", _ABBREV_PLACEHOLDER)
        s = rx.sub(_repl, s)

    for rx in _ABBREV_REGEXES:
        def _abbr_repl(m: re.Match[str]) -> str:
            return m.group(0).replace(".", _ABBREV_PLACEHOLDER)
        s = rx.sub(_abbr_repl, s)

    return s


def _restore_abbreviations(s: str) -> str:
    return s.replace(_ABBREV_PLACEHOLDER, ".")


def _merge_false_boundaries(sentences: List[str]) -> List[str]:
    if not sentences:
        return sentences

    merged: List[str] = []
    i = 0
    while i < len(sentences):
        cur = sentences[i].strip()
        if i + 1 < len(sentences):
            nxt = sentences[i + 1].strip()
            if cur and nxt and _DANGLING_REF_RE.search(cur) and _CONTINUATION_START_RE.search(nxt):
                merged.append(f"{cur} {nxt}".strip())
                i += 2
                continue
        merged.append(cur)
        i += 1
    return merged


def _split_on_list_markers(text: str) -> List[str]:
    """
    Conservative splitting on repeated list markers (useful for long paragraphs).
    """
    markers: List[int] = []
    for rx in [
        re.compile(r"\b\d{1,3}\)"),
        re.compile(r"\(\d{1,3}\)"),
        re.compile(r"\b[A-Za-z]\)"),
        re.compile(r"[•\u2022]"),
    ]:
        for m in rx.finditer(text):
            markers.append(m.start())

    markers = sorted(set(markers))
    if len(markers) < 2:
        return [text]

    split_points = markers[1:]
    out: List[str] = []
    last = 0
    for sp in split_points:
        chunk = text[last:sp].strip()
        if chunk:
            out.append(chunk)
        last = sp

    tail = text[last:].strip()
    if tail:
        out.append(tail)

    return out


def split_into_sentences(
    text: str,
    *,
    split_enumerations: bool = True,
    drop_patterns_for_normalization: Optional[List[str]] = None,
) -> List[str]:
    """
    Legacy recall-first sentence splitter (kept for compatibility).
    """
    s = _normalize_whitespace(text, drop_patterns=drop_patterns_for_normalization)
    if not s:
        return []

    s = _protect_abbreviations(s)
    raw = re.split(r"(?<=[\.\?!])\s+", s)

    sentences = [_restore_abbreviations(x).strip() for x in raw if x and x.strip()]
    sentences = _merge_false_boundaries(sentences)

    if split_enumerations:
        expanded: List[str] = []
        for s2 in sentences:
            for sub in _split_on_list_markers(s2):
                sub = sub.strip()
                if sub:
                    expanded.append(sub)
        sentences = expanded

    return sentences


def is_semantically_meaningful(
    sentence: str,
    *,
    min_chars: int = 10,
    min_alpha_tokens: int = 1,
    max_digit_ratio: float = 0.85,
    drop_patterns: Optional[List[str]] = None,
) -> bool:
    """
    Permissive filter that improves retrieval signal while keeping recall high.
    """
    if not sentence:
        return False
    s = sentence.strip()
    if not s:
        return False

    if drop_patterns:
        for pat in drop_patterns:
            if re.fullmatch(pat, s, flags=re.IGNORECASE):
                return False

    if re.fullmatch(r"[\s\-\–\—\.,;:/()\[\]{}]+", s):
        return False

    if len(s) < min_chars:
        return False

    alpha_tokens = _WORD_RE.findall(s)
    if len(alpha_tokens) < min_alpha_tokens:
        return False

    digits = len(_DIGIT_RE.findall(s))
    if digits / max(1, len(s)) > max_digit_ratio:
        return False

    return True


# ----------------------------- IR-oriented chunking -----------------------------

def _count_tokens(s: str) -> int:
    return len(_TOKEN_RE.findall(s))


def _chunk_items_to_passages(
    items: List[Tuple[int, str]],
    *,
    target_tokens: int = 260,
    overlap_tokens: int = 60,
    min_tokens: int = 40,
    max_tokens: int = 420,
    split_enumerations: bool = True,
) -> List[Tuple[int, str]]:
    """
    Chunk (page, paragraph) items into embedding-friendly passages with overlap.
    The returned page number is the *start page* of the first non-overlap paragraph in the chunk.
    """
    if not items:
        return []

    # Expand overlong paragraphs before chunking (keeps their page).
    expanded_items: List[Tuple[int, str]] = []
    for pg, p in items:
        if _count_tokens(p) > max_tokens and split_enumerations:
            parts = _split_on_list_markers(p)
            expanded_items.extend([(pg, x.strip()) for x in parts if x and x.strip()])
        else:
            expanded_items.append((pg, p))

    passages: List[Tuple[int, str]] = []

    cur_parts: List[str] = []
    cur_tokens = 0
    cur_start_page: Optional[int] = None
    has_real = False  # at least one non-overlap paragraph
    pending_overlap_prefix: Optional[str] = None

    def finalize_current() -> Optional[Tuple[int, str]]:
        nonlocal cur_parts, cur_tokens, cur_start_page, has_real, pending_overlap_prefix
        if not cur_parts or not has_real or cur_start_page is None:
            cur_parts, cur_tokens, cur_start_page, has_real = [], 0, None, False
            return None
        txt = "\n\n".join([x for x in cur_parts if x]).strip()
        start_page = cur_start_page
        cur_parts, cur_tokens, cur_start_page, has_real = [], 0, None, False
        if not txt:
            return None
        if _count_tokens(txt) > max_tokens:
            w = _TOKEN_RE.findall(txt)[:max_tokens]
            txt = " ".join(w)
        return (start_page, txt)

    def compute_overlap_prefix(prev_txt: str) -> str:
        if overlap_tokens <= 0:
            return ""
        words = _TOKEN_RE.findall(prev_txt)
        ov = words[-overlap_tokens:] if len(words) > overlap_tokens else words
        return " ".join(ov)

    for pg, para in expanded_items:
        para = para.strip()
        if not para:
            continue
        pt = _count_tokens(para)
        if pt == 0:
            continue

        # Initialize new chunk with overlap prefix (but don't set start_page yet).
        if not cur_parts and pending_overlap_prefix:
            cur_parts = [pending_overlap_prefix]
            cur_tokens = _count_tokens(pending_overlap_prefix)
            pending_overlap_prefix = None

        if cur_parts and (cur_tokens + pt) > target_tokens and cur_tokens >= min_tokens and has_real:
            finished = finalize_current()
            if finished:
                passages.append(finished)
                pending_overlap_prefix = compute_overlap_prefix(finished[1])

        if cur_start_page is None:
            cur_start_page = pg
        cur_parts.append(para)
        cur_tokens += pt
        has_real = True

    finished = finalize_current()
    if finished:
        passages.append(finished)

    return [(pg, txt) for (pg, txt) in passages if _count_tokens(txt) >= min_tokens]


# ----------------------------- PDF extraction (layout-aware) -----------------------------

def _load_pymupdf_module() -> Any:
    errors: List[str] = []
    for module_name in ("pymupdf", "fitz"):
        try:
            candidate = importlib.import_module(module_name)
            if hasattr(candidate, "Document") or hasattr(candidate, "open"):
                return candidate
        except Exception as e:
            errors.append(f"{module_name}: {type(e).__name__}: {e}")
    raise RuntimeError("PyMuPDF not available. Tried pymupdf/fitz. Errors: " + " | ".join(errors))


def _extract_pages_with_pdfplumber(pdf_path: Path) -> Iterable[Tuple[int, str]]:
    try:
        pdfplumber = importlib.import_module("pdfplumber")
    except Exception as e:
        raise RuntimeError(
            "Neither PyMuPDF nor pdfplumber available. Install one: `pip install pymupdf` or `pip install pdfplumber`.\n"
            f"pdfplumber import error: {type(e).__name__}: {e}"
        ) from e

    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            yield i, (page.extract_text() or "")


def extract_pages_text(pdf_path: Path) -> Iterable[Tuple[int, str]]:
    """
    Text extraction with fallback (legacy interface).
    """
    try:
        pymupdf_mod = _load_pymupdf_module()
        Document = cast(Optional[Any], getattr(pymupdf_mod, "Document", None))
        open_fn = cast(Optional[Any], getattr(pymupdf_mod, "open", None))

        if callable(Document):
            doc = Document(str(pdf_path))
        elif callable(open_fn):
            doc = open_fn(str(pdf_path))
        else:
            raise RuntimeError("PyMuPDF module loaded but provides neither Document nor open().")

        try:
            if getattr(doc, "needs_pass", False):
                raise RuntimeError(f"Encrypted PDF (password needed): {pdf_path.name}")

            for i, page in enumerate(cast(Iterable[Any], doc), start=1):
                try:
                    txt = page.get_text("text", sort=True)  # type: ignore[call-arg]
                except TypeError:
                    txt = page.get_text("text")
                yield i, (txt or "")
        finally:
            close_method = getattr(doc, "close", None)
            if callable(close_method):
                close_method()
    except Exception:
        yield from _extract_pages_with_pdfplumber(pdf_path)


# --- Paragraph extraction helpers ---

_SECTION_START_RE = re.compile(r"^\s*(?:§{1,2}\s*\d+[A-Za-z]?|Art\.?\s*\d+[A-Za-z]?|Artikel\s+\d+)", re.IGNORECASE)
_LIST_ITEM_RE = re.compile(r"^\s*(?:\(\d{1,3}\)|\d{1,3}\)|[A-Za-z]\)|[•\u2022])\s+")
_STRONG_SENT_END_RE = re.compile(r"[\.!?]\s*$")


def _normalize_header_footer_key(s: str) -> str:
    s = _strip_noise_prefixes(s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    s = re.sub(r"\d+", "<d>", s)  # remove page numbers etc.
    return s


def _matches_drop_patterns(s: str, drop_patterns: Optional[List[str]]) -> bool:
    if not drop_patterns:
        return False
    st = s.strip()
    for pat in drop_patterns:
        if re.fullmatch(pat, st, flags=re.IGNORECASE):
            return True
    return False


def _extract_lines_from_page_dict(page_dict: Dict[str, Any]) -> List[Tuple[float, float, float, float, str]]:
    """
    Extract (x0, y0, x1, y1, text) lines from a PyMuPDF page.get_text("dict") result.
    """
    out: List[Tuple[float, float, float, float, str]] = []
    for b in page_dict.get("blocks", []) or []:
        if b.get("type", 0) != 0:
            continue
        for ln in b.get("lines", []) or []:
            spans = ln.get("spans", []) or []
            if not spans:
                continue
            txt = "".join((sp.get("text", "") or "") for sp in spans)
            txt = txt.strip()
            if not txt:
                continue
            x0, y0, x1, y1 = ln.get("bbox", (0.0, 0.0, 0.0, 0.0))
            out.append((float(x0), float(y0), float(x1), float(y1), txt))
    out.sort(key=lambda t: (round(t[1], 2), t[0]))
    return out


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    if q <= 0:
        return v[0]
    if q >= 1:
        return v[-1]
    idx = int(round((len(v) - 1) * q))
    return v[max(0, min(len(v) - 1, idx))]


def _join_lines_into_paragraphs(
    lines: List[Tuple[float, float, float, float, str]],
    *,
    page_width: float,
    page_height: float,
) -> List[str]:
    """
    Convert filtered lines into paragraph strings.
    """
    if not lines:
        return []

    x0s = [x0 for (x0, _, _, _, _) in lines]
    base_x0 = _percentile(x0s, 0.10)  # stable left margin baseline
    indent_threshold = 12.0

    # Estimate "blank line" vertical gap using line heights.
    heights = [max(0.0, y1 - y0) for (_, y0, _, y1, _) in lines]
    line_h = _percentile(heights, 0.50) or 10.0
    gap_threshold = max(0.90 * line_h, 9.0)

    paras: List[str] = []
    cur: List[str] = []

    prev_x0: Optional[float] = None
    prev_y1: Optional[float] = None
    prev_text: str = ""

    def flush():
        nonlocal cur, prev_text
        if not cur:
            return
        txt = "".join(cur).strip()
        txt = _normalize_paragraph_text(txt)
        if txt:
            paras.append(txt)
        cur = []
        prev_text = ""

    for (x0, y0, x1, y1, txt) in lines:
        # Normalize per-line, but keep list markers and section starts intact.
        lt = txt.replace("\u00ad", "").strip()
        if not lt:
            continue

        is_section_start = bool(_SECTION_START_RE.match(lt))
        is_list_item = bool(_LIST_ITEM_RE.match(lt))
        is_indented = (x0 - base_x0) >= indent_threshold

        # Large vertical gap (blank line) -> paragraph break.
        gap = None
        if prev_y1 is not None:
            gap = y0 - prev_y1

        starts_new_para = False
        if not cur:
            starts_new_para = True
        else:
            # Explicit structure markers.
            if is_section_start:
                starts_new_para = True
            # Indentation-based paragraph starts (common in RIS PDFs).
            elif is_indented and (prev_x0 is not None) and (prev_x0 - base_x0) < indent_threshold:
                if _STRONG_SENT_END_RE.search(prev_text) or (gap is not None and gap >= 0):
                    starts_new_para = True
            # Blank-line style separation.
            elif gap is not None and gap > gap_threshold:
                starts_new_para = True
            # Centered heading-ish line (short and centered) -> own paragraph.
            else:
                # A crude center heuristic: left margin far from base and short text
                if (x0 - base_x0) > 60 and len(lt) < 80:
                    starts_new_para = True

        if starts_new_para and cur:
            flush()

        # Append current line to paragraph with robust line-join rules.
        if not cur:
            cur.append(lt)
        else:
            prev = cur[-1]

            # Dehyphenation / hyphen-preserving join:
            # - If previous ends with "-" and current starts lowercase: remove hyphen (word continuation).
            # - Else if previous ends with "-" keep it but join without space (compound with uppercase etc.)
            if prev.endswith("-") and lt and re.match(r"^[a-zäöüß]", lt):
                cur[-1] = prev[:-1] + lt
            elif prev.endswith("-") and lt:
                cur[-1] = prev + lt
            # List items: preserve structure on separate line.
            elif is_list_item or (prev.endswith(":") and is_list_item):
                cur.append("\n" + lt)
            else:
                cur.append(" " + lt)

        prev_x0 = x0
        prev_y1 = y1
        prev_text = lt

    flush()
    return paras


def _normalize_paragraph_text(s: str) -> str:
    """
    Paragraph-level normalization that preserves newlines (used for lists).
    """
    if not s:
        return ""
    # Temporarily protect newlines so we can reuse _normalize_whitespace.
    placeholder = " ⏎ "
    s2 = s.replace("\n", placeholder)
    s2 = _normalize_whitespace(s2, drop_patterns=None)
    s2 = s2.replace(placeholder, "\n")
    # Clean up newline spacing
    s2 = re.sub(r"[ \t]*\n[ \t]*", "\n", s2).strip()
    return s2


def extract_paragraph_items(
    pdf_path: Path,
    *,
    drop_patterns: Optional[List[str]] = None,
    merge_across_pages: bool = True,
    header_footer_scan_ratio: float = 0.10,
    header_footer_min_page_ratio: float = 0.20,
) -> List[Tuple[int, str]]:
    """
    Extract robust layout paragraphs from a RIS PDF.

    Returns: list of (start_page_no, paragraph_text).

    Strategy:
      1) Extract layout lines per page using PyMuPDF dict.
      2) Build a repeating header/footer blacklist using top/bottom bands.
      3) Segment paragraphs with indentation + markers.
      4) Optionally merge paragraphs across page breaks when they look like continuations.
    """
    if drop_patterns is None:
        drop_patterns = list(_DEFAULT_DROP_PATTERNS)

    pymupdf_mod = _load_pymupdf_module()
    Document = cast(Optional[Any], getattr(pymupdf_mod, "Document", None))
    open_fn = cast(Optional[Any], getattr(pymupdf_mod, "open", None))

    if callable(Document):
        doc = Document(str(pdf_path))
    elif callable(open_fn):
        doc = open_fn(str(pdf_path))
    else:
        raise RuntimeError("PyMuPDF module loaded but provides neither Document nor open().")

    try:
        if getattr(doc, "needs_pass", False):
            raise RuntimeError(f"Encrypted PDF (password needed): {pdf_path.name}")

        n_pages = int(getattr(doc, "page_count", 0) or len(cast(List[Any], doc)))
        if n_pages <= 0:
            return []

        # Pass 1: build header/footer blacklist by frequency.
        counts: Counter[str] = Counter()
        for pi in range(n_pages):
            page = cast(Any, doc)[pi]
            pd = page.get_text("dict")
            lines = _extract_lines_from_page_dict(pd)
            h = float(getattr(page.rect, "height", pd.get("height", 0.0)))
            top_y = h * header_footer_scan_ratio
            bot_y = h * (1.0 - header_footer_scan_ratio)

            for (x0, y0, x1, y1, txt) in lines:
                t = txt.strip()
                if not t:
                    continue
                # Consider only top/bottom bands for header/footer candidates.
                if y0 <= top_y or y1 >= bot_y:
                    key = _normalize_header_footer_key(t)
                    if key:
                        counts[key] += 1

        min_count = max(3, int(round(n_pages * header_footer_min_page_ratio)))
        hf_blacklist = {k for (k, c) in counts.items() if c >= min_count and 2 <= len(k) <= 120}

        # Pass 2: extract paragraphs per page and optionally merge across pages.
        out_items: List[Tuple[int, str]] = []
        prev_pg: Optional[int] = None
        prev_txt: Optional[str] = None

        def should_merge(prev: str, cur: str) -> bool:
            # Do not merge if the new paragraph clearly starts a new section/list.
            if _SECTION_START_RE.match(cur) or _LIST_ITEM_RE.match(cur):
                return False
            # Merge if previous paragraph ends without a strong stop, OR clearly ends with a hyphen.
            if prev.rstrip().endswith("-"):
                return True
            if not _STRONG_SENT_END_RE.search(prev):
                # Avoid merging if the next paragraph looks like a heading (center/short).
                if len(cur) < 80 and re.match(r"^[A-ZÄÖÜ0-9]", cur):
                    return False
                return True
            return False

        for pi in range(n_pages):
            page_no = pi + 1
            page = cast(Any, doc)[pi]
            pd = page.get_text("dict")
            lines = _extract_lines_from_page_dict(pd)

            # Filter header/footer and obvious noise lines.
            filtered: List[Tuple[float, float, float, float, str]] = []
            for (x0, y0, x1, y1, txt) in lines:
                t = _strip_noise_prefixes(txt).strip()
                if not t:
                    continue
                if _matches_drop_patterns(t, drop_patterns):
                    continue
                if _normalize_header_footer_key(t) in hf_blacklist:
                    continue
                # Drop pure URL line variants.
                if re.fullmatch(r"https?://\S+", t, flags=re.IGNORECASE):
                    continue
                filtered.append((x0, y0, x1, y1, t))

            paras = _join_lines_into_paragraphs(
                filtered,
                page_width=float(getattr(page.rect, "width", pd.get("width", 0.0))),
                page_height=float(getattr(page.rect, "height", pd.get("height", 0.0))),
            )

            for ptxt in paras:
                if not ptxt:
                    continue
                # Optional cross-page merge.
                if merge_across_pages and prev_txt is not None and prev_pg is not None:
                    if should_merge(prev_txt, ptxt):
                        merged = (prev_txt.rstrip() + " " + ptxt.lstrip()).strip()
                        out_items[-1] = (prev_pg, merged)
                        prev_txt = merged
                        continue

                out_items.append((page_no, ptxt))
                prev_pg, prev_txt = page_no, ptxt

        return out_items
    finally:
        close_method = getattr(doc, "close", None)
        if callable(close_method):
            close_method()


# ----------------------------- Main pipeline -----------------------------

def ris_pdfs_to_parquet(
    input_dir: str | Path = "ris_pdfs",
    output_parquet: str | Path = "ris_sentences.parquet",
    *,
    recursive: bool = False,
    min_chars: int = 25,
    min_alpha_tokens: int = 3,
    max_digit_ratio: float = 0.85,
    drop_patterns: Optional[List[str]] = None,
    print_example_per_pdf: bool = True,
    split_enumerations: bool = True,
    filter_sentences: bool = True,
    unit: str = "passage",
    target_tokens: int = 260,
    overlap_tokens: int = 60,
    min_passage_tokens: int = 40,
    max_passage_tokens: int = 420,
    dedupe_within_pdf: bool = True,
    merge_across_pages: bool = True,
) -> pd.DataFrame:
    """
    Reads all PDFs in input_dir, extracts and filters units, labels them with law_type + start page,
    and writes Parquet. Output schema is unchanged.
    """
    input_dir = Path(input_dir)
    output_parquet = Path(output_parquet)

    if drop_patterns is None:
        drop_patterns = list(_DEFAULT_DROP_PATTERNS)

    unit = unit.lower().strip()
    if unit not in {"sentence", "paragraph", "passage"}:
        raise ValueError("unit must be one of: sentence, paragraph, passage")

    pdf_paths = sorted(input_dir.rglob("*.pdf") if recursive else input_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in: {input_dir.resolve()}")

    rows: List[Dict[str, Any]] = []
    sentence_id = 0
    first_example: Dict[str, Tuple[int, str]] = {}

    for pdf_path in pdf_paths:
        law_type = pdf_path.stem
        seen_texts: Set[str] = set()

        if unit == "sentence":
            for page_no, page_text in extract_pages_text(pdf_path):
                if not page_text or not page_text.strip():
                    continue
                units = split_into_sentences(
                    page_text,
                    split_enumerations=split_enumerations,
                    drop_patterns_for_normalization=drop_patterns,
                )
                for u in units:
                    u = u.strip()
                    if not u:
                        continue
                    if filter_sentences and not is_semantically_meaningful(
                        u,
                        min_chars=min_chars,
                        min_alpha_tokens=min_alpha_tokens,
                        max_digit_ratio=max_digit_ratio,
                        drop_patterns=drop_patterns,
                    ):
                        continue
                    if dedupe_within_pdf and u in seen_texts:
                        continue
                    seen_texts.add(u)

                    if law_type not in first_example:
                        first_example[law_type] = (page_no, u)

                    sentence_id += 1
                    rows.append(
                        {"sentence_id": sentence_id, "law_type": law_type, "page": page_no, "sentence": u, "source_file": pdf_path.name}
                    )

        else:
            # Robust paragraph extraction (layout-aware) across the entire PDF.
            paragraph_items = extract_paragraph_items(
                pdf_path,
                drop_patterns=drop_patterns,
                merge_across_pages=merge_across_pages,
            )

            if split_enumerations and paragraph_items:
                expanded_items: List[Tuple[int, str]] = []
                for pg, ptxt in paragraph_items:
                    parts = _split_on_list_markers(ptxt)
                    expanded_items.extend([(pg, x.strip()) for x in parts if x and x.strip()])
                paragraph_items = expanded_items

            units_items: List[Tuple[int, str]] = (
                paragraph_items if unit == "paragraph"
                else _chunk_items_to_passages(
                    paragraph_items,
                    target_tokens=target_tokens,
                    overlap_tokens=overlap_tokens,
                    min_tokens=min_passage_tokens,
                    max_tokens=max_passage_tokens,
                    split_enumerations=split_enumerations,
                )
            )

            for page_no, u in units_items:
                u = u.strip()
                if not u:
                    continue
                if filter_sentences and not is_semantically_meaningful(
                    u,
                    min_chars=min_chars,
                    min_alpha_tokens=min_alpha_tokens,
                    max_digit_ratio=max_digit_ratio,
                    drop_patterns=drop_patterns,
                ):
                    continue
                if dedupe_within_pdf and u in seen_texts:
                    continue
                seen_texts.add(u)

                if law_type not in first_example:
                    first_example[law_type] = (page_no, u)

                sentence_id += 1
                rows.append(
                    {"sentence_id": sentence_id, "law_type": law_type, "page": page_no, "sentence": u, "source_file": pdf_path.name}
                )

    if print_example_per_pdf:
        print("Example unit per PDF (first retained unit found):")
        for pdf_path in pdf_paths:
            law_type = pdf_path.stem
            if law_type in first_example:
                p, s = first_example[law_type]
                print(f"  - {pdf_path.name}: p.{p}: {s}")
            else:
                print(f"  - {pdf_path.name}: (no units retained)")

    df = pd.DataFrame(rows, columns=["sentence_id", "law_type", "page", "sentence", "source_file"])

    try:
        df.to_parquet(output_parquet, index=False)
    except Exception as e:
        raise RuntimeError(
            f"Failed to write Parquet to {output_parquet.resolve()}.\n"
            "Install a parquet engine, e.g.: `pip install pyarrow` (recommended) or `pip install fastparquet`.\n"
            f"Original error: {type(e).__name__}: {e}"
        ) from e

    print(f"\nWrote {len(df):,} units to: {output_parquet.resolve()}")
    return df


# ----------------------------- CLI -----------------------------

def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract labeled units from RIS PDFs and write Parquet (paragraph-robust).")
    p.add_argument("--input_dir", default="ris_pdfs", help="Folder containing RIS PDF files.")
    p.add_argument("--output", default="ris_sentences.parquet", help="Output Parquet file path.")
    p.add_argument("--recursive", action="store_true", help="Search PDFs recursively.")
    p.add_argument("--no_print_examples", action="store_true", help="Do not print per-PDF example units.")

    p.add_argument("--min_chars", type=int, default=25, help="Minimum character length for a unit.")
    p.add_argument("--min_alpha_tokens", type=int, default=3, help="Minimum count of alphabetic tokens.")
    p.add_argument("--max_digit_ratio", type=float, default=0.85, help="Max allowed digit/char ratio.")

    p.add_argument("--no_split_enumerations", action="store_true", help="Disable splitting on list markers like '1)'.")
    p.add_argument("--no_filter", action="store_true", help="Disable filtering (maximal recall; includes more noise).")

    p.add_argument(
        "--unit",
        choices=["sentence", "paragraph", "passage"],
        default="passage",
        help="Extraction unit: sentence (legacy), paragraph (layout; robust), passage (chunked for embeddings; default).",
    )
    p.add_argument("--target_tokens", type=int, default=260, help="Target token size for passage chunks.")
    p.add_argument("--overlap_tokens", type=int, default=60, help="Token overlap between passages.")
    p.add_argument("--min_passage_tokens", type=int, default=40, help="Minimum token size for retained passages.")
    p.add_argument("--max_passage_tokens", type=int, default=420, help="Hard maximum token size for a passage.")
    p.add_argument("--no_dedupe_within_pdf", action="store_true", help="Disable duplicate removal within each PDF.")
    p.add_argument(
        "--no_merge_across_pages",
        action="store_true",
        help="Disable merging paragraphs that continue across page breaks (paragraph/passage units).",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    ris_pdfs_to_parquet(
        input_dir=args.input_dir,
        output_parquet=args.output,
        recursive=args.recursive,
        min_chars=args.min_chars,
        min_alpha_tokens=args.min_alpha_tokens,
        max_digit_ratio=args.max_digit_ratio,
        print_example_per_pdf=not args.no_print_examples,
        split_enumerations=not args.no_split_enumerations,
        filter_sentences=not args.no_filter,
        unit=args.unit,
        target_tokens=args.target_tokens,
        overlap_tokens=args.overlap_tokens,
        min_passage_tokens=args.min_passage_tokens,
        max_passage_tokens=args.max_passage_tokens,
        dedupe_within_pdf=not args.no_dedupe_within_pdf,
        merge_across_pages=not args.no_merge_across_pages,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
