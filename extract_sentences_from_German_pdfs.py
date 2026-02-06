#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_sentences_from_German_pdfs.py  (IR-optimized adaptation)

Purpose
-------
Extract semantically meaningful text units from RIS law PDFs, label with law type + page number,
and write to a Parquet file. Also prints ONE example unit per PDF.

Downstream compatibility
-----------------------
The output schema is unchanged:
  Columns: sentence_id, law_type, page, sentence, source_file

The column name "sentence" is intentionally kept for compatibility even when
the extraction unit is a paragraph/passage.

IR optimization
--------------
By default, this version extracts *passages* (chunked paragraph-like blocks) rather than
single sentences, because:
  - sentence boundaries are hard to reconstruct reliably from PDFs, especially in legal text
  - embedding retrieval typically performs better with self-contained context windows

You can switch back to legacy behavior with:  --unit sentence
"""

from __future__ import annotations

import argparse
import importlib
import re
import sys
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
    - Dehyphenate across line breaks.
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

    # Dehyphenate across line breaks
    s = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", s)

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


# ----------------------------- IR-oriented paragraph/passage extraction -----------------------------

def _extract_paragraph_candidates_from_page_text(
    raw_page_text: str,
    *,
    drop_patterns: Optional[List[str]] = None,
) -> List[str]:
    """
    Paragraph candidates from a plain page text (fallback when block extraction is not available).
    """
    if not raw_page_text or not raw_page_text.strip():
        return []

    s = raw_page_text.replace("\u00ad", "")
    s = _RIS_INLINE_FOOTER_RE.sub("", s)
    s = _RIS_URL_ONLY_RE.sub("", s)

    kept_lines: List[str] = []
    for ln in s.splitlines():
        ln = ln.rstrip("\n")
        if drop_patterns:
            if any(re.fullmatch(pat, ln.strip(), flags=re.IGNORECASE) for pat in drop_patterns):
                continue
        kept_lines.append(_strip_noise_prefixes(ln))

    cleaned = "\n".join(kept_lines)

    # Prefer blank-line paragraphs if they exist.
    toks = [p.strip() for p in re.split(r"\n\s*\n+", cleaned) if p and p.strip()]
    if len(toks) >= 3:
        paras = [_normalize_whitespace(p, drop_patterns=drop_patterns) for p in toks]
        return [p for p in paras if p]

    # Otherwise merge lines into paragraph-like blocks.
    lines = [ln.strip() for ln in kept_lines if ln.strip()]
    if not lines:
        return []

    merged: List[str] = []
    buf: List[str] = []
    for ln in lines:
        buf.append(ln)
        joined = " ".join(buf)
        strong_end = bool(re.search(r"[\.?!:;]\s*$", ln))
        long_enough = len(joined) >= 240
        if strong_end or long_enough:
            merged.append(" ".join(buf))
            buf = []
    if buf:
        merged.append(" ".join(buf))

    paras = [_normalize_whitespace(p, drop_patterns=drop_patterns) for p in merged]
    return [p for p in paras if p]


def _count_tokens(s: str) -> int:
    return len(_TOKEN_RE.findall(s))


def _chunk_paragraphs_to_passages(
    paragraphs: List[str],
    *,
    target_tokens: int = 260,
    overlap_tokens: int = 60,
    min_tokens: int = 40,
    max_tokens: int = 420,
    split_enumerations: bool = True,
) -> List[str]:
    """
    Chunk paragraph candidates into embedding-friendly passages with overlap.
    """
    if not paragraphs:
        return []

    paras = [p.strip() for p in paragraphs if p and p.strip()]
    if not paras:
        return []

    # Expand overlong paragraphs before chunking.
    expanded: List[str] = []
    for p in paras:
        if _count_tokens(p) > max_tokens and split_enumerations:
            parts = _split_on_list_markers(p)
            expanded.extend([x.strip() for x in parts if x and x.strip()])
        else:
            expanded.append(p)
    paras = expanded

    passages: List[str] = []
    cur_parts: List[str] = []
    cur_tokens = 0

    def finalize_current() -> Optional[str]:
        nonlocal cur_parts, cur_tokens
        if not cur_parts:
            return None
        txt = "\n\n".join(cur_parts).strip()
        cur_parts = []
        cur_tokens = 0
        if not txt:
            return None
        # Hard trim if needed
        if _count_tokens(txt) > max_tokens:
            w = _TOKEN_RE.findall(txt)[:max_tokens]
            txt = " ".join(w)
        return txt

    def overlap_seed(prev: str) -> Tuple[List[str], int]:
        if overlap_tokens <= 0:
            return [], 0
        words = _TOKEN_RE.findall(prev)
        ov = words[-overlap_tokens:] if len(words) > overlap_tokens else words
        if not ov:
            return [], 0
        return [" ".join(ov)], len(ov)

    for para in paras:
        pt = _count_tokens(para)
        if pt == 0:
            continue

        if cur_parts and (cur_tokens + pt) > target_tokens and cur_tokens >= min_tokens:
            finished = finalize_current()
            if finished:
                passages.append(finished)
                cur_parts, cur_tokens = overlap_seed(finished)

        cur_parts.append(para)
        cur_tokens += pt

    finished = finalize_current()
    if finished:
        passages.append(finished)

    # Drop tiny overlap-only chunks
    return [p for p in passages if _count_tokens(p) >= min_tokens]


# ----------------------------- PDF extraction (robust + fallback) -----------------------------

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


def _extract_pages_with_pymupdf(pdf_path: Path) -> Iterable[Tuple[int, str]]:
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


def _extract_pages_with_pymupdf_blocks(pdf_path: Path) -> Iterable[Tuple[int, List[str]]]:
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
                blocks = page.get_text("blocks", sort=True)  # type: ignore[call-arg]
            except TypeError:
                blocks = page.get_text("blocks")

            out: List[str] = []
            for b in blocks or []:
                # (x0, y0, x1, y1, text, block_no, block_type)
                try:
                    txt = b[4]
                except Exception:
                    txt = ""
                if txt:
                    out.append(str(txt))
            yield i, out
    finally:
        close_method = getattr(doc, "close", None)
        if callable(close_method):
            close_method()


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
        yield from _extract_pages_with_pymupdf(pdf_path)
    except Exception:
        yield from _extract_pages_with_pdfplumber(pdf_path)


def extract_pages_blocks(pdf_path: Path) -> Iterable[Tuple[int, List[str]]]:
    """
    Block extraction with fallback:
      - Prefer PyMuPDF blocks
      - Fallback to pdfplumber page text (as single block)
    """
    try:
        yield from _extract_pages_with_pymupdf_blocks(pdf_path)
    except Exception:
        for page_no, txt in extract_pages_text(pdf_path):
            yield page_no, [txt]


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
) -> pd.DataFrame:
    """
    Reads all PDFs in input_dir, extracts and filters units page-by-page,
    labels them with law_type (PDF stem) and page number, writes Parquet.
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
            for page_no, blocks in extract_pages_blocks(pdf_path):
                if not blocks:
                    continue

                # Create paragraph candidates
                if len(blocks) == 1:
                    paras = _extract_paragraph_candidates_from_page_text(blocks[0], drop_patterns=drop_patterns)
                else:
                    paras = []
                    for b in blocks:
                        nb = _normalize_whitespace(b, drop_patterns=drop_patterns)
                        if nb:
                            paras.append(nb)

                if split_enumerations and paras:
                    expanded: List[str] = []
                    for ptxt in paras:
                        parts = _split_on_list_markers(ptxt)
                        expanded.extend([x.strip() for x in parts if x and x.strip()])
                    paras = expanded

                units = paras if unit == "paragraph" else _chunk_paragraphs_to_passages(
                    paras,
                    target_tokens=target_tokens,
                    overlap_tokens=overlap_tokens,
                    min_tokens=min_passage_tokens,
                    max_tokens=max_passage_tokens,
                    split_enumerations=split_enumerations,
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
    p = argparse.ArgumentParser(description="Extract labeled units from RIS PDFs and write Parquet (IR-optimized).")
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
        help="Extraction unit: sentence (legacy), paragraph (blocks), passage (chunked for embeddings; default).",
    )
    p.add_argument("--target_tokens", type=int, default=260, help="Target token size for passage chunks.")
    p.add_argument("--overlap_tokens", type=int, default=60, help="Token overlap between passages.")
    p.add_argument("--min_passage_tokens", type=int, default=40, help="Minimum token size for retained passages.")
    p.add_argument("--max_passage_tokens", type=int, default=420, help="Hard maximum token size for a passage.")
    p.add_argument("--no_dedupe_within_pdf", action="store_true", help="Disable duplicate removal within each PDF.")
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
