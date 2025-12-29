#!/usr/bin/env python3
"""
Extract (meaningful) sentences from RIS law PDFs, label with law type + page number,
and write to a Parquet file. Also prints ONE example sentence per PDF.

Folder layout:
  ris_pdfs/
    ABGB.pdf
    ArbVG.pdf
    StPO.pdf
    ...

Dependencies:
  - pandas
  - PyMuPDF (pip install PyMuPDF)  -> provides module "fitz"
  - (optional fallback) pdfplumber (pip install pdfplumber)
  - parquet engine: pyarrow (recommended) or fastparquet
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict, Any
import argparse
import re
import sys

import pandas as pd


# ----------------------------- Sentence splitting & filtering -----------------------------

_ABBREVIATIONS = [
    # Common German / legal abbreviations that should not trigger sentence splits
    "Abs.", "Art.", "Z.", "Zl.", "lit.", "Nr.", "Ziff.", "idF", "iVm.", "iSd.", "iSv.",
    "zB.", "bzw.", "ua.", "u.a.", "vgl.", "mwN.", "d.h.", "Dr.", "Prof.", "S.", "§§",
]

_DEFAULT_DROP_PATTERNS = [
    # Common RIS-ish headers/footers/noise. Extend as you discover recurring lines.
    r"^\s*Seite\s+\d+(\s+von\s+\d+)?\s*$",
    r"^\s*RIS\s*$",
    r"^\s*Bundesrecht\s+konsolidiert\s*$",
    r"^\s*Zuletzt\s+aktualisiert.*$",
    r"^\s*Stand\s+\d{1,2}\.\d{1,2}\.\d{2,4}\s*$",
    r"^\s*Dokumentnummer.*$",
]


def _normalize_whitespace(s: str) -> str:
    s = s.replace("\u00ad", "")  # soft hyphen
    # de-hyphenate across line breaks: "Verant-\nwortung" -> "Verantwortung"
    s = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", s)
    # remaining newlines -> spaces
    s = re.sub(r"\s*\n\s*", " ", s)
    # collapse whitespace
    s = re.sub(r"[ \t\r\f\v]+", " ", s).strip()
    return s


def _protect_abbreviations(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Replace dots in known abbreviations with a placeholder so sentence splitting
    doesn't break on them.
    """
    placeholder_map: Dict[str, str] = {}
    out = text
    for abbr in _ABBREVIATIONS:
        safe = abbr.replace(".", "∯")  # unlikely char
        if abbr in out:
            placeholder_map[safe] = abbr
            out = out.replace(abbr, safe)
    return out, placeholder_map


def _restore_abbreviations(text: str, placeholder_map: Dict[str, str]) -> str:
    out = text
    for safe, abbr in placeholder_map.items():
        out = out.replace(safe, abbr)
    return out


def split_into_sentences(text: str) -> List[str]:
    """
    Conservative rule-based sentence splitter (dependency-light).
    """
    text = _normalize_whitespace(text)
    if not text:
        return []

    protected, mapping = _protect_abbreviations(text)
    parts = re.split(r"(?<=[.!?])\s+", protected)

    sentences: List[str] = []
    for p in parts:
        p = _restore_abbreviations(p, mapping).strip()
        if p:
            sentences.append(p)
    return sentences


def is_semantically_meaningful(
    sentence: str,
    *,
    min_chars: int = 15,
    min_alpha_tokens: int = 2,
    max_digit_ratio: float = 0.65,
    drop_patterns: Optional[List[str]] = None,
) -> bool:
    """
    Heuristic filter:
      - drop empty, headers/footers, numeric-only lines
      - require at least some alphabetic content
      - reject overly digit-heavy fragments with little language content
    """
    s = sentence.strip()
    if not s:
        return False

    if drop_patterns:
        for pat in drop_patterns:
            if re.match(pat, s, flags=re.IGNORECASE):
                return False

    # Must contain at least one letter (incl. umlauts/ß)
    if not re.search(r"[A-Za-zÄÖÜäöüß]", s):
        return False

    # Numeric/punctuation-only (covers §, etc. if no letters)
    if re.fullmatch(r"[\d\s§\-\–\—\.,;:/()\[\]{}]+", s):
        return False

    # Token heuristics
    alpha_tokens = re.findall(r"[A-Za-zÄÖÜäöüß]{2,}", s)
    if len(s) < min_chars:
        # keep short sentences if they still have enough words
        return len(alpha_tokens) >= min_alpha_tokens

    # Digit ratio heuristic
    digits = sum(ch.isdigit() for ch in s)
    if digits / max(len(s), 1) > max_digit_ratio:
        if len(alpha_tokens) < (min_alpha_tokens + 1):
            return False

    if len(alpha_tokens) < min_alpha_tokens:
        return False

    return True


# ----------------------------- PDF extraction (robust + fallback) -----------------------------

def _extract_pages_with_pymupdf(pdf_path: Path) -> Iterable[Tuple[int, str]]:
    """
    Primary extractor: PyMuPDF (module name: fitz).
    """
    try:
        import fitz  # PyMuPDF
        if not hasattr(fitz, "open"):
            raise ImportError("Imported 'fitz' does not look like PyMuPDF (missing fitz.open).")
    except Exception as e:
        raise RuntimeError(f"PyMuPDF import failed: {e}") from e

    doc = fitz.open(str(pdf_path))
    try:
        if getattr(doc, "needs_pass", False):
            raise RuntimeError(f"Encrypted PDF (password needed): {pdf_path.name}")
        for i, page in enumerate(doc, start=1):  # 1-based pages
            yield i, (page.get_text("text") or "")
    finally:
        doc.close()


def _extract_pages_with_pdfplumber(pdf_path: Path) -> Iterable[Tuple[int, str]]:
    """
    Fallback extractor: pdfplumber.
    """
    import pdfplumber  # optional dependency
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            yield i, (page.extract_text() or "")


def extract_pages_text(pdf_path: Path) -> Iterable[Tuple[int, str]]:
    """
    IMPORTANT: Uses 'yield from' so exceptions raised during iteration are caught,
    enabling a real fallback.
    """
    try:
        yield from _extract_pages_with_pymupdf(pdf_path)
        return
    except Exception as e_pymupdf:
        try:
            yield from _extract_pages_with_pdfplumber(pdf_path)
            return
        except Exception as e_pdfplumber:
            raise RuntimeError(
                f"Failed to extract text from {pdf_path.name}.\n"
                f"PyMuPDF error: {type(e_pymupdf).__name__}: {e_pymupdf}\n"
                f"pdfplumber error: {type(e_pdfplumber).__name__}: {e_pdfplumber}\n"
                f"Tip: install fallback with `pip install pdfplumber`."
            ) from e_pdfplumber


# ----------------------------- Main pipeline -----------------------------

def ris_pdfs_to_parquet(
    input_dir: str | Path = "ris_pdfs",
    output_parquet: str | Path = "ris_sentences.parquet",
    *,
    recursive: bool = False,
    min_chars: int = 15,
    min_alpha_tokens: int = 2,
    max_digit_ratio: float = 0.65,
    drop_patterns: Optional[List[str]] = None,
    print_example_per_pdf: bool = True,
) -> pd.DataFrame:
    """
    Reads all PDFs in input_dir, extracts and filters sentences page-by-page,
    labels them with law_type (PDF stem) and page number, writes Parquet.

    Also prints one example sentence per PDF (the first retained sentence encountered).
    """
    input_dir = Path(input_dir)
    output_parquet = Path(output_parquet)

    if drop_patterns is None:
        drop_patterns = list(_DEFAULT_DROP_PATTERNS)

    pdf_paths = sorted(input_dir.rglob("*.pdf") if recursive else input_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in: {input_dir.resolve()}")

    rows: List[Dict[str, Any]] = []
    sentence_id = 0
    first_example: Dict[str, Tuple[int, str]] = {}  # law_type -> (page, sentence)

    for pdf_path in pdf_paths:
        law_type = pdf_path.stem

        for page_no, page_text in extract_pages_text(pdf_path):
            page_text = _normalize_whitespace(page_text)
            if not page_text:
                continue

            for sent in split_into_sentences(page_text):
                sent = sent.strip()
                if not is_semantically_meaningful(
                    sent,
                    min_chars=min_chars,
                    min_alpha_tokens=min_alpha_tokens,
                    max_digit_ratio=max_digit_ratio,
                    drop_patterns=drop_patterns,
                ):
                    continue

                if law_type not in first_example:
                    first_example[law_type] = (page_no, sent)

                sentence_id += 1
                rows.append(
                    {
                        "sentence_id": sentence_id,
                        "law_type": law_type,
                        "page": page_no,
                        "sentence": sent,
                        "source_file": pdf_path.name,
                    }
                )

    if print_example_per_pdf:
        print("Example sentence per PDF (first retained sentence found):")
        for pdf_path in pdf_paths:
            law_type = pdf_path.stem
            if law_type in first_example:
                page_no, sent = first_example[law_type]
                print(f"- {law_type} (page {page_no}): {sent}")
            else:
                print(f"- {law_type}: [no retained sentence found]")

    df = pd.DataFrame(rows, columns=["sentence_id", "law_type", "page", "sentence", "source_file"])

    # Write Parquet (requires pyarrow or fastparquet)
    try:
        df.to_parquet(output_parquet, index=False)
    except Exception as e:
        raise RuntimeError(
            f"Failed to write Parquet to {output_parquet.resolve()}.\n"
            f"Install a parquet engine, e.g.: `pip install pyarrow` (recommended) or `pip install fastparquet`.\n"
            f"Original error: {type(e).__name__}: {e}"
        ) from e

    print(f"\nWrote {len(df):,} sentences to: {output_parquet.resolve()}")
    return df


# ----------------------------- CLI -----------------------------

def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract labeled sentences from RIS PDFs and write Parquet.")
    p.add_argument("--input_dir", default="ris_pdfs", help="Folder containing RIS PDF files.")
    p.add_argument("--output", default="ris_sentences.parquet", help="Output Parquet file path.")
    p.add_argument("--recursive", action="store_true", help="Search PDFs recursively.")
    p.add_argument("--no_print_examples", action="store_true", help="Do not print per-PDF example sentences.")
    p.add_argument("--min_chars", type=int, default=15, help="Minimum character length for a sentence.")
    p.add_argument("--min_alpha_tokens", type=int, default=2, help="Minimum count of alphabetic tokens.")
    p.add_argument("--max_digit_ratio", type=float, default=0.65, help="Max allowed digit/char ratio.")
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
