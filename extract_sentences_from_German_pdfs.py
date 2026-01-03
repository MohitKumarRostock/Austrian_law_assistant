#!/usr/bin/env python3
"""
Extract (meaningful) sentences from RIS law PDFs, label with law type + page number,
and write to a Parquet file. Also prints ONE example sentence per PDF.

Design goal (v4): maximize recall for semantically relevant content.
- Avoid losing content due to header/footer patterns being merged into real text.
- Avoid false sentence boundaries in Austrian legal references (e.g., "§ 1. Abs. 2").
- Keep filtering conservative (recall-first); default drop patterns only remove
  trivial page/URL artifacts, while common RIS field labels are stripped rather than dropped.

Folder layout (default):
  ris_pdfs/
    ABGB.pdf
    ArbVG.pdf
    StPO.pdf
    ...

Output (default): ris_sentences.parquet
Columns: sentence_id, law_type, page, sentence, source_file
"""

from __future__ import annotations

import argparse
import importlib
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, cast

import pandas as pd


# ----------------------------- Sentence splitting & filtering -----------------------------

_ABBREV_PLACEHOLDER = "∯"  # unlikely to appear in RIS law texts


# Trivial noise lines we can safely drop (FULL line match only).
_DEFAULT_DROP_PATTERNS: List[str] = [
    r"^\s*Seite\s+\d+(\s+von\s+\d+)?\s*$",
    r"^\s*www\.ris\.bka\.gv\.at\s*$",
    r"^\s*RIS\s*$",
]

# Some RIS/metadata headers are better *stripped* than dropped if they appear inline.
# We only apply these at LINE START, before collapsing newlines.
_NOISE_PREFIX_STRIP_REGEXES: List[re.Pattern[str]] = [
    re.compile(r"^\s*Bundesrecht\s+konsolidiert\s*:?\s*", flags=re.IGNORECASE),
    re.compile(r"^\s*Gesamte\s+Rechtsvorschrift(?:\s+für)?\s*:?\s*", flags=re.IGNORECASE),
    re.compile(r"^\s*RIS\s*-\s*Rechtsinformationssystem(?:\s+des\s+Bundes)?\s*:?\s*", flags=re.IGNORECASE),
    re.compile(r"^\s*Rechtsinformationssystem(?:\s+des\s+Bundes)?\s*:?\s*", flags=re.IGNORECASE),
    re.compile(r"^\s*CELEX-?Nr\.?\s*:?\s*", flags=re.IGNORECASE),
    # Common RIS field labels on the title page: keep suggesting value, but strip label.
    re.compile(
        r"^\s*(?:Kundmachungsorgan|Gesetzesnummer|Dokumenttyp|Kurztitel|Langtitel|Abkürzung|Fassung\s+vom|"
        r"Inkrafttretensdatum|Zuletzt\s+geändert\s+durch|Zuletzt\s+aktualisiert|Norm|Anmerkung|Schlagworte|"
        r"Dokumentnummer|Stand)\s*:?\s*",
        flags=re.IGNORECASE,
    ),
]

# RIS inline footer that sometimes appears mid-extraction.
_RIS_INLINE_FOOTER_RE = re.compile(
    r"^\s*www\.ris\.bka\.gv\.at\s+Seite\s+\d+\s+von\s+\d+\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)

_RIS_URL_ONLY_RE = re.compile(r"^\s*www\.ris\.bka\.gv\.at\s*$", flags=re.IGNORECASE | re.MULTILINE)

# Headings: keep §/Art numbers with the following text (avoid creating standalone fragments).
_SECTION_HEADING_LINE_RE = re.compile(
    r"^\s*§\s*\.?\s*(\d+[A-Za-z]?)\s*(?:\.\s*|\s+)",
    flags=re.MULTILINE,
)
_ARTICLE_HEADING_LINE_RE = re.compile(
    r"^\s*(?:Art\.?|Artikel)\s*(\d+[A-Za-z]?|[IVXLCDM]+)\s*(?:\.\s*|\s+|:\s*)",
    flags=re.MULTILINE,
)

# Enumerations at line start: "1." / "a." / "I."  -> "1)" etc.
_ENUM_LINE_RE = re.compile(r"^\s*(\d{1,3})\.\s+", flags=re.MULTILINE)
_ENUM_ALPHA_LINE_RE = re.compile(r"^\s*([A-Za-z])\.\s+", flags=re.MULTILINE)
_ENUM_ROMAN_LINE_RE = re.compile(r"^\s*([IVXLCDM]{1,8})\.\s+", flags=re.MULTILINE)

# Enumerations after ":" or ";" on the same line: "... gilt: 1. ..." -> "... gilt: 1) ..."
_ENUM_AFTER_COLON_SEMI_RE = re.compile(r"([:;])\s*(\d{1,3})\.\s+")

# v4: protect dots in *legal references* that are not sentence ends, e.g. "§ 1. Abs. 2"
# NOTE: These patterns are applied via `_protect_abbreviations`, which replaces '.' inside matches.
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

# Regex-based abbreviation protection (RIS / Austrian statutory texts).
_ABBREVIATION_REGEXES: List[re.Pattern[str]] = [
    *_LEGAL_REF_DOT_REGEXES,

    # Multi-dot abbreviations first (avoid partial matches like "z." before "z.B.")
    re.compile(r"\bz\.\s*B\.", flags=re.IGNORECASE),
    re.compile(r"\bu\.\s*a\.", flags=re.IGNORECASE),
    re.compile(r"\bd\.\s*h\.", flags=re.IGNORECASE),
    re.compile(r"\bi\.\s*d\.\s*R\.", flags=re.IGNORECASE),
    re.compile(r"\bu\.\s*U\.", flags=re.IGNORECASE),

    # Common abbreviations in statutes
    re.compile(r"\bAbs\.", flags=re.IGNORECASE),
    re.compile(r"\bArt\.", flags=re.IGNORECASE),
    re.compile(r"\bZl\.", flags=re.IGNORECASE),
    # Protect "Z." but not when it's part of a multi-dot abbreviation like "z.B."
    re.compile(r"\bZ\.(?!\s*[A-Za-z]\.)", flags=re.IGNORECASE),
    re.compile(r"\blit\.", flags=re.IGNORECASE),
    re.compile(r"\bNr\.", flags=re.IGNORECASE),
    re.compile(r"\bZiff\.", flags=re.IGNORECASE),
    re.compile(r"\biVm\.", flags=re.IGNORECASE),
    re.compile(r"\biSd\.", flags=re.IGNORECASE),
    re.compile(r"\biSv\.", flags=re.IGNORECASE),
    re.compile(r"\bbzw\.", flags=re.IGNORECASE),
    re.compile(r"\bua\.", flags=re.IGNORECASE),
    re.compile(r"\bvgl\.", flags=re.IGNORECASE),
    re.compile(r"\bmwN\.", flags=re.IGNORECASE),

    # Titles / generic
    re.compile(r"\bDr\.", flags=re.IGNORECASE),
    re.compile(r"\bProf\.", flags=re.IGNORECASE),

    # Gazette abbreviations (often appear in amendments/refs)
    re.compile(r"\bBGBl\.", flags=re.IGNORECASE),
    re.compile(r"\bRGBl\.", flags=re.IGNORECASE),
    re.compile(r"\bStGBl\.", flags=re.IGNORECASE),
    re.compile(r"\bdRGBl\.", flags=re.IGNORECASE),
    re.compile(r"\bJGS\.", flags=re.IGNORECASE),
]


# v4: split list items after normalization ("1)" / "a)" / "(1)" / bullets) into separate sentences.
_LIST_MARKER_RE = re.compile(
    r"""(?x)
    (?:
        (?<=\s) | ^
    )
    (?P<marker>
        # Parenthesized items: (1), (2), (a), (I) ...
        \(\s*\d{1,3}\s*\)
        |\(\s*[A-Za-z]\s*\)
        |\(\s*[IVXLCDM]{1,8}\s*\)
        # Non-parenthesized items: 1), a), I)
        |\d{1,3}\)
        |[A-Za-z]\)
        |[IVXLCDM]{1,8}\)
        # Bullets/dashes
        |[•·]
        |[-–—]
    )
    \s+
    """
)


def _strip_noise_prefixes(line: str) -> str:
    """
    Strip known RIS header/field prefixes, but never delete the whole line unless it becomes empty.
    Applies repeatedly (some PDFs stack multiple prefixes).
    """
    s = line
    for _ in range(3):  # prevent pathological loops
        before = s
        for rx in _NOISE_PREFIX_STRIP_REGEXES:
            s = rx.sub("", s)
        if s == before:
            break
        s = s.strip()
        if not s:
            break
    return s.strip()


def _normalize_whitespace(raw: str, *, drop_patterns: Optional[List[str]] = None) -> str:
    """
    Recall-first normalization.
    - Keep original line boundaries long enough to remove/strip RIS headers and page artefacts safely.
    - Dehyphenate across line breaks.
    - Normalize headings and list markers.
    - Collapse remaining whitespace to single spaces.
    """
    if not raw:
        return ""

    s = raw.replace("\u00ad", "")  # soft hyphen

    # Remove RIS inline footers/URL-only lines (multiline).
    s = _RIS_INLINE_FOOTER_RE.sub("", s)
    s = _RIS_URL_ONLY_RE.sub("", s)

    # Drop/strip noise line-by-line BEFORE newlines are collapsed.
    patterns = drop_patterns if drop_patterns is not None else _DEFAULT_DROP_PATTERNS
    cleaned_lines: List[str] = []
    for ln in s.splitlines():
        ln = ln.strip()
        if not ln:
            continue

        # Drop only if the ENTIRE line matches a trivial noise pattern.
        is_noise = False
        for pat in patterns:
            if re.fullmatch(pat, ln, flags=re.IGNORECASE):
                is_noise = True
                break
        if is_noise:
            continue

        ln = _strip_noise_prefixes(ln)
        if ln:
            cleaned_lines.append(ln)

    s = "\n".join(cleaned_lines).strip()
    if not s:
        return ""

    # De-hyphenate across line breaks: "Verant-\nwortung" -> "Verantwortung"
    s = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", s)

    # Normalize headings so §/Art numbers are kept with following text.
    s = _SECTION_HEADING_LINE_RE.sub(r"§\1 ", s)
    s = _ARTICLE_HEADING_LINE_RE.sub(r"Art. \1 ", s)

    # Preserve list markers at start of lines, BEFORE remaining newlines are collapsed.
    s = _ENUM_ROMAN_LINE_RE.sub(r"\1) ", s)
    s = _ENUM_ALPHA_LINE_RE.sub(r"\1) ", s)
    s = _ENUM_LINE_RE.sub(r"\1) ", s)

    # Preserve enumerations after ':' or ';'
    s = _ENUM_AFTER_COLON_SEMI_RE.sub(r"\1 \2) ", s)

    # Remaining newlines -> spaces
    s = re.sub(r"\s*\n\s*", " ", s)

    # Collapse whitespace
    s = re.sub(r"[ \t\r\f\v]+", " ", s).strip()
    return s


def _protect_abbreviations(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Replace dots in known abbreviations / legal reference fragments with a placeholder so
    sentence splitting doesn't break on them.
    """
    placeholder_map: Dict[str, str] = {}
    out = text

    def _repl(m: re.Match[str]) -> str:
        abbr = m.group(0)
        safe = abbr.replace(".", _ABBREV_PLACEHOLDER)
        placeholder_map[safe] = abbr
        return safe

    for rx in _ABBREVIATION_REGEXES:
        out = rx.sub(_repl, out)

    return out, placeholder_map


def _restore_abbreviations(text: str, mapping: Dict[str, str]) -> str:
    out = text
    # Restore longer placeholders first (avoid partial overlaps).
    for safe in sorted(mapping.keys(), key=len, reverse=True):
        out = out.replace(safe, mapping[safe])
    return out


_DANGLING_REF_RE = re.compile(
    r"""
    (?:                           # end-of-fragment patterns that often indicate a false split
        §{1,2}\s*\d+[A-Za-z]?      # § 1 / §§ 1
        |(?:Art\.?|Artikel)\s*\d+[A-Za-z]?  # Art. 10 / Artikel 10
    )
    \.\s*$                         # ends with a dot
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

_CONTINUATION_START_RE = re.compile(
    r"""
    ^\s*
    (?:
        \(?\s*\d{1,3}\s*\)?        # (1) / 1 / ( 1 )
        |Abs\.?|Absatz
        |Z\.?|Ziffer|Ziff\.
        |lit\.?
        |Satz
        |Nr\.?
        |[A-Za-zÄÖÜäöüß]           # any letter (headings like "Geltungsbereich")
        |§{1,2}|Art\.?|Artikel
    )
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)


def _merge_false_boundaries(sentences: List[str]) -> List[str]:
    """
    Repair common false splits created by dots in legal references, e.g.
      "... gemäß § 1." + "Abs. 2 gilt ..." -> one sentence.

    This is intentionally biased toward MERGING rather than dropping.
    """
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


def _split_on_list_markers(sentence: str) -> List[str]:
    """
    Split a sentence into multiple sentences whenever it contains multiple list markers.

    Conservative:
      - Only split when there are at least two markers.
      - Never split at the first marker; keep prefix with item 1.
    """
    matches = list(_LIST_MARKER_RE.finditer(sentence))
    if len(matches) < 2:
        return [sentence]

    # Determine plausible item starts (avoid splitting on a single dash used mid-sentence).
    candidates: List[int] = []
    for m in matches:
        marker = m.group("marker")
        start = m.start("marker")

        if marker in {"-", "–", "—"}:
            if start == 0:
                candidates.append(start)
                continue

            prev = sentence[max(0, start - 2):start]
            if ":" in prev or ";" in prev or "." in prev:
                candidates.append(start)
                continue

            # Avoid splitting hyphenated words like "rechts-".
            if start > 0 and sentence[start - 1].isalpha():
                continue

            candidates.append(start)
        else:
            candidates.append(start)

    candidates = sorted(set(candidates))
    if len(candidates) < 2:
        return [sentence]

    split_points = candidates[1:]
    out: List[str] = []
    last = 0
    for sp in split_points:
        chunk = sentence[last:sp].strip()
        if chunk:
            out.append(chunk)
        last = sp

    tail = sentence[last:].strip()
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
    Recall-first rule-based sentence splitter (dependency-light).

    Notes:
      - Uses normalization that strips (not drops) common RIS prefixes.
      - Protects common abbreviations + legal-reference dots before splitting.
      - Repairs frequent false boundaries via merge step.
      - Optionally splits within long sentences on list markers ("1)", "(1)", "a)", bullets).
    """
    normalized = _normalize_whitespace(text, drop_patterns=drop_patterns_for_normalization)
    if not normalized:
        return []

    protected, mapping = _protect_abbreviations(normalized)

    # Split on punctuation + whitespace, but require a plausible sentence start afterwards.
    # This reduces accidental splits in weird PDF extractions while keeping recall high.
    parts = re.split(r"(?<=[.!?])\s+(?=(?:[A-ZÄÖÜ0-9\"“„\(\[]|§))", protected)

    sentences: List[str] = []
    for p in parts:
        p = _restore_abbreviations(p, mapping).strip()
        if p:
            sentences.append(p)

    sentences = _merge_false_boundaries(sentences)

    if split_enumerations:
        expanded: List[str] = []
        for s in sentences:
            for sub in _split_on_list_markers(s):
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
    Extremely permissive, recall-first filter.

    Objective: do not drop semantically relevant statutory text.

    This function only removes:
      - empty strings
      - strings that are purely punctuation/whitespace
      - trivial RIS/page artefacts when they match *entirely* (fullmatch) against drop_patterns

    The parameters are retained for backwards-compatibility with the CLI, but the current
    implementation deliberately does not apply aggressive length/digit heuristics.
    """
    s = sentence.strip()
    if not s:
        return False

    if drop_patterns:
        for pat in drop_patterns:
            if re.fullmatch(pat, s, flags=re.IGNORECASE):
                return False

    # Drop pure punctuation/whitespace (including common bullet dashes).
    if re.fullmatch(r"[\s\-\–\—\.,;:/()\[\]{}]+", s):
        return False

    return True

# ----------------------------- PDF extraction (robust + fallback) -----------------------------


def _extract_pages_with_pymupdf(pdf_path: Path) -> Iterable[Tuple[int, str]]:
    """
    Primary extractor: PyMuPDF.

    We prefer "sort=True" (when available) to improve reading order and reduce
    accidental header/body interleaving.
    """
    pymupdf_mod: Any = None
    errors: List[str] = []

    for module_name in ("pymupdf", "fitz"):
        try:
            candidate = importlib.import_module(module_name)

            has_document = hasattr(candidate, "Document")
            has_open = hasattr(candidate, "open")
            if not (has_document or has_open):
                raise ImportError(
                    f"Imported '{module_name}' does not look like PyMuPDF (missing Document/open). "
                    "You may have installed the unrelated 'fitz' package from PyPI."
                )
            pymupdf_mod = candidate
            break
        except Exception as e:
            errors.append(f"{module_name}: {e}")

    if pymupdf_mod is None:
        raise RuntimeError("PyMuPDF import failed. Tried pymupdf, fitz. " + " | ".join(errors))

    Document = cast(Optional[Callable[..., Any]], getattr(pymupdf_mod, "Document", None))
    open_fn = cast(Optional[Callable[..., Any]], getattr(pymupdf_mod, "open", None))

    if callable(Document):
        doc = Document(str(pdf_path))
    elif callable(open_fn):
        doc = open_fn(str(pdf_path))
    else:
        raise RuntimeError("PyMuPDF module loaded but provides neither Document nor open().")

    try:
        if getattr(doc, "needs_pass", False):
            raise RuntimeError(f"Encrypted PDF (password needed): {pdf_path.name}")

        for i, page in enumerate(doc, start=1):  # 1-based pages
            # Prefer sort=True if supported by the installed PyMuPDF.
            try:
                txt = page.get_text("text", sort=True)  # type: ignore[call-arg]
            except TypeError:
                txt = page.get_text("text")
            yield i, (txt or "")
    finally:
        doc.close()


def _extract_pages_with_pdfplumber(pdf_path: Path) -> Iterable[Tuple[int, str]]:
    """Fallback extractor: pdfplumber."""
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
    min_chars: int = 10,
    min_alpha_tokens: int = 1,
    max_digit_ratio: float = 0.85,
    drop_patterns: Optional[List[str]] = None,
    print_example_per_pdf: bool = True,
    split_enumerations: bool = True,
    filter_sentences: bool = True,
) -> pd.DataFrame:
    """
    Reads all PDFs in input_dir, extracts and filters sentences page-by-page,
    labels them with law_type (PDF stem) and page number, writes Parquet.

    Pipeline is unchanged:
      PDFs -> page text -> sentence split -> (optional) filter -> Parquet

    Notes for maximal recall:
      - Set --no_filter to keep everything post-split.
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
    first_example: Dict[str, Tuple[int, str]] = {}

    for pdf_path in pdf_paths:
        law_type = pdf_path.stem

        for page_no, page_text in extract_pages_text(pdf_path):
            if not page_text or not page_text.strip():
                continue

            # Normalize inside the splitter (single normalization pass).
            for sent in split_into_sentences(
                page_text,
                split_enumerations=split_enumerations,
                drop_patterns_for_normalization=drop_patterns,
            ):
                sent = sent.strip()
                if not sent:
                    continue

                if filter_sentences and not is_semantically_meaningful(
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
                p, s = first_example[law_type]
                print(f"  - {pdf_path.name}: p.{p}: {s}")
            else:
                print(f"  - {pdf_path.name}: (no sentences retained)")

    df = pd.DataFrame(rows, columns=["sentence_id", "law_type", "page", "sentence", "source_file"])

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

    # Defaults tuned for recall-first behavior.
    p.add_argument("--min_chars", type=int, default=10, help="Minimum character length for a sentence.")
    p.add_argument("--min_alpha_tokens", type=int, default=1, help="Minimum count of alphabetic tokens.")
    p.add_argument("--max_digit_ratio", type=float, default=0.85, help="Max allowed digit/char ratio.")

    p.add_argument("--no_split_enumerations", action="store_true", help="Disable splitting on list markers like '1)'.")
    p.add_argument("--no_filter", action="store_true", help="Disable semantic filtering (maximal recall; includes more noise).")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv or [])

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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
