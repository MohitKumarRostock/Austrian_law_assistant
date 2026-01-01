#!/usr/bin/env python3
"""
RIS Austrian-law sentence extraction test harness.

Place this file next to:
  - extract_sentences_from_German_pdfs.py
  - ris_pdfs/   (optional for PDF smoke tests)

Run:
  python test_harness.py
"""

from __future__ import annotations

from pathlib import Path
import importlib.util
import textwrap


def load_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("ris_extract", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def analyze_text(mod, label: str, raw: str):
    norm = mod._normalize_whitespace(raw)
    sents = mod.split_into_sentences(norm)
    kept = [
        s for s in sents
        if mod.is_semantically_meaningful(
            s,
            drop_patterns=mod._DEFAULT_DROP_PATTERNS,
        )
    ]

    print("=" * 90)
    print(f"{label}")
    print("-" * 90)
    print("RAW:")
    print(raw)
    print("\nNORMALIZED:")
    print(norm)
    print("\nSPLIT SENTENCES:")
    for i, s in enumerate(sents, 1):
        print(f"  {i:02d}. {s}")
    print("\nKEPT (meaningful) SENTENCES:")
    for i, s in enumerate(kept, 1):
        print(f"  {i:02d}. {s}")
    print()

    return norm, sents, kept


def main() -> int:
    script = Path("extract_sentences_from_German_pdfs.py").resolve()
    if not script.exists():
        raise FileNotFoundError(f"Missing {script}. Run this next to the extractor script.")

    mod = load_module(script)

    # -------------------------
    # 1) Representative snippet tests
    # -------------------------

    # A) Section heading normalization: § lines should be preserved as anchors (not dropped)
    raw_a = textwrap.dedent("""\
        § 871.
        Irrtum
        (1) Wer über den Inhalt seiner Erklärung in einem Irrtum befangen war, kann diese anfechten.
        """)
    norm, sents, kept = analyze_text(mod, "A) §-heading anchor retention", raw_a)
    assert any("§871" in s for s in kept), "Expected §871 anchor to appear in kept sentences."

    # B) Article heading normalization: Art / Artikel / roman numerals
    raw_b = textwrap.dedent("""\
        Art 10
        Datenschutz
        (1) Jede Person hat Anspruch auf Geheimhaltung der sie betreffenden personenbezogenen Daten.
        """)
    norm, sents, kept = analyze_text(mod, "B) Art-heading anchor retention (Art 10)", raw_b)
    assert any("Art. 10" in s for s in kept), "Expected 'Art. 10' anchor to appear in kept sentences."

    raw_b2 = textwrap.dedent("""\
        Art. I
        Änderung des Bundes-Verfassungsgesetzes
        (1) In Art. 10 Abs. 1 wird folgender Z 99 eingefügt.
        """)
    norm, sents, kept = analyze_text(mod, "B2) Art-heading anchor retention (Art. I roman)", raw_b2)
    assert any("Art. I" in s for s in kept), "Expected 'Art. I' anchor to appear in kept sentences."

    # C) Abbreviation protection: ensure sentence splitting does not break on common legal abbreviations
    raw_c = (
        "Gem. § 5 Abs. 1 Z. 2 lit. a EStG 1988 ist dies zulässig. "
        "Dies gilt z. B. auch dann, wenn der Nachweis i. d. R. schwierig ist bzw. ua. Ausnahmen greifen. "
        "Kundmachung im BGBl. I Nr. 100/2020."
    )
    norm, sents, kept = analyze_text(mod, "C) Abbreviations should not cause false sentence splits", raw_c)

    # sanity checks: no “dangling abbreviation fragments” like "z." or "B." as standalone sentences
    assert not any(s.strip().lower() in {"z.", "b.", "d.", "h.", "u.", "a.", "i.", "r."} for s in sents), (
        "Abbreviation protection failed: detected dangling abbreviation fragments."
    )

    # D) Enumeration handling: line-start and after ':' / ';'
    raw_d = textwrap.dedent("""\
        Die Voraussetzungen sind:
        1. Der Antrag ist rechtzeitig.
        2. Die Partei ist legitimiert.
        3. Es liegt kein Ausschlussgrund vor.
        """)
    norm, sents, kept = analyze_text(mod, "D) Enumerations should keep markers (1) 2) 3))", raw_d)
    assert any("1)" in s for s in kept), "Expected enumeration marker '1)' to be preserved."

    raw_d2 = "Gilt insbesondere: 1. im Fall A; 2. im Fall B; 3. im Fall C."
    norm, sents, kept = analyze_text(mod, "D2) Inline enumerations after ':' / ';' should keep markers", raw_d2)
    assert any("1)" in s for s in sents), "Expected inline enumeration marker '1)' to be preserved."

    # E) RIS metadata/header/footer lines should be dropped by the meaningfulness filter
    raw_e = textwrap.dedent("""\
        RIS - Rechtsinformationssystem
        Kundmachungsorgan: BGBl. I Nr. 1/2000
        Seite 1 von 10
        """)
    norm, sents, kept = analyze_text(mod, "E) RIS metadata lines should be filtered out", raw_e)
    assert len(kept) == 0, "Expected RIS metadata/header/footer lines to be filtered out."

    # -------------------------
    # 2) Optional PDF smoke test (first pages of a few PDFs)
    # -------------------------
    pdf_dir = Path("ris_pdfs")
    if pdf_dir.exists():
        pdfs = sorted(pdf_dir.glob("*.pdf"))[:3]
        if pdfs:
            print("=" * 90)
            print("PDF SMOKE TEST (first 2 pages, first ~6 kept sentences)")
            print("=" * 90)
            for pdf in pdfs:
                print(f"\n--- {pdf.name} ---")
                shown = 0
                for page_idx, page_text in mod.extract_pages_text(pdf):
                    if page_idx >= 2:
                        break
                    norm = mod._normalize_whitespace(page_text)
                    sents = mod.split_into_sentences(norm)
                    kept = [
                        s for s in sents
                        if mod.is_semantically_meaningful(
                            s,
                            drop_patterns=mod._DEFAULT_DROP_PATTERNS,
                        )
                    ]
                    print(f"Page {page_idx + 1}: extracted={len(sents)} kept={len(kept)}")
                    for s in kept[:3]:
                        print(f"  - {s}")
                        shown += 1
                        if shown >= 6:
                            break
                    if shown >= 6:
                        break
        else:
            print("NOTE: ris_pdfs/ exists but no PDFs found for the smoke test.")
    else:
        print("NOTE: ris_pdfs/ not found; skipping PDF smoke test.")

    print("\nAll harness checks completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
