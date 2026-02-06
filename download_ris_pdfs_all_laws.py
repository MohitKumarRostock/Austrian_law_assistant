#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_ris_pdfs_all_laws.py

Downloads one consolidated German RIS PDF ("Bundesrecht konsolidiert" -> "PDF-Dokument")
for each law abbreviation listed in generate_query_set_austrian_law_extended.py (LAWS).

Key properties
--------------
- Output filenames match LAWS *exactly*: <OUT_DIR>/<LAW>.pdf
- Prefers RIS consolidated PDFs (German by nature); DSGVO uses German EUR-Lex PDF fallback.
- Robust against RIS HTML-escaped hrefs (&amp;), slow responses, transient 5xx/429, read timeouts.
- Safe LAWS loading via AST (no executing the generator), with import fallback.
- Verifies existing PDFs to avoid "skipping" corrupted/HTML files.

Usage
-----
python download_ris_pdfs_all_laws.py \
  --laws_py ./generate_query_set_austrian_law_extended.py \
  --out_dir ./ris_pdfs \
  --sleep 0.6 --timeout 120 --retries 8

Tip: rerun only failures from a previous report:
python download_ris_pdfs_all_laws.py \
  --laws_py ./generate_query_set_austrian_law_extended.py \
  --out_dir ./ris_pdfs \
  --only_failed_from_report ./ris_pdfs/download_report.json \
  --sleep 0.6 --timeout 120 --retries 8
"""

from __future__ import annotations

import argparse
import ast
import html
import importlib.util
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from urllib.parse import quote, urljoin, urlparse, parse_qs

try:
    import requests  # type: ignore
except Exception:
    requests = None  # type: ignore

if TYPE_CHECKING:
    import requests as requests_module

import urllib.request
import urllib.error


RIS_BASE = "https://www.ris.bka.gv.at/"
RIS_SEARCH_TITEL_TMPL = RIS_BASE + "Ergebnis.wxe?Abfrage=Bundesnormen&Titel={q}&ResultPageSize=100"
RIS_SEARCH_SUCHWORTE_TMPL = RIS_BASE + "Ergebnis.wxe?Abfrage=Bundesnormen&Suchworte={q}&ResultPageSize=100"

# §0 NormDokument is used to verify abbreviation ("Abkürzung")
NORMDOC_P0_TMPL = RIS_BASE + "NormDokument.wxe?Abfrage=Bundesnormen&Gesetzesnummer={gn}&Paragraf=0"

# DSGVO fallback (German EUR-Lex PDF)
DSGVO_PDF_URLS_DE = [
    "https://eur-lex.europa.eu/legal-content/DE/TXT/PDF/?uri=CELEX%3A32016R0679",
    "https://eur-lex.europa.eu/legal-content/DE/TXT/PDF/?uri=CELEX%3A02016R0679-20160504",
]


# Extra search terms for abbreviations that are not reliably indexed in RIS Trefferlisten
# (used to improve recall for some laws such as PatG).
LAW_ALIASES: Dict[str, List[str]] = {
    # Patentgesetz 1970
    "PatG": ["Patentgesetz", "Patentgesetz 1970"],
    # Occasionally needed if abbreviations are not indexed as Titel
    "UrlG": ["Urlaubsgesetz"],
    "VersG": ["Versammlungsgesetz", "Versammlungsgesetz 1953"],
    "VerG": ["Vereinsgesetz"],
}

# Known Gesetzesnummer overrides (last-resort, used only if RIS search parsing fails).
# Values verified against RIS "Bundesrecht konsolidiert".
KNOWN_GESETZESNUMMER: Dict[str, str] = {
    "PatG": "10002181",
    "UrlG": "10008376",
    "VersG": "10000249",
}

@dataclass
class FetchResult:
    law: str
    ok: bool
    pdf_path: Optional[str] = None
    search_url: Optional[str] = None
    whole_law_url: Optional[str] = None
    pdf_url: Optional[str] = None
    error: Optional[str] = None


def _norm_abbrev(s: str) -> str:
    s2 = s.upper()
    return re.sub(r"[^0-9A-Z]+", "", s2)


def _abbrev_matches(label: str, ris_abbrev: str) -> bool:
    """
    Tolerant matching for RIS abbreviation variants:
      - "TKG" matches "TKG 2021"
      - "AWG" matches "AWG 2002"
    """
    a = _norm_abbrev(label)
    b = _norm_abbrev(ris_abbrev)
    return (a == b) or b.startswith(a) or a.startswith(b)


def _is_probably_pdf(path: Path) -> bool:
    """Cheap validation to avoid skipping HTML error pages saved as .pdf."""
    try:
        if path.stat().st_size < 1024:
            return False
        with open(path, "rb") as f:
            head = f.read(5)
        return head == b"%PDF-"
    except Exception:
        return False


def _requests_session(user_agent: str) -> Optional[requests_module.Session]:
    if requests is None:
        return None
    s = requests.Session()
    # Force German preference where available (RIS is German anyway)
    s.headers.update({
        "User-Agent": user_agent,
        "Accept-Language": "de-AT,de;q=0.9,en;q=0.1",
    })
    return s


def _http_get_text_once(url: str, timeout: float, user_agent: str, session: Optional[requests_module.Session]) -> str:
    headers = {
        "User-Agent": user_agent,
        "Accept-Language": "de-AT,de;q=0.9,en;q=0.1",
    }
    if session is not None:
        r = session.get(url, timeout=timeout)
        r.raise_for_status()
        r.encoding = r.encoding or "utf-8"
        return r.text

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()

    for enc in ("utf-8", "iso-8859-1", "latin-1"):
        try:
            return raw.decode(enc)
        except Exception:
            pass
    return raw.decode("utf-8", errors="replace")


def _http_download_once(url: str, out_path: Path, timeout: float, user_agent: str, session: Optional[requests_module.Session]) -> Tuple[int, str]:
    headers = {
        "User-Agent": user_agent,
        "Accept-Language": "de-AT,de;q=0.9,en;q=0.1",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")

    if session is not None:
        with session.get(url, timeout=timeout, stream=True) as r:
            r.raise_for_status()
            ct = r.headers.get("Content-Type", "")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
            os.replace(tmp, out_path)
            return int(r.status_code), ct

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        status = int(getattr(resp, "status", 200))
        ct = resp.headers.get("Content-Type", "")
        with open(tmp, "wb") as f:
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                f.write(chunk)
        os.replace(tmp, out_path)
        return status, ct


def _with_retries(fn, *, retries: int, backoff_s: float, max_backoff_s: float, sleep_s: float, retry_on_status: Tuple[int, ...] = (429, 500, 502, 503, 504)):
    """
    Execute fn() with retries. fn should raise on failure (requests raises for HTTP errors).
    Adds exponential backoff and keeps an additional per-request politeness sleep (sleep_s).
    """
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            # backoff
            if attempt < retries:
                delay = min(max_backoff_s, backoff_s * (2 ** attempt))
                time.sleep(delay)
    raise last_exc  # type: ignore


def _http_get_text(url: str, timeout: float, user_agent: str, session: Optional[requests_module.Session], retries: int, backoff_s: float, max_backoff_s: float, politeness_sleep_s: float) -> str:
    def _do():
        return _http_get_text_once(url, timeout=timeout, user_agent=user_agent, session=session)
    text = _with_retries(_do, retries=retries, backoff_s=backoff_s, max_backoff_s=max_backoff_s, sleep_s=politeness_sleep_s)
    if politeness_sleep_s:
        time.sleep(politeness_sleep_s)
    return text


def _http_download(url: str, out_path: Path, timeout: float, user_agent: str, session: Optional[requests_module.Session], retries: int, backoff_s: float, max_backoff_s: float, politeness_sleep_s: float) -> Tuple[int, str]:
    def _do():
        return _http_download_once(url, out_path=out_path, timeout=timeout, user_agent=user_agent, session=session)
    res = _with_retries(_do, retries=retries, backoff_s=backoff_s, max_backoff_s=max_backoff_s, sleep_s=politeness_sleep_s)
    if politeness_sleep_s:
        time.sleep(politeness_sleep_s)
    return res


def _extract_all_complete_norm_hrefs(search_html: str) -> List[str]:
    """
    From Trefferliste HTML, extract hrefs behind the 'Gesamte geltende Rechtsvorschrift' icon.
    RIS uses links containing an image like CompleteNorm.gif.
    """
    patterns = [
        r'<a\s+href="([^"]+)"[^>]*>\s*<img[^>]+CompleteNorm\.gif[^>]*>',
        r'<a\s+href="([^"]+)"[^>]*>\s*<img[^>]+CompleteNorm\.png[^>]*>',
        r'<a\s+href="([^"]+)"[^>]*>\s*<img[^>]+alt="Gesamte\s+geltende\s+Rechtsvorschrift"[^>]*>',
    ]
    hrefs: List[str] = []
    for pat in patterns:
        for m in re.finditer(pat, search_html, flags=re.IGNORECASE | re.DOTALL):
            # RIS often HTML-escapes &amp; inside hrefs -> must unescape
            hrefs.append(html.unescape(m.group(1)))

    seen = set()
    out: List[str] = []
    for h in hrefs:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out


def _extract_pdf_href_from_whole_law_html(whole_law_html: str) -> Optional[str]:
    """
    From the whole-law page (GeltendeFassung), extract the PDF-Dokument href.
    """
    # Unescape early because RIS can embed &amp; in href attributes.
    whole_law_html = html.unescape(whole_law_html)

    idx = whole_law_html.lower().find("andere formate")
    windows = []
    if idx != -1:
        windows.append(whole_law_html[idx : idx + 10000])
    windows.append(whole_law_html)

    patterns = [
        r'<a\s+href="([^"]+)"[^>]*>\s*<img[^>]+pdf\.gif[^>]*alt="PDF-Dokument"[^>]*>',
        r'<a\s+href="([^"]+)"[^>]*>\s*<img[^>]+alt="PDF-Dokument"[^>]*>',
        r'<a\s+href="([^"]+)"[^>]*>\s*PDF-Dokument\s*</a>',
        r'<a\s+href="([^"]+)"[^>]*>\s*<img[^>]+pdf\.gif[^>]*>',
    ]
    for w in windows:
        for pat in patterns:
            m = re.search(pat, w, flags=re.IGNORECASE | re.DOTALL)
            if m:
                return html.unescape(m.group(1))
    return None


def _parse_gesetzesnummer_from_url(url: str) -> Optional[str]:
    try:
        q = parse_qs(urlparse(url).query)
        g = q.get("Gesetzesnummer") or q.get("gesetzesnummer")
        if g:
            return str(g[0])
    except Exception:
        pass
    return None


def _extract_abkuerzung_from_normdoc_p0_html(html_text: str) -> Optional[str]:
    html_text = html.unescape(html_text)
    m = re.search(r"###\s*Abkürzung\s*[\r\n]+([^\r\n<]+)", html_text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    m = re.search(r">Abkürzung<.*?>\s*([^<]+)\s*<", html_text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()

    return None


def _pick_best_whole_law_url_for_label(
    label: str,
    candidate_whole_urls: List[str],
    *,
    timeout: float,
    user_agent: str,
    session: Optional[requests_module.Session],
    retries: int,
    backoff_s: float,
    max_backoff_s: float,
    politeness_sleep_s: float,
) -> Optional[str]:
    """
    Pick the correct whole-law URL among candidates by verifying the §0 NormDokument abbreviation.
    """
    by_gn: Dict[str, str] = {}
    for u in candidate_whole_urls:
        u2 = html.unescape(u)
        gn = _parse_gesetzesnummer_from_url(u2)
        if gn and gn not in by_gn:
            by_gn[gn] = u2

    for gn, whole_url in by_gn.items():
        normdoc_url = NORMDOC_P0_TMPL.format(gn=quote(gn))
        try:
            html_p0 = _http_get_text(normdoc_url, timeout=timeout, user_agent=user_agent, session=session,
                                     retries=retries, backoff_s=backoff_s, max_backoff_s=max_backoff_s, politeness_sleep_s=politeness_sleep_s)
            abk = _extract_abkuerzung_from_normdoc_p0_html(html_p0) or ""
            if _abbrev_matches(label, abk):
                return whole_url
        except Exception:
            continue
    return None


def _search_ris(law: str, *, timeout: float, user_agent: str, session: Optional[requests_module.Session], retries: int, backoff_s: float, max_backoff_s: float, politeness_sleep_s: float) -> Tuple[str, str]:
    """
    Search RIS using multiple query variants (Titel/Suchworte plus optional aliases).
    Returns (search_url, search_html) for the first request that succeeds.
    """
    terms = [law] + LAW_ALIASES.get(law, [])
    urls: List[str] = []
    for t in terms:
        urls.append(RIS_SEARCH_TITEL_TMPL.format(q=quote(t)))
        urls.append(RIS_SEARCH_SUCHWORTE_TMPL.format(q=quote(t)))

    last_err = None
    for u in urls:
        try:
            html_text = _http_get_text(
                u,
                timeout=timeout,
                user_agent=user_agent,
                session=session,
                retries=retries,
                backoff_s=backoff_s,
                max_backoff_s=max_backoff_s,
                politeness_sleep_s=politeness_sleep_s,
            )
            return u, html_text
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"RIS search failed: {last_err}")


def _find_ris_pdf_url_for_law(
    law: str,
    *,
    timeout: float,
    user_agent: str,
    session: Optional[requests_module.Session],
    retries: int,
    backoff_s: float,
    max_backoff_s: float,
    politeness_sleep_s: float
) -> Tuple[str, str, str]:
    """
    Returns (search_url, whole_law_url, pdf_url) for the given law abbreviation.
    """
    search_url, search_html = _search_ris(law, timeout=timeout, user_agent=user_agent, session=session,
                                         retries=retries, backoff_s=backoff_s, max_backoff_s=max_backoff_s, politeness_sleep_s=politeness_sleep_s)
    # Unescape because RIS may return &amp; in hrefs.
    search_html = html.unescape(search_html)

    rel_candidates = _extract_all_complete_norm_hrefs(search_html)

    if not rel_candidates:
        # fallback: extract Gesetzesnummer(s) from the Trefferliste in a tolerant way.
        # RIS sometimes embeds Gesetzesnummer outside of query strings (e.g., in hidden inputs),
        # so we search for the token "Gesetzesnummer" followed by a nearby number.
        gns = re.findall(r"Gesetzesnummer[^0-9]{0,50}([0-9]{5,})", search_html, flags=re.IGNORECASE)
        # De-duplicate while preserving order
        seen = set()
        gns = [g for g in gns if not (g in seen or seen.add(g))]

        # Last resort: hard overrides for a few known problematic abbreviations
        if not gns and law in KNOWN_GESETZESNUMMER:
            gns = [KNOWN_GESETZESNUMMER[law]]

        if not gns:
            raise RuntimeError("No Treffer and could not extract Gesetzesnummer from RIS Trefferliste HTML")

        whole_urls = [
            RIS_BASE + f"GeltendeFassung.wxe?Abfrage=Bundesnormen&Gesetzesnummer={quote(gn)}"
            for gn in gns
        ]
        picked = _pick_best_whole_law_url_for_label(
            law, whole_urls,
            timeout=timeout, user_agent=user_agent, session=session,
            retries=retries, backoff_s=backoff_s, max_backoff_s=max_backoff_s,
            politeness_sleep_s=politeness_sleep_s
        )
        whole_law_url = picked or whole_urls[0]
    else:
        whole_urls = [urljoin(RIS_BASE, r) for r in rel_candidates]
        picked = _pick_best_whole_law_url_for_label(
            law, whole_urls,
            timeout=timeout, user_agent=user_agent, session=session,
            retries=retries, backoff_s=backoff_s, max_backoff_s=max_backoff_s,
            politeness_sleep_s=politeness_sleep_s
        )
        whole_law_url = picked or whole_urls[0]

    whole_html = _http_get_text(whole_law_url, timeout=timeout, user_agent=user_agent, session=session,
                                retries=retries, backoff_s=backoff_s, max_backoff_s=max_backoff_s, politeness_sleep_s=politeness_sleep_s)
    rel_pdf = _extract_pdf_href_from_whole_law_html(whole_html)
    if not rel_pdf:
        raise RuntimeError("Could not locate PDF link on whole-law page (Andere Formate)")
    pdf_url = urljoin(RIS_BASE, rel_pdf)
    return search_url, whole_law_url, pdf_url


def _load_laws_from_generator_ast(laws_py: Path) -> Optional[List[str]]:
    src = laws_py.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src, filename=str(laws_py))
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == "LAWS":
                        laws = ast.literal_eval(node.value)
                        if isinstance(laws, list) and all(isinstance(x, str) for x in laws):
                            return list(laws)
            if isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name) and node.target.id == "LAWS" and node.value is not None:
                    laws = ast.literal_eval(node.value)
                    if isinstance(laws, list) and all(isinstance(x, str) for x in laws):
                        return list(laws)
    except Exception:
        return None
    return None


def _load_laws_from_generator(laws_py: Path) -> List[str]:
    laws = _load_laws_from_generator_ast(laws_py)
    if laws is not None:
        return laws

    # Fallback: import with sys.modules registration (dataclasses-safe)
    module_name = f"law_query_generator_{abs(hash(str(laws_py)))}"
    spec = importlib.util.spec_from_file_location(module_name, str(laws_py))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import laws file: {laws_py}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore
    if not hasattr(mod, "LAWS"):
        raise RuntimeError(f"{laws_py} does not define LAWS")
    laws = getattr(mod, "LAWS")
    if not isinstance(laws, list) or not all(isinstance(x, str) for x in laws):
        raise RuntimeError("LAWS must be List[str]")
    return list(laws)


def _load_failed_laws_from_report(report_path: Path) -> List[str]:
    data = json.loads(report_path.read_text(encoding="utf-8"))
    failed = [x["law"] for x in data if isinstance(x, dict) and x.get("ok") is False and isinstance(x.get("law"), str)]
    return failed


def _download_one(
    law: str,
    out_dir: Path,
    *,
    timeout: float,
    politeness_sleep_s: float,
    overwrite: bool,
    verify_existing: bool,
    user_agent: str,
    session: Optional[requests_module.Session],
    retries: int,
    backoff_s: float,
    max_backoff_s: float,
) -> FetchResult:
    out_path = out_dir / f"{law}.pdf"

    if out_path.exists() and not overwrite:
        if verify_existing and not _is_probably_pdf(out_path):
            # treat as invalid -> redownload
            pass
        else:
            return FetchResult(law=law, ok=True, pdf_path=str(out_path), error="SKIPPED (exists)")

    # Special DSGVO handling: prefer German EUR-Lex PDF, but still try RIS first.
    try:
        search_url, whole_url, pdf_url = _find_ris_pdf_url_for_law(
            law,
            timeout=timeout, user_agent=user_agent, session=session,
            retries=retries, backoff_s=backoff_s, max_backoff_s=max_backoff_s, politeness_sleep_s=politeness_sleep_s
        )
        status, ct = _http_download(
            pdf_url, out_path,
            timeout=timeout, user_agent=user_agent, session=session,
            retries=retries, backoff_s=backoff_s, max_backoff_s=max_backoff_s, politeness_sleep_s=politeness_sleep_s
        )
        # Validate PDF signature
        if not _is_probably_pdf(out_path):
            raise RuntimeError(f"Downloaded content does not look like a PDF (status={status}, content-type={ct})")

        return FetchResult(law=law, ok=True, pdf_path=str(out_path), search_url=search_url, whole_law_url=whole_url, pdf_url=pdf_url)

    except Exception as e:
        if _norm_abbrev(law) == _norm_abbrev("DSGVO"):
            last_err = str(e)
            for url in DSGVO_PDF_URLS_DE:
                try:
                    status, ct = _http_download(
                        url, out_path,
                        timeout=timeout, user_agent=user_agent, session=session,
                        retries=retries, backoff_s=backoff_s, max_backoff_s=max_backoff_s, politeness_sleep_s=politeness_sleep_s
                    )
                    if not _is_probably_pdf(out_path):
                        raise RuntimeError(f"EUR-Lex download not a PDF (status={status}, content-type={ct})")
                    return FetchResult(law=law, ok=True, pdf_path=str(out_path), pdf_url=url, error="FALLBACK: EUR-Lex (DE)")
                except Exception as ee:
                    last_err = f"{last_err} | EUR-Lex: {ee}"
            return FetchResult(law=law, ok=False, pdf_path=str(out_path), error=last_err)

        return FetchResult(law=law, ok=False, pdf_path=str(out_path), error=str(e))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--laws_py", type=str, default="generate_query_set_austrian_law.py",
                    help="Path to generate_query_set_austrian_law.py (must define LAWS).")
    ap.add_argument("--out_dir", type=str, default="ris_pdfs",
                    help="Output directory where <LAW>.pdf files are written.")
    ap.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout per request (seconds).")
    ap.add_argument("--sleep", type=float, default=0.6, help="Politeness sleep between requests (seconds).")
    ap.add_argument("--retries", type=int, default=8, help="Retry count for transient failures/timeouts.")
    ap.add_argument("--backoff", type=float, default=1.2, help="Base backoff seconds (exponential).")
    ap.add_argument("--max_backoff", type=float, default=30.0, help="Max backoff seconds.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing PDFs.")
    ap.add_argument("--no_verify_existing", action="store_true", help="Do not validate existing PDFs; just skip.")
    ap.add_argument("--user_agent", type=str, default="Mozilla/5.0 (compatible; AustrianLawIRBot/1.0)",
                    help="User-Agent header.")
    ap.add_argument("--report_json", type=str, default="download_report.json",
                    help="Write a JSON report to this path (relative to out_dir unless absolute).")
    ap.add_argument("--only_failed_from_report", type=str, default=None,
                    help="If set, download only laws that failed in the given report JSON.")
    args = ap.parse_args()

    laws_py = Path(args.laws_py).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        laws = _load_laws_from_generator(laws_py)
    except Exception as e:
        print(f"[FATAL] Could not load LAWS from {laws_py}: {e}", file=sys.stderr)
        return 2

    if args.only_failed_from_report:
        rp = Path(args.only_failed_from_report).expanduser().resolve()
        try:
            failed = _load_failed_laws_from_report(rp)
            # Preserve LAWS order
            laws = [x for x in laws if x in set(failed)]
            print(f"Loaded {len(laws)} failed laws from report {rp}")
        except Exception as e:
            print(f"[FATAL] Could not load failures from report {rp}: {e}", file=sys.stderr)
            return 2

    session = _requests_session(args.user_agent)

    print(f"Loaded {len(laws)} laws from {laws_py}")
    print(f"Downloading PDFs into {out_dir}")
    print("----")

    results: List[Dict] = []
    ok = 0
    fail = 0

    for i, law in enumerate(laws, start=1):
        print(f"[{i:>3}/{len(laws)}] {law} ... ", end="", flush=True)

        r = _download_one(
            law=law,
            out_dir=out_dir,
            timeout=args.timeout,
            politeness_sleep_s=args.sleep,
            overwrite=args.overwrite,
            verify_existing=(not args.no_verify_existing),
            user_agent=args.user_agent,
            session=session,
            retries=args.retries,
            backoff_s=args.backoff,
            max_backoff_s=args.max_backoff,
        )

        results.append({
            "law": r.law,
            "ok": r.ok,
            "pdf_path": r.pdf_path,
            "search_url": r.search_url,
            "whole_law_url": r.whole_law_url,
            "pdf_url": r.pdf_url,
            "error": r.error,
        })

        if r.ok:
            ok += 1
            if r.error and r.error.startswith("SKIPPED"):
                # Could be skipping an existing valid PDF
                print("skipped (exists)")
            else:
                print("OK")
        else:
            fail += 1
            print("FAILED")
            print(f"      reason: {r.error}")

    report_path = Path(args.report_json)
    if not report_path.is_absolute():
        report_path = out_dir / report_path
    report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("----")
    print(f"Done. OK={ok}, FAILED={fail}")
    print(f"Report: {report_path}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
