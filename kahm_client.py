# kahm_client.py (v11 - general)
#
# Design goals:
# - General-purpose post-processing for KAHM_best_routed retrieval (not tuned to a single query).
# - Optional law-focused retrieval (law_hint / strict_law / quotas) while keeping cross-law diversity possible.
# - Optional lexical (BM25) fallback over your parquet when KAHM retrieval for the hinted law is weak.
# - Minimal assumptions about specific laws; all "domain tuning" is configuration-driven.
#
# Output schema (ready for Austrian law assistant pipeline):
#   List[ (sentence_id, law_type, page, sentence) ]
#
# Dependencies:
#   pip install requests pandas pyarrow
# Optional async client:
#   pip install httpx
#
# Notes:
# - If you regenerated parquet with improved §/Art anchoring, BM25 fallback will surface more relevant, anchored sentences.
# - BM25 indices are cached under `.cache_kahm/` (delete the folder to rebuild).

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from collections import Counter, OrderedDict
import os
import re
import math
import pickle
import heapq

HitTuple = Tuple[int, str, int, str]  # (sentence_id, law_type, page, sentence)


# ----------------------------
# HTTP Clients (sync / async)
# ----------------------------

@dataclass(frozen=True)
class KahmClientConfig:
    base_url: str = "http://127.0.0.1:8000"
    timeout_s: float = 30.0


class KahmClient:
    """Sync client for your FastAPI service."""
    def __init__(self, config: KahmClientConfig = KahmClientConfig()) -> None:
        self.base_url = config.base_url.rstrip("/")
        self.timeout_s = float(config.timeout_s)

    def health(self) -> Dict[str, Any]:
        import requests
        r = requests.get(f"{self.base_url}/health", timeout=5)
        r.raise_for_status()
        return r.json()

    def retrieve_raw(
        self,
        query_text: str,
        *,
        k: int = 10,
        router_k: int = 50,
        cand_k: int = 200,
        maj_frac_thr: float = 0.40,
        maj_lift_thr: float = 5.0,
        top_laws: int = 1,
        law_slots: int = 1,
        lam: float = 0.0,
        mode: str = "hybrid",
        debug: bool = False,
    ) -> Dict[str, Any]:
        import requests

        query_text = (query_text or "").strip()
        if not query_text:
            raise ValueError("query_text must be non-empty.")

        payload: Dict[str, Any] = {
            "query_text": query_text,
            "k": int(k),
            "router_k": int(router_k),
            "cand_k": int(cand_k),
            "maj_frac_thr": float(maj_frac_thr),
            "maj_lift_thr": float(maj_lift_thr),
            "top_laws": int(top_laws),
            "law_slots": int(law_slots),
            "lam": float(lam),
            "mode": str(mode),
            "debug": bool(debug),
        }

        r = requests.post(f"{self.base_url}/v1/retrieve", json=payload, timeout=self.timeout_s)
        if r.status_code != 200:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            raise RuntimeError(f"API error {r.status_code}: {detail}")
        return r.json()


class AsyncKahmClient:
    """Async client (optional). Requires `httpx`."""
    def __init__(self, config: KahmClientConfig = KahmClientConfig()) -> None:
        self.base_url = config.base_url.rstrip("/")
        self.timeout_s = float(config.timeout_s)

    async def health(self) -> Dict[str, Any]:
        try:
            import httpx
        except ImportError as e:
            raise ImportError("AsyncKahmClient requires `httpx` (pip install httpx).") from e

        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{self.base_url}/health")
            r.raise_for_status()
            return r.json()

    async def retrieve_raw(
        self,
        query_text: str,
        *,
        k: int = 10,
        router_k: int = 50,
        cand_k: int = 200,
        maj_frac_thr: float = 0.40,
        maj_lift_thr: float = 5.0,
        top_laws: int = 1,
        law_slots: int = 1,
        lam: float = 0.0,
        mode: str = "hybrid",
        debug: bool = False,
    ) -> Dict[str, Any]:
        try:
            import httpx
        except ImportError as e:
            raise ImportError("AsyncKahmClient requires `httpx` (pip install httpx).") from e

        query_text = (query_text or "").strip()
        if not query_text:
            raise ValueError("query_text must be non-empty.")

        payload: Dict[str, Any] = {
            "query_text": query_text,
            "k": int(k),
            "router_k": int(router_k),
            "cand_k": int(cand_k),
            "maj_frac_thr": float(maj_frac_thr),
            "maj_lift_thr": float(maj_lift_thr),
            "top_laws": int(top_laws),
            "law_slots": int(law_slots),
            "lam": float(lam),
            "mode": str(mode),
            "debug": bool(debug),
        }

        async with httpx.AsyncClient(timeout=self.timeout_s) as c:
            r = await c.post(f"{self.base_url}/v1/retrieve", json=payload)
            r.raise_for_status()
            return r.json()


# ----------------------------
# Retrieval policy/config
# ----------------------------

@dataclass(frozen=True)
class RetrievalPolicy:
    # If you pass law_hint, you can bias selection towards that law.
    strict_law: bool = False
    target_law_hits: Optional[int] = None   # if None, computed from k
    max_other_hits: Optional[int] = None    # if None, computed from k & target_law_hits

    # Non-law gating: require non-law hits to match at least one "core" query token.
    # This is general (query-derived), and prevents leakage from generic legal words.
    gate_other_laws_on_core_tokens: bool = True
    min_core_token_matches_other: int = 1

    # Optional explicit law exclusions (rarely needed if gating is enabled)
    exclude_laws: Optional[Set[str]] = None

    # Fallback triggers for BM25
    lexical_fallback: bool = True
    min_law_api_hits: int = 6
    min_law_api_core_hits: int = 3
    min_law_api_best_overlap: float = 2.0

    # How to choose "core" query tokens:
    # Tokens that appear in too many candidate sentences are considered "generic".
    core_df_frac_thr: float = 0.45        # keep tokens whose df among candidates <= thr
    max_core_tokens: int = 14

    # BM25 retrieval
    parquet_path: str = "ris_sentences.parquet"
    bm25_cache_dir: str = ".cache_kahm"
    bm25_top_n: int = 200
    bm25_keep_if_core_match: bool = True  # keep only BM25 hits that match core tokens


# ----------------------------
# Utilities
# ----------------------------

def group_hits_by_law(hits: Iterable[HitTuple]) -> Dict[str, List[HitTuple]]:
    ordered: "OrderedDict[str, List[HitTuple]]" = OrderedDict()
    for sid, law, page, sent in hits:
        if law not in ordered:
            ordered[law] = []
        ordered[law].append((sid, law, page, sent))
    return dict(ordered)

def law_distribution(hits: Iterable[HitTuple]) -> List[Tuple[str, int]]:
    c = Counter([law for _, law, _, _ in hits])
    return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))

def consensus_fraction(hits: List[HitTuple]) -> float:
    if not hits:
        return 0.0
    dist = law_distribution(hits)
    return float(dist[0][1]) / float(len(hits)) if dist else 0.0


# ----------------------------
# Text normalization + boilerplate
# ----------------------------

_HYPHENS = r"\-\u2010\u2011\u2012\u2013\u2014"
_SOFT_HYPHEN = "\u00ad"
_INLINE_RIS_PAGE = re.compile(r"\s*www\.ris\.bka\.gv\.at\s+Seite\s+\d+\s+von\s+\d+\s*", re.I)

def strip_inline_noise(s: str) -> str:
    if not s:
        return ""
    t = s.replace(_SOFT_HYPHEN, "")
    t = _INLINE_RIS_PAGE.sub(" ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def normalize_for_match(s: str) -> str:
    t = strip_inline_noise(s).lower()
    # de-hyphenate common extraction artifacts: "irr- thum" -> "irrthum"
    t = re.sub(rf"([a-zäöüß])([{_HYPHENS}])\s+([a-zäöüß])", r"\1\3", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

_BOILERPLATE_PATTERNS = [
    re.compile(r"^Von\s+.*\s+überhaupt\.$", re.I),
    re.compile(r"^Abschnitt\s+\d+", re.I),
    re.compile(r"^Hauptstück\s+\d+", re.I),
    re.compile(r"^Teil\s+\d+", re.I),
    re.compile(r"^Titel\s+\d+", re.I),
    re.compile(r"^Inkrafttreten", re.I),
    re.compile(r"^Übergangs", re.I),
    re.compile(r"^Im Gesetz\b", re.I),
    re.compile(r"^\(?\d+\)?\s*Die Bestimmungen des §\.\s*$", re.I),
    re.compile(r"^\(?\d+\)?\s*In den Fällen der §§\.\s*$", re.I),
    re.compile(r"§§\.\s*$"),
    re.compile(r"§\.\s*$"),
]

def is_boilerplate(text: str) -> bool:
    s = strip_inline_noise(text)
    if len(s) < 25:
        return True
    if len(s.split()) < 5:
        return True
    for pat in _BOILERPLATE_PATTERNS:
        if pat.search(s):
            return True
    return False


# ----------------------------
# Law hint inference (general)
# ----------------------------

def infer_law_hint_from_router(router: Dict[str, Any]) -> Optional[str]:
    """
    If router produced a majority law and thresholds, it can be a decent hint.
    We keep it conservative: only accept if maj_frac and maj_lift are present
    AND exceed the router thresholds it reports (if any).
    """
    if not router:
        return None
    law = router.get("maj_law")
    if not law:
        return None
    try:
        maj_frac = float(router.get("maj_frac", 0.0))
        maj_lift = float(router.get("maj_lift", 0.0))
        thr = router.get("thresholds") or {}
        frac_thr = float(thr.get("maj_frac_thr", 1.0))
        lift_thr = float(thr.get("maj_lift_thr", 1e9))
        if maj_frac >= frac_thr and maj_lift >= lift_thr:
            return str(law)
    except Exception:
        return None
    return None


# ----------------------------
# Query tokens + "core token" selection (general)
# ----------------------------

_STOPWORDS_DE = {
    "der","die","das","ein","eine","einer","eines","und","oder","zu","im","in","am","mit","bei","von","vom",
    "für","auf","aus","ist","sind","war","waren","welche","welcher","welches","was","wie","wieso","warum",
    "rechtsfolgen","hat","haben","wegen",
    # common legal glue words
    "abs","absatz","z","ziffer","lit","litera","nr","paragraf","paragraph","art","artikel",
}

def _variants(tok: str) -> Set[str]:
    """
    Small, general-purpose variant expansion helpful for Austrian/German legal text:
    - plural/s-genitive
    - spelling variants: irrtum/irrthum
    - anfecht* stem
    """
    out = {tok}
    if tok.endswith("s") and len(tok) >= 6:
        out.add(tok[:-1])
    if tok == "irrtum":
        out.update({"irrthum", "irrtums", "irrthums"})
    if tok == "irrthum":
        out.update({"irrtum", "irrtums", "irrthums"})
    if tok in {"irrtums", "irrthums"}:
        out.update({"irrtum", "irrthum"})
    if tok.startswith("anfecht"):
        out.add("anfecht")
    return out

def extract_query_tokens(query_text: str) -> Set[str]:
    qn = normalize_for_match(query_text)
    toks = re.findall(r"[a-zäöüß]{4,}", qn)
    base = {t for t in toks if t not in _STOPWORDS_DE}
    # include § numbers if present as stable anchors
    sec = re.findall(r"§\s*(\d+[a-z]?)", query_text, flags=re.IGNORECASE)
    for s in sec:
        base.add(f"§{s.lower()}")
        base.add(s.lower())
    out: Set[str] = set()
    for t in base:
        out |= _variants(t)
    return out

def sentence_token_overlap(sentence: str, tokens: Set[str]) -> int:
    sn = normalize_for_match(sentence)
    return sum(1 for t in tokens if t and t in sn)

def compute_core_tokens(
    query_tokens: Set[str],
    api_sentences: List[str],
    *,
    df_frac_thr: float,
    max_core_tokens: int,
) -> List[str]:
    """
    Choose query tokens that are most discriminative within the current candidate set.
    If a token appears in too many candidate sentences, it is likely generic for this query.
    """
    if not query_tokens:
        return []

    n = max(1, len(api_sentences))
    counts: Dict[str, int] = {t: 0 for t in query_tokens}
    for s in api_sentences:
        sn = normalize_for_match(s)
        for t in query_tokens:
            if t in sn:
                counts[t] += 1

    # df fraction among candidates
    scored = []
    for t, c in counts.items():
        frac = c / float(n)
        # prefer lower df (more specific) but keep tokens that occur at least once if possible
        if c > 0 and frac <= float(df_frac_thr):
            scored.append((frac, -len(t), t))
    scored.sort()

    core = [t for _, _, t in scored][: int(max_core_tokens)]
    if core:
        return core

    # fallback: use tokens that occur at least once, even if common
    scored2 = []
    for t, c in counts.items():
        if c > 0:
            frac = c / float(n)
            scored2.append((frac, -len(t), t))
    scored2.sort()
    return [t for _, _, t in scored2][: int(max_core_tokens)]


# ----------------------------
# Local BM25 index (per law)
# ----------------------------

class LawBm25Index:
    def __init__(
        self,
        law_type: str,
        docs: List[Tuple[int, int, str]],
        postings: Dict[str, List[Tuple[int, int]]],
        idf: Dict[str, float],
        doc_len: List[int],
        avgdl: float,
    ) -> None:
        self.law_type = law_type
        self.docs = docs
        self.postings = postings
        self.idf = idf
        self.doc_len = doc_len
        self.avgdl = avgdl

    @classmethod
    def _tokenize(cls, text: str) -> List[str]:
        t = normalize_for_match(text)
        toks = re.findall(r"[a-zäöüß]{3,}", t)
        base = [x for x in toks if x not in _STOPWORDS_DE]
        expanded: List[str] = []
        for b in base:
            expanded.extend(list(_variants(b)))
        return expanded

    @classmethod
    def build_from_parquet(cls, parquet_path: str, law_type: str) -> "LawBm25Index":
        import pandas as pd
        df = pd.read_parquet(parquet_path, columns=["sentence_id", "law_type", "page", "sentence"])
        df = df[df["law_type"] == law_type].copy()

        docs: List[Tuple[int, int, str]] = []
        postings: Dict[str, List[Tuple[int, int]]] = {}
        doc_len: List[int] = []
        df_counts: Dict[str, int] = {}

        for row in df.itertuples(index=False):
            sid = int(row.sentence_id)
            page = int(row.page)
            sent = strip_inline_noise(str(row.sentence))
            if is_boilerplate(sent):
                continue

            toks = cls._tokenize(sent)
            if not toks:
                continue

            doc_id = len(docs)
            docs.append((sid, page, sent))
            doc_len.append(len(toks))

            tf: Dict[str, int] = {}
            for tok in toks:
                tf[tok] = tf.get(tok, 0) + 1

            for tok, c in tf.items():
                postings.setdefault(tok, []).append((doc_id, c))
                df_counts[tok] = df_counts.get(tok, 0) + 1

        N = len(docs)
        if N == 0:
            raise RuntimeError(f"No docs indexed for law_type={law_type} from {parquet_path}")

        avgdl = sum(doc_len) / float(N)
        idf: Dict[str, float] = {}
        for tok, dfc in df_counts.items():
            idf[tok] = math.log(1.0 + (N - dfc + 0.5) / (dfc + 0.5))

        return cls(law_type, docs, postings, idf, doc_len, avgdl)

    def search(self, query: str, top_n: int = 200, *, k1: float = 1.5, b: float = 0.75) -> List[Tuple[float, HitTuple]]:
        q_toks = self._tokenize(query)
        if not q_toks:
            return []

        q_tf: Dict[str, int] = {}
        for t in q_toks:
            q_tf[t] = q_tf.get(t, 0) + 1

        scores: Dict[int, float] = {}
        for tok, qcount in q_tf.items():
            plist = self.postings.get(tok)
            if not plist:
                continue
            idf = self.idf.get(tok, 0.0)
            for doc_id, tf in plist:
                dl = self.doc_len[doc_id]
                denom = tf + k1 * (1.0 - b + b * (dl / self.avgdl))
                s = idf * ((tf * (k1 + 1.0)) / denom)
                scores[doc_id] = scores.get(doc_id, 0.0) + s * (1.0 + 0.10 * (qcount - 1))

        if not scores:
            return []

        top = heapq.nlargest(top_n, scores.items(), key=lambda kv: kv[1])
        out: List[Tuple[float, HitTuple]] = []
        for doc_id, sc in top:
            sid, page, sent = self.docs[doc_id]
            out.append((float(sc), (sid, self.law_type, int(page), sent)))
        return out


def load_or_build_bm25(parquet_path: str, law_type: str, cache_dir: str) -> LawBm25Index:
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"bm25_{law_type}.pkl")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, LawBm25Index) and obj.law_type == law_type:
                return obj
        except Exception:
            pass
    idx = LawBm25Index.build_from_parquet(parquet_path, law_type)
    with open(cache_file, "wb") as f:
        pickle.dump(idx, f)
    return idx


# ----------------------------
# Candidate merge / rerank / select
# ----------------------------

@dataclass
class Candidate:
    sentence_id: int
    law_type: str
    page: int
    sentence: str
    api_score: Optional[float] = None
    bm25_score: Optional[float] = None
    overlap_core: float = 0.0
    overlap_query: float = 0.0


def merge_candidates(
    api_rows: List[Dict[str, Any]],
    bm25_rows: List[Tuple[float, HitTuple]],
    *,
    query_tokens: Set[str],
    core_tokens: Set[str],
) -> Dict[int, Candidate]:
    by_id: Dict[int, Candidate] = {}

    for r in api_rows:
        sid = int(r["sentence_id"])
        law = str(r["law_type"])
        page = int(r["page"])
        sent = strip_inline_noise(str(r["sentence"]))
        if is_boilerplate(sent):
            continue
        c = by_id.get(sid)
        if c is None:
            c = Candidate(sentence_id=sid, law_type=law, page=page, sentence=sent)
            by_id[sid] = c
        score = float(r.get("score", 0.0))
        c.api_score = score if c.api_score is None else max(c.api_score, score)

    for bm_sc, (sid, law, page, sent) in bm25_rows:
        sent = strip_inline_noise(sent)
        if is_boilerplate(sent):
            continue
        c = by_id.get(int(sid))
        if c is None:
            c = Candidate(sentence_id=int(sid), law_type=str(law), page=int(page), sentence=sent)
            by_id[int(sid)] = c
        sc = float(bm_sc)
        c.bm25_score = sc if c.bm25_score is None else max(c.bm25_score, sc)

    for c in by_id.values():
        c.overlap_query = float(sentence_token_overlap(c.sentence, query_tokens))
        c.overlap_core = float(sentence_token_overlap(c.sentence, core_tokens))

    return by_id


def rerank(cands: Iterable[Candidate], law_hint: Optional[str]) -> List[Candidate]:
    def key(c: Candidate) -> Tuple[int, float, float, float, float]:
        is_law = 1 if (law_hint and c.law_type == law_hint) else 0
        bm = c.bm25_score if c.bm25_score is not None else 0.0
        ap = c.api_score if c.api_score is not None else 0.0
        # General ranking (not law-specific):
        #   law_hint > core overlap > total overlap > bm25 > api
        return (is_law, c.overlap_core, c.overlap_query, bm, ap)

    out = list(cands)
    out.sort(key=key, reverse=True)
    return out


def select_hits(
    ranked: List[Candidate],
    *,
    k: int,
    law_hint: Optional[str],
    policy: RetrievalPolicy,
    core_tokens: Set[str],
) -> List[Candidate]:
    exclude = policy.exclude_laws or set()

    if policy.strict_law and law_hint:
        return [c for c in ranked if c.law_type == law_hint and c.law_type not in exclude][:k]

    if not law_hint:
        out: List[Candidate] = []
        for c in ranked:
            if c.law_type in exclude:
                continue
            out.append(c)
            if len(out) >= k:
                break
        return out

    target_law = policy.target_law_hits
    if target_law is None:
        target_law = max(1, int(round(0.70 * k)))
    target_law = min(target_law, k)

    max_other = policy.max_other_hits
    if max_other is None:
        max_other = k - target_law
    if max_other < 0:
        max_other = 0

    law_cands = [c for c in ranked if c.law_type == law_hint and c.law_type not in exclude]
    other_cands = [c for c in ranked if c.law_type != law_hint and c.law_type not in exclude]

    out: List[Candidate] = []
    out.extend(law_cands[:target_law])

    other_added = 0
    for c in other_cands:
        if len(out) >= k:
            break
        if other_added >= max_other:
            break
        if policy.gate_other_laws_on_core_tokens:
            if sentence_token_overlap(c.sentence, core_tokens) < policy.min_core_token_matches_other:
                continue
        out.append(c)
        other_added += 1

    # backfill with more law hits to keep coherence
    if len(out) < k:
        for c in law_cands[target_law:]:
            if len(out) >= k:
                break
            out.append(c)

    # final backfill (rare): any remaining
    if len(out) < k:
        for c in ranked:
            if len(out) >= k:
                break
            if c.law_type in exclude:
                continue
            if c in out:
                continue
            out.append(c)

    return out[:k]


# ----------------------------
# High-level retrieval for RAG
# ----------------------------

def retrieve_for_rag(
    client: KahmClient,
    query_text: str,
    *,
    k: int = 20,
    prefetch_k: int = 50,
    cand_k: int = 1200,
    law_hint: Optional[str] = None,
    policy: RetrievalPolicy = RetrievalPolicy(),
) -> Tuple[List[HitTuple], Dict[str, List[HitTuple]], Dict[str, Any]]:
    query_text = (query_text or "").strip()
    if not query_text:
        raise ValueError("query_text must be non-empty.")

    k = int(k)
    prefetch_k = max(k, min(int(prefetch_k), 50))

    # 1) Raw KAHM API retrieval
    data = client.retrieve_raw(query_text, k=prefetch_k, cand_k=int(cand_k))
    api_rows = data.get("results", []) or []
    router = data.get("router", {}) or {}

    # 2) Determine law hint (caller > router)
    if not law_hint:
        law_hint = infer_law_hint_from_router(router)

    # 3) Compute query/core tokens from query + current candidates (general)
    q_tokens = extract_query_tokens(query_text)
    api_sents = [str(r.get("sentence", "")) for r in api_rows]
    core_list = compute_core_tokens(
        q_tokens,
        api_sents,
        df_frac_thr=float(policy.core_df_frac_thr),
        max_core_tokens=int(policy.max_core_tokens),
    )
    core_tokens = set(core_list)

    # 4) Decide whether to run BM25 fallback (only meaningful when law_hint is set)
    law_api = [r for r in api_rows if law_hint and str(r.get("law_type")) == law_hint]
    law_api_count = len(law_api)
    law_api_core_hits = sum(1 for r in law_api if sentence_token_overlap(str(r.get("sentence", "")), core_tokens) > 0)
    law_api_best_overlap = float(
        max((sentence_token_overlap(str(r.get("sentence", "")), q_tokens) for r in law_api), default=0)
    )

    do_fallback = (
        bool(policy.lexical_fallback)
        and bool(law_hint)
        and os.path.exists(policy.parquet_path)
        and (
            law_api_count < int(policy.min_law_api_hits)
            or law_api_core_hits < int(policy.min_law_api_core_hits)
            or law_api_best_overlap < float(policy.min_law_api_best_overlap)
        )
    )

    bm25_rows: List[Tuple[float, HitTuple]] = []
    bm25_query: Optional[str] = None
    if do_fallback:
        idx = load_or_build_bm25(policy.parquet_path, str(law_hint), policy.bm25_cache_dir)
        # BM25 query augmentation is general: query + core tokens (discriminative for THIS query)
        bm25_query = query_text + " " + " ".join(core_list[: min(len(core_list), 12)])
        bm25_rows = idx.search(bm25_query, top_n=int(policy.bm25_top_n))
        if policy.bm25_keep_if_core_match and core_tokens:
            bm25_rows = [(sc, h) for sc, h in bm25_rows if sentence_token_overlap(h[3], core_tokens) > 0]

    # 5) Merge, rank, select
    merged = merge_candidates(api_rows, bm25_rows, query_tokens=q_tokens, core_tokens=core_tokens)
    ranked = rerank(merged.values(), law_hint)
    selected = select_hits(ranked, k=k, law_hint=law_hint, policy=policy, core_tokens=core_tokens)

    final_hits: List[HitTuple] = [(c.sentence_id, c.law_type, c.page, c.sentence) for c in selected]
    grouped = group_hits_by_law(final_hits)

    meta: Dict[str, Any] = {
        "router": router,
        "law_hint_used": law_hint,
        "query_tokens_sample": sorted(list(q_tokens))[:16],
        "core_tokens": core_list,
        "api_law_coverage": {
            "law_api_count": law_api_count,
            "law_api_core_hits": int(law_api_core_hits),
            "law_api_best_overlap": float(law_api_best_overlap),
            "thresholds": {
                "min_law_api_hits": int(policy.min_law_api_hits),
                "min_law_api_core_hits": int(policy.min_law_api_core_hits),
                "min_law_api_best_overlap": float(policy.min_law_api_best_overlap),
            },
        },
        "lexical_fallback": {
            "enabled": bool(do_fallback),
            "parquet_exists": os.path.exists(policy.parquet_path),
            "bm25_query": bm25_query,
            "bm25_hits_used": len(bm25_rows),
            "cache_dir": policy.bm25_cache_dir,
        },
        "selection_policy": {
            "strict_law": bool(policy.strict_law),
            "target_law_hits": policy.target_law_hits,
            "max_other_hits": policy.max_other_hits,
            "gate_other_laws_on_core_tokens": bool(policy.gate_other_laws_on_core_tokens),
            "min_core_token_matches_other": int(policy.min_core_token_matches_other),
            "exclude_laws": sorted(list(policy.exclude_laws or set())),
        },
        "final_hit_law_dist_top10": law_distribution(final_hits)[:10],
        "final_consensus_frac": consensus_fraction(final_hits),
    }

    return final_hits, grouped, meta


# ----------------------------
# Demo
# ----------------------------

if __name__ == "__main__":
    client = KahmClient()
    print("Health:", client.health())

    query = "Welche Rechtsfolgen hat eine Anfechtung wegen Irrtums?"
    print("Query:", query)

    # This demo shows the general behavior. For production, you will call `retrieve_for_rag`
    # from your pipeline and pass policy knobs as needed.
    hits, grouped, meta = retrieve_for_rag(
        client,
        query,
        k=20,
        prefetch_k=50,
        cand_k=1200,
        law_hint=None,  # let router hint (if confident) or keep mixed-law
        policy=RetrievalPolicy(
            strict_law=False,
            target_law_hits=None,          # computed from k
            max_other_hits=None,           # computed from k
            gate_other_laws_on_core_tokens=True,
            parquet_path="ris_sentences.parquet",
        ),
    )

    print("Meta:", meta)
    for law, items in grouped.items():
        print(f"\n=== {law} ({len(items)} hits) ===")
        for sid, _, page, sent in items[:6]:
            print(f"{sid} | p.{page} | {sent[:220]}")
