# kahm_best_api.py
# Run: uvicorn kahm_best_api:app --host 0.0.0.0 --port 8000
#
# Implements KAHM_best_routed retrieval:
#  - Router: q_kahm vs KAHM(full) corpus index => law posterior + confidence (maj_frac, maj_lift)
#  - Candidates: q_kahm vs MB corpus index => cand_k candidates
#  - Post-process: hybrid filtering / rerank (defaults match the tuned “kahm_best_routed” idea)

from __future__ import annotations

import os
from dataclasses import dataclass
from collections import Counter
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# Reuse the exact helpers and conventions from your evaluation script.
# This keeps alignment/loading/index behavior consistent.
import evaluate_three_embeddings as e3  # your attached script


# ----------------------------- API models -----------------------------

class RetrieveRequest(BaseModel):
    query_text: str = Field(..., description="User query text (same as query_set['query_text']).")
    k: int = Field(10, ge=1, le=50, description="Number of results to return (top-k).")
    # Optional overrides for KAHM_best_routed behavior:
    router_k: int = Field(50, ge=1, le=500, description="Router neighborhood size (KAHM(full) index).")
    cand_k: int = Field(200, ge=1, le=2000, description="Candidate pool size (KAHM(q→MB) on MB index).")
    maj_frac_thr: float = Field(0.40, ge=0.0, le=1.0, description="Majority fraction threshold for router acceptance.")
    maj_lift_thr: float = Field(5.0, ge=0.0, description="Majority lift threshold (vs corpus law prior) for acceptance.")
    top_laws: int = Field(1, ge=1, le=10, description="How many top router laws to use in law_set.")
    law_slots: int = Field(1, ge=1, le=50, description="How many of the final k slots can be reserved for router laws.")
    lam: float = Field(0.0, ge=0.0, le=5.0, description="Rerank lambda; 0 disables reranking.")
    mode: str = Field("hybrid", description="filter | rerank | hybrid")
    debug: bool = Field(False, description="Include extra routing diagnostics.")


class RetrievalHit(BaseModel):
    rank: int
    score: float
    sentence_id: int
    law_type: str
    page: int
    sentence: str


class RetrieveResponse(BaseModel):
    query_text: str
    k: int
    accepted: bool
    router: Dict[str, Any]
    results: List[RetrievalHit]
    debug: Optional[Dict[str, Any]] = None


# ----------------------------- Engine -----------------------------

@dataclass
class KahmBestEngine:
    # aligned corpus (intersection) row-major arrays
    sentence_ids: np.ndarray          # shape (N,)
    law_arr: np.ndarray               # shape (N,)
    page_arr: np.ndarray              # shape (N,)
    sent_arr: np.ndarray              # shape (N,)

    # indices
    index_mb: Any
    index_kahm: Any

    # models
    idf_pipe: Any
    kahm_model: Any

    # routing prior
    law_prior: Dict[str, float]

    # runtime knobs
    kahm_mode: str
    regress_batch: int
    kahm_n_jobs: int
    kahm_backend: str

    @staticmethod
    def _env_path(name: str, default: str) -> str:
        return str(os.environ.get(name, default)).strip()

    @classmethod
    def from_env(cls) -> "KahmBestEngine":
        # Paths (defaults match evaluate_three_embeddings.py)
        data_dir = cls._env_path("KAHM_DATA_DIR", ".")
        semantic_npz = os.path.join(data_dir, cls._env_path("SEMANTIC_NPZ", e3.DEFAULT_SEMANTIC_NPZ))
        kahm_corpus_npz = os.path.join(data_dir, cls._env_path("KAHM_CORPUS_NPZ", e3.DEFAULT_KAHM_CORPUS_NPZ))
        corpus_parquet = os.path.join(data_dir, cls._env_path("CORPUS_PARQUET", e3.DEFAULT_CORPUS_PARQUET))
        idf_svd_model = os.path.join(data_dir, cls._env_path("IDF_SVD_MODEL", e3.DEFAULT_IDF_SVD_MODEL))
        kahm_model_path = os.path.join(data_dir, cls._env_path("KAHM_MODEL", e3.DEFAULT_KAHM_MODEL))

        # Optional compatibility check inputs (does not affect retrieval if missing)
        idf_svd_npz = os.path.join(data_dir, cls._env_path("IDF_SVD_NPZ", e3.DEFAULT_IDF_SVD_NPZ))
        kahm_precomputed_policy = cls._env_path("KAHM_PRECOMPUTED_POLICY", "warn").lower()  # warn|strict|ignore

        # FAISS knobs
        faiss_index_type = cls._env_path("FAISS_INDEX_TYPE", "hnsw")  # "flat" or "hnsw"
        hnsw_m = int(cls._env_path("HNSW_M", "32"))
        ef_search = int(cls._env_path("HNSW_EF_SEARCH", "128"))
        faiss_threads = int(cls._env_path("FAISS_THREADS", "0"))
        if faiss_threads > 0:
            e3.set_faiss_threads(faiss_threads) # type: ignore[attr-defined]

        # KAHM runtime knobs
        kahm_mode = cls._env_path("KAHM_MODE", e3.DEFAULT_KAHM_MODE)
        regress_batch = int(cls._env_path("KAHM_REGRESS_BATCH", str(e3.DEFAULT_REGRESS_BATCH)))
        kahm_n_jobs = int(cls._env_path("KAHM_N_JOBS", str(e3.DEFAULT_KAHM_N_JOBS)))
        kahm_backend = cls._env_path("KAHM_BACKEND", e3.DEFAULT_KAHM_BACKEND)

        # Load MB corpus embeddings
        ids_mb, emb_mb, _meta_mb = e3.load_npz_bundle(semantic_npz)

        # Load KAHM(full) corpus embeddings (precomputed approx MB-space)
        ids_k, emb_k, meta_k = e3.load_npz_bundle(kahm_corpus_npz)

        # Optional: check NPZ compatibility against IDF NPZ metadata (same logic as eval script)
        if os.path.exists(idf_svd_npz) and kahm_precomputed_policy in ("warn", "strict"):
            # lightweight read of meta from the IDF NPZ without forcing us to keep its embeddings in RAM
            meta_idf: Dict[str, Any] = {}
            try:
                with np.load(idf_svd_npz, allow_pickle=False) as z:
                    for k in z.files:
                        if k in ("sentence_id", "embeddings"):
                            continue
                        v = z[k]
                        if isinstance(v, np.ndarray) and v.shape == ():
                            meta_idf[k] = v.item()
                        else:
                            meta_idf[k] = "<array>"
                ok, reasons = e3.kahm_npz_compatible(
                    meta_k=meta_k,
                    meta_idf=meta_idf,
                    kahm_model_path=kahm_model_path,
                    idf_svd_npz_path=idf_svd_npz,
                    kahm_mode=kahm_mode,
                )
                if not ok:
                    msg = " | ".join(reasons)
                    if kahm_precomputed_policy == "strict":
                        raise RuntimeError(
                            "Precomputed KAHM corpus NPZ appears incompatible with this configuration. "
                            f"Details: {msg}"
                        )
                    # warn mode: continue (API still runs, but you should regenerate the NPZ)
                    print(f"[WARN] KAHM corpus NPZ compatibility check failed: {msg}")
            except Exception as ex:
                if kahm_precomputed_policy == "strict":
                    raise
                print(f"[WARN] KAHM corpus NPZ compatibility check errored: {ex}")

        # Align MB and KAHM corpora by sentence_id intersection (mirrors eval behavior) 
        common_ids, emb_mb_a, emb_k_a = e3.align_by_sentence_id(ids_mb, emb_mb, ids_k, emb_k)

        # Normalize for cosine/IP retrieval
        emb_mb_a = e3.l2_normalize_rows(emb_mb_a)
        emb_k_a = e3.l2_normalize_rows(emb_k_a)

        # Load corpus metadata aligned to common_ids
        law_arr, page_arr, sent_arr = e3.load_corpus_info(corpus_parquet, common_ids)

        # Build FAISS indices (inner product on normalized vectors)
        index_mb = e3.build_faiss_index(emb_mb_a, index_type=faiss_index_type, hnsw_m=hnsw_m, ef_search=ef_search)
        index_kahm = e3.build_faiss_index(emb_k_a, index_type=faiss_index_type, hnsw_m=hnsw_m, ef_search=ef_search)

        # Load query-side models
        idf_pipe = e3.load_idf_svd_model(idf_svd_model)
        kahm_model = e3.load_kahm_model(kahm_model_path)

        # Sentence-level law priors (for lift normalization), same as eval. :contentReference[oaicite:7]{index=7}
        counts = Counter([str(x) for x in law_arr.tolist()])
        total = float(max(1, int(law_arr.size)))
        law_prior = {lw: float(cnt) / total for lw, cnt in counts.items()}

        return cls(
            sentence_ids=common_ids,
            law_arr=law_arr,
            page_arr=page_arr,
            sent_arr=sent_arr,
            index_mb=index_mb,
            index_kahm=index_kahm,
            idf_pipe=idf_pipe,
            kahm_model=kahm_model,
            law_prior=law_prior,
            kahm_mode=kahm_mode,
            regress_batch=regress_batch,
            kahm_n_jobs=kahm_n_jobs,
            kahm_backend=kahm_backend,
        )

    def _embed_query_kahm(self, query_text: str) -> np.ndarray:
        q = str(query_text or "").strip()
        if not q:
            raise ValueError("query_text is empty.")
        q_idf = e3.embed_queries_idf_svd(self.idf_pipe, [q])
        q_kahm = e3.kahm_regress_batched(
            self.kahm_model,
            q_idf,
            mode=str(self.kahm_mode),
            batch_size=int(max(1, self.regress_batch)),
            n_jobs=int(max(1, self.kahm_n_jobs)),
            backend=str(self.kahm_backend),
        )
        return q_kahm  # shape (1, d_mb)

    def _router_info(self, router_laws: List[str]) -> Dict[str, Any]:
        # Mirrors the router posterior / LLR calculation in the eval script. :contentReference[oaicite:8]{index=8}
        if not router_laws:
            return {"counts": {}, "llr": {}, "maj_law": "", "maj_frac": 0.0, "maj_lift": 0.0}

        c = Counter(router_laws)
        maj_law, maj_count = c.most_common(1)[0]
        maj_frac = float(maj_count) / float(len(router_laws))
        prior = float(self.law_prior.get(str(maj_law), 0.0))
        eps = float(e3.DEFAULT_KAHM_BEST_EPS)
        maj_lift = maj_frac / max(prior, eps)

        denom = float(len(router_laws))
        p = {lw: float(cnt) / denom for lw, cnt in c.items()}
        llr = {lw: float(np.log((p[lw] + eps) / (float(self.law_prior.get(lw, 0.0)) + eps))) for lw in p.keys()}
        return {"counts": dict(c), "llr": llr, "maj_law": str(maj_law), "maj_frac": maj_frac, "maj_lift": maj_lift}

    def retrieve_best_routed(
        self,
        *,
        query_text: str,
        k: int,
        router_k: int,
        cand_k: int,
        maj_frac_thr: float,
        maj_lift_thr: float,
        top_laws: int,
        law_slots: int,
        lam: float,
        mode: str,
        debug: bool,
    ) -> RetrieveResponse:
        k = int(k)
        router_k = int(max(k, router_k))
        cand_k = int(max(k, cand_k))

        mode = str(mode).strip().lower()
        if mode not in ("filter", "rerank", "hybrid"):
            mode = "hybrid"

        q_kahm = self._embed_query_kahm(query_text)

        # Candidate generation: q_kahm vs MB corpus index :contentReference[oaicite:9]{index=9}
        cand_scores, cand_idx = e3.faiss_search(self.index_mb, q_kahm, cand_k)
        cand_scores = cand_scores[0].astype(np.float32, copy=False)
        cand_idx = cand_idx[0].astype(np.int64, copy=False)

        # Router: q_kahm vs KAHM(full) corpus index :contentReference[oaicite:10]{index=10}
        router_scores, router_idx = e3.faiss_search(self.index_kahm, q_kahm, router_k)
        router_idx = router_idx[0].astype(np.int64, copy=False)

        router_laws = [str(self.law_arr[i]) for i in router_idx.tolist() if int(i) >= 0]
        info = self._router_info(router_laws)

        accepted = bool(info["counts"]) and (float(info["maj_frac"]) >= float(maj_frac_thr)) and (float(info["maj_lift"]) >= float(maj_lift_thr))

        # If router not confident, fall back to plain KAHM(q→MB) top-k
        if not accepted:
            picked = [int(i) for i in cand_idx.tolist() if int(i) >= 0][:k]
            results = self._format_hits(picked, cand_scores, cand_idx)
            return RetrieveResponse(
                query_text=query_text,
                k=k,
                accepted=False,
                router={
                    "maj_law": info["maj_law"],
                    "maj_frac": info["maj_frac"],
                    "maj_lift": info["maj_lift"],
                    "thresholds": {"maj_frac_thr": maj_frac_thr, "maj_lift_thr": maj_lift_thr},
                },
                results=results,
                debug=(self._debug_block(info, cand_idx, cand_scores, router_idx, router_scores[0]) if debug else None),
            )

        # Router accepted: law-aware selection (mirrors _build_kahm_best_idx) :contentReference[oaicite:11]{index=11}
        law_ranked = [lw for lw, _ in Counter(router_laws).most_common(max(1, int(top_laws)))]
        law_set = set(law_ranked)

        cand_laws = self.law_arr[cand_idx]
        order = np.arange(cand_idx.shape[0], dtype=np.int64)

        if mode in ("rerank", "hybrid") and float(lam) != 0.0:
            llr = info["llr"]
            bonuses = np.zeros((cand_idx.shape[0],), dtype=np.float32)
            for j in range(cand_idx.shape[0]):
                bonuses[j] = float(llr.get(str(cand_laws[j]), 0.0))
            rerank_scores = cand_scores + float(lam) * bonuses
            order = np.argsort(-rerank_scores, kind="mergesort")  # stable
            cand_scores_eff = rerank_scores
        else:
            cand_scores_eff = cand_scores

        ordered_idx = cand_idx[order]
        ordered_laws = cand_laws[order]
        ordered_scores = cand_scores_eff[order]

        picked: List[int] = []
        picked_scores: List[float] = []

        if mode in ("filter", "hybrid"):
            cap = int(max(1, min(int(law_slots), int(k))))
            for j, sid in enumerate(ordered_idx.tolist()):
                if len(picked) >= cap:
                    break
                if sid < 0:
                    continue
                if str(ordered_laws[j]) in law_set:
                    picked.append(int(sid))
                    picked_scores.append(float(ordered_scores[j]))

        if len(picked) < k:
            seen = set(picked)
            for j, sid in enumerate(ordered_idx.tolist()):
                if sid < 0 or sid in seen:
                    continue
                picked.append(int(sid))
                picked_scores.append(float(ordered_scores[j]))
                seen.add(int(sid))
                if len(picked) >= k:
                    break

        results = self._format_hits_with_scores(picked[:k], picked_scores[:k])

        return RetrieveResponse(
            query_text=query_text,
            k=k,
            accepted=True,
            router={
                "maj_law": info["maj_law"],
                "maj_frac": info["maj_frac"],
                "maj_lift": info["maj_lift"],
                "top_laws": law_ranked,
                "mode": mode,
                "lam": float(lam),
                "law_slots": int(law_slots),
                "thresholds": {"maj_frac_thr": maj_frac_thr, "maj_lift_thr": maj_lift_thr},
            },
            results=results,
            debug=(self._debug_block(info, cand_idx, cand_scores, router_idx, router_scores[0]) if debug else None),
        )

    def _format_hits(self, picked: List[int], cand_scores: np.ndarray, cand_idx: np.ndarray) -> List[RetrievalHit]:
        # If we picked by index, recover the original FAISS scores from the candidate arrays.
        score_map = {int(idx): float(sc) for idx, sc in zip(cand_idx.tolist(), cand_scores.tolist()) if int(idx) >= 0}
        return self._format_hits_with_scores(picked, [score_map.get(int(i), float("nan")) for i in picked])

    def _format_hits_with_scores(self, picked: List[int], scores: List[float]) -> List[RetrievalHit]:
        out: List[RetrievalHit] = []
        for r, (i, s) in enumerate(zip(picked, scores), start=1):
            out.append(
                RetrievalHit(
                    rank=r,
                    score=float(s),
                    sentence_id=int(self.sentence_ids[int(i)]),
                    law_type=str(self.law_arr[int(i)]),
                    page=int(self.page_arr[int(i)]),
                    sentence=str(self.sent_arr[int(i)]),
                )
            )
        return out

    def _debug_block(
        self,
        info: Dict[str, Any],
        cand_idx: np.ndarray,
        cand_scores: np.ndarray,
        router_idx: np.ndarray,
        router_scores: np.ndarray,
    ) -> Dict[str, Any]:
        # Keep it bounded to avoid huge payloads.
        return {
            "router_counts_top10": dict(Counter([str(self.law_arr[i]) for i in router_idx.tolist() if i >= 0]).most_common(10)),
            "router_llr_top10": dict(sorted(info.get("llr", {}).items(), key=lambda kv: -abs(float(kv[1])))[:10]),
            "cand_preview": [
                {
                    "idx": int(i),
                    "sentence_id": int(self.sentence_ids[int(i)]) if int(i) >= 0 else -1,
                    "law": str(self.law_arr[int(i)]) if int(i) >= 0 else "",
                    "score": float(s),
                }
                for i, s in zip(cand_idx[:20].tolist(), cand_scores[:20].tolist())
                if int(i) >= 0
            ],
            "router_preview": [
                {
                    "idx": int(i),
                    "sentence_id": int(self.sentence_ids[int(i)]) if int(i) >= 0 else -1,
                    "law": str(self.law_arr[int(i)]) if int(i) >= 0 else "",
                    "score": float(s),
                }
                for i, s in zip(router_idx[:20].tolist(), router_scores[:20].tolist())
                if int(i) >= 0
            ],
        }


# ----------------------------- FastAPI app -----------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.engine = KahmBestEngine.from_env()
    except Exception as ex:
        # Make startup failures explicit.
        raise RuntimeError(f"Failed to initialize KAHM_best_routed engine: {ex}") from ex
    yield


app = FastAPI(
    title="KAHM_best_routed Retrieval API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> Dict[str, Any]:
    eng: KahmBestEngine = app.state.engine
    return {
        "status": "ok",
        "n_sentences": int(eng.sentence_ids.size),
        "kahm_mode": eng.kahm_mode,
    }


@app.post("/v1/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    eng: KahmBestEngine = app.state.engine
    try:
        return eng.retrieve_best_routed(
            query_text=req.query_text,
            k=req.k,
            router_k=req.router_k,
            cand_k=req.cand_k,
            maj_frac_thr=req.maj_frac_thr,
            maj_lift_thr=req.maj_lift_thr,
            top_laws=req.top_laws,
            law_slots=req.law_slots,
            lam=req.lam,
            mode=req.mode,
            debug=req.debug,
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {ex}")


# Optional convenience GET endpoint
@app.get("/v1/retrieve", response_model=RetrieveResponse)
def retrieve_get(
    query_text: str = Query(...),
    k: int = Query(10, ge=1, le=50),
    debug: bool = Query(False),
) -> RetrieveResponse:
    eng: KahmBestEngine = app.state.engine
    try:
        return eng.retrieve_best_routed(
            query_text=query_text,
            k=k,
            router_k=50,
            cand_k=200,
            maj_frac_thr=0.40,
            maj_lift_thr=5.0,
            top_laws=1,
            law_slots=1,
            lam=0.0,
            mode="hybrid",
            debug=debug,
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {ex}")
