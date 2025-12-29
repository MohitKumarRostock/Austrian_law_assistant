#!/usr/bin/env python3
#compare_embedding_npz.py
import argparse
import os
import sys
from pathlib import Path
import numpy as np


# -------------------------
# Utilities
# -------------------------

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, eps, None)


def _safe_scalar(v):
    try:
        if np.ndim(v) == 0:
            return v.item()
        return {"shape": tuple(v.shape), "dtype": str(v.dtype)}
    except Exception:
        return "<unreadable>"


def load_npz_embeddings(path: str):
    """
    Loads an embedding bundle saved as NPZ.
    Requires keys: 'sentence_id' (N,) and 'embeddings' (N,D).
    Returns (ids:int64, emb:float32, meta:dict, keys:list[str]).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with np.load(str(p), allow_pickle=False) as d:
        keys = list(d.keys())
        if "sentence_id" not in d or "embeddings" not in d:
            raise KeyError(
                f"{path} must contain 'sentence_id' and 'embeddings'. Found keys: {keys}"
            )

        ids = d["sentence_id"]
        emb = d["embeddings"]

        meta = {}
        for k in keys:
            if k not in ("sentence_id", "embeddings"):
                meta[k] = _safe_scalar(d[k])

    ids = np.asarray(ids, dtype=np.int64)
    emb = np.asarray(emb, dtype=np.float32)

    if ids.ndim != 1:
        raise ValueError(f"{path}: sentence_id must be 1D; got shape {ids.shape}")
    if emb.ndim != 2:
        raise ValueError(f"{path}: embeddings must be 2D; got shape {emb.shape}")
    if emb.shape[0] != ids.shape[0]:
        raise ValueError(
            f"{path}: mismatch: embeddings has {emb.shape[0]} rows but sentence_id has {ids.shape[0]} entries"
        )

    return ids, emb, meta, keys


def summarize_bundle(label: str, path: str, ids: np.ndarray, emb: np.ndarray, meta: dict, keys: list):
    print(f"\n[{label}] {path}")
    print(f"  keys: {keys}")
    print(f"  N={len(ids)}  D={emb.shape[1]}  ids.dtype={ids.dtype}  emb.dtype={emb.dtype}")
    print(f"  embeddings stats: min={np.min(emb):.6g} max={np.max(emb):.6g} mean={np.mean(emb):.6g}")

    finite = np.isfinite(emb)
    if not finite.all():
        bad = int(np.size(emb) - finite.sum())
        print(f"  WARNING: embeddings contain {bad} non-finite values (NaN/inf).")

    uniq = np.unique(ids)
    if len(uniq) != len(ids):
        print(f"  WARNING: sentence_id contains duplicates: unique={len(uniq)} vs N={len(ids)}")
    else:
        print("  sentence_id: unique")

    if meta:
        print("  metadata:")
        for k in sorted(meta.keys()):
            print(f"    - {k}: {meta[k]}")
    else:
        print("  metadata: (none)")


def align_by_sentence_id(ids_a, emb_a, ids_b, emb_b):
    """
    Align by intersection of sentence_id. Uses the first occurrence if duplicates exist.
    Returns (common_ids_sorted, A_aligned, B_aligned).
    """
    idx_a = {}
    for i, sid in enumerate(ids_a):
        if sid not in idx_a:
            idx_a[sid] = i

    idx_b = {}
    for i, sid in enumerate(ids_b):
        if sid not in idx_b:
            idx_b[sid] = i

    common = np.array(sorted(set(idx_a.keys()) & set(idx_b.keys())), dtype=np.int64)
    if common.size == 0:
        raise ValueError("No overlapping sentence_id values between the two files.")

    a_rows = np.fromiter((idx_a[sid] for sid in common), dtype=np.int64, count=common.size)
    b_rows = np.fromiter((idx_b[sid] for sid in common), dtype=np.int64, count=common.size)

    return common, emb_a[a_rows], emb_b[b_rows]


# -------------------------
# Similarity metrics
# -------------------------

def procrustes_orthogonal(Xn: np.ndarray, Yn: np.ndarray) -> np.ndarray:
    """
    Orthogonal Procrustes: W = argmin ||XW - Y||_F s.t. W^T W = I.
    Assumes Xn, Yn are normalized for cosine comparison.
    """
    M = Xn.T @ Yn
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    return U @ Vt


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear CKA between X and Y (row-wise examples), with centering.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)

    XY = Xc.T @ Yc
    XX = Xc.T @ Xc
    YY = Yc.T @ Yc

    hsic_xy = np.sum(XY * XY)
    hsic_xx = np.sum(XX * XX)
    hsic_yy = np.sum(YY * YY)
    return float(hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-12))


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    """
    Spearman correlation via rank-transform then Pearson correlation.
    """
    ra = a.argsort().argsort().astype(np.float64)
    rb = b.argsort().argsort().astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = (np.linalg.norm(ra) * np.linalg.norm(rb)) + 1e-12
    return float((ra @ rb) / denom)


def rsa_pair_sample_spearman(Xn: np.ndarray, Yn: np.ndarray, pairs: int, seed: int) -> float:
    """
    RSA-like comparison without O(N^2): sample random pairs (i,j), compare cosine similarities.
    Returns Spearman correlation between sampled similarity values.
    """
    rng = np.random.default_rng(seed)
    n = Xn.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 vectors for RSA sampling.")

    pairs = int(pairs)
    if pairs <= 0:
        raise ValueError("--pairs must be > 0")

    i = rng.integers(0, n, size=pairs, dtype=np.int64)
    j = rng.integers(0, n, size=pairs, dtype=np.int64)

    # Avoid i == j by resampling collisions
    mask = (i == j)
    while mask.any():
        j[mask] = rng.integers(0, n, size=int(mask.sum()), dtype=np.int64)
        mask = (i == j)

    sx = np.sum(Xn[i] * Xn[j], axis=1)
    sy = np.sum(Yn[i] * Yn[j], axis=1)
    return spearman_corr(sx, sy)


# -------------------------
# FAISS-based neighborhood + self-retrieval
# -------------------------

def _require_faiss():
    try:
        import faiss  # type: ignore
        return faiss
    except Exception as e:
        raise RuntimeError(
            "FAISS is not available. Install faiss-cpu (or faiss-gpu) or use --skip-knn/--skip-self."
        ) from e


def knn_overlap_faiss(Xn: np.ndarray, Yn: np.ndarray, k: int):
    """
    Compute neighborhood overlap metrics using FAISS inner-product search (cosine if normalized).
    Returns mean recall@k and mean jaccard@k over queries.
    """
    faiss = _require_faiss()

    d = Xn.shape[1]
    index_x = faiss.IndexFlatIP(d)
    index_y = faiss.IndexFlatIP(d)

    Xf = Xn.astype(np.float32, copy=False)
    Yf = Yn.astype(np.float32, copy=False)
    index_x.add(Xf)
    index_y.add(Yf)

    # Search top (k+1) to drop self match at rank 0
    _, ix = index_x.search(Xf, k + 1)
    _, iy = index_y.search(Yf, k + 1)

    ix = ix[:, 1:]
    iy = iy[:, 1:]

    n = Xn.shape[0]
    recall = np.empty(n, dtype=np.float32)
    jacc = np.empty(n, dtype=np.float32)

    for t in range(n):
        sx = set(ix[t].tolist())
        sy = set(iy[t].tolist())
        inter = len(sx & sy)
        union = len(sx | sy)
        recall[t] = inter / k
        jacc[t] = (inter / union) if union else 0.0

    return float(recall.mean()), float(jacc.mean())


def self_retrieval_metrics(db_vecs: np.ndarray, q_vecs: np.ndarray, topk: int):
    """
    FAISS self-retrieval: build index on db_vecs, query with q_vecs.
    Assumes row i in q corresponds to row i in db (true after alignment).
    Returns Recall@1, Recall@topk, MRR.
    """
    faiss = _require_faiss()

    N, D = db_vecs.shape
    if topk <= 0:
        raise ValueError("topk must be > 0")
    if topk > N:
        topk = N

    index = faiss.IndexFlatIP(D)
    dbf = db_vecs.astype(np.float32, copy=False)
    qf = q_vecs.astype(np.float32, copy=False)
    index.add(dbf)

    _, I = index.search(qf, topk)

    correct = np.arange(N, dtype=np.int64)
    hit_any = np.zeros(N, dtype=bool)
    hit_1 = (I[:, 0] == correct)

    rr = np.zeros(N, dtype=np.float32)
    for i in range(N):
        where = np.where(I[i] == correct[i])[0]
        if where.size:
            hit_any[i] = True
            rr[i] = 1.0 / float(where[0] + 1)

    r1 = float(hit_1.mean())
    rk = float(hit_any.mean())
    mrr = float(rr.mean())
    return r1, rk, mrr


# -------------------------
# File auto-detection
# -------------------------

def autodetect_npz_files(cwd: Path):
    npzs = sorted(cwd.glob("*.npz"))
    if len(npzs) == 0:
        raise FileNotFoundError(f"No .npz files found in {cwd}")
    if len(npzs) == 2:
        return str(npzs[0]), str(npzs[1])

    # Heuristics if more than 2 files:
    # Prefer names containing 'mb'/'mixedbread' for A and 'kahm' for B.
    def score_a(p: Path):
        s = p.name.lower()
        score = 0
        if "mixedbread" in s: score += 5
        if "mxbai" in s: score += 3
        if "mb" in s: score += 2
        if "embedding" in s: score += 1
        return score

    def score_b(p: Path):
        s = p.name.lower()
        score = 0
        if "kahm" in s: score += 6
        if "regress" in s: score += 2
        if "approx" in s: score += 1
        return score

    best_a = max(npzs, key=score_a)
    remaining = [p for p in npzs if p != best_a]
    best_b = max(remaining, key=score_b) if remaining else best_a

    return str(best_a), str(best_b)


# -------------------------
# Main
# -------------------------

def stats(x: np.ndarray):
    return {
        "mean": float(np.mean(x)),
        "p10": float(np.quantile(x, 0.10)),
        "p50": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Compare two embedding NPZ files (align by sentence_id, evaluate similarity).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--a", default="embedding_index.npz", help="Path to MB embeddings.")
    ap.add_argument("--b", default="embedding_index_kahm_mixedbread_approx.npz", help="Path to KAHM embeddings.")
    ap.add_argument("--sample", type=int, default=50000, help="Max aligned rows to sample for evaluation.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")
    ap.add_argument("--pairs", type=int, default=200000, help="Number of random pairs for RSA sampling.")
    ap.add_argument("--knn", type=int, nargs="*", default=[10, 50, 100], help="k values for kNN overlap.")
    ap.add_argument("--skip-knn", action="store_true", help="Skip kNN overlap computation.")
    ap.add_argument("--self-topk", type=int, default=10, help="Top-k for self-retrieval metrics.")
    ap.add_argument("--skip-self", action="store_true", help="Skip self-retrieval computation.")
    args = ap.parse_args()

    if args.a is None or args.b is None:
        a_path, b_path = autodetect_npz_files(Path.cwd())
        if args.a is None:
            args.a = a_path
        if args.b is None:
            args.b = b_path
        print(f"[Auto-detect] Using A={args.a}")
        print(f"[Auto-detect] Using B={args.b}")

    ids_a, emb_a, meta_a, keys_a = load_npz_embeddings(args.a)
    ids_b, emb_b, meta_b, keys_b = load_npz_embeddings(args.b)

    summarize_bundle("A", args.a, ids_a, emb_a, meta_a, keys_a)
    summarize_bundle("B", args.b, ids_b, emb_b, meta_b, keys_b)

    common_ids, A, B = align_by_sentence_id(ids_a, emb_a, ids_b, emb_b)

    print(f"\n[Alignment]")
    print(f"  Intersection size: {len(common_ids)}")
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"Dimension mismatch: A D={A.shape[1]} vs B D={B.shape[1]}")

    # Sample for speed
    rng = np.random.default_rng(args.seed)
    n = A.shape[0]
    if n > args.sample:
        idx = rng.choice(n, size=args.sample, replace=False)
        A = A[idx]
        B = B[idx]
        common_ids = common_ids[idx]
        print(f"  Sampled to N={A.shape[0]} for evaluation")
    else:
        print(f"  Using all N={n} aligned vectors for evaluation")

    # Normalize (cosine/IP comparable)
    An = l2_normalize(A)
    Bn = l2_normalize(B)

    # Similarity metrics
    raw_cos = np.sum(An * Bn, axis=1)

    W = procrustes_orthogonal(An, Bn)
    A_al = An @ W
    aligned_cos = np.sum(A_al * Bn, axis=1)

    cka = linear_cka(An, Bn)

    # Keep RSA pairs bounded relative to N to avoid excessive work
    # Rule of thumb: up to ~20*N pairs is plenty for stable estimates, capped by --pairs.
    rsa_pairs = min(int(args.pairs), max(1000, int(An.shape[0] * 20)))
    rsa = rsa_pair_sample_spearman(An, Bn, pairs=rsa_pairs, seed=args.seed)

    s_raw = stats(raw_cos)
    s_al = stats(aligned_cos)

    print("\n[Similarity Results]")
    print(f"  Raw pointwise cosine:      mean={s_raw['mean']:.4f}  p10={s_raw['p10']:.4f}  p50={s_raw['p50']:.4f}  p90={s_raw['p90']:.4f}")
    print(f"  Procrustes-aligned cosine: mean={s_al['mean']:.4f}  p10={s_al['p10']:.4f}  p50={s_al['p50']:.4f}  p90={s_al['p90']:.4f}")
    print(f"  Linear CKA (centered):     {cka:.4f}")
    print(f"  RSA Spearman (pair-sampled, pairs={rsa_pairs}): {rsa:.4f}")

    # kNN overlap
    if not args.skip_knn:
        print("\n[kNN Neighborhood Overlap (FAISS; cosine via L2 normalization)]")
        for k in args.knn:
            if k <= 0:
                continue
            if k >= An.shape[0]:
                print(f"  Skipping k={k} (k must be < N)")
                continue
            try:
                recall_k, jacc_k = knn_overlap_faiss(An, Bn, k=k)
                print(f"  k={k:4d}: Recall@k={recall_k:.4f}  Jaccard@k={jacc_k:.4f}")
            except RuntimeError as e:
                print(f"  WARNING: {e}")
                break

    # Self-retrieval
    if not args.skip_self:
        topk = int(args.self_topk)
        if topk <= 0:
            raise ValueError("--self-topk must be > 0")

        print(f"\n[Self-Retrieval Quality (FAISS; cosine via L2 normalization; topk={topk})]")
        try:
            # Query KAHM against MB DB
            r1, rk, mrr = self_retrieval_metrics(db_vecs=An, q_vecs=Bn, topk=topk)
            print(f"  Query=B, DB=A: Recall@1={r1:.4f}  Recall@{topk}={rk:.4f}  MRR={mrr:.4f}")

            # Query MB against KAHM DB
            r1, rk, mrr = self_retrieval_metrics(db_vecs=Bn, q_vecs=An, topk=topk)
            print(f"  Query=A, DB=B: Recall@1={r1:.4f}  Recall@{topk}={rk:.4f}  MRR={mrr:.4f}")
        except RuntimeError as e:
            print(f"  WARNING: {e}")

    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
