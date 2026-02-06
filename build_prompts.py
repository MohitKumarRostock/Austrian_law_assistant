#!/usr/bin/env python3
"""
1) Read query embeddings (train/test) from NPZ
2) Read sentence embeddings from embedding_index.npz
3) Build FAISS IndexFlatIP on L2-normalized sentence embeddings (cosine via inner product)
4) Retrieve top-k sentences per query from ris_sentences.parquet
5) Build a prompt (query + ranked excerpts)
6) Save prompt datasets (parquet + jsonl) for downstream generation

pip install numpy pandas pyarrow tqdm faiss-cpu
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, List, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def _to_str_array(x: np.ndarray) -> np.ndarray:
    if x.dtype == object:
        return np.array([v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v) for v in x], dtype=object)
    if np.issubdtype(x.dtype, np.bytes_):
        return np.array([v.decode("utf-8") for v in x], dtype=object)
    return x.astype(object)


def l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norms, eps)


def load_queries_npz(path: str,
                    id_key: str = "query_id",
                    text_key: str = "query_text",
                    emb_key: str = "embeddings") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    qid = z[id_key]
    qtxt = _to_str_array(z[text_key])
    emb = np.asarray(z[emb_key], dtype=np.float32)
    return qid, qtxt, emb


def load_sent_npz(path: str,
                  id_key: str = "sentence_id",
                  emb_key: str = "embeddings") -> Tuple[np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    sid = z[id_key]
    emb = np.asarray(z[emb_key], dtype=np.float32)
    return sid, emb


def read_sentences_parquet(path: str) -> pd.DataFrame:
    cols = ["sentence_id", "law_type", "page", "sentence", "source_file"]
    df = pd.read_parquet(path, columns=cols)
    return df.set_index("sentence_id", drop=False)


def make_prompt(
    query_text: str,
    retrieved: pd.DataFrame,
    max_sentence_chars: int,
    summary_language: str,
) -> str:
    lines: List[str] = []
    lines.append(
        f"Dir wird eine Nutzeranfrage sowie potenziell relevante Auszüge aus österreichischem Recht gegeben.\n"
        f"Aufgabe: Verfasse eine knappe, sachliche Zusammenfassung, die die Anfrage ausschließlich anhand der bereitgestellten Auszüge beantwortet.\n"
        f"Schreibe die Zusammenfassung auf {summary_language}. Wenn die Auszüge nicht ausreichen, gib an, welche Informationen fehlen.\n"
    )
    lines.append("Query:")
    lines.append(query_text.strip())
    lines.append("\nExcerpts (ranked):")

    for i, (_, row) in enumerate(retrieved.iterrows(), start=1):
        sent = str(row.get("sentence", "")).strip().replace("\n", " ")
        if len(sent) > max_sentence_chars:
            print(f"Truncating sentence_id={row.get('sentence_id','')} from {len(sent)} to {max_sentence_chars} chars")
            sent = sent[: max_sentence_chars - 3].rstrip() + "..."
        lines.append(
            f"{i}. [law_type={row.get('law_type','')} | page={row.get('page','')} | source_file={row.get('source_file','')}] "
            f"{sent}"
        )

    lines.append("\nAnswer:")
    return "\n".join(lines)


@dataclass
class PromptRecord:
    split: str
    record_id: str          # unique id used for batch custom_id
    query_id: Any
    query_text: str
    prompt: str
    retrieved_sentence_ids: List[Any]
    retrieved_scores: List[float]


def build_split(
    split: str,
    query_npz: str,
    sent_ids: np.ndarray,
    sent_emb_norm: np.ndarray,
    sent_df: pd.DataFrame,
    top_k: int,
    max_sentence_chars: int,
    summary_language: str,
    query_id_key: str,
    query_text_key: str,
    query_emb_key: str,
) -> List["PromptRecord"]:
    """
    Builds PromptRecord list for a split by:
      - loading query embeddings
      - L2-normalizing query embeddings
      - exact top-k retrieval with FAISS IndexFlat + inner-product (cosine on normalized vectors)
      - assembling prompt text with retrieved sentence metadata

    Notes:
      - sent_emb_norm is assumed to already be L2-normalized.
      - FAISS prefers contiguous float32 arrays; we enforce that.
      - We type `index` as Any to avoid Pylance/pyright stub/signature issues with faiss.
    """
    import faiss  # type: ignore

    # Load queries
    qids, qtxts, qemb = load_queries_npz(
        query_npz,
        id_key=query_id_key,
        text_key=query_text_key,
        emb_key=query_emb_key,
    )

    # Normalize queries for cosine similarity
    qemb_norm = l2_normalize(qemb)

    # FAISS requires float32 + contiguous arrays
    sent_mat = np.ascontiguousarray(sent_emb_norm, dtype=np.float32)
    q_mat = np.ascontiguousarray(qemb_norm, dtype=np.float32)

    if sent_mat.ndim != 2 or q_mat.ndim != 2:
        raise ValueError(f"Embeddings must be 2D. sent={sent_mat.shape}, query={q_mat.shape}")
    if sent_mat.shape[1] != q_mat.shape[1]:
        raise ValueError(f"Dim mismatch. sent_dim={sent_mat.shape[1]} query_dim={q_mat.shape[1]}")

    d = int(sent_mat.shape[1])

    # Exact cosine retrieval: inner product on L2-normalized vectors
    index: Any = faiss.IndexFlat(d, faiss.METRIC_INNER_PRODUCT)
    index.add(sent_mat)

    scores, idxs = index.search(q_mat, int(top_k))  # (N,k), (N,k)

    records: List[PromptRecord] = []
    for i in tqdm(range(len(qtxts)), desc=f"[{split}] prompts"):
        qid = qids[i]
        qtext = str(qtxts[i])

        ind = idxs[i].tolist()
        sc = scores[i].astype(float).tolist()

        # Map FAISS row indices -> sentence_id
        retrieved_ids = [sent_ids[j] for j in ind]

        # Lookup sentence metadata in parquet dataframe (indexed by sentence_id)
        retrieved_rows = sent_df.reindex(retrieved_ids)

        # Drop missing rows (in case parquet and embedding ids don't perfectly match)
        missing = retrieved_rows["sentence_id"].isna()
        if bool(missing.any()):
            keep = (~missing).to_numpy().tolist()
            retrieved_rows = retrieved_rows.loc[~missing]
            retrieved_ids = [rid for rid, k in zip(retrieved_ids, keep) if k]
            sc = [s for s, k in zip(sc, keep) if k]

        prompt = make_prompt(
            query_text=qtext,
            retrieved=retrieved_rows,
            max_sentence_chars=max_sentence_chars,
            summary_language=summary_language,
        )

        record_id = f"{split}:{i}"  # stable + unique even if query_id overlaps
        records.append(
            PromptRecord(
                split=split,
                record_id=record_id,
                query_id=qid,
                query_text=qtext,
                prompt=prompt,
                retrieved_sentence_ids=retrieved_ids,
                retrieved_scores=sc,
            )
        )

    return records


def json_safe(obj):
    """Convert numpy scalars/arrays (and nested structures) into JSON-serializable Python types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def save_records(records: List[PromptRecord], out_dir: str, split: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, f"{split}_prompts.jsonl")
    parquet_path = os.path.join(out_dir, f"{split}_prompts.parquet")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            payload = json_safe(asdict(r))
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    pd.DataFrame([asdict(r) for r in records]).to_parquet(parquet_path, index=False)
    print(f"Wrote {jsonl_path}")
    print(f"Wrote {parquet_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_npz", default="queries_embedding_index_train.npz")
    ap.add_argument("--test_npz", default="queries_embedding_index_test.npz")
    ap.add_argument("--sent_npz", default="embedding_index.npz")
    ap.add_argument("--sent_parquet", default="ris_sentences.parquet")
    ap.add_argument("--output_dir", default=".")

    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--max_sentence_chars", type=int, default=5000)
    ap.add_argument("--summary_language", default="English")

    ap.add_argument("--query_id_key", default="query_id")
    ap.add_argument("--query_text_key", default="query_text")
    ap.add_argument("--query_emb_key", default="embeddings")
    ap.add_argument("--sent_id_key", default="sentence_id")
    ap.add_argument("--sent_emb_key", default="embeddings")

    args = ap.parse_args()

    sent_ids, sent_emb = load_sent_npz(args.sent_npz, id_key=args.sent_id_key, emb_key=args.sent_emb_key)
    sent_emb_norm = l2_normalize(sent_emb)

    sent_df = read_sentences_parquet(args.sent_parquet)

    train_records = build_split(
        split="train",
        query_npz=args.train_npz,
        sent_ids=sent_ids,
        sent_emb_norm=sent_emb_norm,
        sent_df=sent_df,
        top_k=args.top_k,
        max_sentence_chars=args.max_sentence_chars,
        summary_language=args.summary_language,
        query_id_key=args.query_id_key,
        query_text_key=args.query_text_key,
        query_emb_key=args.query_emb_key,
    )
    save_records(train_records, args.output_dir, "train")

    test_records = build_split(
        split="test",
        query_npz=args.test_npz,
        sent_ids=sent_ids,
        sent_emb_norm=sent_emb_norm,
        sent_df=sent_df,
        top_k=args.top_k,
        max_sentence_chars=args.max_sentence_chars,
        summary_language=args.summary_language,
        query_id_key=args.query_id_key,
        query_text_key=args.query_text_key,
        query_emb_key=args.query_emb_key,
    )
    save_records(test_records, args.output_dir, "test")


if __name__ == "__main__":
    main()
