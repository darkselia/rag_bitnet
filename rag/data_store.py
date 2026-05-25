import json
from typing import Dict, List, Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from common import config
from common.text_utils import tokenize
from rag.models import get_embed_model

rows: List[Dict] = []
corpus: List[str] = []
index: Optional[faiss.IndexFlatIP] = None
embeddings: Optional[np.ndarray] = None
bm25: Optional[BM25Okapi] = None


def _load_jsonl(path) -> List[Dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _save_jsonl(path, data: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _norm_path(path: str) -> str:
    return path.replace("\\", "/")


def save_index() -> None:
    _save_jsonl(config.CORPUS_JSONL, rows)
    if embeddings is not None:
        np.save(config.EMB_NPY, embeddings)
    elif config.EMB_NPY.exists():
        config.EMB_NPY.unlink()
    if index is not None:
        try:
            faiss.write_index(index, str(config.FAISS_INDEX))
        except Exception as e:  # keep server alive on faiss write issues
            print("faiss write warning:", e)
    elif config.FAISS_INDEX.exists():
        config.FAISS_INDEX.unlink()
    with open(config.BM25_JSON, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False)
    with open(config.META_JSON, "w", encoding="utf-8") as f:
        json.dump(config.meta_payload(len(rows)), f)


def load_index():
    global rows, corpus, embeddings, index, bm25
    rows = _load_jsonl(config.CORPUS_JSONL)
    corpus = [r.get("text", "") for r in rows]

    embeddings = np.load(config.EMB_NPY) if config.EMB_NPY.exists() else None

    if config.FAISS_INDEX.exists():
        try:
            index_obj = faiss.read_index(str(config.FAISS_INDEX))
        except Exception as e:
            print("faiss load warning:", e)
            index_obj = None
    else:
        index_obj = None
    if index_obj is not None and index_obj.ntotal != len(corpus):
        print(f"faiss load warning: index size {index_obj.ntotal} != corpus size {len(corpus)}")
        index_obj = None
    if embeddings is not None and embeddings.shape[0] != len(corpus):
        print(f"embeddings load warning: embeddings size {embeddings.shape[0]} != corpus size {len(corpus)}")
        embeddings = None
    index = index_obj

    bm25_corpus = corpus
    if config.BM25_JSON.exists():
        try:
            with open(config.BM25_JSON, "r", encoding="utf-8") as f:
                loaded_bm25_corpus = json.load(f)
            if loaded_bm25_corpus and len(loaded_bm25_corpus) == len(corpus):
                bm25_corpus = loaded_bm25_corpus
        except Exception as e:
            print("bm25 load warning:", e)
    bm25_tokens = [tokenize(c) for c in bm25_corpus]
    bm25_obj = BM25Okapi(bm25_tokens) if bm25_tokens else None
    bm25 = bm25_obj


load_index()


def rebuild_indices(batch_size: int = config.EMBED_BATCH) -> None:
    global corpus, embeddings, index, bm25
    rows[:] = [r for r in rows if r.get("text")]
    corpus = [r["text"] for r in rows]
    print(f"[index] rebuilding indices for {len(corpus)} chunks", flush=True)
    if not corpus:
        embeddings = None
        index = None
        bm25 = None
        save_index()
        print("[index] saved empty indices", flush=True)
        return

    em = get_embed_model()
    encs = []
    for i in range(0, len(corpus), batch_size):
        batch = corpus[i:i + batch_size]
        print(f"[index] embedding batch {i // batch_size + 1}/{(len(corpus) + batch_size - 1) // batch_size}",
              flush=True)
        enc = em.encode(batch, batch_size=len(batch), show_progress_bar=False, normalize_embeddings=True)
        encs.append(enc)
    embeddings = np.vstack(encs).astype("float32")

    dim = embeddings.shape[1]
    cpu_index = faiss.IndexFlatIP(dim)
    cpu_index.add(embeddings)
    index = cpu_index

    bm25 = BM25Okapi([tokenize(c) for c in corpus])
    save_index()
    print("[index] rebuild complete", flush=True)


def remove_chunks_by_paths(paths: set[str]) -> int:
    global rows
    if not paths:
        return 0

    normalized_paths = {_norm_path(path) for path in paths}
    before = len(rows)
    rows = [
        r for r in rows
        if _norm_path(str(r.get("meta", {}).get("path", ""))) not in normalized_paths
    ]
    removed = before - len(rows)
    if removed:
        rebuild_indices()
    return removed


def replace_chunks_by_paths(paths: set[str], chunks: List[Dict]) -> int:
    global rows
    normalized_paths = {_norm_path(path) for path in paths}
    kept_rows = [
        r for r in rows
        if _norm_path(str(r.get("meta", {}).get("path", ""))) not in normalized_paths
    ]
    existing_ids = {r["id"] for r in kept_rows}
    new_items = [c for c in chunks if c["id"] not in existing_ids]

    rows = kept_rows + new_items
    rebuild_indices()
    return len(new_items)


def add_chunks(chunks: List[Dict]) -> int:
    existing_ids = {r["id"] for r in rows}
    new_items = [c for c in chunks if c["id"] not in existing_ids]
    if not new_items:
        return 0
    rows.extend(new_items)
    rebuild_indices()
    return len(new_items)
