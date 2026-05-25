import json
from typing import Dict, List, Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from core import config
from core.models import get_embed_model
from core.text_utils import tokenize

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


def save_index() -> None:
    if rows:
        _save_jsonl(config.CORPUS_JSONL, rows)
    if embeddings is not None:
        np.save(config.EMB_NPY, embeddings)
    if index is not None:
        try:
            faiss.write_index(index, str(config.FAISS_INDEX))
        except Exception as e:  # keep server alive on faiss write issues
            print("faiss write warning:", e)
    if corpus:
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
    index = index_obj

    if config.BM25_JSON.exists():
        with open(config.BM25_JSON, "r", encoding="utf-8") as f:
            bm25_corpus = json.load(f)
        bm25_obj = BM25Okapi([tokenize(c) for c in bm25_corpus])
    else:
        bm25_obj = None
    bm25 = bm25_obj


load_index()


def rebuild_indices(batch_size: int = config.EMBED_BATCH) -> None:
    global corpus, embeddings, index, bm25
    corpus = [r["text"] for r in rows]
    if not corpus:
        embeddings = None
        index = None
        bm25 = None
        save_index()
        return

    em = get_embed_model()
    encs = []
    for i in range(0, len(corpus), batch_size):
        batch = corpus[i:i + batch_size]
        enc = em.encode(batch, batch_size=len(batch), show_progress_bar=False, normalize_embeddings=True)
        encs.append(enc)
    embeddings = np.vstack(encs).astype("float32")

    dim = embeddings.shape[1]
    cpu_index = faiss.IndexFlatIP(dim)
    cpu_index.add(embeddings)
    index = cpu_index

    bm25 = BM25Okapi([tokenize(c) for c in corpus])
    save_index()


def add_chunks(chunks: List[Dict]) -> None:
    existing_ids = {r["id"] for r in rows}
    new_items = [c for c in chunks if c["id"] not in existing_ids]
    if not new_items:
        return
    rows.extend(new_items)
    rebuild_indices()
