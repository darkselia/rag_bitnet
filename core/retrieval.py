from typing import Dict, List, Optional

import numpy as np
from starlette.concurrency import run_in_threadpool

from core import config
from core import data_store
from core.text_utils import best_excerpt, tokenize


async def retrieve_async(query: str, k_vec: int = 8, k_bm25: int = 8, k_final: int = 6, sim_threshold: float = 0.0,
                         sources: Optional[List[str]] = None, site_domain: Optional[str] = None) -> List[Dict]:
    # reuse sync implementation in a thread to avoid blocking event loop
    return await run_in_threadpool(
        retrieve,
        query,
        k_vec,
        k_bm25,
        k_final,
        sim_threshold,
        sources,
        site_domain,
    )


def retrieve(query: str, k_vec: int = 8, k_bm25: int = 8, k_final: int = 6, sim_threshold: float = 0.0,
             sources: Optional[List[str]] = None, site_domain: Optional[str] = None) -> List[Dict]:
    if data_store.index is None or data_store.bm25 is None or len(data_store.corpus) == 0:
        return []

    allowed_ids = None
    if sources:
        allowed_ids = {i for i, r in enumerate(data_store.rows) if r.get("kind") in set(sources)}
    if site_domain:
        allowed_ids = {i for i, r in enumerate(data_store.rows) if r.get("kind") == "site" and site_domain in r.get("source", "")} if allowed_ids is None else {i for i in allowed_ids if data_store.rows[i].get("kind") == "site" and site_domain in data_store.rows[i].get("source", "")}

    from core.models import get_embed_model, get_reranker

    em = get_embed_model()
    q_emb = em.encode([query], normalize_embeddings=True).astype("float32")
    D, I = data_store.index.search(q_emb, k_vec)
    vec_hits = []
    for i, s in zip(I[0], D[0]):
        if i == -1:
            continue
        if allowed_ids is not None and i not in allowed_ids:
            continue
        vec_hits.append((int(i), float(s)))

    bm_scores = data_store.bm25.get_scores(tokenize(query))
    bm_idx = np.argsort(bm_scores)[::-1][:k_bm25]
    bm_hits = []
    for i in bm_idx:
        if allowed_ids is not None and i not in allowed_ids:
            continue
        bm_hits.append((int(i), float(bm_scores[i])))

    scores = {}
    for i, s in vec_hits + bm_hits:
        prev = scores.get(i, 0.0)
        scores[i] = max(prev, s)
    candidates = [(i, scores[i]) for i in scores]
    vec_dict = dict(vec_hits)
    candidates = [(i, s) for i, s in candidates if s >= (sim_threshold if i in vec_dict else 0.0)]
    if not candidates:
        return []

    candidates_sorted = sorted(candidates, key=lambda x: x[1], reverse=True)
    top_for_rerank = candidates_sorted[: max(config.CROSS_RERANKER_TOPK, k_final * 3)]

    reranker = get_reranker()
    if reranker:
        pairs = [[query, best_excerpt(query, data_store.corpus[i], max_chars=1000)] for i, _ in top_for_rerank]
        rerank_scores = reranker.predict(pairs, batch_size=config.RERANKER_BATCH)
        ranked = sorted(zip([i for i, _ in top_for_rerank], rerank_scores), key=lambda x: x[1], reverse=True)
        top = ranked[:k_final]
    else:
        top = sorted(top_for_rerank, key=lambda x: x[1], reverse=True)[:k_final]

    return [{
        "id": data_store.rows[i]["id"],
        "text": data_store.rows[i]["text"],
        "excerpt": best_excerpt(query, data_store.rows[i]["text"]),
        "score": float(score),
        "source": data_store.rows[i]["source"],
        "kind": data_store.rows[i].get("kind"),
        "meta": data_store.rows[i].get("meta", {}),
    } for i, score in top]
