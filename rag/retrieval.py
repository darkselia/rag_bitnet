from typing import Dict, List, Optional
from urllib.parse import urlsplit

import numpy as np
from starlette.concurrency import run_in_threadpool

from common import config
from common.text_utils import best_excerpt, tokenize
from rag import data_store

RRF_K = 60.0
RERANK_RELATIVE_KEEP = 0.25


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


def _allowed_indices(rows: List[Dict], sources: Optional[List[str]], site_domain: Optional[str]) -> Optional[set[int]]:
    allowed_ids = None
    if sources:
        allowed_ids = {i for i, r in enumerate(rows) if r.get("kind") in set(sources)}
    if site_domain:
        domain = site_domain.lower().rstrip(".")
        site_ids = {
            i for i, r in enumerate(rows)
            if r.get("kind") == "site" and (
                    (host := (urlsplit(r.get("source", "")).hostname or "").lower().rstrip(".")) == domain
                    or host.endswith("." + domain)
            )
        }
        allowed_ids = site_ids if allowed_ids is None else allowed_ids & site_ids
    return allowed_ids


def _rrf_scores(*ranked_lists: List[int]) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
    return scores


def _trim_weak_rerank_hits(ranked: List[tuple[int, float]], k_final: int) -> List[tuple[int, float]]:
    if not ranked:
        return []
    top = ranked[:k_final]
    best_score = top[0][1]
    if best_score <= 0:
        return top
    threshold = best_score * RERANK_RELATIVE_KEEP
    kept = [item for item in top if item[1] >= threshold]
    return kept or top[:1]


def retrieve(query: str, k_vec: int = 8, k_bm25: int = 8, k_final: int = 6, sim_threshold: float = 0.0,
             sources: Optional[List[str]] = None, site_domain: Optional[str] = None) -> List[Dict]:
    if k_final <= 0:
        return []
    state = data_store.snapshot()
    if state.index is None or len(state.corpus) == 0:
        return []

    allowed_ids = _allowed_indices(state.rows, sources, site_domain)
    if allowed_ids is not None and not allowed_ids:
        return []

    from rag.models import get_embed_model, get_reranker

    em = get_embed_model()
    q_emb = em.encode([query], normalize_embeddings=True).astype("float32")
    vec_search_k = max(k_vec, k_final)
    if allowed_ids is not None:
        vec_search_k = len(state.corpus)
    vec_search_k = min(len(state.corpus), vec_search_k)

    D, I = state.index.search(q_emb, vec_search_k)
    vec_ranked = []
    for i, s in zip(I[0], D[0]):
        if i == -1:
            continue
        if allowed_ids is not None and i not in allowed_ids:
            continue
        if sim_threshold != 0 and float(s) < sim_threshold:
            continue
        vec_ranked.append(int(i))
        if len(vec_ranked) >= max(k_vec, k_final):
            break

    bm_ranked = []
    if state.bm25 is not None:
        bm_scores = state.bm25.get_scores(tokenize(query))
        bm_pool = max(k_bm25, k_final * 20, config.CROSS_RERANKER_TOPK * 3)
        for i in np.argsort(bm_scores)[::-1]:
            if bm_scores[i] <= 0:
                break
            if allowed_ids is not None and i not in allowed_ids:
                continue
            bm_ranked.append(int(i))
            if len(bm_ranked) >= bm_pool:
                break
        bm_ranked = bm_ranked[:max(k_bm25, k_final)]

    fused_scores = _rrf_scores(vec_ranked, bm_ranked)
    if not fused_scores:
        return []

    candidate_limit = max(config.CROSS_RERANKER_TOPK, k_final * 3)
    top_for_rerank = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:candidate_limit]

    reranker = get_reranker()
    if reranker:
        pairs = [[query, best_excerpt(query, state.corpus[i], max_chars=1000)] for i, _ in top_for_rerank]
        rerank_scores = reranker.predict(pairs, batch_size=config.RERANKER_BATCH)
        ranked = sorted(zip([i for i, _ in top_for_rerank], rerank_scores), key=lambda x: x[1], reverse=True)
        top = _trim_weak_rerank_hits([(i, float(score)) for i, score in ranked], k_final)
    else:
        top = [(i, float(score)) for i, score in top_for_rerank[:k_final]]

    return [{
        "id": state.rows[i]["id"],
        "text": state.rows[i]["text"],
        "excerpt": best_excerpt(query, state.rows[i]["text"]),
        "score": float(score),
        "source": state.rows[i]["source"],
        "kind": state.rows[i].get("kind"),
        "meta": state.rows[i].get("meta", {}),
    } for i, score in top]
