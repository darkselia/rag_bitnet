from typing import Optional

from sentence_transformers import CrossEncoder, SentenceTransformer

from common import config

_embed_model: Optional[SentenceTransformer] = None
_reranker: Optional[CrossEncoder] = None
_device = "cpu"  # adjust if GPU available


def get_embed_model() -> SentenceTransformer | None:
    global _embed_model
    if _embed_model is None:
        print(f"[init] loading embed model {config.EMBED_MODEL} on device {_device}")
        _embed_model = SentenceTransformer(config.EMBED_MODEL)
    return _embed_model


def get_reranker() -> Optional[CrossEncoder]:
    global _reranker
    if not config.USE_CROSS_RERANKER:
        return None
    if _reranker is None:
        print(f"[init] loading cross-encoder reranker {config.RERANKER_MODEL} on device {_device}")
        _reranker = CrossEncoder(config.RERANKER_MODEL, device=_device)
    return _reranker

