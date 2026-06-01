import re
from typing import List

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "come", "does", "from", "in", "is", "it", "of", "on",
    "or", "question", "that", "the", "this", "to", "what", "when", "where", "who", "why", "with",
}


def tokenize(text: str) -> List[str]:
    return re.findall(r"[^\W_]+", text.lower(), flags=re.UNICODE)


def query_terms(query: str) -> List[str]:
    terms = []
    seen = set()
    for term in tokenize(query):
        if len(term) < 3 or term in _STOPWORDS or term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return terms


def best_excerpt(query: str, text: str, max_chars: int = 1000) -> str:
    if len(text) <= max_chars:
        return text

    text_lower = text.lower()
    terms = query_terms(query)
    if not terms:
        return text[:max_chars].strip()

    candidate_starts = {0}
    for term in terms:
        start = 0
        while True:
            pos = text_lower.find(term, start)
            if pos == -1:
                break
            candidate_starts.add(max(0, pos - max_chars // 3))
            start = pos + len(term)

    best_start = 0
    best_score = -1
    for start in candidate_starts:
        end = min(len(text), start + max_chars)
        window = text_lower[start:end]
        score = sum(1 for term in terms if term in window)
        if score > best_score:
            best_score = score
            best_start = start

    excerpt = text[best_start:best_start + max_chars].strip()
    if best_start > 0:
        excerpt = "..." + excerpt
    if best_start + max_chars < len(text):
        excerpt += "..."
    return excerpt
