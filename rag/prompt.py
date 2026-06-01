from typing import Dict, List

SYSTEM = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    "You answer questions using only the provided context.\n"
    "Return only the final answer, without repeating the question or context.\n"
    "If the answer is present, give a complete answer using all relevant details from the context.\n"
    "When the question asks what someone said, include the full relevant utterance.\n"
    "If the context does not contain the answer, reply exactly: Couldn't find the answer in the database.\n"
    "<|eot_id|>"
)
SYSTEM1 = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    "Ты отвечаешь на вопросы, используя только полученный контекст.\n"
    "Ответ должен быть полным со всеми необходимыми деталями, представленными в контексте.\n"
    "<|eot_id|>"
)
ASSISTANT = "<|start_header_id|>assistant<|end_header_id|>"


def build_prompt(query: str, contexts: List[Dict]) -> str:
    ctx_text = "\n\n---\n\n".join(f"[{c['id']}]\n{c['text']}" for c in contexts)
    # ctx_text = "\n\n---\n\n".join(f"[{c['id']}]\n{c.get('excerpt') or c['text']}" for c in contexts)
    user = (
        "<|start_header_id|>user<|end_header_id|>\n"
        "Context:\n"
        f"{ctx_text}\n\n"
        f"Question: {query}\n"
        "Answer:\n"
        "<|eot_id|>"
    )
    prompt = SYSTEM + "\n" + user + "\n" + ASSISTANT + "\n"
    return prompt
