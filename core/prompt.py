SYSTEM = (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
          f"You are a helpful assistant. Your answers are short "
          f"<|eot_id|>")
ASSISTANT = '<|start_header_id|>assistant<|end_header_id|>'


def build_prompt(query: str) -> str:
    user = (f"<|start_header_id|>user<|end_header_id|>"
            f"{query}<|eot_id|>")

    prompt = SYSTEM + "\n" + user + "\n" + ASSISTANT + "\n"
    return prompt
