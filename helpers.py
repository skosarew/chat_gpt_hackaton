import tiktoken

from constants import GPT_MODEL


def num_tokens(text: str) -> int:
    encoding = tiktoken.encoding_for_model(GPT_MODEL)

    return len(encoding.encode(text))
