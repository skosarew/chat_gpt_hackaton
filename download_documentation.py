import os

import contentful
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from constants import BATCH_SIZE, EMBEDDING_MODEL, MAX_TOKENS
from helpers import num_tokens

load_dotenv()
api_key = os.environ.get('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)


def get_pages():
    contentful_client = contentful.Client(
        os.environ.get('CONTENTFUL_SPACE_ID'),
        os.environ.get('CONTENTFUL_TOKEN'),
        timeout_s=10,
    )
    entries = contentful_client.entries({'content_type': 'article', 'limit': 1000})

    return [entry.raw['fields']['title'] + '\n\n' + entry.raw['fields']['body'] for entry in entries.items]


def split_page(page):
    num_tokens_in_string = num_tokens(page)
    if num_tokens_in_string <= MAX_TOKENS:
        return [page]
    else:
        return page[:len(page) // 2], page[len(page) // 2:]


pages = get_pages()
pages = [section for page in pages for section in split_page(page)]

embeddings = [
    e.embedding
    for batch_start in range(0, len(pages), BATCH_SIZE)
    for e in client.embeddings.create(model=EMBEDDING_MODEL, input=pages[batch_start:batch_start + BATCH_SIZE]).data
]

df = pd.DataFrame({"text": pages, "embedding": embeddings})

SAVE_PATH = "./data.csv"
df.to_csv(SAVE_PATH, index=False)
