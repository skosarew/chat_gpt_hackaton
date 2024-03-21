import os

import contentful
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from constants import (BATCH_SIZE, CONTENTFUL_LIMIT, EMBEDDING_MODEL,
                       PAGE_TOKEN_LIMIT)
from helpers import num_tokens

load_dotenv()
api_key = os.environ.get('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

contentful_client = contentful.Client(
    os.environ.get('CONTENTFUL_SPACE_ID'),
    os.environ.get('CONTENTFUL_TOKEN'),
    timeout_s=10,
)


def get_entries(content_type):
    all_entries = []
    skip = 0

    entries = contentful_client.entries(
        {
            'content_type': content_type,
            'skip': skip,
            'limit': CONTENTFUL_LIMIT,
        }
    )
    all_entries.extend(entries.items)

    if entries.total > entries.limit:
        while entries.total > entries.limit:
            skip += CONTENTFUL_LIMIT
            entries = contentful_client.entries(
                {
                    'content_type': content_type,
                    'skip': skip,
                    'limit': CONTENTFUL_LIMIT
                }
            )
            all_entries.extend(entries.items)

    return all_entries


def get_pages():
    article_entries = get_entries('article')
    fragment_entries = get_entries('fragment')
    combined_entries = [entry for entry in article_entries + fragment_entries if 'body' in entry.raw['fields']]

    return [
        entry.raw['fields']['title'] + '\n\n' + entry.raw['fields']['body']
        for entry in combined_entries
    ]


def split_page(page):
    num_tokens_in_page = num_tokens(page)
    if num_tokens_in_page <= PAGE_TOKEN_LIMIT:
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
