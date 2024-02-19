import os

import contentful
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from constants import EMBEDDING_MODEL
from helpers import num_tokens

BATCH_SIZE = 1000  # you can submit up to 2048 embedding inputs per request

load_dotenv()
api_key = os.environ.get('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

contentful_client = contentful.Client(
    os.environ.get('CONTENTFUL_SPACE_ID'),
    os.environ.get('CONTENTFUL_TOKEN'),
)
entries = contentful_client.entries({'content_type': 'article', 'limit': 1000})
entries_items = entries.items

content = [entry.raw['fields']['title'] + '\n\n' + entry.raw['fields']['body'] for entry in entries.items]


def split_strings_from_subsection(string, max_tokens):
    num_tokens_in_string = num_tokens(string)
    if num_tokens_in_string <= max_tokens:
        return [string]
    else:
        return string[:len(string) // 2], string[len(string) // 2:]


pages = []
for section in content:
    pages.extend(split_strings_from_subsection(section, max_tokens=1600))

embeddings = []
for batch_start in range(0, len(pages), BATCH_SIZE):
    batch_end = batch_start + BATCH_SIZE
    batch = pages[batch_start:batch_end]
    print(f"Batch {batch_start} to {batch_end - 1}")

    response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)

    for i, be in enumerate(response.data):
        assert i == be.index  # double check embeddings are in same order as input

    batch_embeddings = [e.embedding for e in response.data]
    embeddings.extend(batch_embeddings)

df = pd.DataFrame({"text": pages, "embedding": embeddings})

SAVE_PATH = "./data.csv"
df.to_csv(SAVE_PATH, index=False)
