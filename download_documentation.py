import os

import contentful
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from constants import EMBEDDING_MODEL, MAX_TOKENS
from helpers import num_tokens

BATCH_SIZE = 1000  # you can submit up to 2048 embedding inputs per request

load_dotenv()
api_key = os.environ.get('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)


def get_content():
    contentful_client = contentful.Client(
        os.environ.get('CONTENTFUL_SPACE_ID'),
        os.environ.get('CONTENTFUL_TOKEN'),
    )
    entries = contentful_client.entries({'content_type': 'article', 'limit': 1000})

    return [entry.raw['fields']['title'] + '\n\n' + entry.raw['fields']['body'] for entry in entries.items]


def split_strings_from_subsection(string):
    num_tokens_in_string = num_tokens(string)
    if num_tokens_in_string <= MAX_TOKENS:
        return [string]
    else:
        return string[:len(string) // 2], string[len(string) // 2:]


content = get_content()
pages = [y for section in content for y in split_strings_from_subsection(section)]

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
