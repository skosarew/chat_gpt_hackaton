import os

import contentful
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from constants import EMBEDDING_MODEL, MAX_TOKENS, BATCH_SIZE
from helpers import num_tokens

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

embeddings = [
    e.embedding
    for batch_start in range(0, len(pages), BATCH_SIZE)
    for e in client.embeddings.create(model=EMBEDDING_MODEL, input=pages[batch_start:batch_start + BATCH_SIZE]).data
]

df = pd.DataFrame({"text": pages, "embedding": embeddings})

SAVE_PATH = "./data.csv"
df.to_csv(SAVE_PATH, index=False)
