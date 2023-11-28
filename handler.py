import os

import pandas as pd
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from scipy import spatial

load_dotenv()
api_key = os.environ.get('OPENAI_API_KEY')

# models
EMBEDDING_MODEL = 'text-embedding-ada-002'
GPT_MODEL = 'gpt-3.5-turbo'


class OpenAPIHandler:
    def __init__(self, question):
        self.question = question

    def make_question(
        self,
        df: pd.DataFrame,
        token_budget: int = 4096 - 500,
        model: str = 'gpt-4',
    ) -> str:
        strings, _ = self.strings_ranked_by_relatedness(self.question, df)
        question = f'\n\nQuestion: {self.question}'
        message = ''
        for string in strings:
            if (
                self.num_tokens(message + string + question, model=model)
                > token_budget
            ):
                break
            else:
                message += string

        return message + question

    def strings_ranked_by_relatedness(
        self,
        query: str,
        df: pd.DataFrame,
        relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
        top_n: int = 100,
    ) -> tuple[list[str], list[float]]:
        client = OpenAI(api_key=api_key)

        query_embedding_response = client.embeddings.create(model=EMBEDDING_MODEL, input=query)
        query_embedding = query_embedding_response.data[0].embedding
        strings_and_relatednesses = [
            (row['text'], relatedness_fn(query_embedding, row['embedding']))
            for i, row in df.iterrows()
        ]
        strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
        strings, relatednesses = zip(*strings_and_relatednesses)

        return strings[:top_n], relatednesses[:top_n]

    def num_tokens(self, text: str, model: str = GPT_MODEL) -> int:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
