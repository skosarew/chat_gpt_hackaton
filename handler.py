import ast

import pandas as pd
from scipy import spatial

from constants import EMBEDDING_MODEL
from helpers import num_tokens

df = pd.read_csv('./data.csv')
df['embedding'] = df['embedding'].apply(ast.literal_eval)


class OpenAPIHandler:
    def __init__(self, client, question):
        self.client = client
        self.question = question

    def make_question(self, token_budget: int = 4096 - 500) -> str:
        strings, _ = self.strings_ranked_by_relatedness()
        question = f'\n\nQuestion: {self.question}'
        message = ''
        for string in strings:
            if num_tokens(message + string + question) > token_budget:
                break
            else:
                message += string

        return message + question

    def strings_ranked_by_relatedness(
        self,
        relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
        top_n: int = 100,
    ) -> tuple[list[str], list[float]]:
        query_embedding_response = self.client.embeddings.create(model=EMBEDDING_MODEL, input=self.question)
        query_embedding = query_embedding_response.data[0].embedding

        strings_and_relatednesses = [
            (row['text'], relatedness_fn(query_embedding, row['embedding']))
            for i, row in df.iterrows()
        ]
        strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
        strings, relatednesses = zip(*strings_and_relatednesses)

        return strings[:top_n], relatednesses[:top_n]
