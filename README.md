Adjust Hackaton 2023
====================
### Team lead simulator
An advanced knowledge base for client-facing team to look up information per product by our knowledge base, compiled from docs.

### Description
Question answering is done using embeddings-based search.

__Model__: gpt-4 8,192 tokens (~10 pages)

Preparation of Search Data (Performed once per document)

1. Collect: Downloading several hundred of Adjust documentation.
   1. Chunk: Dividing documents into concise, mostly self-contained sections suitable for embedding.
   2. Embed: Utilizing the OpenAI API to embed each section.
   3. Store: Saving the generated embeddings (for extensive datasets, consider using a vector database).
2. Searching (Performed once per query)
   1. Given a user's question, generate an embedding for the query using the OpenAI API.
   2. Utilizing the embeddings, rank the text sections based on their relevance to the query.
3. Asking (Performed once per query)
   1. Insert the question along with the most relevant sections into a message addressed to GPT.
   2. Retrieve GPT's response.
