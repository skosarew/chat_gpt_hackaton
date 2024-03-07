GPT_MODEL = 'gpt-4-1106-preview'
EMBEDDING_MODEL = 'text-embedding-3-small'
PAGE_TOKEN_LIMIT = 8191
QUESTION_TOKEN_LIMIT = 4096
# 2048 is the maximum value, but for text-embedding-3-small there is another restriction on input tokens:
# tokens per min (TPM): Limit 1000000
BATCH_SIZE = 1000

CONTENTFUL_LIMIT = 1000
