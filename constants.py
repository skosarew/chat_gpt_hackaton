GPT_MODEL = 'gpt-4-1106-preview'
EMBEDDING_MODEL = 'text-embedding-3-small'
MAX_TOKENS = 8191
# openai.RateLimitError: Error code: 429 - {‘error’: {‘message’: ‘Request too large for text-embedding-3-small
# in organization org-xxx on tokens per min (TPM): Limit 1000000, Requested 1026047.
# The input or output tokens must be reduced in order to run successfully.
# Visit https://platform.openai.com/account/rate-limits to learn more.’}}
BATCH_SIZE = 1000  # 2048 is a maximum value, but to handle RateLimitError this value must be reduced

CONTENTFUL_LIMIT = 1000
