# Compromises
This document contains compromises I'm making along the way due to time constraints. Essentially what I'd do differently.

* The `is_collection_exist` method is too rudimentary - can definitely be improved. This is a simple function to save time with re-embedding and ingestion.
* Grader could be async.
* A reranking agent would be better alongside a conditional loop.
* Hybrid search with bm25 for faster better retrieval.
* Missing observability as a service (in Docker-compose)
* Handling retries with LiteLLM - tenacity would give us better control