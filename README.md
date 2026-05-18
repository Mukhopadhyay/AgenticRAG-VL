# Agentic RAG

## Getting Started

```bash
uv sync
```

## Agents

### 1. Planner Agent

Responsibility
* Understand query
* Rewrite / improve retrieval query
* Decide retrieval strategy

### 2. Retriever Agent

Uses the retrieval class

Returns
* top-k-docs
* metadata

### 3. Relevance Grader Agent

For filtering out irrelevant chunks

### 4. Answerer Agent

Generates the final answer from filtered docs