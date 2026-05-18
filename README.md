# Agentic RAG

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![LangChain](https://img.shields.io/badge/langchain-%231C3C3C.svg?style=for-the-badge&logo=langchain&logoColor=white) ![LangGraph](https://img.shields.io/badge/langgraph-%231C3C3C.svg?style=for-the-badge&logo=langgraph&logoColor=white)

A production-ready multi-agent RAG pipeline built with **LangChain**, **LangGraph**, and **DeepAgents**. The system answers natural-language questions over a document corpus using a structured retry loop and an autonomous DeepAgent as a fallback for hard queries.

---

## Agent Roles

| Agent | File | Role |
|-------|------|------|
| **Planner** | `src/agents/planner.py` | Rewrites the user query into a retrieval-friendly form. Adds a retry hint on subsequent attempts. |
| **Retriever** | `src/agents/retriever.py` | Fetches semantically similar document chunks from Qdrant using the configured embedding model and MMR search. |
| **Grader** | `src/agents/grader.py` | Scores each retrieved chunk with a yes/no LLM judgment and discards irrelevant documents. |
| **Answerer** | `src/agents/answerer.py` | Synthesises a grounded answer from the filtered documents, refusing to answer from context it cannot find. |
| **Verifier** | `src/agents/verifier.py` | Checks whether the generated answer is fully supported by the source documents and signals retry or pass. |
| **DeepAgent** | `src/agents/deep_agent.py` | Autonomous escalation agent powered by DeepAgents. Plans its own multi-step retrieval strategy, calls `search_documents` as many times as needed, and produces a final answer from evidence. Triggered after the LangGraph retry loop has failed once. |

---

## LangGraph + DeepAgents Flow

The full flow is documented in [docs/workflow.md](docs/workflow.md).

```
START
  │
  ▼
Planner ──► Retriever ──► Grader
                             │
              ┌──────────────┼──────────────────┐
           (no docs,      (no docs,          (docs found)
           retry=0)       retry≥1)               │
              │               │                  ▼
              └──► Planner   DeepAgent ◄─── Answerer ──► Verifier
                                │               (not verified,   │
                                │                retry≥1)        │
                                ▼                                 │
                               END ◄──────────────── (verified / retries exhausted)
```

The **LangGraph** graph owns the state machine — routing, retry counters, and conditional edges. The **DeepAgent** operates as a self-directed node: it receives the original query, plans its own search strategy using its built-in todo tool, calls `search_documents` iteratively, and returns a synthesised answer directly to `END`.

---

## Getting Started

### Prerequisites

- Python ≥ 3.10
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Docker & Docker Compose (for the Qdrant vector store)
- API key for a supported LLM provider (Groq, Gemini, OpenRouter, etc.)

### 1 — Clone and install

```bash
git clone <repo-url>
cd AgenticRAG-VL

# with uv (recommended)
uv sync

# or with pip
pip install -e .
```

### 2 — Configure environment

Copy the example and fill in the required values:

```bash
cp .env.example .env   # if present, otherwise create .env manually
```

Minimum required variables:

```dotenv
LLM_API_KEY=<your-key>
# Optional overrides (defaults shown)
LLM_MODEL=groq/llama-3.1-8b-instant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=ViralLens
```

All settings are defined in `src/config.py` and can be overridden via environment variables.

### 3 — Start the vector store

```bash
docker compose up -d qdrant
```

### 4 — Ingest documents

Place your PDF files in `data/raw/`, then run:

```bash
cd src
python -c "from scripts.ingest import ingest; ingest()"
```

### 5 — Run a query

```bash
cd src
python main.py
```

### 6 — Run with Docker (full stack)

```bash
docker compose up --build zensical
```

This starts Qdrant, ingests documents, and launches the Chainlit chat interface on `http://localhost:8000`.

---

## Project Structure

```
src/
├── agents/          # Individual agent classes (Planner, Retriever, Grader,
│                    #   Answerer, Verifier, DeepAgent)
├── graph/           # LangGraph state, node functions, and workflow
├── llm/             # LiteLLM-backed chat model factory with rate-limit retry
├── rag/             # Embedding, chunking, vector store, and retriever helpers
└── scripts/         # Ingestion entrypoint
data/raw/            # Source documents (PDFs)
docs/                # Extended documentation
```

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | `groq/llama-3.1-8b-instant` | LiteLLM model string for all agents |
| `LLM_API_KEY` | — | Provider API key |
| `LLM_API_BASE` | — | Override base URL (e.g. for LM Studio) |
| `LLM_TEMPERATURE` | `0.7` | Sampling temperature |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | HuggingFace embedding model |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `QDRANT_COLLECTION` | `ViralLens` | Collection name |
| `SEARCH_TYPE` | `mmr` | Retrieval strategy (`mmr` or `similarity`) |
| `SEARCH_K` | `10` | Number of documents to retrieve |
| `MAX_RETRIES` | `2` | Maximum LangGraph retry loops before hard stop |
| `VERBOSE` | `true` | Print agent activity to console |
