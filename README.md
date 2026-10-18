# GraphRAG

A GraphRAG (knowledge-graph-backed retrieval-augmented generation) backend. Documents are
ingested into a Neo4j knowledge graph (entities + relationships, extracted by Gemini), and
questions are answered by combining vector search with graph traversal instead of plain
chunk-similarity search — so multi-hop questions ("how is X connected to Y?") can be answered
even when no single chunk mentions both facts.

## How it works

1. **Ingest** (`POST /documents`): a PDF/text file is chunked, each chunk is sent to Gemini to
   extract entities and relationships, everything is embedded (Gemini embeddings) and written
   into Neo4j as a graph: `Document -[:HAS_CHUNK]-> Chunk -[:MENTIONS]-> Entity -[:RELATES_TO]-> Entity`.
2. **Query** (`POST /query`): the question is embedded, matched against entity vectors to find
   seed entities, the graph is traversed 1-2 hops out from those entities to pull in connected
   facts and their source chunks, and Gemini synthesizes an answer grounded in that context.

## Setup

```bash
cp .env.example .env
# edit .env and set GEMINI_API_KEY

docker compose up -d        # starts Neo4j on bolt://localhost:7687 (browser: http://localhost:7474)

uv venv
uv pip install -e ".[dev]"
```

## Run the API

```bash
uv run uvicorn graphrag.api.app:app --app-dir src --reload
```

- `GET /health` — checks Neo4j connectivity
- `POST /documents` — multipart file upload (`file`), ingests a PDF/txt document
- `POST /query` — `{"question": "..."}` → `{"answer": ..., "sources": [...], "entities_used": [...]}`

Example:

```bash
curl -F file=@sample.pdf localhost:8000/documents
curl -X POST localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"question": "Where does Jane Smith work and how is that connected to Beta Inc?"}'
```

## CLI ingestion

```bash
uv run python scripts/ingest_cli.py path/to/document.pdf
```

## Tests

```bash
uv run pytest
```

Tests use mocked Gemini/Neo4j clients — no API key or running database required.
