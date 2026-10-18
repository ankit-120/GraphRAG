from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from graphrag.api.routes import router
from graphrag.models import IngestResponse, SourceChunk
from graphrag.retrieval.local_search import LocalSearchContext


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)

    app.state.driver = MagicMock()
    app.state.pipeline = MagicMock()
    app.state.local_search = MagicMock()
    app.state.generator = MagicMock()

    return TestClient(app), app


def test_health_runs_a_query_against_the_driver(client):
    test_client, app = client
    session = MagicMock()
    app.state.driver.session.return_value.__enter__.return_value = session

    response = test_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    session.run.assert_called_once_with("RETURN 1")


def test_ingest_document_calls_pipeline_and_returns_counts(client):
    test_client, app = client
    app.state.pipeline.ingest.return_value = IngestResponse(
        document_id="doc-1", chunk_count=1, entity_count=2, relationship_count=1
    )

    response = test_client.post(
        "/documents", files={"file": ("sample.txt", b"Acme Corp acquired Beta Inc.")}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == "doc-1"
    assert body["chunk_count"] == 1
    assert body["entity_count"] == 2


def test_query_returns_answer_sources_and_entities(client):
    test_client, app = client
    context = LocalSearchContext(
        entities=[{"name": "jane smith", "type": "Person", "description": "desc"}],
        relationships=[],
        chunks=[SourceChunk(chunk_id="chunk-1", document_id="doc-1", text="Jane works at Acme.")],
    )
    app.state.local_search.search.return_value = context
    app.state.generator.generate.return_value = "Jane Smith works at Acme Corp."

    response = test_client.post("/query", json={"question": "Where does Jane work?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Jane Smith works at Acme Corp."
    assert body["entities_used"] == ["jane smith"]
    assert body["sources"][0]["chunk_id"] == "chunk-1"
