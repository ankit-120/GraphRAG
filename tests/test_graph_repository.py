from unittest.mock import MagicMock

from graphrag.graph.repository import GraphRepository
from graphrag.models import ExtractedEntity, ExtractedRelationship


def make_driver_with_session():
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__.return_value = session
    driver.session.return_value.__exit__.return_value = False
    return driver, session


def test_write_entity_normalizes_name_and_sets_params():
    driver, session = make_driver_with_session()
    repo = GraphRepository(driver)

    entity = ExtractedEntity(name="  Jane Smith  ", type="Person", description="An engineer")
    repo.write_entity(chunk_id="chunk-1", entity=entity, embedding=[0.1, 0.2])

    session.run.assert_called_once()
    _, kwargs = session.run.call_args
    assert kwargs["name"] == "jane smith"
    assert kwargs["type"] == "Person"
    assert kwargs["chunk_id"] == "chunk-1"
    assert kwargs["embedding"] == [0.1, 0.2]


def test_write_relationship_normalizes_source_and_target():
    driver, session = make_driver_with_session()
    repo = GraphRepository(driver)

    relationship = ExtractedRelationship(
        source="Jane Smith", target="Acme Corp", type="WORKS_AT", description="desc"
    )
    repo.write_relationship(relationship)

    _, kwargs = session.run.call_args
    assert kwargs["source"] == "jane smith"
    assert kwargs["target"] == "acme corp"
    assert kwargs["type"] == "WORKS_AT"


def test_vector_search_entities_returns_records_as_dicts():
    driver, session = make_driver_with_session()
    session.run.return_value = [
        {"name": "acme corp", "type": "Organization", "description": "desc", "score": 0.9}
    ]
    repo = GraphRepository(driver)

    results = repo.vector_search_entities([0.1, 0.2], top_k=5)

    assert results == [
        {"name": "acme corp", "type": "Organization", "description": "desc", "score": 0.9}
    ]


def test_expand_neighborhood_runs_three_queries_and_assembles_result():
    driver, session = make_driver_with_session()
    session.run.side_effect = [
        [{"name": "acme corp", "type": "Organization", "description": "desc"}],
        [{"source": "jane smith", "target": "acme corp", "type": "WORKS_AT", "description": "d"}],
        [{"id": "chunk-1", "document_id": "doc-1", "text": "some text"}],
    ]
    repo = GraphRepository(driver)

    result = repo.expand_neighborhood(["jane smith"], hops=2)

    assert result["entities"][0]["name"] == "acme corp"
    assert result["relationships"][0]["type"] == "WORKS_AT"
    assert result["chunks"][0]["id"] == "chunk-1"
    assert session.run.call_count == 3
