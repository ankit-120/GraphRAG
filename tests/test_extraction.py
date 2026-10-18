import json
from unittest.mock import MagicMock

from graphrag.ingestion.extraction import EntityExtractor


def _fake_response(entities, relationships):
    response = MagicMock()
    response.text = json.dumps({"entities": entities, "relationships": relationships})
    return response


def test_extract_parses_entities_and_relationships(sample_text):
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = _fake_response(
        entities=[
            {"name": "Acme Corp", "type": "Organization", "description": "A company"},
            {"name": "Jane Smith", "type": "Person", "description": "An engineer"},
        ],
        relationships=[
            {
                "source": "Jane Smith",
                "target": "Acme Corp",
                "type": "WORKS_AT",
                "description": "Jane works at Acme",
            }
        ],
    )

    extractor = EntityExtractor(fake_client, model="fake-model")
    result = extractor.extract(sample_text)

    assert len(result.entities) == 2
    assert result.entities[0].name == "Acme Corp"
    assert len(result.relationships) == 1
    assert result.relationships[0].type == "WORKS_AT"

    fake_client.models.generate_content.assert_called_once()
    _, kwargs = fake_client.models.generate_content.call_args
    assert kwargs["model"] == "fake-model"
    assert sample_text in kwargs["contents"]


def test_extract_handles_empty_result():
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = _fake_response(
        entities=[], relationships=[]
    )

    extractor = EntityExtractor(fake_client, model="fake-model")
    result = extractor.extract("Nothing interesting here.")

    assert result.entities == []
    assert result.relationships == []
