#!/usr/bin/env python3
import argparse
import pathlib
import sys

from google import genai

from graphrag.config import get_settings
from graphrag.embeddings import EmbeddingClient
from graphrag.graph.client import create_driver
from graphrag.graph.repository import GraphRepository
from graphrag.graph.schema import init_schema
from graphrag.ingestion.extraction import EntityExtractor
from graphrag.pipeline import IngestionPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a document into the GraphRAG knowledge graph")
    parser.add_argument("path", type=pathlib.Path, help="Path to a .txt or .pdf file")
    args = parser.parse_args()

    settings = get_settings()
    driver = create_driver(settings)
    init_schema(driver, settings.embedding_dimensions)

    genai_client = genai.Client(api_key=settings.gemini_api_key)
    repository = GraphRepository(driver)
    embedding_client = EmbeddingClient(genai_client, settings.gemini_embedding_model)
    extractor = EntityExtractor(genai_client, settings.gemini_llm_model)
    pipeline = IngestionPipeline(
        repository, extractor, embedding_client, settings.chunk_size, settings.chunk_overlap
    )

    content = args.path.read_bytes()
    result = pipeline.ingest(args.path.name, content)
    print(result.model_dump_json(indent=2))

    driver.close()


if __name__ == "__main__":
    sys.exit(main())
