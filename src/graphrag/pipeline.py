import uuid

from graphrag.embeddings import EmbeddingClient
from graphrag.graph.repository import GraphRepository
from graphrag.ingestion.chunking import chunk_text
from graphrag.ingestion.extraction import EntityExtractor
from graphrag.ingestion.image_description import ImageDescriber
from graphrag.ingestion.loaders import load_text
from graphrag.models import IngestResponse


class IngestionPipeline:
    def __init__(
        self,
        repository: GraphRepository,
        extractor: EntityExtractor,
        embedding_client: EmbeddingClient,
        chunk_size: int,
        chunk_overlap: int,
        image_describer: ImageDescriber | None = None,
        min_image_dimension: int = 64,
    ) -> None:
        self._repository = repository
        self._extractor = extractor
        self._embedding_client = embedding_client
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._image_describer = image_describer
        self._min_image_dimension = min_image_dimension

    def ingest(self, filename: str, content: bytes) -> IngestResponse:
        document_id = str(uuid.uuid4())
        text = load_text(
            filename, content, self._image_describer, self._min_image_dimension
        )
        chunks = chunk_text(text, document_id, self._chunk_size, self._chunk_overlap)

        self._repository.create_document(document_id, filename)

        entity_count = 0
        relationship_count = 0

        for chunk in chunks:
            chunk.embedding = self._embedding_client.embed([chunk.text])[0]
            self._repository.write_chunk(document_id, chunk)

            extraction = self._extractor.extract(chunk.text)
            if extraction.entities:
                entity_embeddings = self._embedding_client.embed(
                    [f"{e.name}: {e.description}" for e in extraction.entities]
                )
                for entity, embedding in zip(extraction.entities, entity_embeddings):
                    self._repository.write_entity(chunk.id, entity, embedding)
                entity_count += len(extraction.entities)

            for relationship in extraction.relationships:
                self._repository.write_relationship(relationship)
            relationship_count += len(extraction.relationships)

        return IngestResponse(
            document_id=document_id,
            chunk_count=len(chunks),
            entity_count=entity_count,
            relationship_count=relationship_count,
        )
