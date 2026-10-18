from graphrag.embeddings import EmbeddingClient
from graphrag.graph.repository import GraphRepository
from graphrag.models import SourceChunk


class LocalSearchContext:
    def __init__(self, entities: list[dict], relationships: list[dict], chunks: list[SourceChunk]) -> None:
        self.entities = entities
        self.relationships = relationships
        self.chunks = chunks

    def entity_names(self) -> list[str]:
        return [e["name"] for e in self.entities]

    def to_prompt_context(self) -> str:
        entity_lines = "\n".join(
            f"- {e['name']} ({e['type']}): {e['description']}" for e in self.entities
        )
        relationship_lines = "\n".join(
            f"- {r['source']} --{r['type']}--> {r['target']}: {r['description']}"
            for r in self.relationships
        )
        chunk_lines = "\n\n".join(f"[{c.chunk_id}] {c.text}" for c in self.chunks)
        return (
            f"Entities:\n{entity_lines}\n\n"
            f"Relationships:\n{relationship_lines}\n\n"
            f"Source excerpts:\n{chunk_lines}"
        )


class LocalSearch:
    def __init__(
        self,
        repository: GraphRepository,
        embedding_client: EmbeddingClient,
        top_k: int,
        hops: int,
    ) -> None:
        self._repository = repository
        self._embedding_client = embedding_client
        self._top_k = top_k
        self._hops = hops

    def search(self, question: str, top_k: int | None = None) -> LocalSearchContext:
        query_embedding = self._embedding_client.embed_query(question)
        seed_entities = self._repository.vector_search_entities(
            query_embedding, top_k or self._top_k
        )
        seed_names = [e["name"] for e in seed_entities]
        if not seed_names:
            return LocalSearchContext(entities=[], relationships=[], chunks=[])

        neighborhood = self._repository.expand_neighborhood(seed_names, self._hops)
        chunks = [
            SourceChunk(chunk_id=c["id"], document_id=c["document_id"], text=c["text"])
            for c in neighborhood["chunks"]
        ]
        return LocalSearchContext(
            entities=neighborhood["entities"],
            relationships=neighborhood["relationships"],
            chunks=chunks,
        )
