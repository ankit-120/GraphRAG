from neo4j import Driver

from graphrag.models import Chunk, ExtractedEntity, ExtractedRelationship


def _normalize(name: str) -> str:
    return name.strip().lower()


class GraphRepository:
    def __init__(self, driver: Driver) -> None:
        self._driver = driver

    def create_document(self, document_id: str, source: str) -> None:
        with self._driver.session() as session:
            session.run(
                "MERGE (d:Document {id: $id}) SET d.source = $source",
                id=document_id,
                source=source,
            )

    def write_chunk(self, document_id: str, chunk: Chunk) -> None:
        with self._driver.session() as session:
            session.run(
                """
                MATCH (d:Document {id: $document_id})
                MERGE (c:Chunk {id: $id})
                SET c.text = $text, c.document_id = $document_id, c.embedding = $embedding
                MERGE (d)-[:HAS_CHUNK]->(c)
                """,
                document_id=document_id,
                id=chunk.id,
                text=chunk.text,
                embedding=chunk.embedding,
            )

    def write_entity(
        self, chunk_id: str, entity: ExtractedEntity, embedding: list[float]
    ) -> None:
        with self._driver.session() as session:
            session.run(
                """
                MERGE (e:Entity {name: $name})
                ON CREATE SET e.type = $type, e.description = $description, e.embedding = $embedding
                ON MATCH SET e.description = CASE
                    WHEN size(e.description) < size($description) THEN $description
                    ELSE e.description
                END
                WITH e
                MATCH (c:Chunk {id: $chunk_id})
                MERGE (c)-[:MENTIONS]->(e)
                """,
                name=_normalize(entity.name),
                type=entity.type,
                description=entity.description,
                embedding=embedding,
                chunk_id=chunk_id,
            )

    def write_relationship(self, relationship: ExtractedRelationship) -> None:
        with self._driver.session() as session:
            session.run(
                """
                MERGE (source:Entity {name: $source})
                MERGE (target:Entity {name: $target})
                MERGE (source)-[r:RELATES_TO {type: $type}]->(target)
                SET r.description = $description
                """,
                source=_normalize(relationship.source),
                target=_normalize(relationship.target),
                type=relationship.type,
                description=relationship.description,
            )

    def vector_search_entities(
        self, embedding: list[float], top_k: int
    ) -> list[dict]:
        with self._driver.session() as session:
            result = session.run(
                """
                CALL db.index.vector.queryNodes('entity_embedding', $top_k, $embedding)
                YIELD node, score
                RETURN node.name AS name, node.type AS type, node.description AS description, score
                """,
                embedding=embedding,
                top_k=top_k,
            )
            return [dict(record) for record in result]

    def expand_neighborhood(self, entity_names: list[str], hops: int) -> dict:
        with self._driver.session() as session:
            entity_names_result = session.run(
                f"""
                MATCH (start:Entity) WHERE start.name IN $names
                MATCH (start)-[:RELATES_TO*0..{hops}]-(related:Entity)
                RETURN DISTINCT related.name AS name, related.type AS type, related.description AS description
                """,
                names=entity_names,
            )
            entities = [dict(record) for record in entity_names_result]
            names = [e["name"] for e in entities]

            relationships_result = session.run(
                """
                MATCH (source:Entity)-[r:RELATES_TO]->(target:Entity)
                WHERE source.name IN $names AND target.name IN $names
                RETURN DISTINCT source.name AS source, target.name AS target, r.type AS type, r.description AS description
                """,
                names=names,
            )
            relationships = [dict(record) for record in relationships_result]

            chunks_result = session.run(
                """
                MATCH (e:Entity)<-[:MENTIONS]-(c:Chunk)
                WHERE e.name IN $names
                RETURN DISTINCT c.id AS id, c.document_id AS document_id, c.text AS text
                """,
                names=names,
            )
            chunks = [dict(record) for record in chunks_result]

            return {"entities": entities, "relationships": relationships, "chunks": chunks}
