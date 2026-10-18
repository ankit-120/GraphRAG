from pydantic import BaseModel, Field


class Chunk(BaseModel):
    id: str
    document_id: str
    text: str
    embedding: list[float] | None = None


class ExtractedEntity(BaseModel):
    name: str
    type: str
    description: str = ""


class ExtractedRelationship(BaseModel):
    source: str
    target: str
    type: str
    description: str = ""


class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)


class IngestResponse(BaseModel):
    document_id: str
    chunk_count: int
    entity_count: int
    relationship_count: int


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None


class SourceChunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    entities_used: list[str]
