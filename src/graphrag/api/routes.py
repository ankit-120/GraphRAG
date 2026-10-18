from fastapi import APIRouter, Request, UploadFile

from graphrag.models import IngestResponse, QueryRequest, QueryResponse

router = APIRouter()


@router.get("/health")
def health(request: Request) -> dict:
    with request.app.state.driver.session() as session:
        session.run("RETURN 1")
    return {"status": "ok"}


@router.post("/documents", response_model=IngestResponse)
async def ingest_document(request: Request, file: UploadFile) -> IngestResponse:
    content = await file.read()
    pipeline = request.app.state.pipeline
    return pipeline.ingest(file.filename, content)


@router.post("/query", response_model=QueryResponse)
def query(request: Request, body: QueryRequest) -> QueryResponse:
    local_search = request.app.state.local_search
    generator = request.app.state.generator

    context = local_search.search(body.question, body.top_k)
    answer = generator.generate(body.question, context)

    return QueryResponse(
        answer=answer,
        sources=context.chunks,
        entities_used=context.entity_names(),
    )
