from contextlib import asynccontextmanager

from fastapi import FastAPI
from google import genai

from graphrag.config import get_settings
from graphrag.embeddings import EmbeddingClient
from graphrag.generation import AnswerGenerator
from graphrag.graph.client import create_driver
from graphrag.graph.repository import GraphRepository
from graphrag.graph.schema import init_schema
from graphrag.ingestion.extraction import EntityExtractor
from graphrag.ingestion.image_description import ImageDescriber
from graphrag.pipeline import IngestionPipeline
from graphrag.retrieval.local_search import LocalSearch


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    driver = create_driver(settings)
    init_schema(driver, settings.embedding_dimensions)

    genai_client = genai.Client(api_key=settings.gemini_api_key)
    repository = GraphRepository(driver)
    embedding_client = EmbeddingClient(genai_client, settings.gemini_embedding_model)
    extractor = EntityExtractor(genai_client, settings.gemini_llm_model)
    generator = AnswerGenerator(genai_client, settings.gemini_llm_model)
    local_search = LocalSearch(
        repository,
        embedding_client,
        settings.local_search_top_k,
        settings.local_search_hops,
    )
    image_describer = (
        ImageDescriber(genai_client, settings.gemini_llm_model)
        if settings.enable_image_descriptions
        else None
    )
    pipeline = IngestionPipeline(
        repository,
        extractor,
        embedding_client,
        settings.chunk_size,
        settings.chunk_overlap,
        image_describer,
        settings.min_image_dimension,
    )

    app.state.driver = driver
    app.state.pipeline = pipeline
    app.state.local_search = local_search
    app.state.generator = generator

    yield

    driver.close()


def create_app() -> FastAPI:
    app = FastAPI(title="GraphRAG API", lifespan=lifespan)

    from graphrag.api.routes import router

    app.include_router(router)
    return app


app = create_app()
