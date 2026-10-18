from google import genai
from tenacity import retry, stop_after_attempt, wait_exponential

from graphrag.models import ExtractionResult

_PROMPT_TEMPLATE = """\
Extract entities and relationships from the following text for a knowledge graph.

Rules:
- Entities: real-world things worth linking across documents (people, organizations, \
locations, products, concepts). Use a short canonical "type" (e.g. Person, \
Organization, Location, Concept).
- Relationships: factual connections stated or clearly implied in the text, using the \
exact entity names you extracted as source/target, with a short UPPER_SNAKE_CASE type \
(e.g. WORKS_AT, ACQUIRED, FOUNDER_OF, LOCATED_IN).
- Only extract what the text actually supports. Do not invent facts.

Text:
\"\"\"
{text}
\"\"\"
"""


class EntityExtractor:
    def __init__(self, client: genai.Client, model: str) -> None:
        self._client = client
        self._model = model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def extract(self, text: str) -> ExtractionResult:
        response = self._client.models.generate_content(
            model=self._model,
            contents=_PROMPT_TEMPLATE.format(text=text),
            config={
                "response_mime_type": "application/json",
                "response_schema": ExtractionResult,
            },
        )
        return ExtractionResult.model_validate_json(response.text)
