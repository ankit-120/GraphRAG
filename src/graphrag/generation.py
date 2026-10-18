from google import genai
from tenacity import retry, stop_after_attempt, wait_exponential

from graphrag.retrieval.local_search import LocalSearchContext

_PROMPT_TEMPLATE = """\
Answer the question using only the knowledge graph context below. Cite source \
excerpts by their [id] when you use them. If the context doesn't contain the \
answer, say so.

Context:
{context}

Question: {question}

Answer:
"""


class AnswerGenerator:
    def __init__(self, client: genai.Client, model: str) -> None:
        self._client = client
        self._model = model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def generate(self, question: str, context: LocalSearchContext) -> str:
        if not context.entities:
            return "I don't have enough information in the knowledge graph to answer that."
        prompt = _PROMPT_TEMPLATE.format(
            context=context.to_prompt_context(), question=question
        )
        response = self._client.models.generate_content(model=self._model, contents=prompt)
        return response.text
