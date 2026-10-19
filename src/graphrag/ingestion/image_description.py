from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

NO_CONTENT_MARKER = "NONE"

_PROMPT = f"""\
Describe this image factually, for inclusion as text in a document.

- If it is a chart or graph: state its type, the axis labels, and the exact value for \
each category/data point you can read.
- If it is a diagram or figure with informational content: describe what it shows.
- If it is purely decorative (a logo, icon, divider, bullet, background pattern) with \
no informational content, respond with exactly: {NO_CONTENT_MARKER}
"""


class ImageDescriber:
    def __init__(self, client: genai.Client, model: str) -> None:
        self._client = client
        self._model = model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def describe(self, image_bytes: bytes, mime_type: str) -> str:
        response = self._client.models.generate_content(
            model=self._model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                _PROMPT,
            ],
        )
        return response.text.strip()
