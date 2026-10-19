from io import BytesIO
from typing import TYPE_CHECKING

from PIL import Image
from pypdf import PdfReader

from graphrag.ingestion.image_description import NO_CONTENT_MARKER

if TYPE_CHECKING:
    from graphrag.ingestion.image_description import ImageDescriber

_DEFAULT_MIME_TYPE = "image/png"


def load_text(
    filename: str,
    content: bytes,
    image_describer: "ImageDescriber | None" = None,
    min_image_dimension: int = 64,
) -> str:
    if filename.lower().endswith(".pdf"):
        return _load_pdf(content, image_describer, min_image_dimension)
    return content.decode("utf-8", errors="replace")


def _load_pdf(
    content: bytes,
    image_describer: "ImageDescriber | None",
    min_image_dimension: int,
) -> str:
    reader = PdfReader(BytesIO(content))
    pages = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if image_describer is not None:
            image_descriptions = _describe_page_images(
                page, image_describer, min_image_dimension
            )
            if image_descriptions:
                page_text = f"{page_text}\n\n" + "\n\n".join(image_descriptions)
        pages.append(page_text)
    return "\n\n".join(pages)


def _describe_page_images(page, image_describer, min_image_dimension: int) -> list[str]:
    descriptions = []
    for image_file in page.images:
        pil_image = image_file.image
        if pil_image is None:
            continue
        width, height = pil_image.size
        if width < min_image_dimension or height < min_image_dimension:
            continue

        mime_type = Image.MIME.get(pil_image.format, _DEFAULT_MIME_TYPE)
        description = image_describer.describe(image_file.data, mime_type)
        if description and description.strip().upper() != NO_CONTENT_MARKER:
            descriptions.append(f"[Image: {description.strip()}]")
    return descriptions
