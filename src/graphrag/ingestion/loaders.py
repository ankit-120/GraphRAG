from io import BytesIO

from pypdf import PdfReader


def load_text(filename: str, content: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        return _load_pdf(content)
    return content.decode("utf-8", errors="replace")


def _load_pdf(content: bytes) -> str:
    reader = PdfReader(BytesIO(content))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)
