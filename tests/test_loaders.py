from dataclasses import dataclass
from unittest.mock import MagicMock

from PIL import Image

from graphrag.ingestion.loaders import load_text


@dataclass
class FakeImageFile:
    image: Image.Image
    data: bytes


class FakePage:
    def __init__(self, text: str, images: list[FakeImageFile]) -> None:
        self._text = text
        self.images = images

    def extract_text(self) -> str:
        return self._text


class FakeReader:
    def __init__(self, pages: list[FakePage]) -> None:
        self.pages = pages


def _patch_reader(monkeypatch, pages: list[FakePage]) -> None:
    monkeypatch.setattr(
        "graphrag.ingestion.loaders.PdfReader", lambda _content: FakeReader(pages)
    )


def _large_image() -> Image.Image:
    return Image.new("RGB", (200, 200))


def _tiny_image() -> Image.Image:
    return Image.new("RGB", (10, 10))


def test_load_text_plain_text_file_ignores_image_describer():
    result = load_text("notes.txt", b"Acme Corp acquired Beta Inc.", image_describer=MagicMock())
    assert result == "Acme Corp acquired Beta Inc."


def test_load_pdf_without_image_describer_returns_plain_text(monkeypatch):
    page = FakePage("Acme Corp acquired Beta Inc.", images=[FakeImageFile(_large_image(), b"img")])
    _patch_reader(monkeypatch, [page])

    result = load_text("report.pdf", b"fake-pdf-bytes", image_describer=None)

    assert result == "Acme Corp acquired Beta Inc."


def test_load_pdf_appends_description_for_large_image(monkeypatch):
    page = FakePage("Revenue grew in Q3.", images=[FakeImageFile(_large_image(), b"chart-bytes")])
    _patch_reader(monkeypatch, [page])

    describer = MagicMock()
    describer.describe.return_value = "Bar chart: Q1=$2M, Q2=$3.5M, Q3=$6M, Q4=$4M."

    result = load_text("report.pdf", b"fake-pdf-bytes", image_describer=describer, min_image_dimension=64)

    describer.describe.assert_called_once_with(b"chart-bytes", "image/png")
    assert "Revenue grew in Q3." in result
    assert "[Image: Bar chart: Q1=$2M, Q2=$3.5M, Q3=$6M, Q4=$4M.]" in result


def test_load_pdf_skips_small_images_without_calling_describer(monkeypatch):
    page = FakePage("Some text.", images=[FakeImageFile(_tiny_image(), b"icon-bytes")])
    _patch_reader(monkeypatch, [page])

    describer = MagicMock()

    result = load_text("report.pdf", b"fake-pdf-bytes", image_describer=describer, min_image_dimension=64)

    describer.describe.assert_not_called()
    assert result == "Some text."


def test_load_pdf_skips_decorative_images_marked_none(monkeypatch):
    page = FakePage("Some text.", images=[FakeImageFile(_large_image(), b"logo-bytes")])
    _patch_reader(monkeypatch, [page])

    describer = MagicMock()
    describer.describe.return_value = "NONE"

    result = load_text("report.pdf", b"fake-pdf-bytes", image_describer=describer, min_image_dimension=64)

    assert result == "Some text."
