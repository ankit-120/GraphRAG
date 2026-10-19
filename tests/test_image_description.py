from unittest.mock import MagicMock

from graphrag.ingestion.image_description import NO_CONTENT_MARKER, ImageDescriber


def test_describe_sends_image_bytes_and_mime_type_and_returns_text():
    fake_client = MagicMock()
    fake_response = MagicMock()
    fake_response.text = "Bar chart showing Q1=$2M, Q2=$3.5M, Q3=$6M, Q4=$4M."
    fake_client.models.generate_content.return_value = fake_response

    describer = ImageDescriber(fake_client, model="fake-vision-model")
    result = describer.describe(b"fake-image-bytes", "image/png")

    assert result == "Bar chart showing Q1=$2M, Q2=$3.5M, Q3=$6M, Q4=$4M."
    fake_client.models.generate_content.assert_called_once()
    _, kwargs = fake_client.models.generate_content.call_args
    assert kwargs["model"] == "fake-vision-model"
    parts = kwargs["contents"]
    assert parts[0].inline_data.data == b"fake-image-bytes"
    assert parts[0].inline_data.mime_type == "image/png"


def test_describe_returns_none_marker_for_decorative_images():
    fake_client = MagicMock()
    fake_response = MagicMock()
    fake_response.text = f"  {NO_CONTENT_MARKER}  "
    fake_client.models.generate_content.return_value = fake_response

    describer = ImageDescriber(fake_client, model="fake-vision-model")
    result = describer.describe(b"fake-icon-bytes", "image/png")

    assert result == NO_CONTENT_MARKER
