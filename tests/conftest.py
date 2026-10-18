import pytest


@pytest.fixture
def sample_text() -> str:
    return (
        "Acme Corp acquired Beta Inc in 2023. "
        "Beta Inc's founder, Jane Smith, then joined Acme Corp as Head of Engineering. "
        "Acme Corp is headquartered in Austin."
    )
