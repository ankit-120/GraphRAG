from graphrag.ingestion.chunking import chunk_text, split_sentences


def test_split_sentences_splits_on_terminal_punctuation():
    text = "Hello world. How are you? I'm fine!"
    assert split_sentences(text) == ["Hello world.", "How are you?", "I'm fine!"]


def test_split_sentences_empty_text_returns_empty_list():
    assert split_sentences("   ") == []


def test_chunk_text_single_chunk_when_under_size(sample_text):
    chunks = chunk_text(sample_text, document_id="doc-1", chunk_size=1000, chunk_overlap=150)
    assert len(chunks) == 1
    assert chunks[0].document_id == "doc-1"
    assert "Acme Corp" in chunks[0].text


def test_chunk_text_splits_when_over_size():
    sentences = [f"Sentence number {i} is here." for i in range(20)]
    text = " ".join(sentences)
    chunks = chunk_text(text, document_id="doc-2", chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.document_id == "doc-2"


def test_chunk_text_overlap_carries_sentences_between_chunks():
    sentences = [f"Sentence number {i} is here." for i in range(20)]
    text = " ".join(sentences)
    chunks = chunk_text(text, document_id="doc-3", chunk_size=100, chunk_overlap=30)
    assert chunks[0].text.split(".")[-2].strip() in chunks[1].text
