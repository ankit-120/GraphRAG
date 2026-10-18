import re
import uuid

from graphrag.models import Chunk

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def chunk_text(
    text: str, document_id: str, chunk_size: int = 1000, chunk_overlap: int = 150
) -> list[Chunk]:
    sentences = split_sentences(text)
    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        if current_len + len(sentence) > chunk_size and current:
            chunks.append(_make_chunk(current, document_id))
            current, current_len = _carry_overlap(current, chunk_overlap)
        current.append(sentence)
        current_len += len(sentence) + 1

    if current:
        chunks.append(_make_chunk(current, document_id))

    return chunks


def _make_chunk(sentences: list[str], document_id: str) -> Chunk:
    return Chunk(id=str(uuid.uuid4()), document_id=document_id, text=" ".join(sentences))


def _carry_overlap(sentences: list[str], chunk_overlap: int) -> tuple[list[str], int]:
    overlap: list[str] = []
    overlap_len = 0
    for sentence in reversed(sentences):
        if overlap_len >= chunk_overlap:
            break
        overlap.insert(0, sentence)
        overlap_len += len(sentence) + 1
    return overlap, overlap_len
