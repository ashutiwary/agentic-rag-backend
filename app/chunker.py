from __future__ import annotations
import re


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping fixed-size chunks.

    Args:
        text:       Raw text to split.
        chunk_size: Maximum characters per chunk.
        overlap:    Characters repeated at the start of each
                    subsequent chunk to preserve boundary context.

    Returns:
        List of non-empty text chunks.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    if not text or not text.strip():
        return []

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Avoid splitting mid-word
        if end < len(text):
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        next_start = end - overlap

        # Guarantee forward progress: a short word-boundary chunk can make
        # `end - overlap` land at or before `start`, which would loop forever.
        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


# Split on sentence-ending punctuation followed by whitespace.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    """
    Break text into sentences. Splits on paragraph boundaries first,
    then on sentence-ending punctuation, so two paragraphs are never
    merged into one sentence.
    """
    sentences: list[str] = []
    for block in text.split("\n"):
        block = block.strip()
        if not block:
            continue
        for sentence in _SENTENCE_RE.split(block):
            sentence = sentence.strip()
            if sentence:
                sentences.append(sentence)
    return sentences


def semantic_chunk(
    text: str,
    max_chars: int,
    breakpoint_percentile: float,
) -> list[str]:
    """
    Split text on meaning boundaries.

    Each sentence is embedded, the cosine distance between neighbouring
    sentences is measured, and a chunk boundary is placed wherever that
    distance lands in the top (100 - breakpoint_percentile) percent, i.e.
    where the topic shifts most. A running size cap (max_chars) forces a
    break before any chunk grows too large.

    Args:
        text:                  Raw text to split.
        max_chars:             Hard upper bound on chunk length.
        breakpoint_percentile: Distance percentile that triggers a split.
                               Higher means fewer, larger chunks.

    Returns:
        List of non-empty text chunks. Falls back to fixed-size splitting
        for any single chunk that still exceeds max_chars.
    """
    if not text or not text.strip():
        return []

    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return _cap(sentences[:1] or [text.strip()], max_chars)

    # Imported lazily so importing this module does not load the
    # embedding model when only fixed-size chunking is used.
    import numpy as np
    from app.embedder import embed

    vectors = np.asarray(embed(sentences), dtype=float)
    norms   = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = vectors / norms

    # Cosine distance between each pair of neighbouring sentences.
    distances = [1.0 - float(unit[i - 1] @ unit[i]) for i in range(1, len(sentences))]
    threshold = float(np.percentile(distances, breakpoint_percentile))

    chunks: list[str] = []
    current: list[str] = [sentences[0]]
    current_len = len(sentences[0])

    for i in range(1, len(sentences)):
        prospective = current_len + 1 + len(sentences[i])
        topic_shift = distances[i - 1] > threshold

        if topic_shift or prospective > max_chars:
            chunks.append(" ".join(current))
            current     = [sentences[i]]
            current_len = len(sentences[i])
        else:
            current.append(sentences[i])
            current_len = prospective

    if current:
        chunks.append(" ".join(current))

    return _cap(chunks, max_chars)


def _cap(chunks: list[str], max_chars: int) -> list[str]:
    """Fixed-split any chunk that still exceeds max_chars, drop empties."""
    capped: list[str] = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        if len(chunk) > max_chars:
            capped.extend(chunk_text(chunk, chunk_size=max_chars, overlap=0))
        else:
            capped.append(chunk)
    return capped


def chunk_document(text: str) -> list[str]:
    """
    Chunk text using the strategy set in config (semantic or fixed).
    Read at call time so a live settings change takes effect.
    """
    import app.config as config

    if config.CHUNK_STRATEGY == "semantic":
        return semantic_chunk(
            text,
            max_chars=config.MAX_CHUNK_SIZE,
            breakpoint_percentile=config.SEMANTIC_BREAKPOINT_PERCENTILE,
        )

    return chunk_text(text, chunk_size=config.CHUNK_SIZE, overlap=config.OVERLAP)