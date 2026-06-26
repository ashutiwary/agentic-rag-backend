from __future__ import annotations


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