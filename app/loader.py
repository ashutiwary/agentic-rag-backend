from __future__ import annotations
import os
import fitz          # pymupdf
import docx          # python-docx
import pandas as pd


def load_pdf(filepath: str) -> str:
    """Extract plain text from all pages of a PDF."""
    document = fitz.open(filepath)
    try:
        pages = [page.get_text() for page in document]
    finally:
        document.close()
    return "\n".join(pages)


def load_docx(filepath: str) -> str:
    """Extract paragraph text from a Word document."""
    document   = docx.Document(filepath)
    paragraphs = [
        p.text for p in document.paragraphs if p.text.strip()
    ]
    return "\n".join(paragraphs)


def load_csv(filepath: str) -> str:
    """
    Convert each CSV row into a human-readable text string.
    Format: "column1: value1 | column2: value2 | ..."
    """
    df   = pd.read_csv(filepath)
    rows = []
    for _, row in df.iterrows():
        parts = [
            f"{col}: {val}"
            for col, val in row.items()
            if pd.notna(val) and str(val).strip()
        ]
        if parts:
            rows.append(" | ".join(parts))
    return "\n".join(rows)


def load_txt(filepath: str) -> str:
    """Read a plain text file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


# Supported formats registry
_LOADERS: dict[str, callable] = {
    ".pdf":  load_pdf,
    ".docx": load_docx,
    ".csv":  load_csv,
    ".txt":  load_txt,
}

SUPPORTED_FORMATS = set(_LOADERS.keys())


def load_document(filepath: str) -> str:
    """
    Detect file format from extension and return extracted text.
    This is the single entry point for all other modules.

    Raises:
        ValueError: If the file extension is not supported.
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()

    if ext not in _LOADERS:
        raise ValueError(
            f"Unsupported format '{ext}'. "
            f"Supported: {sorted(SUPPORTED_FORMATS)}"
        )

    return _LOADERS[ext](filepath)