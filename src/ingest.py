"""Extract text from uploaded PDFs and split it into overlapping chunks for embedding."""
from dataclasses import dataclass

from pypdf import PdfReader


@dataclass
class Chunk:
    text: str
    source: str  # filename
    page: int    # 1-indexed page number the chunk starts on


def extract_pages(file, filename: str) -> list[tuple[int, str]]:
    """Read a PDF (file-like object or path) and return a list of (page_number, text)."""
    reader = PdfReader(file)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i, text))
    return pages


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Split text into overlapping character-based chunks.

    Character-based rather than token-based to avoid pulling in a tokenizer
    just for chunking; chunk_size is generous enough that this is a reasonable
    approximation for typical research-paper prose.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def process_pdf(file, filename: str, chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    """Extract and chunk a single PDF, tagging each chunk with its source filename and page."""
    chunks = []
    for page_num, page_text in extract_pages(file, filename):
        for piece in chunk_text(page_text, chunk_size, overlap):
            if piece.strip():
                chunks.append(Chunk(text=piece, source=filename, page=page_num))
    return chunks
