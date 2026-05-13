"""PDF ingestion: load documents and split into overlapping chunks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_chunk(
    pdf_dir: str | Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
) -> List[Document]:
    """Load every PDF in *pdf_dir* and return a flat list of text chunks.

    Args:
        pdf_dir: Directory that contains the source PDF files.
        chunk_size: Target character count per chunk.
        chunk_overlap: Characters shared between adjacent chunks to preserve context.

    Returns:
        List of LangChain ``Document`` objects, one per chunk.

    Raises:
        FileNotFoundError: If *pdf_dir* does not exist.
        ValueError: If no PDF files are found in *pdf_dir*.
    """
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in {pdf_dir}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks: List[Document] = []
    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        file_chunks = splitter.split_documents(pages)
        # Attach source filename so retrieval results are traceable.
        for chunk in file_chunks:
            chunk.metadata.setdefault("source", pdf_path.name)
        chunks.extend(file_chunks)

    return chunks
