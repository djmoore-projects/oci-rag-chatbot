"""Tests for the ingestion pipeline (pdf_loader and embedder)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.ingestion.embedder import _build_oci_config, build_embeddings
from src.ingestion.pdf_loader import load_and_chunk

# ---------------------------------------------------------------------------
# pdf_loader tests
# ---------------------------------------------------------------------------


def _make_pages(texts: list[str]) -> list[Document]:
    return [Document(page_content=t, metadata={"source": "test.pdf"}) for t in texts]


@pytest.fixture
def pdf_dir(tmp_path: Path) -> Path:
    """Create a temp directory with a dummy PDF file."""
    (tmp_path / "sample.pdf").touch()
    return tmp_path


@patch("src.ingestion.pdf_loader.PyPDFLoader")
def test_chunking_produces_non_empty_output(mock_loader_cls, pdf_dir: Path):
    """Each loaded PDF must produce at least one chunk."""
    long_text = "PropTech is transforming real estate. " * 50
    mock_loader = MagicMock()
    mock_loader.load.return_value = _make_pages([long_text])
    mock_loader_cls.return_value = mock_loader

    chunks = load_and_chunk(pdf_dir, chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 0, "Expected at least one chunk from a non-empty document"


@patch("src.ingestion.pdf_loader.PyPDFLoader")
def test_chunks_respect_size_boundary(mock_loader_cls, pdf_dir: Path):
    """No chunk should exceed chunk_size + overlap characters by a large margin."""
    long_text = "word " * 500
    mock_loader = MagicMock()
    mock_loader.load.return_value = _make_pages([long_text])
    mock_loader_cls.return_value = mock_loader

    chunk_size = 300
    chunks = load_and_chunk(pdf_dir, chunk_size=chunk_size, chunk_overlap=30)

    for chunk in chunks:
        assert len(chunk.page_content) <= chunk_size * 1.5, (
            f"Chunk too large: {len(chunk.page_content)} chars"
        )


@patch("src.ingestion.pdf_loader.PyPDFLoader")
def test_source_metadata_attached(mock_loader_cls, pdf_dir: Path):
    """Every chunk must carry a 'source' metadata key."""
    mock_loader = MagicMock()
    mock_loader.load.return_value = _make_pages(["Some content about real estate."] * 5)
    mock_loader_cls.return_value = mock_loader

    chunks = load_and_chunk(pdf_dir)

    assert all("source" in c.metadata for c in chunks), "Missing 'source' in chunk metadata"


def test_missing_directory_raises():
    with pytest.raises(FileNotFoundError):
        load_and_chunk("/nonexistent/path/to/pdfs")


@patch("src.ingestion.pdf_loader.PyPDFLoader")
def test_empty_directory_raises(mock_loader_cls, tmp_path: Path):
    """A directory with no PDFs should raise ValueError."""
    with pytest.raises(ValueError, match="No PDF files found"):
        load_and_chunk(tmp_path)


# ---------------------------------------------------------------------------
# embedder tests
# ---------------------------------------------------------------------------


def test_oci_config_raises_on_missing_env(monkeypatch: pytest.MonkeyPatch):
    """_build_oci_config should raise EnvironmentError when env vars are absent."""
    for key in ["OCI_USER", "OCI_FINGERPRINT", "OCI_TENANCY", "OCI_REGION", "OCI_KEY_FILE"]:
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(OSError, match="Missing OCI environment variables"):
        _build_oci_config()


@patch("src.ingestion.embedder.GenerativeAiInferenceClient")
@patch("src.ingestion.embedder.OCIGenAIEmbeddings")
def test_build_embeddings_passes_client(mock_embeddings_cls, mock_client_cls):
    """build_embeddings should inject the inference client directly into OCIGenAIEmbeddings."""
    fake_config = {
        "user": "u", "fingerprint": "f", "tenancy": "t",
        "region": "r", "key_file": "k",
    }
    fake_client = MagicMock()
    mock_client_cls.return_value = fake_client

    build_embeddings(oci_config=fake_config)

    mock_embeddings_cls.assert_called_once()
    call_kwargs = mock_embeddings_cls.call_args.kwargs
    assert call_kwargs["client"] is fake_client
