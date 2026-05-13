"""Tests for retrieval (vector_store, query_engine) and RAG chain generation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from src.generation.rag_chain import build_rag_chain
from src.retrieval.query_engine import assemble_context, similarity_search
from src.retrieval.vector_store import build_vector_store, ingest_documents

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _docs(n: int) -> list[Document]:
    return [
        Document(
            page_content=f"Chunk {i}: PropTech insight number {i}.",
            metadata={"source": f"doc{i}.pdf"},
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# vector_store tests
# ---------------------------------------------------------------------------


@patch("src.retrieval.vector_store.OracleVS")
def test_build_vector_store_returns_oracle_vs(mock_oracle_vs_cls):
    """build_vector_store should instantiate OracleVS with COSINE distance."""
    from langchain_community.vectorstores.utils import DistanceStrategy

    fake_conn = MagicMock()
    fake_embeddings = MagicMock()
    mock_oracle_vs_cls.return_value = MagicMock()

    build_vector_store(fake_conn, fake_embeddings)

    mock_oracle_vs_cls.assert_called_once()
    kwargs = mock_oracle_vs_cls.call_args.kwargs
    assert kwargs["distance_strategy"] == DistanceStrategy.COSINE


def test_ingest_documents_returns_correct_count():
    """ingest_documents should return the exact number of chunks passed in."""
    fake_vs = MagicMock()
    docs = _docs(42)

    count = ingest_documents(fake_vs, docs)

    assert count == 42
    fake_vs.add_documents.assert_called_once_with(docs)


# ---------------------------------------------------------------------------
# query_engine tests
# ---------------------------------------------------------------------------


def test_similarity_search_returns_k_results():
    """similarity_search must return exactly k documents."""
    k = 4
    fake_vs = MagicMock()
    fake_vs.similarity_search.return_value = _docs(k)

    results = similarity_search(fake_vs, "What is PropTech?", k=k)

    assert len(results) == k
    fake_vs.similarity_search.assert_called_once_with("What is PropTech?", k=k)


def test_similarity_search_returns_fewer_when_store_is_small():
    """When fewer than k docs exist, the function returns what the store provides."""
    fake_vs = MagicMock()
    fake_vs.similarity_search.return_value = _docs(2)

    results = similarity_search(fake_vs, "query", k=10)

    assert len(results) == 2


def test_assemble_context_joins_page_content():
    """assemble_context should concatenate chunk texts with the separator."""
    docs = _docs(3)
    context = assemble_context(docs, separator=" | ")

    for doc in docs:
        assert doc.page_content in context
    assert " | " in context


# ---------------------------------------------------------------------------
# RAG chain tests
# ---------------------------------------------------------------------------


@patch("src.generation.rag_chain.create_retrieval_chain")
@patch("src.generation.rag_chain.create_stuff_documents_chain")
@patch("src.generation.rag_chain.ChatOCIGenAI")
@patch("src.generation.rag_chain.GenerativeAiInferenceClient")
def test_rag_chain_produces_non_empty_answer(
    mock_client_cls,
    mock_llm_cls,
    mock_doc_chain_fn,
    mock_retrieval_chain_fn,
):
    """The RAG chain's invoke call should produce a non-empty 'answer' string."""
    fake_config = {"tenancy": "ocid.tenancy.test"}
    fake_vs = MagicMock()
    fake_vs.as_retriever.return_value = MagicMock()

    fake_chain = MagicMock()
    fake_chain.invoke.return_value = {
        "input": "What is PropTech?",
        "context": [Document(page_content="PropTech uses digital technology.")],
        "answer": "PropTech uses digital technology to transform real estate.",
    }
    mock_retrieval_chain_fn.return_value = fake_chain

    chain = build_rag_chain(fake_vs, fake_config)
    result = chain.invoke({"input": "What is PropTech?"})

    assert result["answer"], "RAG chain returned an empty answer"
    assert len(result["answer"]) > 10


@patch("src.generation.rag_chain.create_retrieval_chain")
@patch("src.generation.rag_chain.create_stuff_documents_chain")
@patch("src.generation.rag_chain.ChatOCIGenAI")
@patch("src.generation.rag_chain.GenerativeAiInferenceClient")
def test_retrieval_increases_with_more_chunks(
    mock_client_cls,
    mock_llm_cls,
    mock_doc_chain_fn,
    mock_retrieval_chain_fn,
):
    """Passing a larger k to build_rag_chain should set k on the retriever."""
    fake_config = {"tenancy": "ocid.tenancy.test"}
    fake_vs = MagicMock()
    fake_vs.as_retriever.return_value = MagicMock()
    mock_retrieval_chain_fn.return_value = MagicMock()

    build_rag_chain(fake_vs, fake_config, retriever_k=8)

    fake_vs.as_retriever.assert_called_once_with(search_kwargs={"k": 8})
