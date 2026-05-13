"""Similarity search and context assembly over the Oracle vector store."""

from __future__ import annotations

from typing import List, Tuple

from langchain_community.vectorstores import OracleVS
from langchain_core.documents import Document


def similarity_search(
    vector_store: OracleVS,
    query: str,
    k: int = 4,
) -> List[Document]:
    """Return the *k* most relevant chunks for *query*.

    Args:
        vector_store: Initialised ``OracleVS`` instance.
        query: Natural-language question from the user.
        k: Number of chunks to retrieve.

    Returns:
        Ordered list of ``Document`` objects (most similar first).
    """
    return vector_store.similarity_search(query, k=k)


def similarity_search_with_score(
    vector_store: OracleVS,
    query: str,
    k: int = 4,
) -> List[Tuple[Document, float]]:
    """Return chunks paired with their cosine similarity scores.

    Scores closer to 1.0 indicate higher relevance. Useful for evaluation
    and for building score-gated retrieval pipelines.
    """
    return vector_store.similarity_search_with_score(query, k=k)


def assemble_context(docs: List[Document], separator: str = "\n\n---\n\n") -> str:
    """Concatenate retrieved chunk texts into a single context string."""
    return separator.join(doc.page_content for doc in docs)
