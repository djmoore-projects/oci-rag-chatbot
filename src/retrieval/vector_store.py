"""Oracle Autonomous Database 23ai vector store interface."""

from __future__ import annotations

import os

import oracledb
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_community.vectorstores import OracleVS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document

_TABLE_NAME = "PROPTECH_KNOWLEDGE"
_EMBEDDING_DIM = 1024


def connect_oracle() -> oracledb.Connection:
    """Open a TLS connection to Oracle Autonomous Database using env vars.

    Required env vars:
        ORACLE_USER, ORACLE_PASSWORD, ORACLE_DSN,
        ORACLE_WALLET_LOCATION, ORACLE_WALLET_PASSWORD
    """
    return oracledb.connect(
        user=os.environ["ORACLE_USER"],
        password=os.environ["ORACLE_PASSWORD"],
        dsn=os.environ["ORACLE_DSN"],
        wallet_location=os.environ["ORACLE_WALLET_LOCATION"],
        wallet_password=os.environ["ORACLE_WALLET_PASSWORD"],
    )


def build_vector_store(
    conn: oracledb.Connection,
    embeddings: OCIGenAIEmbeddings,
    table_name: str = _TABLE_NAME,
    embedding_dim: int = _EMBEDDING_DIM,
) -> OracleVS:
    """Initialise (or connect to an existing) Oracle vector table.

    The table schema stores chunk text as CLOB, metadata as JSON, and
    the embedding as a native VECTOR column — enabling hybrid SQL+vector queries
    that are impossible in standalone vector databases like Pinecone.

    Args:
        conn: Active oracledb connection.
        embeddings: Configured embedding function.
        table_name: Oracle table name for vector storage.
        embedding_dim: Dimension of the embedding model in use.

    Returns:
        ``OracleVS`` instance ready for ``add_documents`` or ``as_retriever``.
    """
    return OracleVS(
        client=conn,
        embedding_function=embeddings,
        table_name=table_name,
        distance_strategy=DistanceStrategy.COSINE,
        params={"embedding_dim": embedding_dim},
    )


def ingest_documents(
    vector_store: OracleVS,
    chunks: list[Document],
) -> int:
    """Add *chunks* to *vector_store* and return the count of stored vectors."""
    vector_store.add_documents(chunks)
    return len(chunks)
