"""
Stub out heavy OCI/Oracle/LangChain packages so the test suite runs
without any cloud credentials or vendor SDKs installed.

langchain-core and langchain-text-splitters are installed for real
because they provide Document and RecursiveCharacterTextSplitter.
Everything else is replaced with MagicMocks before any src/ import occurs.
"""

import sys
from unittest.mock import MagicMock

_STUBS = [
    "oci",
    "oci.generative_ai",
    "oci.generative_ai_inference",
    "oracledb",
    "langchain_community",
    "langchain_community.document_loaders",
    "langchain_community.embeddings",
    "langchain_community.vectorstores",
    "langchain_community.vectorstores.utils",
    "langchain_community.chat_models",
    "langchain_community.chat_models.oci_generative_ai",
    "langchain_classic",
    "langchain_classic.chains",
    "langchain_classic.chains.combine_documents",
]

for _mod in _STUBS:
    sys.modules.setdefault(_mod, MagicMock())
