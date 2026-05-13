"""LangChain RAG chain wiring Cohere Command R to the Oracle retriever."""

from __future__ import annotations

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain_community.vectorstores import OracleVS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from oci.generative_ai_inference import GenerativeAiInferenceClient

_SYSTEM_PROMPT = (
    "You are a knowledgeable PropTech assistant. "
    "Answer the question using only the context provided below. "
    "If the context does not contain enough information, say so rather than guessing.\n\n"
    "{context}"
)

_DEFAULT_MODEL = "cohere.command-r-08-2024"
_DEFAULT_ENDPOINT = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"


def build_rag_chain(
    vector_store: OracleVS,
    oci_config: dict,
    model_id: str = _DEFAULT_MODEL,
    service_endpoint: str = _DEFAULT_ENDPOINT,
    max_tokens: int = 1000,
    temperature: float = 0.3,
    retriever_k: int = 4,
) -> Runnable:
    """Construct a retrieval-augmented generation chain.

    The chain:
      1. Embeds the user question and retrieves *retriever_k* chunks from Oracle 23ai.
      2. Stuffs those chunks into the system prompt context.
      3. Calls Cohere Command R via OCI GenAI and returns the response.

    Args:
        vector_store: Initialised ``OracleVS`` with ingested documents.
        oci_config: OCI authentication config dict.
        model_id: OCI GenAI chat model identifier.
        service_endpoint: Regional OCI GenAI inference endpoint URL.
        max_tokens: Maximum tokens in the generated response.
        temperature: Sampling temperature (lower = more deterministic).
        retriever_k: Number of chunks to retrieve per query.

    Returns:
        Runnable chain; call with ``chain.invoke({"input": "your question"})``.
        Returns a dict with keys ``"input"``, ``"context"``, and ``"answer"``.
    """
    inference_client = GenerativeAiInferenceClient(
        oci_config,
        service_endpoint=service_endpoint,
    )
    llm = ChatOCIGenAI(
        model_id=model_id,
        compartment_id=oci_config["tenancy"],
        client=inference_client,
        model_kwargs={"max_tokens": max_tokens, "temperature": temperature},
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        ("human", "{input}"),
    ])

    doc_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever(search_kwargs={"k": retriever_k})
    return create_retrieval_chain(retriever, doc_chain)
