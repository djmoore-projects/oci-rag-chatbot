"""OCI GenAI Cohere embedding client factory."""

from __future__ import annotations

import os

from langchain_community.embeddings import OCIGenAIEmbeddings
from oci.generative_ai_inference import GenerativeAiInferenceClient

_DEFAULT_MODEL = "cohere.embed-english-v3.0"
_DEFAULT_ENDPOINT = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"


def _build_oci_config() -> dict:
    """Assemble OCI config from environment variables.

    Required env vars:
        OCI_USER, OCI_FINGERPRINT, OCI_TENANCY, OCI_REGION, OCI_KEY_FILE
    """
    required = ["OCI_USER", "OCI_FINGERPRINT", "OCI_TENANCY", "OCI_REGION", "OCI_KEY_FILE"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise OSError(f"Missing OCI environment variables: {missing}")

    return {
        "user": os.environ["OCI_USER"],
        "fingerprint": os.environ["OCI_FINGERPRINT"],
        "tenancy": os.environ["OCI_TENANCY"],
        "region": os.environ["OCI_REGION"],
        "key_file": os.environ["OCI_KEY_FILE"],
    }


def build_embeddings(
    model_id: str = _DEFAULT_MODEL,
    service_endpoint: str = _DEFAULT_ENDPOINT,
    oci_config: dict | None = None,
) -> OCIGenAIEmbeddings:
    """Return an authenticated ``OCIGenAIEmbeddings`` instance.

    Credentials are read from environment variables by default so that
    no secrets appear in notebook or application code.

    Args:
        model_id: Cohere embedding model identifier on OCI GenAI.
        service_endpoint: Regional OCI GenAI inference endpoint URL.
        oci_config: Override dict; if ``None`` the env-var-based config is used.

    Returns:
        Configured ``OCIGenAIEmbeddings`` ready for ``embed_documents`` calls.
    """
    config = oci_config or _build_oci_config()
    inference_client = GenerativeAiInferenceClient(
        config,
        service_endpoint=service_endpoint,
    )
    return OCIGenAIEmbeddings(
        model_id=model_id,
        compartment_id=config["tenancy"],
        client=inference_client,
    )
