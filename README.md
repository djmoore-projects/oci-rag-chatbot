OCI RAG Chatbot

Enterprise-grade Retrieval-Augmented Generation on Oracle Cloud

Overview

This project is a production-style Retrieval-Augmented Generation (RAG) system built on Oracle Cloud Infrastructure (OCI). It demonstrates how enterprises can deploy a secure, scalable, and compliant AI assistant that answers questions over private documents using vector search inside Oracle Database 26ai.

The system ingests PDFs, generates embeddings using Cohere on OCI GenAI, stores vectors in Oracle Autonomous Database 26ai, and serves intelligent responses via LangChain.

This mirrors how regulated enterprises (finance, healthcare, government, real estate) deploy internal AI copilots without exposing data to public LLMs.

⸻

Architecture

Ingestion Pipeline
	1.	PDF documents are loaded and chunked
	2.	Text is embedded using Cohere Embeddings on OCI GenAI
	3.	Vectors are stored in Oracle Autonomous Database 23ai (OCI Vector Store)

Query Pipeline
	1.	User submits a question
	2.	LangChain performs similarity search in Oracle DB
	3.	Relevant chunks are retrieved
	4.	Context is sent to the LLM for response generation

Core Stack
	•	Oracle Cloud Infrastructure (OCI)
	•	Oracle Autonomous Database 26ai (Vector Search)
	•	OCI Generative AI (Cohere)
	•	LangChain
	•	Python

⸻

Why Oracle DB 26ai Vector Search Matters

Unlike Pinecone or other standalone vector databases, Oracle DB 26ai stores vectors natively inside the same enterprise database that runs mission-critical workloads.

This enables:
	•	Vector search + SQL in one query
	•	Row-level security and access controls
	•	Auditing, backups, compliance
	•	Enterprise data governance

This project demonstrates how RAG can be deployed in environments that require SOC-2, HIPAA, or financial compliance.

⸻

What This Demonstrates

This repository showcases the core skills required for modern AI Solutions Engineer / Architect roles:
	•	Designing RAG architectures
	•	Embedding pipelines
	•	Vector database design
	•	Secure enterprise AI deployment
	•	LLM orchestration with LangChain
	•	Cloud-native implementation on OCI

⸻

How to Run
	1.	Configure OCI credentials locally
	2.	Install dependencies
	3.	Run the ingestion notebook to embed documents
	4.	Run the chatbot notebook to query the system

(Details in /notebooks)

⸻

Use Cases
	•	Internal knowledge assistants
	•	Real estate due diligence copilots
	•	Legal document search
	•	Financial research tools
	•	Customer support AI

⸻

Author

Built by Derek Moore 
AI Solutions Engineer | Cloud Architect | RAG Systems Builder
