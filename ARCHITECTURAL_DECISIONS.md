# Architectural Decisions

This document outlines the key architectural decisions made during the development of the Aircraft Maintenance RAG system. It serves as a reference for understanding the design choices and their rationale.

## Decision 1: Model Selection
**Choice:** Ollama with Llama 3.1:8b  
**Rationale:** Ollama provides a robust platform for deploying large language models locally, ensuring data privacy and low latency. Llama 3.1:8b is perfect due to its ability of running on personal computers while having a large context window size of 8192 tokens by default. See [Model Card](https://huggingface.co/mozilla-ai/Meta-Llama-3.1-8B-llamafile) for more details.

## Decision 2: Vector Store
**Choice:** ChromaDB  
**Rationale:** ChromaDB offers efficient storage and retrieval of vector embeddings for moderate scale applications. It is easy to integrate with LangChain and provides good enough performance.
