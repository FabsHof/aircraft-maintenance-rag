# Architectural Decisions

This document outlines the key architectural decisions made during the development of the Aircraft Maintenance RAG system. It serves as a reference for understanding the design choices and their rationale.

## Decision 1: Model Selection
**Choice:** LM Studio with Llama 3.1:8b  
**Rationale:** LM Studio provides a robust platform for deploying large language models locally, ensuring data privacy and low latency. Llama 3.1:8b is perfect due to its ability of running on personal computers while having a large context window size of 8192 tokens by default. See [Model Card](https://huggingface.co/mozilla-ai/Meta-Llama-3.1-8B-llamafile) for more details.

**Choice:** Gemma 3:4b for visual understanding  
**Rationale:** Gemma 3:4b is selected for its capability to process and understand images, which is essential for interpreting diagrams and visual content in aircraft maintenance documents. Its integration with LM Studio ensures seamless operation within the existing architecture.

## Decision 2: Vector Store
**Choice:** ChromaDB  
**Rationale:** ChromaDB offers efficient storage and retrieval of vector embeddings for moderate scale applications. It is easy to integrate with LangChain and provides good enough performance.

## Decision 3: Document Processing
**Choice:** Use PyMuPDFLoader with LLMImageBlobParser instead of PyPDFLoader
**Rationale:** PyPDFLoader still has an unfixed bug regarding image extraction: https://github.com/langchain-ai/langchain/issues/15730