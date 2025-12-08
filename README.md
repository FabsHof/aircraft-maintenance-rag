# aircraft-maintenance-rag

A multi-agent retrieval-augmented generation (RAG) system for aircraft maintenance documentation using LangChain, LangGraph, LangSmith, Ollama.

## 🤖 agents

The system employs multiple specialized agents, that handle different aspects of aircraft maintenance:
- **maintenance technician agent**
- **design engineer agent**
- **safety officer agent**
- **pilot / operator agent**

## 📑 documents

The system processes various types of aircraft maintenance documents for the Airbus A320 family, including:
- maintenance procedures
- troubleshooting guides
- technical specifications
- engineering analysis
- incident reports
- airworthiness directives
- flight manuals
- operating procedures

Additional documents and/ or projects might be added by creating or editing the config file (see command `make load_data`).

## 🔧 installation guide

Follow these steps to set up the development environment:

1. install python 3.10 or higher
2. install uv package manager: see https://docs.astral.sh/uv/getting-started/installation/
3. get dependencies: `uv sync`
4. activate virtual environment: `source .venv/bin/activate`
5. install ollama: https://ollama.com/download
6. download model: `ollama pull llama3.1:8b`
7. configure environment variables: `cp .env.example .env` and adjust values as needed

## ⚙️ configuration

The project uses environment variables for configuration. Copy `.env.example` to `.env` and customize the values:

- **LM_STUDIO_URL**: URL for LM Studio API endpoint (default: `http://localhost:1234/v1`)
- **VISUAL_MODEL_NAME**: Model name for image parsing (default: `google/gemma-3-4b`)
- **EMBEDDING_MODEL_NAME**: HuggingFace model for embeddings (default: `sentence-transformers/all-MiniLM-L6-v2`)
- **EMBEDDING_DEVICE**: Device for embedding model (default: `cpu`, use `cuda` for GPU)
- **CHUNK_SIZE**: Text chunk size for splitting (default: `150`)
- **CHUNK_OVERLAP**: Overlap between chunks (default: `30`)
- **BATCH_SIZE**: Batch size for document ingestion (default: `5000`)

## 🚀 getting started

There is a central Makefile to run common tasks. Here are some useful commands:

- `make load_data` Load raw data files as specified in the configuration file `config/data_config.json`. Another config file might be specified by utilizing the `--config` argument, e.g., `make load_data --config=config/another_config.json`.
- `make process_data` Clean and preprocess raw data files, storing the results in the `data/clean` directory.
- `make split_and_store_data` Split the cleaned data into chunks and store them in the vector store. Provide `--clear_db` or `--dry-run` flags as needed.

## 🏗️ architectural decisions

See [ARCHITECTURAL_DECISIONS.md](ARCHITECTURAL_DECISIONS.md) for details on key architectural decisions made during the development of this system.