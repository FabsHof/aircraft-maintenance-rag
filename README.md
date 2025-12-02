# aircraft-maintenance-rag

A retrieval-augmented generation (RAG) system for aircraft maintenance documentation using LangChain, Ollama, and ChromaDB.

## 📑 documents

The system processes various types of aircraft maintenance documents, from the [FAA: Aviation Handbooks and Manuals website](https://www.faa.gov/regulations_policies/handbooks_manuals/aviation).

## 🔧 installation guide

Follow these steps to set up the development environment:

1. install python 3.10 or higher
2. install uv package manager: see https://docs.astral.sh/uv/getting-started/installation/
3. get dependencies: `uv sync`
4. activate virtual environment: `source .venv/bin/activate`
5. install ollama: https://ollama.com/download
6. download model: `ollama pull llama3.1:8b`

## 🚀 getting started

There is a central Makefile to run common tasks. Here are some useful commands:

- `make load_data` Load raw data files into the `data/raw` directory.
- `make process_data` Clean and preprocess raw data files, storing the results in the `data/clean` directory.
- `make split_and_store_data` Split the cleaned data into chunks and store them in the vector store.

## 🏗️ architectural decisions

See [ARCHITECTURAL_DECISIONS.md](ARCHITECTURAL_DECISIONS.md) for details on key architectural decisions made during the development of this system.