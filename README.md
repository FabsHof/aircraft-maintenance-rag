# aircraft-maintenance-rag
RAG tool for aircraft documentation

## 🔧 installation guide

1. install python 3.10 or higher
2. install uv package manager: see https://docs.astral.sh/uv/getting-started/installation/
3. get dependencies: `uv sync`
4. activate virtual environment: `source .venv/bin/activate`
5. install ollama: https://ollama.com/download
6. download model: `ollama pull llama3.1:8b`

## 🚀 getting started

1. load data: `python src/data/load_data.py`