import os
from os import path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import VectorStore
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters.base import TextSplitter
from collections.abc import Iterable
from langchain_core.documents import Document
from src.util.logging import log_message
import argparse

def extract_and_split_documents(file_path: str, text_splitter: TextSplitter) -> Iterable[Document]:
    '''
    Extract and split documents from a PDF file.
    arguments:
        file_path: Path to the PDF file.
        text_splitter: An instance of TextSplitter to split the documents.
    returns:
        A list of split documents.
    '''
    loader = PyPDFLoader(file_path=file_path, mode='page')
    documents = loader.lazy_load()
    documents = text_splitter.split_documents(documents)
    return documents


def ingest_raw_data(files: list, text_splitter: TextSplitter, vectorstore: VectorStore, dry_run: bool = False) -> None:
    '''
    Ingest raw data files into the vector store.
    arguments:
        files: List of file paths to ingest.
        text_splitter: An instance of TextSplitter to split the documents.
        vectorstore: An instance of VectorStore to add documents to.
        dry_run: If True, do not actually add documents to the vector store.
    '''
    for file in files:
        log_message(f'Ingesting file: {file}')
        documents = extract_and_split_documents(file, text_splitter)
        if not dry_run:
            vectorstore.add_documents(documents)
            vectorstore.persist()
            log_message(f'Successfully ingested {file} with {len(documents)} documents.')
        else:
            log_message(f'Dry run enabled. Skipping ingestion for {file}. Found {len(documents)} documents.')
    log_message('Ingestion process completed successfully.')

def main(clear_db: bool = False, dry_run: bool = False) -> None:
    clean_dir = path.join('data', 'clean')
    chroma_persist_directory = 'chroma_db'
    if not path.exists(clean_dir):
        log_message(f'Clean data directory {clean_dir} does not exist. Please run the data processing step first.')
        return
    files = [path.join(clean_dir, f) for f in os.listdir(clean_dir) if f.endswith('.pdf')]
    if not files:
        log_message(f'No PDF files found in {clean_dir} to ingest.')
        return
    
    embeddings = OllamaEmbeddings(model='llama3.1:8b')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    vectorstore = Chroma(persist_directory=chroma_persist_directory, embedding_function=embeddings)
    if clear_db:
        log_message('Clearing existing vector store database...')
        vectorstore.delete_collection()
        log_message('Existing vector store database cleared.')
    ingest_raw_data(files, text_splitter, vectorstore, dry_run=dry_run)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ingest clean data into vector store.')
    parser.add_argument('--clear_db', '-c', action=argparse.BooleanOptionalAction, help='Clear existing vector store before ingestion.')
    parser.add_argument('--dry_run', '-d', action=argparse.BooleanOptionalAction, help='Perform a dry run without actual ingestion.')
    args = parser.parse_args()
    
    main(clear_db=args.clear_db, dry_run=args.dry_run)