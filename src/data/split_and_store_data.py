import os
import json
import itertools
from os import path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders.parsers import LLMImageBlobParser
from langchain_core.vectorstores import VectorStore
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters.base import TextSplitter
from langchain_core.documents import Document
from src.util.logging import log_info
import argparse
from typing import Generator

def extract_and_split_documents(file_path: str, text_splitter: TextSplitter, visual_model: ChatOpenAI, metadata: dict = None) -> Generator[Document, None, None]:
    '''
    Extract and split documents from a given file using PyMuPDFLoader and a text splitter.
    arguments:
        file_path: Path to the file to be processed.
        text_splitter: An instance of TextSplitter to split the documents.
        visual_model: A visual model for parsing images.
        metadata: Optional metadata to attach to each document.
    returns:
        A generator yielding split Document instances.
    '''
    loader = PyMuPDFLoader(
        file_path=file_path,
        mode='page',
        images_inner_format='text',
        extract_tables='markdown',
        extract_images=True,
        images_parser=LLMImageBlobParser(model=visual_model)
    )
    # In order to support large documents, it is done in a storage-efficient manner
    documents = loader.lazy_load()
    
    # Update metadata on each document as it's loaded
    def update_document_metadata(docs):
        for doc in docs:
            if metadata:
                doc.metadata.update({
                    'role': metadata.get('role', 'document'),
                    'doc_type': metadata.get('doc_type', 'unknown'),
                    'source': metadata.get('source', 'unknown'),
                })
            doc.metadata.update({
                'title': metadata.get('title', path.basename(file_path)) if metadata else path.basename(file_path),
            })
            yield doc
    
    documents_with_metadata = update_document_metadata(documents)
    split_documents = text_splitter.split_documents(documents_with_metadata)
    
    for doc in split_documents:
        yield doc

def ingest_documents_in_batches(documents: Generator[Document, None, None], vectorstore: VectorStore, batch_size: int = 5000) -> int:
    '''
    Ingest documents into the vector store in batches.
    arguments:
        documents: Generator of Document instances to ingest.
        vectorstore: An instance of VectorStore to add documents to.
        batch_size: Number of documents to process in each batch.
    returns:
        Total number of documents ingested.
    '''
    total_ingested = 0
    batch_num = 1
    
    while True:
        batch_docs = list(itertools.islice(documents, batch_size))
        if not batch_docs:
            break
        vectorstore.add_documents(batch_docs)
        total_ingested += len(batch_docs)
        log_info(f'Ingested batch {batch_num}: {len(batch_docs)} documents')
        batch_num += 1
    
    return total_ingested

def ingest_raw_data(files: list, text_splitter: TextSplitter, vectorstore: VectorStore, visual_model: ChatOpenAI, metadata: dict = None, ingestion_batch_size: int = 5000, dry_run: bool = False) -> None:
    '''
    Ingest raw data files into the vector store using optimized batch processing.
    arguments:
        files: List of file paths to ingest.
        text_splitter: An instance of TextSplitter to split the documents.
        vectorstore: An instance of VectorStore to add documents to.
        metadata: Optional metadata to attach to each document.
        ingestion_batch_size: Number of documents to process in each batch. Must not exceed 5000.
        visual_model: A visual model for parsing images.
        dry_run: If True, do not actually add documents to the vector store.
    '''
    # Batch size must not exceed ChromaDB's limit (5000)
    if ingestion_batch_size > 5000:
        ingestion_batch_size = 5000
        log_info('Batch size exceeds ChromaDB limit. Setting batch size to 5000.')
    
    for file in files:
        if not path.exists(file):
            log_info(f'⚡️ File {file} does not exist. Skipping.')
            continue
        log_info(f'Ingesting file: {file}')
        split_documents = extract_and_split_documents(file, text_splitter, visual_model, metadata=metadata)
        
        if not dry_run:
            total_docs = ingest_documents_in_batches(split_documents, vectorstore, ingestion_batch_size)
            log_info(f'Successfully ingested {file} with {total_docs} documents.')
        else:
            # For dry run, count documents without ingesting
            total_docs = sum(1 for _ in split_documents)
            log_info(f'Dry run enabled. Skipping ingestion for {file}. Found {total_docs} documents.')
    
    log_info('Ingestion process completed successfully.')

def get_configs_and_files(config_files: list[str], clean_dir: str) -> tuple[list[dict], list[str]]:
    '''
    For each configuration file, determine the corresponding clean data files to ingest.
    arguments:
        config_files: List of configuration file paths.
        clean_dir: Base directory where clean data files are stored.
    returns:
        A list of tuples, each containing the configuration dictionary and the list of file paths to ingest.
    '''
    
    configs_and_files = []
    for config_file in config_files:
        log_info(f'Using configuration file: {config_file}')
        with open(config_file, 'r') as cf:
            config_json = json.load(cf)
            config_project = config_json.get('project', 'default_project')
            config_version = config_json.get('version', '1.0')
            log_info(f'Project: {config_project}, Version: {config_version}')
            files_from_config = config_json.get('documents', [])
            if not files_from_config:
                log_info(f'No documents specified in configuration file: {config_file}. Skipping.')
                continue
        
            base_dir = path.join(clean_dir, config_project, config_version)
            # Get the directory for each document specified in the config as well as the filenames
            files = []
            for doc in files_from_config:
                directory = doc.get('directory', '')
                filename = doc.get('filename', '')
                current_dir = path.join(base_dir, directory, filename)
                files.append(current_dir)
        
            if not files:
                log_info(f'No PDF files found in {base_dir} to ingest.')
                continue
        configs_and_files.append((config_file, files))
    return configs_and_files

def main(clear_db: bool = False, dry_run: bool = False, lm_studio_url: str = 'http://localhost:1234/v1', visual_model_name: str = 'google/gemma-3-4b', visual_model_temperature: float = 0.1, embedding_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', embedding_device: str = 'cpu', embedding_chunk_size: int = 1000, embedding_chunk_overlap: int = 200, ingestion_batch_size: int = 5000) -> None:
    clean_dir = path.join('data', 'clean')
    config_path = path.join('config')

    if not path.exists(clean_dir):
        log_info(f'Clean data directory {clean_dir} does not exist. Please run the data processing step first.')
        return
    
    if not path.exists(config_path):
        log_info(f'Configuration directory {config_path} does not exist. Please provide a valid configuration.')
        return

    config_files = [path.join(config_path, f) for f in os.listdir(config_path) if f.endswith('.json')]
    if not config_files:
        log_info(f'No configuration files found in {config_path}. Please provide at least one configuration file.')
        return
    
    configs_and_files = get_configs_and_files(config_files, clean_dir)

    if not configs_and_files:
        log_info('No valid configurations and files found for ingestion.')
        return
    
    # Initialize models and components
    log_info('Initializing models and components...')
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={'device': embedding_device}
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=embedding_chunk_size,
        chunk_overlap=embedding_chunk_overlap,
        separators=['\n\n', '\n', '(?<=\. )', ' ', '']
    )
    visual_model = ChatOpenAI(
        model=visual_model_name,
        temperature=visual_model_temperature,
        base_url=lm_studio_url,
        api_key=''
    )

    for config_file, files in configs_and_files:
        log_info(f'Starting ingestion for configuration: {config_file}')
        with open(config_file, 'r') as cf:
            config_json = json.load(cf)
            project = config_json.get('project', 'default_project')
            version = config_json.get('version', '1.0')
            collection_name = f'{project}_{version}_collection'
            vectorstore = Chroma(
                collection_name=collection_name,
                persist_directory='chroma_db',
                embedding_function=embedding_model
            )
    
        if clear_db:
            log_info('Clearing existing vector store database...')
            vectorstore.reset_collection()
            log_info('Existing vector store database cleared.')
    
        ingest_raw_data(files, text_splitter, vectorstore, visual_model, ingestion_batch_size=ingestion_batch_size, dry_run=dry_run)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ingest clean data into vector store.')
    parser.add_argument('--clear_db', '-c', action=argparse.BooleanOptionalAction, help='Clear existing vector store before ingestion.')
    parser.add_argument('--dry_run', '-d', action=argparse.BooleanOptionalAction, help='Perform a dry run without actual ingestion.')
    parser.add_argument('--lm_studio_url', type=str, default='http://localhost:1234/v1', help='URL for the LM Studio instance.')
    parser.add_argument('--visual_model_name', type=str, default='google/gemma-3-4b', help='Name of the visual model to use.')
    parser.add_argument('--visual_model_temperature', type=float, default=0.1, help='Temperature for the visual model.')
    parser.add_argument('--batch_size', type=int, default=5000, help='Batch size for ingestion.')
    parser.add_argument('--embedding_model_name', type=str, default='sentence-transformers/all-MiniLM-L6-v2', help='Name of the embedding model to use.')
    parser.add_argument('--embedding_device', type=str, default='cpu', help='Device for the embedding model (e.g., cpu, cuda).')
    parser.add_argument('--chunk_size', type=int, default=1000, help='Chunk size for text splitting.')
    parser.add_argument('--chunk_overlap', type=int, default=200, help='Chunk overlap for text splitting.')
    args = parser.parse_args()
    
    main(clear_db=args.clear_db, dry_run=args.dry_run, lm_studio_url=args.lm_studio_url, visual_model_name=args.visual_model_name, visual_model_temperature=args.visual_model_temperature, embedding_model_name=args.embedding_model_name, embedding_device=args.embedding_device, embedding_chunk_size=args.chunk_size, embedding_chunk_overlap=args.chunk_overlap, ingestion_batch_size=args.batch_size)