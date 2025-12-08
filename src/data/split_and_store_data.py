import os
from os import path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders.parsers import LLMImageBlobParser
from langchain_core.vectorstores import VectorStore
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters.base import TextSplitter
from collections.abc import Iterable
from langchain_core.documents import Document
from src.util.logging import log_info
import argparse

# Load environment variables
load_dotenv()

def extract_and_split_documents(file_path: str, text_splitter: TextSplitter, visual_model: ChatOpenAI) -> list[Document]:
    '''
    Extract and split documents from a PDF file using PyMuPDFLoader with image extraction.
    Uses variant 6 configuration: extract_images=True, load_lazy=False, batch_ingestion=True, manual_ids=False
    arguments:
        file_path: Path to the PDF file.
        text_splitter: An instance of TextSplitter to split the documents.
        visual_model: A visual model for parsing images.
    returns:
        A list of split documents.
    '''
    loader = PyMuPDFLoader(
        file_path=file_path,
        mode='page',
        images_inner_format='text',
        extract_tables='markdown',
        extract_images=True,
        images_parser=LLMImageBlobParser(model=visual_model)
    )
    
    # Load documents (not lazy loading for better performance)
    documents = loader.load()
    split_documents = text_splitter.split_documents(documents)
    
    return split_documents

def generate_document_ids(documents: list[Document]) -> list[str]:
    '''
    Generate unique IDs for documents based on their metadata.
    arguments:
        documents: List of documents to generate IDs for.
    returns:
        A list of unique document IDs.
    '''
    ids = []
    idx = 0
    last_page = -1
    
    if not documents:
        return ids
    
    file_name = documents[0].metadata.get('file_path', '').split('/')[-1]
    title = documents[0].metadata.get('title', file_name.replace('.pdf', ''))
    
    for doc in documents:
        page = doc.metadata.get('page', 0)
        if page != last_page:
            idx = 0
            last_page = page
        else:
            idx += 1
        doc_id = f'{title}_page{page}_chunk{idx}'
        ids.append(doc_id)
    
    return ids

def ingest_documents_in_batches(documents: list[Document], vectorstore: VectorStore, batch_size: int = 5000) -> None:
    '''
    Ingest documents into the vector store in batches.
    arguments:
        documents: List of documents to ingest.
        vectorstore: An instance of VectorStore to add documents to.
        batch_size: Number of documents to process in each batch.
    '''
    total_docs = len(documents)
    
    for i in range(0, total_docs, batch_size):
        batch_docs = documents[i:i+batch_size]
        vectorstore.add_documents(batch_docs)
        log_info(f'Ingested batch {i // batch_size + 1}: {len(batch_docs)} documents')

def ingest_raw_data(files: list, text_splitter: TextSplitter, vectorstore: VectorStore, visual_model: ChatOpenAI, dry_run: bool = False) -> None:
    '''
    Ingest raw data files into the vector store using optimized batch processing.
    arguments:
        files: List of file paths to ingest.
        text_splitter: An instance of TextSplitter to split the documents.
        vectorstore: An instance of VectorStore to add documents to.
        visual_model: A visual model for parsing images.
        dry_run: If True, do not actually add documents to the vector store.
    '''
    batch_size = int(os.getenv('BATCH_SIZE', 5000))
    
    for file in files:
        log_info(f'Ingesting file: {file}')
        split_documents = extract_and_split_documents(file, text_splitter, visual_model)
        
        if not dry_run:
            ingest_documents_in_batches(split_documents, vectorstore, batch_size)
            log_info(f'Successfully ingested {file} with {len(split_documents)} documents.')
        else:
            log_info(f'Dry run enabled. Skipping ingestion for {file}. Found {len(split_documents)} documents.')
    
    log_info('Ingestion process completed successfully.')

def main(clear_db: bool = False, dry_run: bool = False) -> None:
    clean_dir = path.join('data', 'clean')

    lm_studio_url = os.getenv('LM_STUDIO_URL', 'http://localhost:1234/v1')
    visual_model_name = os.getenv('VISUAL_MODEL_NAME', 'google/gemma-3-4b')
    
    model_name = os.getenv('EMBEDDING_MODEL_NAME', 'sentence-transformers/all-MiniLM-L6-v2')
    device = os.getenv('EMBEDDING_DEVICE', 'cpu')

    chunk_size = int(os.getenv('CHUNK_SIZE', '150'))
    chunk_overlap = int(os.getenv('CHUNK_OVERLAP', '30'))
    
    if not path.exists(clean_dir):
        log_info(f'Clean data directory {clean_dir} does not exist. Please run the data processing step first.')
        return
    
    files = [path.join(clean_dir, f) for f in os.listdir(clean_dir) if f.endswith('.pdf')]
    if not files:
        log_info(f'No PDF files found in {clean_dir} to ingest.')
        return
    
    # Initialize models and components
    log_info('Initializing models and components...')
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device}
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=['\n\n', '\n', '(?<=\. )', ' ', '']
    )
    visual_model = ChatOpenAI(
        model=visual_model_name,
        temperature=0.0,
        base_url=lm_studio_url,
        api_key=''
    )
    vectorstore = Chroma(
        collection_name='aircraft_maintenance_docs',
        persist_directory='chroma_db',
        embedding_function=embedding_model
    )
    
    if clear_db:
        log_info('Clearing existing vector store database...')
        vectorstore.reset_collection()
        log_info('Existing vector store database cleared.')
    
    ingest_raw_data(files, text_splitter, vectorstore, visual_model, dry_run=dry_run)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ingest clean data into vector store.')
    parser.add_argument('--clear_db', '-c', action=argparse.BooleanOptionalAction, help='Clear existing vector store before ingestion.')
    parser.add_argument('--dry_run', '-d', action=argparse.BooleanOptionalAction, help='Perform a dry run without actual ingestion.')
    args = parser.parse_args()
    
    main(clear_db=args.clear_db, dry_run=args.dry_run)