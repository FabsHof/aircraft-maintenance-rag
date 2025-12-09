from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever
from dotenv import load_dotenv
import os
import argparse
from src.util.logging import log_info

load_dotenv()

def get_retriever(search_type: str, chroma_persist_directory: str, collection_name: str, k: int = 5) -> VectorStoreRetriever:
    '''
    Create and return a Chroma retriever based on the specified search type.
    arguments:
        search_type: Type of search to perform ('similarity', 'mmr', or 'hybrid').
        chroma_persist_directory: Directory where Chroma database is persisted.
        collection_name: Name of the Chroma collection to use.
        k: Number of documents to retrieve.
    returns: Configured Chroma retriever.
    '''
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'}
    )
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=chroma_persist_directory, 
        embedding_function=embeddings
    )

    return vectorstore.as_retriever(search_type=search_type, search_kwargs={'k': k})

def main(search_type: str, chroma_persist_directory: str, collection_name: str) -> None:
    retriever = get_retriever(search_type, chroma_persist_directory, collection_name)
    docs = retriever.invoke('What is A320 hydraulic pressure?')
    for i, doc in enumerate(docs):
        log_info(f'Document {i+1}:\n{doc.metadata.get("source", "Unknown")}:\n{doc.page_content[:100]}\n')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a Chroma retriever.')
    parser.add_argument('--search_type', type=str, default='similarity', help='Type of search to perform (similarity, mmr, hybrid)')
    parser.add_argument('--chroma_persist_directory', type=str, default='chroma_db', help='Directory where Chroma database is persisted')
    parser.add_argument('--collection_name', type=str, default='airbus_a320_1.0_collection', help='Name of the Chroma collection to use (<project>_<version>_collection)')
    args = parser.parse_args()

    main(args.search_type, args.chroma_persist_directory, args.collection_name)