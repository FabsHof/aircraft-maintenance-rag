import logging
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama

logging.basicConfig(level=logging.INFO)

def get_current_timestamp():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def log_message(message):
    timestamp = get_current_timestamp()
    logging.info(f'[{timestamp}] {message}')

chroma_persist_directory = 'chroma_persist'
file_path = 'data/manuals/airship_pilot_manual.pdf'
loader = PyPDFLoader(file_path)

initial_time = datetime.now()
now = datetime.now()
docs = loader.load()
then = datetime.now()
log_message(f'Document loading time: {then - now}')
log_message(f'Number of documents loaded: {len(docs)}')

now = datetime.now()
embeddings = OllamaEmbeddings(model='llama3.1:8b')
vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=chroma_persist_directory)
then = datetime.now()
log_message(f'Vector store creation time: {then - now}')

now = datetime.now()
query = 'What is the recommended maintenance schedule for the equipment?'
result_docs = vectorstore.similarity_search(query, k=3)
then = datetime.now()
log_message(f'Similarity search time: {then - now}')
log_message(f'Number of results retrieved: {len(result_docs)}')

llm = ChatOllama(model='llama3.1:8b', temperature=0.5)
now = datetime.now()
prompt = f'Based on the following documents, {query}\n\n{result_docs}'
response = llm.invoke(prompt)
then = datetime.now()
log_message(f'QA response time: {then - now}')
log_message(f'QA Response: {response}')

final_time = datetime.now()
log_message(f'Total process time: {final_time - initial_time}')
log_message('Process completed successfully.')