from datetime import datetime
import itertools
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.parsers import LLMImageBlobParser
from langchain_huggingface import HuggingFaceEmbeddings
from src.util.logging import log_info
from langchain_community.document_loaders import PyMuPDFLoader

file_path = 'data/raw/airbus_a320/1.0/maintenance/ac_43.13-1b_w-chg1.pdf'
chroma_persist_directory = 'chroma_db'
lm_studio_url = 'http://localhost:1234/v1'

setup_start = datetime.now()
embedding_model = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs={'device': 'cpu'}
)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=30,
    separators=['\n\n', '\n', '(?<=\. )', ' ', '']
)

vectorstore = Chroma(
    collection_name='aircraft_maintenance_docs',
    persist_directory=chroma_persist_directory, 
    embedding_function=embedding_model
)

visual_model = ChatOpenAI(
    model='google/gemma-3-4b',
    temperature=0.0,
    base_url=lm_studio_url,
    api_key=''
)
setup_end = datetime.now()
log_info(f'Setup time: {setup_end - setup_start}')

results = {}

variants = list(itertools.product([True, False], repeat=4))
for i, (extract_images, load_lazy, batch_ingestion, manual_ids) in enumerate(variants):
    log_info(f'\n=============\nStarting variant {i+1}/{len(variants)}: extract_images={extract_images}, load_lazy={load_lazy}, batch_ingestion={batch_ingestion}, manual_ids={manual_ids}')

    results[i] = {}

    vectorstore.reset_collection()  # Ensure collection is reset

    loader = PyMuPDFLoader(
        file_path=file_path, 
        mode='page',
        images_inner_format='text',
        extract_tables='markdown',
        extract_images=extract_images,
        images_parser=LLMImageBlobParser(model=visual_model)
    )

    initial_time = datetime.now()
    if not load_lazy:
        documents = loader.load()
    else:
        documents = loader.lazy_load()
    results[i]['load_time'] = datetime.now() - initial_time
    log_info(f'Document loading time: {results[i]["load_time"]}')

    split_start = datetime.now()
    split_documents = text_splitter.split_documents(documents)
    split_end = datetime.now()
    results[i]['split_time'] = split_end - split_start
    log_info(f'Document splitting time: {results[i]["split_time"]}')


    ingest_start = datetime.now()

    if not manual_ids:
        ids = []
        idx = 0
        last_page = -1
        log_info(f'{split_documents[0].metadata}')
        file_name = split_documents[0].metadata.get('file_path').split('/')[-1]
        title = split_documents[0].metadata.get('title', file_name.replace('.pdf', ''))
        for split_doc in split_documents:
            page = split_doc.metadata.get('page', 0)
            if page != last_page:
                idx = 0
                last_page = page
            else:
                idx += 1
            id = f'{title}_page{page}_chunk{idx}'
            ids.append(id)
    else:
        ids = None

    # Always use batching to respect ChromaDB's max batch size limit
    # The batch_ingestion parameter controls the batch size
    chunksize = 20 if batch_ingestion else 5000  # Use smaller chunks for batch_ingestion=True
    total_docs = len(split_documents)
    for j in range(0, total_docs, chunksize):
        batch_docs = split_documents[j:j+chunksize]
        if ids is not None:
            batch_ids = ids[j:j+chunksize]
        else:
            batch_ids = None
        new_ids = vectorstore.add_documents(batch_docs, ids=batch_ids)

    ingest_end = datetime.now()
    results[i]['ingest_time'] = ingest_end - ingest_start
    log_info(f'Data ingestion time: {results[i]["ingest_time"]}')

    # inference_start = datetime.now()
    # query = 'What is the recommended maintenance schedule for the landing gear?'
    # prompt = f'Based on the following documents, {query}\n\nDocuments: {documents}'
    # llm = ChatOpenAI(
    #     model='google/gemma-3-4b',
    #     temperature=0.0,
    #     base_url=lm_studio_url,
    #     api_key=''
    # )
    # response = llm.invoke(prompt)
    # inference_end = datetime.now()
    # log_info(f'Inference time: {inference_end - inference_start}')
    # log_info(f'QA Response: {response}')

log_info('All variants completed. Summary of results:')
for i, res in results.items():
    log_info(f'Variant {i}: Load Time: {res["load_time"]}, Split Time: {res["split_time"]}, Ingest Time: {res["ingest_time"]}')

# setup_time = datetime.now()

# # Ingest data
# ingest_raw_data([file_path], text_splitter, vectorstore)
# ingestion_time = datetime.now()
# log_info(f'Setup time: {setup_time - initial_time}')
# log_info(f'Ingestion time: {ingestion_time - setup_time}')

# # Perform similarity search and QA
# query = 'What is the recommended maintenance schedule for the equipment?'
# result_docs = vectorstore.similarity_search(query, k=3)
# then = datetime.now()
# log_info(f'Similarity search time: {then - now}')
# log_info(f'Number of results retrieved: {len(result_docs)}')

# llm = ChatOllama(model='llama3.1:8b', temperature=0.5)
# now = datetime.now()
# prompt = f'Based on the following documents, {query}\n\n{result_docs}'
# response = llm.invoke(prompt)
# then = datetime.now()
# log_info(f'QA response time: {then - now}')
# log_info(f'QA Response: {response}')

# final_time = datetime.now()
# log_info(f'Total process time: {final_time - initial_time}')
# log_info('Process completed successfully.')