from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from src.retrieval.basic_retriever import get_retriever
from langchain_core.vectorstores import VectorStoreRetriever
from src.util.logging import log_info
from dotenv import load_dotenv
import os
import argparse

load_dotenv()

def format_docs(docs) -> str:
    '''Format retrieved documents for inclusion in the prompt.'''
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', 'Unknown')
        content_snippet = doc.page_content[:200].replace('\n', ' ')
        formatted.append(f'Document {i+1} (Source: {source}):\n{content_snippet}...')
    return "\n\n".join(formatted)

def get_basic_chain(retriever: VectorStoreRetriever, llm: ChatOpenAI, template: str):
    prompt = ChatPromptTemplate.from_template(template)
    chain = {'documents': retriever, 'question': RunnablePassthrough()} | prompt | llm | StrOutputParser()
    return chain


def main(lm_studio_url: str, queries: list[str], model_name: str, search_type: str, chroma_persist_directory: str, collection_name: str) -> None:
    llm = ChatOpenAI(
        model=model_name,
        temperature=0.1,
        base_url=lm_studio_url,
        api_key=''
    )
    template = '''
    You are an expert on the Airbus A320-family aircraft maintenance and operations. Answer the questions ONLY using the provided documents. 
    Cite your sources for every answer using the 'numeric-comp' footnote style from LaTex.
    If the answer is not contained in the documents, respond with "I don't have enough information in the provided documents to answer that question."
    
    <documents>
    {documents}
    </documents>

    <question>
    {question}
    </question>

    Answer: 
    '''
    chain = get_basic_chain(get_retriever(search_type, chroma_persist_directory, collection_name), llm, template)

    for query in queries:
        log_info(f'Processing query: {query}\n')
        for chunk in chain.stream(query):
            print(chunk, end='', flush=True)
        print('\n')

if __name__ == '__main__':
    lm_studio_url = os.getenv('LM_STUDIO_URL', 'http://localhost:1234/v1')

    parser = argparse.ArgumentParser(description='Run basic retrieval chain.')
    parser.add_argument('--model_name', type=str, default='google/gemma-3-4b', help='Name of the language model to use')
    parser.add_argument('--queries', type=str, nargs='+', default=['Describe the A320 landing gear maintenance schedule'], help='List of queries to run')
    parser.add_argument('--search_type', type=str, default='similarity', help='Type of search to use in retriever')
    parser.add_argument('--chroma_persist_directory', type=str, default='chroma_db', help='Directory for Chroma persistence')
    parser.add_argument('--collection_name', type=str, default='airbus_a320_1.0_collection', help='Name of the collection to use')
    
    args = parser.parse_args()

    main(lm_studio_url, args.queries, args.model_name, args.search_type, args.chroma_persist_directory, args.collection_name)