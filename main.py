from src.retrieval.basic_retriever import get_retriever
from src.retrieval.basic_chain import get_basic_chain
from src.util.logging import log_info
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import argparse

load_dotenv()
# Disable parallelism in tokenizers to avoid warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def main(model_name: str = 'google/gemma-3-4b', temperature: float = 0.1) -> None:
    log_info('✈️  Welcome to the Airbus A320 RAG System!\nAsk your questions about Airbus A320 maintenance and operations.\n(Type "quit" to exit)')
    
    retriever = get_retriever(
        search_type='similarity',
        chroma_persist_directory='chroma_db',
        collection_name='airbus_a320_1.0_collection'
    )
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        base_url=os.getenv('LM_STUDIO_URL', 'http://localhost:1234/v1'),
        api_key=''
    )
    template = '''
    You are an expert on the Airbus A320-family aircraft maintenance and operations. Answer the questions based ONLY on the provided documents.
    Cite at least every paragraph with your sources in the format [1], [2], etc., and provide a bibliography at the end listing the sources. For example, 
    [1]: Show here the field 'title' from document metadata instead of its id.
    If the documents do not contain enough information, respond with "I don't have enough information in the provided documents to answer that question."

    <documents>
    {documents}
    </documents>
    
    <question>
    {question}
    </question>
    '''
    chain = get_basic_chain(retriever, llm, template)
    
    # Interactive loop
    while True:
        query = input('\n📝 Your question (or "quit"): ')
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        print('\n🤖 Assistant: ', end="", flush=True)
        for chunk in chain.stream(query):
            print(chunk, end="", flush=True)
        print('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run A320 RAG System.')
    parser.add_argument('--model_name', type=str, default='google/gemma-3-4b', help='Name of the language model to use')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for the language model')
    args = parser.parse_args()

    main(model_name=args.model_name, temperature=args.temperature)