import os
import json
import yaml

from openai import OpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
# import anthropic
from openai.types.chat import ChatCompletionMessage

with open('./config.yaml', 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    openai_config = config['openai']
    claude_config = config['claude']
    

OPENAI_BASE_URL = openai_config['base_url']
OPENAI_API_KEY = openai_config['api_key']
OPENAI_EMBEDDING_MODEL = openai_config['embedding_model']
OPENAI_EMBEDDING_BASE_URL = openai_config['embedding_base_url']
OPENAI_EMBEDDING_API_KEY = openai_config['embedding_api_key']
OPENAI_COMPLETION_MODEL = openai_config['completion_model']

CLAUDE_API_KEY = claude_config['api_key']
CLAUDE_BASE_URL = claude_config['base_url']
CLAUDE_COMPLETION_MODEL = claude_config['completion_model']

USE_COMFYBENCH_WORKFLOW = config['use_comfybench_workflow']

workspace = "workflow_comfybench" if USE_COMFYBENCH_WORKFLOW else "workflow"

def retrieve_references(requirement, model='OpenAI', count=3):
    with open(f'./dataset/{workspace}/meta.json') as meta_file:
        metadata = json.load(meta_file)

    documents = []
    for name, path in metadata.items():
        with open(path['description'], 'r') as desc_file:
            desc = desc_file.read()
        meta = {
            'name': name,
            'description': path['description'],
            workspace: path[workspace],
            'code': path['code'],
            'markdown': path['markdown']
        }
        documents.append(Document(desc, metadata=meta))

    if model == 'OpenAI':
        embedding = OpenAIEmbeddings(
            model=OPENAI_EMBEDDING_MODEL,
            base_url=OPENAI_EMBEDDING_BASE_URL,
            api_key=OPENAI_EMBEDDING_API_KEY
        )
    elif model == 'BERT':
        embedding = HuggingFaceEmbeddings(
            model_name='google-bert/bert-base-uncased'
        )
    elif model == 'MPNet':
        embedding = HuggingFaceEmbeddings(
            model_name='facebook/m2m100_418M'
        )
    elif model == 'BAAI':
        embedding = HuggingFaceEmbeddings(
            model_name='BAAI/bge-large-en'
        )
    elif model == 'Qwen':
        embedding = HuggingFaceEmbeddings(
            model_name='Alibaba-NLP/gte-Qwen2-1.5B-instruct'
        )

    database = f'./dataset/{workspace}/db/{model}'
    if os.path.exists(database):
        vectors = Chroma(
            embedding_function=embedding,
            persist_directory=database
        )
    else:
        vectors = Chroma.from_documents(
            documents=documents,
            embedding=embedding,
            persist_directory=database
        )

    retriever = vectors.as_retriever(
        search_kwargs={"k": count}
    )

    references = retriever.invoke(requirement)
    return references


def invoke_completion(message):
    client = OpenAI(
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY
    )

    try:
        response = client.chat.completions.create(
            model=OPENAI_COMPLETION_MODEL,
            messages=[{
                'role': 'user',
                'content': message
            }]
        )
        # answer = response.choices[0].message
        answer = response.choices[0].message
        print(answer) 
        usage = response.usage

    except Exception as error:
        answer = ChatCompletionMessage(content=f'Error: {error}', role='assistant')
        usage = None

    return answer, usage

def invoke_completion_claude(message):
    # client = anthropic.Anthropic(
    #     api_key=ANTHROPIC_API_KEY,
    # )
    client = OpenAI(
        base_url=CLAUDE_BASE_URL,
        api_key=CLAUDE_API_KEY
    )
    try:
        # response = client.messages.create(
        #     model=ANTHROPIC_COMPLETION_MODEL,
        #     messages=[{
        #         "role": "user", 
        #         "content": message
        #     }]
        # )
        response = client.chat.completions.create(
            model=CLAUDE_COMPLETION_MODEL,
            messages=[{
                'role': 'user',
                'content': message
            }]
        )
    
        # answer = response.content[0].text
        answer = response.choices[0].message
        usage = response.usage

    except Exception as error:
        answer = ChatCompletionMessage(content=f'Error: {error}', role='assistant')
        usage = None

    return answer, usage

def invoke_vision(message: any) -> tuple[str, any]:
    client = OpenAI(
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY
    )

    try:
        response = client.chat.completions.create(
            model=OPENAI_COMPLETION_MODEL,
            messages=message,
            # temperature=0.5
        )
        answer = response.choices[0].message.content
        usage = response.usage

    except Exception as error:
        answer = f'Error: {error}'
        usage = None

    return answer, usage