from PyPDF2 import PdfReader
import json
from uuid import uuid4
import chromadb
from chromadb.utils import embedding_functions
import argparse
from langchain.document_loaders import PyPDFLoader

persist_directory = ("vectordbs/")
chromadb_client = chromadb.PersistentClient(path=persist_directory)

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
model_name="BAAI/bge-large-en-v1.5",
device="cuda",
normalize_embeddings=True
)


prompt_template_2 = """<s> [INST] Below is a question, followed by a list of context that may be related to the question. WITHOUT relying on your own knowledge, give a detailed answer, only using information from the context below. When presenting your answer, strictly follow this format:
Answer: Your Answer goes here
Question: {question}
Context:
{context} 
Previous Chats:
{previous_chats_str} [/INST]"""

prompt_template_openai = """Below is a question, followed by a list of context that may be related to the question. WITHOUT relying on your own knowledge, give a detailed answer, only using information from the context below. 
Question: {question}
Context:
{context} 
Previous Chats:
{previous_chats_str}"""


prompt_template_openai_fusion = """Below is a question, generate 4 related queries. Follow this guide: Identify the question's main topic, then rephrase it using synonyms, alternative sentence structures, and related terms, ensuring each version maintains the original intent.
Response in JSON format:
{{
    "Query 1": "Your Query 1 goes here",
    "Query 2": "Your Query 2 goes here",
    "Query 3": "Your Query 3 goes here",
    "Query 4": "Your Query 4 goes here"
}}
Question: {question}"""

'''
prompt_template = """USER: Below is a question, followed by a list of context that may be related to the question. WITHOUT relying on your own knowledge, give a detailed answer, only using information from the context below. When presenting your answer, strictly follow this format:
Answer: Your Answer goes here
Question: {question}
Context:
{context}
ASSISTANT:"""

prompt_template = """<|im_start|>system
Below is a question, followed by a list of context that may be related to the question. WITHOUT relying on your own knowledge, give a detailed answer, only using information from the context below.<|im_end|>
<|im_start|>user
Below is a question, followed by a list of context that may be related to the question. WITHOUT relying on your own knowledge, give a detailed answer, only using information from the context below. When presenting your answer, strictly follow this format:
Answer: Your Answer goes here
Question: {question}
Context:
{context}<|im_end|>
<|im_start|>assistant

prompt_template_1 = """<s> [INST] Below is a question, followed by a list of references that may be related to the question. DO NOT answer the question, instead choose ONLY one reference that help answer the question. When presenting your selected references, strictly follow this format:
[
    {{ "Reference 1": "Reference ID" }},
    {{ "Reference 2": "Reference ID" }},
    {{ "Reference n": "Reference ID" }}
]
/List

You MUST use /List to indicate the end of your list of references.
For example, if you think a reference is relevant, your response should look like this. Do not include anything else in your response:
[
    {{ "Reference": "Reference ID" }},
]
/List

If there are no references that help answer the question, your response should look like this. Do not include anything else in your response:
[
    {{ }}
]
/List

Question: {question}
Context
{context} [/INST]"""

"""
'''

prompt_template_1 = """<s> [INST] Below is a question, followed by a list of references that may be related to the question. DO NOT answer the question, instead choose ONLY one or two reference that help answer the question. When presenting your selected references, strictly follow this format:
[
    {{ "Reference 1": "Reference ID" }},
    {{ "Reference 2": "Reference ID" }},
    {{ "Reference n": "Reference ID" }}
]
/List

You MUST use /List to indicate the end of your list of references.
For example, if you think Reference 1 and Reference 3 are relevant, your response should look like this. Do not include anything else in your response:
[
    {{ "Reference 1": "Reference ID" }},
    {{ "Reference 3": "Reference ID" }}
]
/List
If there are no references that help answer the question, your response should look like this. Do not include anything else in your response:
[
    {{ }}
]
/List

Question: {question}
Context
{context} [/INST]"""

'''
prompt_template_selection = """<|im_start|>system
Below is a question, followed by a list of contexts that may be related to the question. DO NOT answer the question, instead choose all the contexts that help answer the question.<|im_end|>
<|im_start|>user
Below is a question, followed by a list of contexts that may be related to the question. DO NOT answer the question, instead choose all the contexts that help answer the question. When presenting your selected contexts, strictly follow this format:
[
    {{ "Reference 1": "Reference ID" }},
    {{ "Reference 2": "Reference ID" }},
    {{ "Reference n": "Reference ID" }}
]

For example, if you think Reference 1 and Reference 3 are relevant, your response should look like this, and nothing else:
[
    {{ "Reference 1": "Reference ID" }},
    {{ "Reference 3": "Reference ID" }}
]

Question: {question}
Context
{context}<|im_end|>
<|im_start|>assistant
"""


prompt_template = """<|im_start|>system
Below is a question, followed by a list of context that may be related to the question. WITHOUT relying on your own knowledge, give a detailed answer, only using information from the context below. You are not required to use all of the context, use the contexts that will help you answer the question.<|im_end|>
<|im_start|>user
Below is a question, followed by a list of context that may be related to the question. WITHOUT relying on your own knowledge, give a detailed answer, only using information from the context below. You are not required to use all of the context, use the contexts that will help you answer the question. When presenting your answer, strictly follow this format:
Answer: Your Answer goes here [EoA]
Context used in answer:
[
    {{ "Reference 1": {{
        "Name": "Context Name"
    }} }},
    {{ "Reference 2": {{
        "Name": "Context Name"
    }} }},
    {{ "Reference n": {{
        "Name": "Context Name"
    }} }}
]
Question: {question}
Context
{context}<|im_end|>
<|im_start|>assistant
"""



prompt_template = """SYSTEM: Below is a question, followed by a list of context that may be related to the question. WITHOUT relying on your own knowledge, give a detailed answer, only using information from the context below. You are not required to use all of the context, use the contexts that will help you answer the question.
USER: Below is a question, followed by a list of context that may be related to the question. WITHOUT relying on your own knowledge, give a detailed answer, only using information from the context below. You are not required to use all of the context, use the contexts that will help you answer the question. When presenting your answer, strictly follow this format:
Answer: Your Answer goes here [EoA]
Context used in answer:
[
    {{ "Reference 1": {{
        "Name": "Context Name"
    }} }},
    {{ "Reference 2": {{
        "Name": "Context Name"
    }} }},
    {{ "Reference n": {{
        "Name": "Context Name"
    }} }}
]
Question: {question}
Context
{context}
ASSISTANT: """
'''

def create_docs_from_results(results):
    # Flatten the lists
    documents = [doc for sublist in results["documents"] for doc in sublist]
    metadatas = [meta for sublist in results["metadatas"] for meta in sublist]
    distances = [dist for sublist in results["distances"] for dist in sublist]
    ids = [id for sublist in results["ids"] for id in sublist]

    # Create the list of dictionaries
    docs = []
    for document, metadata, distance, id in zip(documents, metadatas, distances, ids):
        doc = {
            "document": document,
            "metadata": metadata,
            "distance": distance,
            "id": id
        }
        docs.append(doc)

    return docs


def format_context(results):
    # Flatten the lists
    documents = [doc for sublist in results["documents"] for doc in sublist]
    metadatas = [meta for sublist in results["metadatas"] for meta in sublist]
    ids = [id for sublist in results["ids"] for id in sublist]

    # Create the list of formatted context strings
    contexts = []
    count = 0
    for document, metadata, id in zip(documents, metadatas, ids):
        count += 1
        context_str = f"Reference {count}:\nReference ID: {id}\n{document}\nFrom: {metadata['name']}\n"
        contexts.append(context_str)

    # Join the contexts into one string with separators
    formatted_context = "\n".join(contexts)

    return formatted_context

def filter_and_format_context(results):
    # Flatten the lists
    documents = [doc for sublist in results["documents"] for doc in sublist]
    metadatas = [meta for sublist in results["metadatas"] for meta in sublist]
    ids = [id for sublist in results["ids"] for id in sublist]

    # Create the list of formatted context strings
    contexts = []
    relevant_pages = []
    count = 0
    for document, metadata, id in zip(documents, metadatas, ids):
        count += 1
        context_str = f"{document}\nFrom: {metadata['name']}\n"
        name = metadata['name']
        page_number = metadata['page']
        pdf_id = metadata['pdf_id']
        relevant_pages.append({"name": name, "pdf_id": pdf_id, "page": page_number})
        contexts.append(context_str)

    # Join the contexts into one string with separators
    formatted_context = "\n".join(contexts)

    return formatted_context, relevant_pages

def filter_and_format_context_fusion(docs):
    # Create the list of formatted context strings
    contexts = []
    relevant_pages = []
    count = 0
    for doc in docs:
        document = doc["document"]
        metadata = doc["metadata"]

        context_str = f"{document}\nFrom: {metadata['name']}\n"
        name = metadata['name']
        page_number = metadata['page']
        pdf_id = metadata['pdf_id']
        relevant_pages.append({"name": name, "pdf_id": pdf_id, "page": page_number})
        contexts.append(context_str)

    # Join the contexts into one string with separators
    formatted_context = "\n".join(contexts)

    return formatted_context, relevant_pages



def load_and_split_document(file_path):
    # Load document if file path is provided
    if file_path is not None:
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        for page in pages:
            print(page)
            print("-----")
    return pages


def add_to_vectordb(texts, name, pdf_id, page_number):
    vectordb = chromadb_client.get_collection(name="LightGPT_BGE", embedding_function=sentence_transformer_ef)

    count = vectordb.count()

    for text in texts:
        print(text)
        if text != "":
            count += 1
            vectordb.add(ids=[str(count)], documents=[text], metadatas=[{"name": name, "pdf_id": pdf_id, "page": page_number}])

    print("Added {} documents to the vectordb".format(len(texts)))

    return "success"
