import chromadb
from embeddings import SFRMistralEmbeddingFunction
from server_utils import create_docs_from_results
from chromadb.utils import embedding_functions
import json
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import TokenTextSplitter

text_splitter = TokenTextSplitter( # Create an instance of the CharacterTextSplitter class (https://python.langchain.com/docs/modules/data_connection/document_transformers/split_by_token#tiktoken)
    chunk_size=1000, chunk_overlap=200
) # This function splits text into chunks of 2000 characters, with an overlap of 200 characters

text_splitter_2 = TokenTextSplitter( # Create an instance of the CharacterTextSplitter class (https://python.langchain.com/docs/modules/data_connection/document_transformers/split_by_token#tiktoken)
    chunk_size=300, chunk_overlap=50
) # This function splits text into chunks of 2000 characters, with an overlap of 200 characters

sentence_transformer_ef2 = embedding_functions.SentenceTransformerEmbeddingFunction( # Create an instance of the SentenceTransformerEmbeddingFunction class (https://docs.trychroma.com/embeddings)
model_name="intfloat/multilingual-e5-large",
device="cuda", # Specify the device to use, in this case, GPU
normalize_embeddings=True # Specify whether to normalize the embeddings, meaning that the embeddings will be scaled to have a norm of 1
)

persist_directory = ("vectordbs/") # Directory where the vector database is stored
chromadb_client = chromadb.PersistentClient(path=persist_directory) # Create a ChromaDB client in that directory

with open('documents.json', 'r') as f:
    documents = json.load(f)

try:
    chromadb_client.delete_collection(name="LightGPT_e5_multilingual_split")
except:
    pass

new_vectordb = chromadb_client.create_collection(name="LightGPT_e5_multilingual_split", embedding_function=sentence_transformer_ef2)

for doc in documents:
    current_id = new_vectordb.count() + 1
    text = doc["document"]
    metadata = doc["metadata"]
    doc_id = doc["id"]

    split_document = text_splitter.split_text(text)
    for child_document in split_document:
        metadata["parent_text"] = child_document
        child_texts = text_splitter_2.split_text(child_document)
    
        for child_text in child_texts:
            new_vectordb.add(ids=[str(current_id)] , documents=[child_text], metadatas=[metadata])
            print(f"Added document {doc_id} to the new vector database")
            current_id += 1



