import chromadb
from embeddings import SFRMistralEmbeddingFunction
from server_utils import create_docs_from_results
from chromadb.utils import embedding_functions
import json

sentence_transformer_ef2 = embedding_functions.SentenceTransformerEmbeddingFunction( # Create an instance of the SentenceTransformerEmbeddingFunction class (https://docs.trychroma.com/embeddings)
model_name="intfloat/multilingual-e5-large",
device="cuda", # Specify the device to use, in this case, GPU
normalize_embeddings=True # Specify whether to normalize the embeddings, meaning that the embeddings will be scaled to have a norm of 1
)

import chromadb.utils.embedding_functions as embedding_functions
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key="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",
                model_name="text-embedding-3-large"
            )

persist_directory = ("vectordbs/") # Directory where the vector database is stored
chromadb_client = chromadb.PersistentClient(path=persist_directory) # Create a ChromaDB client in that directory

with open('documents.json', 'r') as f:
    documents = json.load(f)

try:
    chromadb_client.delete_collection(name="LightGPT_openai")
except:
    pass

new_vectordb = chromadb_client.create_collection(name="LightGPT_openai", embedding_function=openai_ef)

for doc in documents:
    metadata = doc["metadata"]
    text = doc["document"]
    doc_id = doc["id"]
    new_vectordb.add(ids=[doc_id], documents=[text], metadatas=[metadata])

    print(f"Added document {doc_id} to the new vector database")

