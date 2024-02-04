import chromadb
#from embeddings import SFRMistralEmbeddingFunction
from server_utils import create_docs_from_results
from chromadb.utils import embedding_functions
import json

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction( # Create an instance of the SentenceTransformerEmbeddingFunction class (https://docs.trychroma.com/embeddings)
model_name="model9/", # Specify the name of the SentenceTransformer model, in this case, model9 which is locally stored
device="cuda", # Specify the device to use, in this case, GPU
normalize_embeddings=True # Specify whether to normalize the embeddings, meaning that the embeddings will be scaled to have a norm of 1
) 


persist_directory = ("vectordbs/") # Directory where the vector database is stored
chromadb_client = chromadb.PersistentClient(path=persist_directory) # Create a ChromaDB client in that directory

vectordb = chromadb_client.get_collection(name="LightGPT", embedding_function=sentence_transformer_ef)

number_of_texts = vectordb.count() + 1000
print(f"Number of texts: {number_of_texts}")
all_documents = vectordb.query(query_texts=["yes"], n_results=number_of_texts)
print(f"Number of documents: {len(all_documents)}")
documents = create_docs_from_results(all_documents)
print(f"Number of documents: {len(documents)}")

# save the documents as a json list
with open('documents.json', 'w') as f:
    json.dump(documents, f)