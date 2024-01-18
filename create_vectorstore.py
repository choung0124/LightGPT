import json
from uuid import uuid4
import chromadb
from chromadb.utils import embedding_functions
import argparse
import os

from server_utils import load_and_split_document

articles_with_pdf_files_directory = "pdfs/"
persist_directory = "vectordbs/"
chromadb_client = chromadb.PersistentClient(path=persist_directory)

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="model9/",
    device="cuda",
    normalize_embeddings=True
)

vectordb = chromadb_client.create_collection(name="LightGPT",
                                             embedding_function=sentence_transformer_ef,
                                             get_or_create=True)

id_counter = 0
for file in os.listdir(articles_with_pdf_files_directory):
    if file.endswith(".pdf"):
        file_path = os.path.join(articles_with_pdf_files_directory, file)

        # Load and split the document
        try:
            texts = load_and_split_document(str(file_path))
        except Exception as e:
            print(f"Failed to load or split document {file}: {e}")
            continue

        # Get article name and link once per PDF
        print("file name: " + file)

        for text in texts:
            text_str = text.page_content
            page_number = text.metadata["page"]
            if text_str.strip():  # Check if text is not just whitespace
                id_counter += 1
                vectordb.add(ids=[str(id_counter)], documents=[text_str], metadatas=[{"name": file, "page": page_number}])

print("All documents processed and added to the database.")
