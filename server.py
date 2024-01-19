# Import necessary modules and functions
import uvicorn
from fastapi import FastAPI, UploadFile, File, WebSocket, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from uuid import uuid4
from server_utils import (
    load_and_split_document,
    add_to_vectordb,
    prompt_template,
    prompt_template_selection,
    create_docs_from_results,
    create_docs_from_results_formatted,
    filtered_create_docs_from_results_formatted
)
from pathlib import Path
import shutil
from langchain.text_splitter import CharacterTextSplitter
import gc
from chromadb.utils import embedding_functions
import chromadb
from fastapi.responses import StreamingResponse
import GPUtil
import random
import asyncio
import websockets
import json
from urllib.parse import urlencode
import httpx
import aiohttp
from fastapi.responses import FileResponse
from fastapi import HTTPException
import os
import fitz  # PyMuPDF
from fastapi.staticfiles import StaticFiles
from os import listdir
from os.path import isfile, join

### FastAPI ###############################################################################################################

# Initialize FastAPI app
# Fast API 이란? https://velog.io/@cho876/%EC%9A%94%EC%A6%98-%EB%9C%A8%EA%B3%A0%EC%9E%88%EB%8B%A4%EB%8A%94-FastAPI
app = FastAPI()

# Set CORS (Cross-Origin Resource Sharing) policy 
# CORS 이란? https://inpa.tistory.com/entry/WEB-%F0%9F%93%9A-CORS-%F0%9F%92%AF-%EC%A0%95%EB%A6%AC-%ED%95%B4%EA%B2%B0-%EB%B0%A9%EB%B2%95-%F0%9F%91%8F
# FastAPI CORS 설정: https://fastapi.tiangolo.com/ko/tutorial/cors/
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount a directory to serve static files (for thumbnails)
# FastAPI Static Files: https://fastapi.tiangolo.com/tutorial/static-files/
app.mount("/thumbnails", StaticFiles(directory="thumbnails"), name="thumbnails")

### ChromaDB ##############################################################################################################

# ChromaDB 한글: https://opentutorials.org/module/6369/32376
persist_directory = ("vectordbs/") # Directory where the vector database is stored
chromadb_client = chromadb.PersistentClient(path=persist_directory) # Create a ChromaDB client in that directory

# Create an embedding function(this function will be used to convert text to vectors)
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction( # Create an instance of the SentenceTransformerEmbeddingFunction class (https://docs.trychroma.com/embeddings)
model_name="model9/", # Specify the name of the SentenceTransformer model, in this case, model9 which is locally stored
device="cuda", # Specify the device to use, in this case, GPU
normalize_embeddings=True # Specify whether to normalize the embeddings, meaning that the embeddings will be scaled to have a norm of 1
) 

# Load the vector database, and save it to the variable vectordb
vectordb = chromadb_client.get_collection(name="LightGPT", embedding_function=sentence_transformer_ef)

### Langchain #############################################################################################################

text_splitter = CharacterTextSplitter.from_tiktoken_encoder( # Create an instance of the CharacterTextSplitter class (https://python.langchain.com/docs/modules/data_connection/document_transformers/split_by_token#tiktoken)
    chunk_size=2000, chunk_overlap=200
) # This function splits text into chunks of 2000 characters, with an overlap of 200 characters

############################################################################################################################

prompts = {} # Define a dictionary to store prompts

### Uploading PDF Enpoint #################################################################################################

# Define a Pydantic model for PDF upload
# Pydantic 이란? https://lsjsj92.tistory.com/650
class PDF(BaseModel):
    file: UploadFile = File(...) 
    name: str
    link: str
    
# Endpoint for uploading PDF files
@app.post("/upload_pdf/")
async def upload_pdf(pdf: PDF): 
    # Get the file from the request
    file = pdf.file
    # Define the directory where you want to save the file
    directory = Path("pdfs/")

    # Create the directory if it doesn't exist
    directory.mkdir(parents=True, exist_ok=True)

    # Create the full path to the file
    file_path = directory / file.filename

    # Save the uploaded file to the directory
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer) 

    texts = load_and_split_document(str(file_path)) # Load and split the document

    result = add_to_vectordb(texts, pdf.name, pdf.link)

    if result == "success":
        del vectordb
        gc.collect()
        vectordb = chromadb_client.get_collection(name="LightGPT", embedding_function=sentence_transformer_ef)

    return result
    
### Uploading Text Enpoint ################################################################################################

class Text(BaseModel):
    string: str
    name: str
    link: str

@app.post("/upload_text/")
async def upload_text(text: Text):
    texts = text_splitter.split_text(text.string)
    result = add_to_vectordb(texts, text.name, text.link)

    if result == "success":
        del vectordb
        gc.collect()
        vectordb = chromadb_client.get_collection(name="LightGPT", embedding_function=sentence_transformer_ef)

    return result

class Question(BaseModel):
    question: str

@app.post("/ask/")
async def ask(question: Question, request: Request):
    context = vectordb.query(query_texts=[question.question], n_results=7)
    formatted_context = create_docs_from_results_formatted(context)
    prompt_id = str(uuid4())
    prompt = prompt_template_selection.format(context=formatted_context, question=question.question)
    
    prompts[prompt_id] = {
        "prompt": prompt,
        "context": context,
        "question": question.question
    }

    FirstResult = stream_full_no_yield(prompt_id)
    try:
        response = await asyncio.wait_for(FirstResult, timeout=100)
    except asyncio.TimeoutError:
        response = "No results found"
    except Exception as e:
        response = "No results found"
        print(e)

    if response != "No results found" and response != "":
        try:
            start_index = response.find('[')
            end_index = response.rfind(']') + 1  # Find the last ']' and include it

            # Extract the JSON list string
            json_list_str = response[start_index:end_index]

            # Parse the JSON list
            relevant_contexts = json.loads(json_list_str)
            reference_ids = []
            for context_dict in relevant_contexts:
                for _, reference_id in context_dict.items():
                    reference_ids.append(reference_id)

        except Exception as e:
            print(f"Error processing JSON: {e}")
            reference_ids = []

    prompt_id_2 = str(uuid4())
    new_formatted_context, relevant_pages = filtered_create_docs_from_results_formatted(context, reference_ids)

    print("relevant_pages: " + str(relevant_pages))

    prompt2 = prompt_template.format(context=new_formatted_context, question=question.question)

    prompts[prompt_id_2] = {
        "prompt": prompt2,
        "context": new_formatted_context,
        "question": question.question,
        "relevant_pages": relevant_pages
    }

    url = str(request.url_for("streaming_response", prompt_id=prompt_id_2))
    return {
            "url": url,
            "relevant_pages": relevant_pages,
            }

class PDFName(BaseModel):
    name: str

# one pdf at a time
@app.post("/get_pdf/")
async def get_pdf(pdf_name: PDFName):
    pdf_path = f"pdfs/{pdf_name.name}"  # Ensure the file extension is correct
    try:
        response = FileResponse(path=pdf_path, filename=pdf_name.name)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"File not found: {pdf_name.name}")

    return response

def generate_thumbnail(pdf_path):
    thumbnail_path = "thumbnails/" + pdf_path.split('/')[-1].replace('.pdf', '.png')

    # Check if thumbnail already exists
    if not isfile(thumbnail_path):
        try:
            doc = fitz.open(pdf_path)
            if doc.page_count > 0:
                page = doc.load_page(0)  # first page
                pix = page.get_pixmap()
                pix.save(thumbnail_path)
            else:
                print(f"Document is empty: {pdf_path}")
                return None
        except Exception as e:
            print(f"Error generating thumbnail for {pdf_path}: {e}")
            return None

    return thumbnail_path


@app.get("/list_pdfs/")
async def list_pdfs():
    pdf_directory = "pdfs/"
    pdf_dicts = []
    pdf_files = [f for f in listdir(pdf_directory) if isfile(join(pdf_directory, f)) and f.lower().endswith('.pdf')]

    for file in pdf_files:
        pdf_path = join(pdf_directory, file)
        thumbnail_path = generate_thumbnail(pdf_path)
        if thumbnail_path:
            # Create a URL for the thumbnail
            thumbnail_url = f"/thumbnails/{thumbnail_path.split('/')[-1]}"
            pdf_dicts.append({"name": file, "image": thumbnail_url})

    return pdf_dicts

async def stream_generator(prompt_id: str):
    if prompt_id not in prompts:
        yield json.dumps({"error": "Invalid prompt ID"})
        return

    prompt = prompts[prompt_id]["prompt"]
    x_api_key = "67d8ece657b29e5219190ee2ba7eb2db"
    headers = {"x-api-key": x_api_key, "Content-Type": "application/json"}

    request_data = {
        "model": "brucethemoose_Yi-34B-200K-DARE-merge-v7-4bpw-exl2-fiction",
        "prompt": prompt,
        "max_tokens": 2048,
        "stream": True,
        "temperature": 1.3,
        "min_p": 0.02,
        "token_repetition_penalty": 1.25,
    }

    url = "http://192.168.100.131:5000/v1/completions"
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=request_data) as response:
            async for chunk in response.content.iter_any():
                if chunk.startswith(b'data: '):
                    chunk = chunk[6:]  # Remove 'data: ' prefix
                    try:
                        chunk_data = json.loads(chunk.decode("utf-8"))
                        text = chunk_data["choices"][0]["text"]
                        yield text
                    except json.JSONDecodeError:
                        # Handle possible JSON decoding error
                        continue


async def stream_full_no_yield(prompt_id: str):
    if prompt_id not in prompts:
        return json.dumps({"error": "Invalid prompt ID"})

    prompt = prompts[prompt_id]["prompt"]
    x_api_key = "67d8ece657b29e5219190ee2ba7eb2db"
    headers = {"x-api-key": x_api_key, "Content-Type": "application/json"}

    request_data = {
        "model": "brucethemoose_Yi-34B-200K-DARE-merge-v7-4bpw-exl2-fiction",
        "prompt": prompt,
        "max_tokens": 128,
        "stream": True,
        "temperature": 1.3,
        "min_p": 0.03,
        "token_repetition_penalty": 1.7,
        "stop": ["/List"]
    }

    url = "http://192.168.100.131:5005/v1/completions"
    full_text = ""
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=request_data) as response:
            async for chunk in response.content.iter_any():
                if chunk.startswith(b'data: '):
                    chunk = chunk[6:]
                    try:
                        chunk_data = json.loads(chunk.decode("utf-8"))
                        text = chunk_data["choices"][0]["text"]
                        print(text, end="")
                        full_text += text
                    except json.JSONDecodeError:
                        continue
    return full_text



@app.get("/streaming_response/{prompt_id}")
async def streaming_response(prompt_id: str):
    return StreamingResponse(stream_generator(prompt_id), media_type="application/json")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)