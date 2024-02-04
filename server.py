# Import necessary modules and functions
import uvicorn
from fastapi import FastAPI, UploadFile, File, WebSocket, Request, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from uuid import uuid4
from server_utils import (
    load_and_split_document,
    add_to_vectordb,
    prompt_template_1,
    prompt_template_2,
    format_context,
    filter_and_format_context,
    create_docs_from_results
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
<<<<<<< HEAD
from embeddings import SFRMistralEmbeddingFunction
from pypdf import PdfReader
=======
>>>>>>> a8aceff6a7ad15ecef67934104177a0f22c5b422

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

<<<<<<< HEAD
chat_persist_directory = ("history/")
chat_client = chromadb.PersistentClient(path=chat_persist_directory)
# Create an embedding function(this function will be used to convert text to vectors)

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction( # Create an instance of the SentenceTransformerEmbeddingFunction class (https://docs.trychroma.com/embeddings)
model_name="BAAI/bge-large-en-v1.5", # Specify the name of the SentenceTransformer model, in this case, model9 which is locally stored
=======
# Create an embedding function(this function will be used to convert text to vectors)
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction( # Create an instance of the SentenceTransformerEmbeddingFunction class (https://docs.trychroma.com/embeddings)
model_name="model9/", # Specify the name of the SentenceTransformer model, in this case, model9 which is locally stored
>>>>>>> a8aceff6a7ad15ecef67934104177a0f22c5b422
device="cuda", # Specify the device to use, in this case, GPU
normalize_embeddings=True # Specify whether to normalize the embeddings, meaning that the embeddings will be scaled to have a norm of 1
) 

# Load the vector database, and save it to the variable vectordb
<<<<<<< HEAD
vectordb = chromadb_client.create_collection(name="LightGPT_BGE", embedding_function=sentence_transformer_ef, get_or_create=True)

### Langchain #############################################################################################################

text_splitter = CharacterTextSplitter.from_tiktoken_encoder( # Create an instance of the CharacterTextSplitter class (https://python.langchain.com/docs/modules/data_connection/document_transformers/split_by_token#tiktoken)
    chunk_size=2000, chunk_overlap=200
) # This function splits text into chunks of 2000 characters, with an overlap of 200 characters

### Initializing ########################################################################################################

# Create a dictionary to store prompts
prompts = {}
questions = {}
last_chats = {}

### Uploading Text Enpoint ################################################################################################
=======
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
>>>>>>> a8aceff6a7ad15ecef67934104177a0f22c5b422

# create image from first page of pdf
def generate_thumbnail(pdf_id, zoom=1):
    pdf_path = f"pdfs/{pdf_id}.pdf"

    pdf_document = fitz.open(pdf_path)
    first_page = pdf_document[0]

<<<<<<< HEAD
    zoom_x = zoom
    zoom_y = zoom
    mat = fitz.Matrix(zoom_x, zoom_y)

    pix = first_page.get_pixmap(matrix=mat)  # Correct method name here
    pix.save(f"thumbnails/{pdf_id}.png")
=======
    # Save the uploaded file to the directory
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer) 

    texts = load_and_split_document(str(file_path)) # Load and split the document
>>>>>>> a8aceff6a7ad15ecef67934104177a0f22c5b422

    return pdf_id

class Page(BaseModel):
    text: str
    pageNumber: int

<<<<<<< HEAD
class Document(BaseModel):
=======
    return result
    
### Uploading Text Enpoint ################################################################################################

class Text(BaseModel):
    string: str
>>>>>>> a8aceff6a7ad15ecef67934104177a0f22c5b422
    name: str
    pages: List[Page]
    pdf_id: str

class Document2(BaseModel):
    name: str
    pages: List[Page]

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    pdf_id = str(uuid4())
    # Saving PDF
    directory = Path("pdfs/")
    directory.mkdir(parents=True, exist_ok=True)
    file_path = directory / f"{pdf_id}.pdf"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    # Generating Thumbnail
    try:
        image_id = generate_thumbnail(pdf_id)
    except Exception as e:
        print(e)
        image_id = ""

    if image_id and image_id != "": 
        return {"pdf_id": pdf_id}
    else:
        return {"pdf_id": pdf_id}

@app.post("/upload_text/")
async def upload_text(document: Document):
    vectordb = chromadb_client.get_collection(name="LightGPT_BGE",
                                                embedding_function=sentence_transformer_ef)
    # Generating Thumbnail
    pdf_id = document.pdf_id

    for page in document.pages:
        split_texts = text_splitter.split_text(page.text)
        add_to_vectordb(split_texts, document.name, pdf_id, page.pageNumber)


    del vectordb
    gc.collect()
    vectordb = chromadb_client.get_collection(name="LightGPT_BGE", 
                                                    embedding_function=sentence_transformer_ef)

    return {"message": "success"}

@app.post("/upload_pdf_auto/")
async def upload_pdf_auto(file: UploadFile = File(...)):
    reader = PdfReader(file.file)
    pages = reader.pages
    meta = reader.metadata

    title = meta["/Title"]
    if not title and title == "":
        title = file.filename

    # construct Document
    document = Document2(name=title, pages=[])
    for page in pages:
        text = page.extract_text()
        pageNumber = page.page_number
        document.pages.append(Page(text=text, pageNumber=pageNumber))

    return document

### Deleting PDF Endpoint #################################################################################################

class PDF_id(BaseModel):
    pdf_id: str

@app.post("/delete_pdf/")
async def delete_pdf(pdf: PDF_id):
    vectordb.delete(where={"pdf_id": pdf.pdf_id})
    os.remove(f"pdfs/{pdf.pdf_id}.pdf")
    return {"message": "success"}

### Start Session Endpoint ################################################################################################

class Session(BaseModel):
    session_id: str

@app.post("/start_session/")
async def start_session(session: Session):
    # Create a unique ID for the session
    session_id = session.session_id
    # Store the session ID in the prompts dictionary, using the session ID as the key
    Chat_VectorDB = chat_client.create_collection(name=session_id, embedding_function=sentence_transformer_ef, get_or_create=True)
    # Return the session ID
    last_chats[session_id] = []

    return {"session_id": session_id}
    
### Asking Question Endpoint ###############################################################################################

class Question(BaseModel):
    question: str
    session_id: str

@app.post("/ask/")
async def ask(question: Question, request: Request):
    ### First Prompt + First LLM Server Request ############################################################################
    
    # Query all documents related to the question
    initial_context = vectordb.query(query_texts=[question.question], 
                                        n_results=3)

    ### Get previous chats ################################################################################################

    # Get the session ID from the question
    session_id = question.session_id
    # Get the chat vector database from the session ID
    Chat_VectorDB = chat_client.get_collection(name=session_id, embedding_function=sentence_transformer_ef)
    # Query all documents related to the question

    previous_chats = []

    chat_context = Chat_VectorDB.query(query_texts=[question.question], n_results=5)

    if chat_context and len(chat_context) > 0:
        chat_context = create_docs_from_results(chat_context)

        for chat in chat_context:
            previous_chats.append(chat["document"])

        if session_id in last_chats:
            last_two_chats = last_chats[session_id][-2:]

            previous_chats += last_two_chats

    previous_chats_str = "\n".join(previous_chats)

    ### Second Prompt + Second LLM Server Request #######################################################################
    
    # Once we have the reference IDs of the most relevant context, we can use them to filter the initial context we queried at the start
    filtered_and_formatted_context, relevant_pages = filter_and_format_context(initial_context)
    # Create the second prompt, using .format() to insert the filtered context and question into the prompt template
    # This prompt will be used to ask the LLM to generate an answer to the question, using the filtered context
    prompt_for_answering = prompt_template_2.format(context=filtered_and_formatted_context, 
                                                        question=question.question,
                                                        previous_chats_str=previous_chats_str)
    
    # Unique ID for the second prompt
    prompt_id_2 = str(uuid4())
    # Store the second prompt in the prompts dictionary using the prompt ID as the key
    prompts[prompt_id_2] = {"prompt": prompt_for_answering, "session_id": question.session_id}

    questions[prompt_id_2] = question.question
    
    # Unlike the first prompt, we need to send a streaming response to the frontend, so we need to create a URL for the streaming response
    # The code below creates a URL for the /streaming_response/ endpoint, and passes the prompt ID as a query parameter
    url = str(request.url_for("streaming_response", prompt_id=prompt_id_2))

    # Return the URL and the relevant context used to generate the answer
    return {"url": url, "relevant_pages": relevant_pages}
    
### LLM Server Request ####################################################################################################

# This function sends the second prompt to the LLM Server, and returns the response in chunks
# The response is a String, which contains the generated answer
# This function collects the response in chunks, and yields each chunk
# The chunks are returned one at a time, as a String

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
    # Check if the prompt ID is valid
    if prompt_id not in prompts:
        # If the prompt ID is not valid, yield an error message
        yield json.dumps({"error": "Invalid prompt ID"})
        return
    
    # Get the prompt from the prompts dictionary
    prompt = prompts[prompt_id]

    prompt_text = prompt["prompt"]
    # Prepare the prompt to be sent to the LLM Server
    headers, request_data, url = prepare_request_data(prompt_text, mode='generator')

    full_text = ""

    session_id = prompt["session_id"]

    # Send the prompt to the LLM Server, and get the response
    async with aiohttp.ClientSession() as session:
        # Send the prompt to the LLM Server, and get the response
        async with session.post(url, headers=headers, json=request_data) as response:
            # Loop through the response
            async for chunk in response.content.iter_any():
                # Check if the chunk starts with 'data: '
                if chunk.startswith(b'data: '):
                    chunk = chunk[6:]  # Remove 'data: ' prefix
                    # Try to parse the chunk as JSON
                    try:
                        chunk_data = json.loads(chunk.decode("utf-8"))
                        # Extract the generated text from the JSON
                        text = chunk_data["choices"][0]["text"]

                        full_text += text
                        # Yield the generated text
                        yield text
                    # If there is an error parsing the JSON, skip the chunk
                    except json.JSONDecodeError:
                        continue

    Chat_VectorDB = chat_client.get_collection(name=session_id, embedding_function=sentence_transformer_ef)
    Final_text = f"Question: {questions[prompt_id]}\nAnswer: {full_text}\n\n"
    Chat_VectorDB.add(ids=[str(Chat_VectorDB.count()+1)], documents=[Final_text])
    last_chats[session_id].append(Final_text)

### Prepare Request Data #################################################################################################

# This function prepares the prompt to be sent to the LLM Server
# The prompt is sent as a JSON object, with the format expected by the LLM Server
def prepare_request_data(prompt, mode='full'):
    # Set the API key and headers
    x_api_key = "67d8ece657b29e5219190ee2ba7eb2db"
    headers = {"x-api-key": x_api_key, "Content-Type": "application/json"}
    # Set the request data and URL
    if mode == 'full':
        request_data = {
            "model": "brucethemoose_Yi-34B-200K-DARE-merge-v7-4bpw-exl2-fiction",
            "prompt": prompt,
            "max_tokens": 128,
            "stream": True,
            "temperature": 1.0,
            "min_p": 0.05,
            "token_repetition_penalty": 1.7,
            "stop": ["/List"]
        }
        url = "http://192.168.100.131:5005/v1/completions"
    else:  # mode == 'generator'
        request_data = {
            "model": "brucethemoose_Yi-34B-200K-DARE-merge-v7-4bpw-exl2-fiction",
            "prompt": prompt,
            "max_tokens": 2048,
            "stream": True,
            "temperature": 1.0,
            "min_p": 0.05,
            "token_repetition_penalty": 1.3,
        }
        url = "http://192.168.100.131:5000/v1/completions"
    return headers, request_data, url

@app.get("/streaming_response/{prompt_id}")
async def streaming_response(prompt_id: str):
    return StreamingResponse(stream_generator(prompt_id), media_type="application/json")

### PDF Endpoint ############################################################################################################

class PDFName(BaseModel):
    name: str

# one pdf at a time
@app.post("/get_pdf/")
async def get_pdf(pdf_name: PDFName):
    pdf_path = f"pdfs/{pdf_name.name}.pdf"
    try:
        response = FileResponse(path=pdf_path, filename=pdf_name.name)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"File not found: {pdf_name.name}")

    return response

@app.get("/list_pdfs/")
async def list_pdfs():
    max_results = vectordb.count() + 1000
    print(f"max_results: {max_results}")
    all_documents = vectordb.query(query_texts=["yes"], n_results=max_results)
    documents = create_docs_from_results(all_documents)
    pdfs = {}
    for document in documents:
        print(document)
        name = document["metadata"]["name"]
        pdf_id = document["metadata"]["pdf_id"]
        image = f"thumbnails/{pdf_id}.png"

        pdfs[name] = {"name": name, "pdf_id": pdf_id, "image": image}

    return {"pdfs": pdfs}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)