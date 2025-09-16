from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
import ollama
import os
import asyncio

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify domains like ["http://localhost:5500"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Folder where PDFs live (adjust this to your secure location)
PDF_FOLDER = "./pdfs"

# Pydantic schema for requests
class QARequest(BaseModel):
    filename: str    # PDF file name only, no path
    question: str
    model: str = "phi3:3.8b" #llama model


def get_pdf_path(filename: str) -> str:
    # Prevent path traversal attacks
    safe_filename = os.path.basename(filename)
    full_path = os.path.join(PDF_FOLDER, safe_filename)
    if not os.path.isfile(full_path):
        raise HTTPException(status_code=404, detail="PDF file not found")
    return full_path


def extract_pdf_text(file_path: str) -> str:
    text = ""
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {str(e)}")
    if not text.strip():
        raise HTTPException(status_code=400, detail="PDF contains no readable text")
    return text


def build_prompt(question: str, context: str) -> str:
    return f"""You are a helpful assistant. Use the following context to answer the question directly and concisely. Do NOT start your answer with phrases like 'According to the PDF or The PDF lists the following packages not use any type of phase contains pdf in this pdf as pdf according pdf from yourself' or 'Based on the PDF'. Just provide the answer if need add names according to pdf for better answering.

Context:
\"\"\"
{context}
\"\"\"

Question: {question}
Answer:"""


@app.post("/ask")
def ask_pdf_question(req: QARequest):
    pdf_path = get_pdf_path(req.filename)
    context = extract_pdf_text(pdf_path)
    prompt = build_prompt(req.question, context)
    try:
        response = ollama.chat(
            model=req.model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama error: {str(e)}")
    return {"answer": response["message"]["content"]}


@app.post("/ask-stream")
async def ask_pdf_question_stream(req: QARequest):
    pdf_path = get_pdf_path(req.filename)
    context = extract_pdf_text(pdf_path)
    prompt = build_prompt(req.question, context)

    try:
        stream = ollama.chat(
            model=req.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama error: {str(e)}")

    async def event_generator():
        # Stream each chunk to client
        for chunk in stream:
            content = chunk["message"]["content"]
            yield content
            await asyncio.sleep(0.01)  # allow other tasks to run

    return StreamingResponse(event_generator(), media_type="text/plain")


# Simple root route
@app.get("/")
def root():
    return {"message": "PDF QA Chatbot API is running"}
