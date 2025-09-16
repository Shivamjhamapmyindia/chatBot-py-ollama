from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
import ollama
import os
import asyncio

app = FastAPI()

# === Enable CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Constants ===
PDF_FOLDER = "./pdfs"

# === Request schema ===
class QARequest(BaseModel):
    filename: str
    question: str
    model: str = "qwen3:0.6b"  # Default model
    

# === Utilities ===
def get_pdf_path(filename: str) -> str:
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
    return f"""You are a helpful assistant. Use the following context to answer the question directly and concisely. 
 
 If user Greets you, respond with a greeting. If user says thanks, respond with a welcome message.

Do NOT start your answer with phrases like 'According to the PDF', 'This PDF contains', 'Based on the PDF', or any variation mentioning 'PDF'. 

Provide a clear, concise, and relevant answer. If the answer is long, summarize it to be between 200 and 300 words, focusing on key points only.


Context:
\"\"\"
{context}
\"\"\"

Question: {question}
Answer:"""

# === Regular (non-streaming) endpoint ===
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
            thinking=False
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama error: {str(e)}")
    return {"answer": response["message"]["content"]}

# === Streaming endpoint with cancel support ===
@app.post("/ask-stream")
async def ask_pdf_question_stream(req: QARequest, request: Request):
    pdf_path = get_pdf_path(req.filename)
    context = extract_pdf_text(pdf_path)
    prompt = build_prompt(req.question, context)

    try:
        stream = ollama.chat(
            model=req.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            think=False
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama error: {str(e)}")

    async def event_generator():
        try:
            for chunk in stream:
                # 🚨 Stop streaming if client disconnects (e.g., user hit Stop or closed browser)
                if await request.is_disconnected():
                    print("Client disconnected. Aborting stream.")
                    break
                content = chunk["message"]["content"]
                yield content
                await asyncio.sleep(0.01)  # Yield control to event loop
        except asyncio.CancelledError:
            print("Streaming cancelled by server.")
        except Exception as e:
            print(f"Error during streaming: {e}")

    return StreamingResponse(event_generator(), media_type="text/plain")

# === Root route ===
@app.get("/")
def root():
    return {"message": "PDF QA Chatbot API is running"}
