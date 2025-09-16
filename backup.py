import PyPDF2
import ollama

# === Step 1: Read PDF and extract text ===
def extract_pdf_text(file_path):
    text = ""
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text()
    return text


# === Step 2: Ask question using Ollama ===
# def ask_ollama(question, context, model="llama3"):
#     prompt = f"""You are a helpful assistant. Use the following context from a PDF to answer the question.

# Context:
# \"\"\"
# {context}
# \"\"\"

# Question: {question}
# Answer:"""

#     response = ollama.chat(
#         model=model,
#         messages=[
#             {"role": "user", "content": prompt}
#         ]
#     )
#     return response['message']['content']
def ask_ollama(question, context, model="llama3"):
    prompt = f"""You are a helpful assistant. Use the following context from a PDF to answer the question.

Context:
\"\"\"
{context}
\"\"\"

Question: {question}
Answer:"""

    print("Bot: ", end='', flush=True)

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        stream=True  # <-- enable streaming
    )

    full_response = ""
    for chunk in response:
        content = chunk['message']['content']
        print(content, end='', flush=True)
        full_response += content

    print()  # for newline after full message
    return full_response



# === Step 3: Main chatbot loop ===
def chat_with_pdf(pdf_path, model="llama3.1:8b"):
    print("Reading PDF...")
    context = extract_pdf_text(pdf_path)

    print("PDF loaded. You can now ask questions (type 'exit' to quit).")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        answer = ask_ollama(user_input, context, model)
        print("Bot:", answer)


# === Run the chatbot ===
if __name__ == "__main__":
    pdf_path = "abc_travel_agency_brochure.pdf"  # Change this to your PDF path
    chat_with_pdf(pdf_path)
