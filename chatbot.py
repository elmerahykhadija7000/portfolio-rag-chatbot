import os
from fastapi import FastAPI, Request, Form, Depends, Cookie, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from typing import Optional
import uuid
import json

# --- 1. Load API key ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# --- 2. Configure Gemini ---
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)

# --- 3. Embeddings HuggingFace ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- 4. Load your data.txt ---
with open("data.txt", "r", encoding="utf-8") as f:
    text_data = f.read()

docs = [Document(page_content=text_data)]

# --- 5. Split text into chunks ---
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splitted_docs = splitter.split_documents(docs)

# --- 6. Index into Chroma ---
vectorstore = Chroma.from_documents(splitted_docs, embeddings, collection_name="my_collection")

# --- 7. Retriever ---
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# --- 8. Conversation memory storage ---
# Dictionary to store conversation histories by session_id
conversation_histories = {}

# --- 9. FastAPI App ---
app = FastAPI()

# Configuration des templates et fichiers statiques
templates = Jinja2Templates(directory="templates")

# Servir les images depuis templates/img
app.mount("/img", StaticFiles(directory="templates/img"), name="images")

# Servir les assets (PDF, etc.) depuis templates/assets  
app.mount("/assets", StaticFiles(directory="templates/assets"), name="assets")

# Servir tous les autres fichiers statiques depuis templates (CSS, JS, etc.)
app.mount("/static", StaticFiles(directory="templates"), name="static")

# Add this line for Vercel deployment
root_path = os.environ.get("ROOT_PATH", "")
if root_path:
    app.root_path = root_path

# --- Prompt template with conversation history ---
prompt_template = """
You are an assistant that answers questions about Khadija El Merahy.
- Detect the language of the user's input automatically.
- Detect the user's intention automatically:
    * Greeting
    * Thanks
    * Goodbye
    * Question about Khadija
    * Unrelated input

- Interpret the user's input correctly and respond based on the intention and language.
- Structure your responses with Markdown for better readability for affichement of skills, projects, and experiences, in list format when necessary.
- Respond based on intention and language:
- If the user asks you a social question like how are you or i love you (just examples), respond with the perfect answer.
- If the user greets you, respond with a greeting in the same language, mentioning a question with the same language of his request of how you can help him about Khadija.
- If the user thanks you, respond with "You're welcome!" in the same language.
- If the user says goodbye, respond with "Goodbye! Have a great day!" in the same language.
- If the user provides unrelated input, respond with "I can only answer questions about Khadija's profile, skills, projects, and experience." in the same language.

- If the user asks anything about Khadija, answer using ALL the provided sources, concisely in 1-2 sentences.
- Warning: don't use any greetings, thanks, goodbye in your answer and answer with the same language please, and if the user asks you but they have a grammatical or orthographic mistake, correct it and answer it.
- Do NOT add unnecessary information outside the user's question.
- IMPORTANT: Use the conversation history to maintain context and provide relevant responses.

Chat History:
{chat_history}

User input:
{question}

Sources retrieved:
{context}

Answer:
"""

# --- 10. Helper function to get or create session ---
def get_session_id(session_id: Optional[str] = Cookie(None), response: Response = None):
    if not session_id:
        session_id = str(uuid.uuid4())
        response.set_cookie(key="session_id", value=session_id)
        # Initialize conversation history for new session
        conversation_histories[session_id] = []
    elif session_id not in conversation_histories:
        # Initialize conversation history for existing but unknown session
        conversation_histories[session_id] = []
    return session_id

# --- 11. Routes ---
@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request, session_id: str = Depends(get_session_id)):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask")
async def ask_question(
    question: str = Form(...),
    session_id: str = Depends(get_session_id)
):
    # Get the conversation history for this session
    chat_history = conversation_histories[session_id]
    
    # Format chat history for the prompt
    formatted_history = ""
    for entry in chat_history:
        formatted_history += f"Human: {entry['human']}\nAssistant: {entry['ai']}\n\n"
    
    # Get relevant documents
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Create prompt with history
    prompt = PromptTemplate(
        input_variables=["question", "context", "chat_history"],
        template=prompt_template
    )
    
    # Generate answer
    chain_input = {
        "question": question,
        "context": context,
        "chat_history": formatted_history
    }
    
    response = llm.invoke(prompt.format(**chain_input))
    
    # Update conversation history
    chat_history.append({
        "human": question,
        "ai": response.content
    })
    
    # Limit history length to prevent token overflow
    if len(chat_history) > 10:
        chat_history = chat_history[-10:]
    
    # Save updated history
    conversation_histories[session_id] = chat_history
    
    return {"answer": response.content}

@app.get("/download/{filename}")
async def download_file(filename: str):
    from fastapi.responses import FileResponse
    file_path = f"templates/assets/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        return {"error": "File not found"}

# --- 12. Run the server properly ---
if __name__ == "__main__":
    import sys
    import subprocess
    
    print("🚀 Starting CV Assistant server with memory at http://127.0.0.1:8000")
    subprocess.run([sys.executable, "-m", "uvicorn", "chatbot:app", "--reload", "--host", "127.0.0.1", "--port", "8000"])
