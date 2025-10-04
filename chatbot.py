import os
from fastapi import FastAPI, Request, Form
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

# --- 8. Prompt template ---
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
- Structure your responses with Markdown for better readability for affichement of skills,projects,and experiences,in list format when necessary.
- Respond based on intention and language:
-if the user asks you a social question like how are you or i love you (just exemples), respond with the perfect answer.
- If the user greets you, respond with a greeting in the same language, mentioning a question with the same language of his request of how you can help him about Khadija.
- If the user thanks you, respond with "You're welcome!" in the same language.
- If the user says goodbye, respond with "Goodbye! Have a great day!" in the same language.
- If the user provides unrelated input, respond with "I can only answer questions about Khadija's profile, skills, projects, and experience." in the same language.

- If the user asks anything about Khadija, answer using ALL the provided sources, concisely in 1-2 sentences.
warning d'ont use any greetings,thanks,goodbye in your answer and answer with the same language please , and if the user ask you but he have a mistake grammaticaly or orthographiqlly correct it and answer it .
- Do NOT add unnecessary information outside the user's question.

User input:
{question}

Sources retrieved:
{context}

Answer:
"""

prompt = PromptTemplate(input_variables=["question", "context"], template=prompt_template)

# --- Modern LangChain approach instead of LLMChain ---
chain = (
    {"question": RunnablePassthrough(), "context": lambda question: "\n\n".join([doc.page_content for doc in retriever.get_relevant_documents(question)])}
    | prompt
    | llm
)

# --- 9. FastAPI App ---
app = FastAPI()

# MODIFICATION PRINCIPALE: Configuration des templates et fichiers statiques
templates = Jinja2Templates(directory="templates")

# Servir les images depuis templates/img
app.mount("/img", StaticFiles(directory="templates/img"), name="images")

# Servir les assets (PDF, etc.) depuis templates/assets  
app.mount("/assets", StaticFiles(directory="templates/assets"), name="assets")

# Servir tous les autres fichiers statiques depuis templates (CSS, JS, etc.)
app.mount("/static", StaticFiles(directory="templates"), name="static")

# --- 10. Routes ---
@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    # MODIFICATION: Utiliser Jinja2Templates au lieu de lire directement le fichier
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask")
async def ask_question(question: str = Form(...)):
    response = chain.invoke(question)
    return {"answer": response.content}

# Route supplémentaire pour servir des fichiers spécifiques si nécessaire
@app.get("/download/{filename}")
async def download_file(filename: str):
    from fastapi.responses import FileResponse
    file_path = f"templates/assets/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        return {"error": "File not found"}

# --- 11. Run the server properly ---
if __name__ == "__main__":
    import sys
    import subprocess
    
    print("🚀 Starting CV Assistant server at http://127.0.0.1:8000")
    # CORRECTION: Utiliser le bon nom de module
    subprocess.run([sys.executable, "-m", "uvicorn", "chatbot:app", "--reload", "--host", "127.0.0.1", "--port", "8000"])
