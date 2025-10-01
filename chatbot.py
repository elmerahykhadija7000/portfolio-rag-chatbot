import json
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langdetect import detect, LangDetectException
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

# Load environment variables
load_dotenv()
key = os.getenv("GOOGLE_API_KEY")
if not key:
    raise ValueError("GOOGLE_API_KEY is not set in .env file")
genai.configure(api_key=key)

# Language detection
def detect_language(text):
    try:
        lang = detect(text.strip())
        if lang.startswith('ar'):
            return "ar", "أجب باللغة العربية."
        elif lang.startswith('fr') or lang in ['ca', 'ht']:
            return "fr", "Réponds en français."
        elif lang.startswith('en'):
            return "en", "Answer in English."
    except LangDetectException:
        pass

    if any(char in text for char in 'اأإبتثجحخدذرزسشصضطظعغفقكلمنهوي'):
        return "ar", "أجب باللغة العربية."

    text_lower = text.lower()
    common_french_words = [
        'bonjour', 'salut', 'merci', 'oui', 'non', 'je', 'tu', 'il', 'elle', 'nous',
        'vous', 'ils', 'elles', 'suis', 'es', 'est', 'sommes', 'etes', 'sont',
        'le', 'la', 'les', 'un', 'une', 'des', 'du', 'au', 'aux',
        'ce', 'cette', 'ces', 'mon', 'ton', 'son', 'ma', 'ta', 'sa',
        'comment', 'pourquoi', 'quand', 'ou', 'qui', 'que', 'quoi',
        'avec', 'sans', 'pour', 'par', 'dans', 'sur', 'sous', 'chez', 'voici', 'voila'
    ]
    words = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in text_lower).split()
    french_word_count = sum(1 for word in words if word in common_french_words)
    french_patterns = ['je suis', 'tu es', 'il est', 'c est', "j ai", "qu est", "puis je", "pouvez vous"]
    for pattern in french_patterns:
        if pattern in text_lower.replace("'", " "):
            return "fr", "Réponds en français."
    if french_word_count >= 1 or any(ord(c) > 127 and c.isalpha() for c in text):
        return "fr", "Réponds en français."
    return "en", "Answer in English."

# Load CV data from JSON
def load_cv_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Create documents from CV JSON
def create_documents_from_json(cv_data):
    documents = []
    personal_info = f"Name: {cv_data['name']}\nLocation: {cv_data['location']}\nEmail: {cv_data['email']}\nPhone: {cv_data['phone']}\nProfile: {cv_data['profile']}\n"
    documents.append({"text": personal_info, "metadata": {"source": "personal_information"}})

    education_text = "EDUCATION:\n"
    for education in cv_data['education']:
        education_text += f"• {education['institution']} - {education['degree']} ({education['period']})\n"
        for detail in education['details']:
            education_text += f"  - {detail}\n"
    documents.append({"text": education_text, "metadata": {"source": "education"}})

    exp_text = "WORK EXPERIENCE:\n"
    for exp in cv_data['work_experience']:
        exp_text += f"• {exp['company']} - {exp['position']} ({exp['period']})\n"
        for mission in exp['missions']:
            exp_text += f"  - {mission}\n"
    documents.append({"text": exp_text, "metadata": {"source": "experience"}})

    projects_text = "PROJECTS:\n"
    for project in cv_data['projects']:
        projects_text += f"• {project['name']} ({', '.join(project['technologies'])})\n  {project['description']}\n"
    documents.append({"text": projects_text, "metadata": {"source": "projects"}})

    skills_text = "TECHNICAL SKILLS:\n"
    skills_text += f"• Languages: {', '.join(cv_data['technical_skills']['languages'])}\n"
    skills_text += f"• Tools and libraries: {', '.join(cv_data['technical_skills']['tools_and_libraries'])}\n"
    skills_text += "• Areas of expertise:\n"
    for domain in cv_data['technical_skills']['domains']:
        skills_text += f"  - {domain}\n"
    documents.append({"text": skills_text, "metadata": {"source": "technical_skills"}})

    cert_text = "CERTIFICATIONS:\n"
    for cert in cv_data['certifications']:
        cert_text += f"• {cert['title']} - {cert['organization']} ({cert['duration']})\n"
        for content in cert['content']:
            cert_text += f"  - {content}\n"
    documents.append({"text": cert_text, "metadata": {"source": "certifications"}})

    other_text = "LANGUAGES:\n"
    for language, level in cv_data['languages'].items():
        other_text += f"• {language.capitalize()}: {level}\n"
    other_text += "\nSOFT SKILLS:\n"
    for skill in cv_data['soft_skills']:
        other_text += f"• {skill}\n"
    documents.append({"text": other_text, "metadata": {"source": "languages_and_soft_skills"}})
    return documents

# Create vector index
def create_vector_store(documents):
    texts = [doc["text"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)

# Generate response
def answer_question(question, vector_store, cv_data):
    response_lang, language_instruction = detect_language(question)
    question_lowered = question.lower()

    greetings = {
        "en": {
            "hello": "Hello! How can I help you with Khadija?",
            "hi": "Hi there! How can I assist you about Khadija?",
            "thanks": "You're welcome! Anything else about Khadija?",
            "thank you": "You're welcome! Anything else?",
            "bye": "Goodbye! Feel free to come back.",
            "goodbye": "Goodbye! Have a great day!"
        },
        "fr": {
            "bonjour": "Bonjour ! Comment puis-je vous aider à propos de Khadija ?",
            "salut": "Salut ! Comment puis-je vous assister à propos de Khadija ?",
            "merci": "De rien ! Autre chose à propos de Khadija ?",
            "au revoir": "Au revoir ! N'hésitez pas à revenir.",
            "by": "Au revoir ! Passez une excellente journée !"
        },
        "ar": {
            "مرحبا": "مرحبا! كيف يمكنني مساعدتك بخصوص خديجة؟",
            "شكرا": "على الرحب والسعة! هل هناك شيء آخر؟",
            "وداعا": "مع السلامة! لا تتردد في العودة."
        }
    }
    for lang, phrases in greetings.items():
        for phrase, response in phrases.items():
            if phrase in question_lowered:
                return response

    if any(keyword in question_lowered for keyword in ['internship', 'stage', 'intern', 'stagiaire', 'تدريب', 'متدرب']):
        exp_docs = vector_store.similarity_search("work experience professional internship stage", k=2)
        has_exp_doc = any(doc.metadata.get('source') == 'experience' for doc in exp_docs)
        if not has_exp_doc:
            try:
                exp_docs = vector_store.get(where={"source": "experience"})["documents"]
                exp_docs = [{"page_content": doc} for doc in exp_docs]
            except:
                exp_docs = []
                for term in ["internship", "work experience", "professional experience"]:
                    more_docs = vector_store.similarity_search(term, k=1)
                    exp_docs.extend(more_docs)
        other_docs = vector_store.similarity_search(question, k=1)
        relevant_docs = exp_docs + other_docs
    elif any(keyword in question_lowered for keyword in ['project', 'projects', 'projet', 'projets', 'مشروع', 'مشاريع']):
        project_docs = vector_store.similarity_search("PROJECTS: telecommunications network e-commerce library", k=3)
        has_project_doc = any(doc.metadata.get('source') == 'projects' for doc in project_docs)
        if not has_project_doc:
            try:
                project_docs = vector_store.get(where={"source": "projects"})["documents"]
                project_docs = [{"page_content": doc} for doc in project_docs]
            except:
                project_docs = []
                for project_name in ["Telecommunications Network", "Library Management", "E-commerce"]:
                    more_docs = vector_store.similarity_search(project_name, k=1)
                    project_docs.extend(more_docs)
        other_docs = vector_store.similarity_search(question, k=1)
        relevant_docs = project_docs + other_docs
    else:
        relevant_docs = vector_store.similarity_search(question, k=3)

    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    if any(keyword in question_lowered for keyword in ['internship', 'stage', 'intern', 'stagiaire', 'تدريب']):
        exp_info = "WORK EXPERIENCE:\n"
        for exp in cv_data['work_experience']:
            exp_info += f"• {exp['company']} - {exp['position']} ({exp['period']})\n"
            for mission in exp['missions']:
                exp_info += f"  - {mission}\n"
        context = exp_info + "\n\n" + context
    elif any(keyword in question_lowered for keyword in ['project', 'projects', 'projet', 'projets']):
        projects_info = "PROJECTS:\n"
        for project in cv_data['projects']:
            projects_info += f"• {project['name']} ({', '.join(project['technologies'])})\n  {project['description']}\n"
        context = projects_info + "\n\n" + context

    if response_lang == "ar":
        system_role = f"أنا DijaBot، المساعد الافتراضي لخديجة. كيف يمكنني مساعدتك بخصوص خديجة؟ استخدم فقط المعلومات أدناه للإجابة."
        no_info_msg = "عذرًا، لا أملك هذه المعلومة."
    elif response_lang == "fr":
        system_role = f"Je suis DijaBot, l'assistant virtuel de Khadija. Comment puis-je vous aider à propos de Khadija ? Utilise uniquement les informations ci-dessous pour répondre."
        no_info_msg = "Je ne dispose pas de هذه المعلومات."
    else:
        system_role = f"I am DijaBot, Khadija's virtual assistant. How can I help you about Khadija? Use only the information below to answer."
        no_info_msg = "I don't have this information."

    prompt = f"""
{system_role}

Informations utiles sur Khadija :
{context}

Question : {question}

Instruction de langue : {language_instruction}

Si l'information n'est pas disponible, dis : "{no_info_msg}".
Sois clair et concis.
"""

    try:
        model = genai.GenerativeModel(model_name="gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error during generation: {str(e)}"

# Load data at startup
print("Loading CV data...")
cv_data = load_cv_data("data.json")
print("Preparing documents...")
documents = create_documents_from_json(cv_data)
print("Creating vector index...")
vector_store = create_vector_store(documents)
print("✅ CV Assistant is ready!")

# FastAPI setup
app = FastAPI(title="CV Assistant API", description="Ask questions about Khadija El Merahy's CV")
app.mount("/img", StaticFiles(directory="templates/img"), name="img")
app.mount("/assets", StaticFiles(directory="templates/assets"), name="assets")
templates = Jinja2Templates(directory="templates")

class QuestionRequest(BaseModel):
    question: str

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        answer = answer_question(request.question, vector_store, cv_data)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "OK", "message": "CV Assistant is running!"}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting CV Assistant server at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)