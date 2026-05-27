from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from huggingface_hub import hf_hub_download
import fasttext
from sentence_transformers import SentenceTransformer
import joblib
from backend.preprocessing.text_extraction import analyze_language, create_embeddings
from pydantic import BaseModel

logistic_model = None
language_model = None
detection_model = None
embedding_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global logistic_model
    logistic_model = joblib.load("logistic_model.pkl")

    global language_model
    language_model = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")

    global detection_model
    detection_model = fasttext.load_model(language_model)
    
    global embedding_model
    embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    print("Models loaded")

    yield

    print("Shutting down")

app = FastAPI(lifespan=lifespan)

origins = ["http://127.0.0.1:5500", "http://localhost:5500"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class NewsArticle(BaseModel):
    text: str

@app.get("/")
def home():
    return {
        "message": "API is working"
    }

@app.post("/check")
def check(article: NewsArticle):
    text_list = []
    text_list.append(article.text)

    segments, placeholder, language = analyze_language(text_list, detection_model)
    embeddings, suspicious_factors = create_embeddings(segments, embedding_model)

    embedding_list = []
    embedding_list.append(embeddings)

    prediction = logistic_model.predict(embedding_list)
    score = logistic_model.predict_proba(embedding_list)[:, 1][0]

    return {
        "prediction": prediction.tolist(),
        "score": float(score)
    }
