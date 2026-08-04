from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from huggingface_hub import hf_hub_download
import fasttext
from sentence_transformers import SentenceTransformer
import joblib
from backend.preprocessing.text_extraction import segment_text_and_detect_language, create_embeddings
from pydantic import BaseModel

# Models are loaded during application startup instead of on every request
# to avoid repeatedly loading large ML models during inference.
logistic_model = None
language_model = None
detection_model = None
embedding_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Loads ML models when the FastAPI application starts
    and performs cleanup when the application shuts down.
    """

    # Load trained classifier and embedding/language models once at startup
    # so prediction requests can run without initialization overhead.

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

# Allow requests from the local frontend development server.
# Required because the frontend and backend run on different ports.
origins = ["http://127.0.0.1:5500", "http://localhost:5500"]

# Enables communication between the frontend application and FastAPI backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class NewsArticle(BaseModel):
    """
    Request schema containing the article text to classify.
    """
    text: str

@app.get("/")
def home():
    return {
        "message": "API is working"
    }

@app.post("/check")
def check(article: NewsArticle):
    """
    Processes user-provided article text through the inference pipeline:
    segments the text, detects language, generates semantic embeddings,
    and returns the model prediction and confidence score.
    """

    text_list = []
    text_list.append(article.text)

    # Prepare user-provided article text for inference by splitting it
    # into segments and detecting the language before embedding generation.
    segments, placeholder, language = segment_text_and_detect_language(text_list, detection_model)

    # Convert processed article segments into fixed-size embeddings
    # used as input features for the classifier.
    embeddings, suspicious_factors = create_embeddings(segments, embedding_model)

    embedding_list = []
    embedding_list.append(embeddings)

    prediction = logistic_model.predict(embedding_list)
    score = logistic_model.predict_proba(embedding_list)[:, 1][0]

    return {
        "prediction": prediction.tolist(),
        "score": float(score)
    }
