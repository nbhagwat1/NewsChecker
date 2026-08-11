from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import joblib
from backend.preprocessing.text_extraction import segment_text_and_detect_language, create_embedding
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModel
import torch

# Models are loaded during application startup instead of on every request
# to avoid repeatedly loading large ML models during inference.
logistic_model = None
tokenizer = None
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
    global tokenizer
    global embedding_model

    try: 
        print("Loading classifier...", flush=True)
        logistic_model = joblib.load("models/logistic_model_v3.pkl")

        print("Loading embedding model...", flush=True)

        MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        embedding_model = AutoModel.from_pretrained(
            MODEL_NAME,
            dtype=torch.float16
        )

        embedding_model.eval()

        print("Models loaded successfully", flush=True)

    except Exception as e:
        print(f"Failed to load models: {e}")

        # Stop application startup because the API cannot perform inference
        # without the required ML models.
        raise

    # Application runs while paused at this point. Cleanup code executes after shutdown.
    yield

    print("Shutting down")

app = FastAPI(lifespan=lifespan)

# Allow requests from the local frontend development server.
# Required because the frontend and backend run on different ports.
origins = ["http://127.0.0.1:5500", "http://localhost:5500", "https://news-checker-kappa.vercel.app"]

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
    text: str = Field(min_length=1)

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

    try: 
        text_list = [article.text]

        # Prepare user-provided article text for inference by splitting it
        # into segments and detecting the language before embedding generation.
        segments, _, _ = segment_text_and_detect_language(text_list, None)

        # Convert processed article segments into fixed-size embeddings
        # used as input features for the classifier.
        average_embedding = create_embedding(segments, tokenizer, embedding_model)

        embedding_list = []
        embedding_list.append(average_embedding)

        prediction = logistic_model.predict(embedding_list)
        score = logistic_model.predict_proba(embedding_list)[:, 1][0]

        return {
            "prediction": prediction.tolist(),
            "score": float(score)
        }
    
    except Exception as e:
        print(f"Failed to process article: {e}")

        # Convert internal processing failures into an HTTP response
        # so the frontend can display a user-friendly error message.
        raise HTTPException(
            status_code=500,
            detail="Unable to process article."
        )
