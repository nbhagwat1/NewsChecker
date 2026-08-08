import psutil

process = psutil.Process()

def memory(label):
    print(
        f"{label}: "
        f"{process.memory_info().rss / 1024 / 1024:.2f} MB",
        flush=True
    )

memory("Before imports")

import fastapi
memory("After fastapi")

from fastapi.middleware.cors import CORSMiddleware
memory("After CORS")

from contextlib import asynccontextmanager
memory("After contextlib")

from sentence_transformers import SentenceTransformer
memory("After sentence-transformers")

import joblib
memory("After joblib")

from backend.preprocessing.text_extraction import (
    segment_text_and_detect_language,
    create_embeddings
)
memory("After text_extraction")

from pydantic import BaseModel, Field
memory("After pydantic")