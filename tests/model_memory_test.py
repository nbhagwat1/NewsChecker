import psutil
from sentence_transformers import SentenceTransformer

process = psutil.Process()

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print(
    f"Memory before model: "
    f"{process.memory_info().rss / 1024 / 1024:.2f} MB",
    flush=True
)

print("Loading MiniLM...", flush=True)

model = SentenceTransformer(
    MODEL_NAME,
    model_kwargs={"torch_dtype": "float16"}
)

print(
    f"Memory after model: "
    f"{process.memory_info().rss / 1024 / 1024:.2f} MB",
    flush=True
)

texts = [
    "This is a test article about technology and artificial intelligence.",
    "The weather is expected to remain sunny throughout the weekend.",
]

embeddings = model.encode(
    texts,
    show_progress_bar=False
)

print(f"Embedding shape: {embeddings.shape}", flush=True)