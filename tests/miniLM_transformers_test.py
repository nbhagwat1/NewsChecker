import psutil
import torch
from transformers import AutoTokenizer, AutoModel

process = psutil.Process()

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def memory(label):
    print(
        f"{label}: "
        f"{process.memory_info().rss / 1024 / 1024:.2f} MB",
        flush=True
    )


memory("Before imports")

print("Loading tokenizer...", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

memory("After tokenizer")

print("Loading MiniLM...", flush=True)

model = AutoModel.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16
)

model.eval()

memory("After MiniLM")