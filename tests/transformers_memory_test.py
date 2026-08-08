import os
import psutil
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

process = psutil.Process(os.getpid())


def memory_mb():
    return process.memory_info().rss / 1024 / 1024


print(f"Memory before imports: {memory_mb():.2f} MB", flush=True)

print("Loading tokenizer...", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"Memory after tokenizer: {memory_mb():.2f} MB", flush=True)

print("Loading MPNet...", flush=True)

model = AutoModel.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16
)

model.eval()

print(f"Memory after MPNet: {memory_mb():.2f} MB", flush=True)

print("Model loaded successfully.", flush=True)