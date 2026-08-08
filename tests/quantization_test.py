import psutil
import torch
from transformers import AutoTokenizer, AutoModel

process = psutil.Process()

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

print(
    f"Memory before imports: "
    f"{process.memory_info().rss / 1024 / 1024:.2f} MB",
    flush=True
)

print("Loading tokenizer...", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(
    f"Memory after tokenizer: "
    f"{process.memory_info().rss / 1024 / 1024:.2f} MB",
    flush=True
)

print("Loading FP16 MPNet...", flush=True)

model = AutoModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16
)

model.eval()

print(
    f"Memory after MPNet: "
    f"{process.memory_info().rss / 1024 / 1024:.2f} MB",
    flush=True
)