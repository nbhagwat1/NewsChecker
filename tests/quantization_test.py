import psutil
import torch
from transformers import AutoTokenizer, AutoModel

def main():
    """
    Measures memory usage while loading the MPNet tokenizer and model.

    This function records the process's memory usage before loading the
    tokenizer, after loading the tokenizer, and after loading the
    all-mpnet-base-v2 model using half-precision values. The measurements
    help determine how much memory is required by each component.

    Args:
        None

    Returns:
        None
    """
    
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

if __name__ == "__main__":
    main()