import os
import psutil
import torch
from transformers import AutoTokenizer, AutoModel

def memory_mb(process):
    """
    Returns the current memory usage of a process in megabytes.

    Args:
        process: Process whose memory usage will be measured.

    Returns:
        float: Current memory usage in megabytes.
    """

    return process.memory_info().rss / 1024 / 1024

def main():
    """
    Measures memory usage while loading the MPNet tokenizer and model.

    This function records the process's memory usage before loading the
    tokenizer, after loading the tokenizer, and after loading the
    all-mpnet-base-v2 model. It then confirms that the model loaded
    successfully.

    Args:
        None

    Returns:
        None
    """

    MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

    process = psutil.Process(os.getpid())

    print(f"Memory before imports: {memory_mb(process):.2f} MB", flush=True)

    print("Loading tokenizer...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Memory after tokenizer: {memory_mb(process):.2f} MB", flush=True)

    print("Loading MPNet...", flush=True)

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16
    )

    model.eval()

    print(f"Memory after MPNet: {memory_mb(process):.2f} MB", flush=True)

    print("Model loaded successfully.", flush=True)

if __name__ == "__main__":
    main()