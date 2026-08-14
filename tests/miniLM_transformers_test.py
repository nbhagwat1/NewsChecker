import psutil
import torch
from transformers import AutoTokenizer, AutoModel

def memory(label, process):
    """
    Prints the current memory usage of a process.

    Args:
        label (str): Description identifying the point at which memory
            usage is measured.
        process: Process whose memory usage will be measured.

    Returns:
        None
    """

    print(
        f"{label}: "
        f"{process.memory_info().rss / 1024 / 1024:.2f} MB",
        flush=True
    )

def main():
    """
    Measures memory usage while loading the tokenizer and model.

    This function records the process's memory usage before loading the
    tokenizer, after loading the tokenizer, and after loading the
    MiniLM model. The measurements help determine how much memory is
    required by each component.

    Args:
        None

    Returns:
        None
    """
    
    process = psutil.Process()

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    memory("Before imports", process)

    print("Loading tokenizer...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    memory("After tokenizer", process)

    print("Loading MiniLM...", flush=True)

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16
    )

    model.eval()

    memory("After MiniLM", process)