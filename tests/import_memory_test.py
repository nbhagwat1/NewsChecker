import psutil

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
    Measures the application's memory usage as major dependencies are imported.

    This function records the process's memory usage before importing
    dependencies and after each major import. The measurements help identify
    which dependencies contribute most to the application's memory usage.

    Args:
        None

    Returns:
        None
    """
    
    process = psutil.Process()

    memory("Before imports", process)

    import fastapi
    memory("After fastapi", process)

    from fastapi.middleware.cors import CORSMiddleware
    memory("After CORS", process)

    from contextlib import asynccontextmanager
    memory("After contextlib", process)

    from sentence_transformers import SentenceTransformer
    memory("After sentence-transformers", process)

    import joblib
    memory("After joblib", process)

    from backend.preprocessing.text_extraction import (
        segment_text_and_detect_language,
        create_embedding
    )
    memory("After text_extraction", process)

    from pydantic import BaseModel, Field
    memory("After pydantic", process)

if __name__ == "__main__":
    main()