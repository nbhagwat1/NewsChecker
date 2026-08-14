from backend.preprocessing.text_extraction import get_content, segment_text_and_detect_language, create_embedding
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from backend.model.train import load_datasets, unpack_dataset

def main():
    """
    Verifies that the embedding generation pipeline continues to produce
    valid embeddings after processing was interrupted and resumed.

    This function processes a portion of the original dataset using the
    same preprocessing and embedding generation steps used by the main
    pipeline. The newly generated embeddings are then compared against
    the previously saved training, validation, and testing datasets to
    check whether they are already present.

    Args:
        None

    Returns:
        None
    """

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"   
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    embedding_model = AutoModel.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16
    )
    embedding_model.eval()

    news_data = pd.read_csv("data/original/FakeNewsNet.csv")
    news_data = news_data.sample(frac=1, random_state=42).reset_index(drop=True)
    test_data = news_data.iloc[23186:]
    article_links = test_data['news_url'].tolist()

    valid_embeddings = []

    for link in article_links:
        content, _, text_list, _, _ = get_content(link)
        if content:
            segment_list, _, _ = segment_text_and_detect_language(text_list, None)
            average_embedding = create_embedding(segment_list, tokenizer, embedding_model)
            valid_embeddings.append(average_embedding)

    train_dataset, validate_dataset, test_dataset = load_datasets()
    
    X_train_dataset, _ = unpack_dataset(train_dataset)
    X_validate_dataset, _ = unpack_dataset(validate_dataset)
    X_test_dataset, _ = unpack_dataset(test_dataset)

    # Check whether the newly generated embeddings are already present
    # in the saved datasets, confirming that the pipeline produced
    # consistent results after processing was resumed.
    for embedding in valid_embeddings:
        print((embedding in X_train_dataset) or (embedding in X_validate_dataset) or (embedding in X_test_dataset))

if __name__ == "__main__":
    main()