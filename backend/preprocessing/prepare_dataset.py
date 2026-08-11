from transformers import logging
logging.set_verbosity_error()
import os
import numpy as np
import pandas as pd
from backend.preprocessing.text_extraction import get_content, segment_text_and_detect_language, create_embedding
import random
from multiprocessing import Pool
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = None
embedding_model = None
iteration = 0

def main():
    """
    Processes the original article dataset through the ML preprocessing pipeline.

    This function loads the original news article dataset, extracts article
    content from URLs, generates semantic embeddings, filters invalid results,
    and creates cleaned datasets for model training and evaluation.

    The preprocessing pipeline uses multiprocessing to process articles
    concurrently. After processing, the resulting article embeddings and
    labels are split into training, validation, and testing datasets using
    a 70/15/15 split. Each dataset is verified to contain both real and fake
    news examples before being saved as NumPy files for later model training.

    The function also reports preprocessing statistics, including successful
    and failed article extractions, embedding quality issues, and detected
    article languages.

    Args:
        None
    
    Returns:
        None
    """

    news_data = pd.read_csv("data/original/FakeNewsNet.csv")

    # Randomly shuffle the dataset before further processing while making
    # sure it is shuffled the same way every time the program runs.
    news_data = news_data.sample(frac=1, random_state=42).reset_index(drop=True)
    test_data = news_data.iloc[:]

    article_links = test_data['news_url'].tolist()
    article_labels = test_data['real'].tolist()

    X_final_data = []
    y_final_data = []
    failed_data = []

    # Process multiple articles at the same time to reduce the total
    # preprocessing time. Six worker processes are used to balance
    # performance and system resource usage.
    pool = Pool(processes=6, initializer=initialize_models)
    results = pool.map(process_article_wrapper, zip(article_links, article_labels))
    pool.close()
    pool.join()

    # Separate successful and failed processing results. Store the
    # successfully processed article segments, their corresponding target
    # labels, and detected languages in separate lists, while storing
    # failed articles in their own list.
    failed_data = [result for result in results if result[0] == "Fail"]
    X_final_data = [result[2][0] for result in results if result[0] == "Success"]
    y_final_data = [result[2][1] for result in results if result[0] == "Success"]

    # Print summary statistics about the preprocessing pipeline to verify
    # that the dataset was processed successfully and to identify potential
    # issues before training the model.

    print(f"Length of successful data: {len(X_final_data)}")
    print(f"Length of failed data: {len(failed_data)}")
    print("-----------------")
    print("FAILED DATA STATS")
    print(f"HTTP request failed: {len([failure for failure in failed_data if failure[1] == 'HTTP request failed'])}")
    print(f"Pipeline failed: {len([failure for failure in failed_data if failure[1] == 'Text cleanup function got rid of everything'])}")
    print(f"Translation failed: {len([failure for failure in failed_data if failure[1] == 'Translation failed'])}")
    print(f"Other reason: {len([failure for failure in failed_data if (failure[1] != 'HTTP request failed' and failure[1] != 'Text cleanup function got rid of everything' and failure[1] != 'Translation failed')])}")
    print("-----------------")

    new_final_data = list(zip(X_final_data, y_final_data))
    print(f"Embeddings with invalid shape: {len([individual_article for individual_article in new_final_data if individual_article[0].shape != (384,)])}")
    print(f"Embeddings with NaN: {len([individual_article for individual_article in new_final_data if np.isnan(individual_article[0]).any()])}")
    print(f"Embeddings with INF: {len([individual_article for individual_article in new_final_data if np.isinf(individual_article[0]).any()])}")

    valid_data = [individual_article for individual_article in new_final_data if (individual_article[0].shape == (384,) and (not np.isnan(individual_article[0]).any()) and (not np.isinf(individual_article[0]).any()))]
    new_final_data = valid_data
    print(f"Length of new data: {len(new_final_data)}")

    zero_count = [zero_label for zero_label in y_final_data if zero_label == 0]
    one_count = [one_label for one_label in y_final_data if one_label == 1]
    print(f"Number of fake articles: {zero_count}")
    print(f"Number of real articles: {one_count}")

    # Split the processed dataset into training, validation, and testing
    # datasets for model development and evaluation.

    training_set = []
    validating_set = []
    testing_set = []

    contains_fake_article = False
    while contains_fake_article == False:
        # If a previous split did not give every dataset at least one real
        # article and one fake article, reshuffle the data and try again.
        random.shuffle(new_final_data)

        training_count = int(0.7 * len(new_final_data))
        validating_count = int(0.15 * len(new_final_data))

        training_set = new_final_data[0:training_count]
        validating_set = new_final_data[training_count:(training_count + validating_count)]
        testing_set = new_final_data[(training_count + validating_count): len(new_final_data)]

        training_set_contains_fake = False
        training_set_contains_real = False
        validating_set_contains_fake = False
        validating_set_contains_real = False
        testing_set_contains_fake = False
        testing_set_contains_real = False

        # Verify that each dataset contains at least one real article and one
        # fake article so every stage of model development uses both classes.
        for article_tuple in training_set:
            if article_tuple[1] == 0:
                training_set_contains_fake = True
            elif article_tuple[1] == 1:
                training_set_contains_real = True
        for article_tuple in validating_set:
            if article_tuple[1] == 0:
                validating_set_contains_fake = True
            elif article_tuple[1] == 1:
                validating_set_contains_real = True
        for article_tuple in testing_set:
            if article_tuple[1] == 0:
                testing_set_contains_fake = True
            elif article_tuple[1] == 1:
                testing_set_contains_real = True
        
        contains_fake_article = (training_set_contains_fake and training_set_contains_real) and (validating_set_contains_fake and validating_set_contains_real) and (testing_set_contains_fake and testing_set_contains_real)

        training_set_contains_fake = False
        training_set_contains_real = False
        validating_set_contains_fake = False
        validating_set_contains_real = False
        testing_set_contains_fake = False
        testing_set_contains_real = False

    # Save the processed training, validation, and testing datasets as NumPy files so
    # the processed data can be loaded later without rerunning the preprocessing
    # pipeline.

    # Create the path to the clean data directory relative to the project
    # location so the output files are saved consistently regardless of
    # where the script is run from.
    CURRENT_LOCATION = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_LOCATION, "..", ".."))
    DATA_LOCATION = os.path.join(PROJECT_ROOT, "data", "clean")
    os.makedirs(DATA_LOCATION, exist_ok=True)

    # Convert the datasets into NumPy object arrays so the article
    # embeddings and their labels can be saved together.
    training_set = np.array(training_set, dtype=object)
    validating_set = np.array(validating_set, dtype=object)
    testing_set = np.array(testing_set, dtype=object)

    # Save each dataset as a NumPy file so it can be loaded directly during
    # model training and evaluation.
    np.save(os.path.join(DATA_LOCATION, "new_training_dataset.npy"), training_set, allow_pickle=True)
    np.save(os.path.join(DATA_LOCATION, "new_validating_dataset.npy"), validating_set, allow_pickle=True)
    np.save(os.path.join(DATA_LOCATION, "new_testing_dataset.npy"), testing_set, allow_pickle=True)

    print("PROCESSING FINISHED SUCCESSFULLY")

def initialize_models():
    """
    Initializes the machine learning models required by the pipeline.

    This function loads the language detection model and the sentence
    embedding model used during article processing. The models are initialized
    once and stored globally so they can be reused across multiple article
    processing tasks without repeatedly loading large model files.

    Args:
        None

    Returns:
        None
    """

    # Initialize the models once before processing articles to avoid repeatedly
    # loading large models and slowing down the pipeline.

    global tokenizer
    global embedding_model

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    embedding_model = AutoModel.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16
    )

    embedding_model.eval()

def process_article_wrapper(article_tuple):
    """
    Wrapper function that allows article processing with multiprocessing.

    This function unpacks an article URL and its corresponding label from
    a tuple and passes them to the standard article processing function.
    It is used with multiprocessing.Pool.map() because worker functions
    require a single input argument.

    Args:
        article_tuple (tuple[str, str]): Tuple containing an article URL
            and its binary truthfulness label (0 = fake news, 1 = real 
            news).
    
    Returns:
        str: String indicating whether processing succeeded ("Success")
            or failed ("Fail").
        str: Error message explaining why processing failed, or None if
            processing was successful.
        tuple[np.ndarray, int, str] | str: Processing output returned by 
            the article processing function, containing the article 
            embedding, label, and detected language on success, or the 
            failed article URL otherwise.
    """

    return process_article(article_tuple[0], article_tuple[1])

def process_article(link, label):
    """
    Processes a single article URL through the training pipeline.

    This function runs an article through the preprocessing pipeline by
    extracting its content, splitting the content into smaller segments,
    detecting the article language, generating an embedding, and pairing the
    embedding with the article's binary truthfulness label.

    Args:
        link (str): URL of the article to process.
        label (int): Binary label representing the article's truthfulness
            (0 = fake news, 1 = real news).
    
    Returns:
        str: Status indicating whether processing succeeded ("Success")
            or failed ("Fail").
        str: Error message explaining why processing failed, or None if
            the pipeline completed successfully.
        tuple[np.ndarray, int, str] | str: A tuple containing the article 
            embedding, its binary truthfulness label, and detected language 
            if processing succeeds. Otherwise, contains the URL of the 
            article that failed processing.
    """

    global iteration
    print(f"Iteration {iteration} is beginning!")
    iteration += 1
    content, _, text_list, _, reason = get_content(link)
    if content is None:
        return "Fail", reason, link
    else:
        segment_list, _, _ = segment_text_and_detect_language(text_list, None)

        average_embedding = create_embedding(segment_list, tokenizer, embedding_model)

        return "Success", None, (average_embedding, label)

if __name__ == "__main__":
    main()