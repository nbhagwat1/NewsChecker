from transformers import logging
logging.set_verbosity_error()
import os
import numpy as np
import pandas as pd
from backend.preprocessing.text_extraction import get_content, segment_text_and_detect_language, create_embeddings
import random
from huggingface_hub import hf_hub_download
import fasttext
from sentence_transformers import SentenceTransformer
from multiprocessing import Pool

language_model = None
detection_model = None
embedding_model = None
iteration = 0

def main():
    """
    Runs the entire original dataset of article URLs through the training
    pipeline to produce the new and cleaned dataset, prints information relevant
    to the training pipeline's performance, splits the new and cleaned dataset
    into training, validation, and testing datasets, and saves the training,
    validation, and testing datasets as .npy files.

    The method loads the .csv file that contains the original dataset and retrieves
    the set of article URLs and their respective binary labels that signify their
    truthfulness. The method then creates a pool of six worker processes that concurrently
    process the article URLs and their respective binary truthfulness labels. Once the
    entire original dataset has been processed, the method prints important information
    related to the training pipeline's performance, such as how many article URLs it
    successfully processed, how many article URLs it didn't successfully process, and
    how many article URLs it didn't successfully process because of a certain reason.
    After that, the method takes the article URLs that were successfully processed and
    adds their resulting embeddings to one dataset and their respective binary
    truthfulness labels to another dataset. Once that is done, the method pairs each
    embedding with its respective binary truthfulness label and creates a new dataset
    consisting of those pairs using the built-in zip() function. The method then splits 
    that dataset into a training dataset, a validation dataset, and a testing dataset
    using a 70/15/15 split, ensuring that each dataset contains at least one embedding
    that captures the meaning of a fake news article. Finally, the method saves each of
    the three datasets (the training dataset, the validation dataset, and the testing
    dataset) as .npy files in the `data/clean` directory of the NewsChecker project.

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
    language_data = [result[2][2] for result in results if result[0] == "Success"]

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
    print(f"Embeddings with invalid shape: {len([individual_article for individual_article in new_final_data if individual_article[0].shape != (768,)])}")
    print(f"Embeddings with NaN: {len([individual_article for individual_article in new_final_data if np.isnan(individual_article[0]).any()])}")
    print(f"Embeddings with INF: {len([individual_article for individual_article in new_final_data if np.isinf(individual_article[0]).any()])}")

    valid_data = [individual_article for individual_article in new_final_data if (individual_article[0].shape == (768,) and (not np.isnan(individual_article[0]).any()) and (not np.isinf(individual_article[0]).any()))]
    new_final_data = valid_data
    print(f"Length of new data: {len(new_final_data)}")
    print("-----------------")
    print("LANGUAGE DATA")
    print(f"Number of English articles: {len([statistic for statistic in language_data if statistic == 'eng'])}")
    print(f"Number of non-English articles: {len([statistic for statistic in language_data if statistic != 'eng'])}")
    print("-----------------")

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
    np.save(os.path.join(DATA_LOCATION, "training_dataset.npy"), training_set, allow_pickle=True)
    np.save(os.path.join(DATA_LOCATION, "validating_dataset.npy"), validating_set, allow_pickle=True)
    np.save(os.path.join(DATA_LOCATION, "testing_dataset.npy"), testing_set, allow_pickle=True)

    print("PROCESSING FINISHED SUCCESSFULLY")

def initialize_models():
    """
    Initializes all of the models that the pipeline needs to
    process the articles in the original dataset.

    This method loads and initializes the models that the
    pipeline uses to detect the article's text's language
    and generate an embedding that captures the meaning of
    the article.

    Args:
        None

    Returns:
        None
    """

    # Initialize the models once before processing articles to avoid repeatedly
    # loading large models and slowing down the pipeline.

    global language_model
    global detection_model
    global embedding_model
    
    language_model = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
    detection_model = fasttext.load_model(language_model)
    embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

def process_article_wrapper(article_tuple):
    """
    Wrapper function used by multiprocessing workers.

    Calls the standard article processing function so that articles can be
    processed concurrently using Pool.map().

    Args:
        article_tuple (tuple[str, str]): A tuple that contains an article's URL and a binary label that
        represents the article's truthfulness.
    
    Returns:
        str: A string that states whether or not the training pipeline successfully processed the article URL.
        str: A string that explains why the training pipeline failed to process the article URL (if necessary).
        tuple[np.ndarray, int, str]: A tuple that contains the single embedding that captures the meaning of the
        article, a binary label that represents the truthfulness of the article, and a string that states the
        language of the article's text.
    """

    return process_article(article_tuple[0], article_tuple[1])

def process_article(link, label):
    """
    Runs the URL of an article through the training pipeline and returns
    the finishing result along with the article's respective binary label
    that represents its truthfulness.

    This method takes the inputted article URL, extracts its main article
    content, divides this main article content into segments of up to 
    300 words, detects the language of this main article content, and
    generates a single embedding that captures the meaning of the article.
    The method then returns this single embedding, a string that represents
    the detected language of the article, and a binary number that represents
    the truthfulness of the article (0 = fake news, 1 = real news).

    Args:
        link (str): The article's URL.
        label (int): A binary label that represents the article's truthfulness.
    
    Returns:
        str: A string that states whether or not the training pipeline successfully processed the article URL.
        str: A string that explains why the training pipeline failed to process the article URL (if necessary).
        tuple[np.ndarray, int, str]: A tuple that contains the single embedding that captures the meaning of the
        article, a binary label that represents the truthfulness of the article, and a string that states the
        language of the article's text.
    """

    global iteration
    print(f"Iteration {iteration} is beginning!")
    iteration += 1
    content, title, text_list, additional_information, reason = get_content(link)
    if content is None:
        return "Fail", reason, link
    else:
        translated_content, failed_reason, language = segment_text_and_detect_language(text_list, detection_model)

        average_embedding, flags = create_embeddings(translated_content, embedding_model)

        return "Success", None, (average_embedding, label, language)

if __name__ == "__main__":
    main()