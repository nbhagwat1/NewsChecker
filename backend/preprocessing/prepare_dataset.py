from transformers import logging
logging.set_verbosity_error()
import os
import numpy as np
import pandas as pd
from backend.preprocessing.text_extraction import get_content, analyze_language, create_embeddings
import random
import psutil
from huggingface_hub import hf_hub_download
import fasttext
from sentence_transformers import SentenceTransformer
from multiprocessing import Pool

language_model = None
detection_model = None
embedding_model = None
iteration = 0

def main():
    news_data = pd.read_csv("data/original/FakeNewsNet.csv")

    # print(os.cpu_count())

    news_data = news_data.sample(frac=1, random_state=42).reset_index(drop=True)
    test_data = news_data.iloc[:]

    article_links = test_data['news_url'].tolist()
    article_labels = test_data['real'].tolist()

    # print(len(article_links))

    X_final_data = []
    y_final_data = []
    failed_data = []

    # process = psutil.Process(os.getpid())

    pool = Pool(processes=6, initializer=initialize_models)
    results = pool.map(process_article_wrapper, zip(article_links, article_labels))
    pool.close()
    pool.join()

    failed_data = [result for result in results if result[0] == "Fail"]
    X_final_data = [result[2][0] for result in results if result[0] == "Success"]
    y_final_data = [result[2][1] for result in results if result[0] == "Success"]
    language_data = [result[2][2] for result in results if result[0] == "Success"]

    # print(f"Length of valid articles: {len(final_data)}")
    # print(f"Length of invalid articles: {len(failed_data)}")

    '''
    fake_article_count = 0
    for valid_article in final_data:
        if valid_article['label'] == 0:
            fake_article_count += 1
    
    print("\n")
    print(f"Fake articles: {fake_article_count}")
    print(f"Real articles: {len(final_data) - fake_article_count}")
    '''

    # print(f"{len(X_final_data)} vs. {len(y_final_data)}")

    print(f"Length of successful data: {len(X_final_data)}")
    print(f"Length of failed data: {len(failed_data)}")
    print("-----------------")
    print("FAILED DATA STATS")
    print(f"HTTP request failed: {len([failure for failure in failed_data if failure[1] == 'HTTP request failed'])}")
    print(f"Pipeline failed: {len([failure for failure in failed_data if failure[1] == 'Text cleanup function got rid of everything'])}")
    print(f"Translation failed: {len([failure for failure in failed_data if failure[1] == 'Translation failed'])}")
    print(f"Other reason: {len([failure for failure in failed_data if (failure[1] != 'HTTP request failed' and failure[1] != 'Text cleanup function got rid of everything' and failure[1] != 'Translation failed')])}")
    print("-----------------")

    training_set = []
    validating_set = []
    testing_set = []

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

    contains_fake_article = False
    while contains_fake_article == False:
        random.shuffle(new_final_data)

        training_count = int(0.7 * len(new_final_data))
        validating_count = int(0.15 * len(new_final_data))
        # testing_count = len(new_final_data) - training_count - validating_count

        training_set = new_final_data[0:training_count]
        validating_set = new_final_data[training_count:(training_count + validating_count)]
        testing_set = new_final_data[(training_count + validating_count): len(new_final_data)]

        training_set_contains_fake = False
        training_set_contains_real = False
        validating_set_contains_fake = False
        validating_set_contains_real = False
        testing_set_contains_fake = False
        testing_set_contains_real = False

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

    CURRENT_LOCATION = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_LOCATION, "..", ".."))
    DATA_LOCATION = os.path.join(PROJECT_ROOT, "data", "clean")
    os.makedirs(DATA_LOCATION, exist_ok=True)

    training_set = np.array(training_set, dtype=object)
    validating_set = np.array(validating_set, dtype=object)
    testing_set = np.array(testing_set, dtype=object)

    np.save(os.path.join(DATA_LOCATION, "training_dataset.npy"), training_set, allow_pickle=True)
    np.save(os.path.join(DATA_LOCATION, "validating_dataset.npy"), validating_set, allow_pickle=True)
    np.save(os.path.join(DATA_LOCATION, "testing_dataset.npy"), testing_set, allow_pickle=True)

    print("PROCESSING FINISHED SUCCESSFULLY")

def initialize_models():
    global language_model
    global detection_model
    global embedding_model
    
    language_model = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
    detection_model = fasttext.load_model(language_model)
    embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

def process_article_wrapper(article_tuple):
    return process_article(article_tuple[0], article_tuple[1])

def process_article(link, label):
    global iteration
    print(f"Iteration {iteration} is beginning!")
    iteration += 1
    content, title, text_list, additional_information, reason = get_content(link)
    if content is None:
        return "Fail", reason, link
    else:
        translated_content, failed_reason, language = analyze_language(text_list, detection_model)

        average_embedding, flags = create_embeddings(translated_content, embedding_model)

        return "Success", None, (average_embedding, label, language)

        '''
        final_data.append({
            "link": link,
            "label": label,
            "article_text": title + "\n\n" + " ".join(translated_content),
            "embeddings": embeddings,
            "flags": flags,
            "split": None
        })
        '''

if __name__ == "__main__":
    main()