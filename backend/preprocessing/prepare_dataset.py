import os
import numpy as np
import pandas as pd
from backend.preprocessing.text_extraction import get_content, analyze_language, create_embeddings
import random
import psutil
from huggingface_hub import hf_hub_download
import fasttext
from sentence_transformers import SentenceTransformer

def main():
    news_data = pd.read_csv("data/original/FakeNewsNet.csv")

    # print(os.cpu_count())

    news_data = news_data.sample(frac=1, random_state=42).reset_index(drop=True)
    test_data = news_data.iloc[1500:1540]

    article_links = test_data['news_url'].tolist()
    article_labels = test_data['real'].tolist()

    # print(len(article_links))

    X_final_data = []
    y_final_data = []
    failed_data = []

    language_model = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
    detection_model = fasttext.load_model(language_model)
    embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    process = psutil.Process(os.getpid())
    i = 1
    for link, label in zip(article_links, article_labels):
        print(f"Iteration {i}/{len(list(zip(article_links, article_labels)))} is starting!")

        content, title, text_list, additional_information, reason = get_content(link)
        if content is None:
            failed_data.append(link)
            print("Processing failed: Could not get cleaned text")
        else:
            translated_content, failed_reason = analyze_language(text_list, detection_model)
            if translated_content is None:
                failed_data.append(link)
                print("Processing failed: Could not translate text")
                continue

            average_embedding, flags = create_embeddings(translated_content, embedding_model)

            X_final_data.append(average_embedding)
            y_final_data.append(label)

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
        print(f"Article {i} complete")
        i += 1
        print("RAM usage (MB):", process.memory_info().rss / 1024 / 1024)

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

    training_set = []
    validating_set = []
    testing_set = []

    new_final_data = list(zip(X_final_data, y_final_data))

    contains_fake_article = False
    while contains_fake_article == False:
        random.shuffle(new_final_data)

        training_count = int(0.7 * len(new_final_data))
        validating_count = int(0.15 * len(new_final_data))
        # testing_count = len(new_final_data) - training_count - validating_count

        training_set = new_final_data[0:training_count]
        validating_set = new_final_data[training_count:(training_count + validating_count)]
        testing_set = new_final_data[(training_count + validating_count): len(new_final_data)]

        valid_training_set = False
        valid_validating_set = False
        valid_testing_set = False

        for article_tuple in training_set:
            if article_tuple[1] == 0:
                valid_training_set = True
        for article_tuple in validating_set:
            if article_tuple[1] == 0:
                valid_validating_set = True
        for article_tuple in testing_set:
            if article_tuple[1] == 0:
                valid_testing_set = True
        
        contains_fake_article = valid_training_set and valid_validating_set and valid_testing_set

        valid_training_set = False
        valid_validating_set = False
        valid_testing_set = False

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

if __name__ == "__main__":
    main()