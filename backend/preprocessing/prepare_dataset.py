import os
import numpy as np
import pandas as pd
from backend.preprocessing.text_extraction import get_content, analyze_language, create_embeddings
import random

def main():
    news_data = pd.read_csv("data/original/FakeNewsNet.csv")

    news_data = news_data.sample(frac=1, random_state=42).reset_index(drop=True)
    test_data = news_data.iloc[1500:1600]

    article_links = test_data['news_url'].tolist()
    article_labels = test_data['real'].tolist()

    final_data = []
    failed_data = []

    i = 1
    for link, label in zip(article_links, article_labels):
        content, title, list, additional_information, reason = get_content(link)
        if content is None:
            failed_data.append({
                "link": link,
                "reason": reason
            })
        else:
            translated_content, failed_reason = analyze_language(list)
            if translated_content is None:
                failed_data.append({
                    "link": link,
                    "reason": failed_reason
                })
                continue

            embeddings, flags = create_embeddings(translated_content)

            final_data.append({
                "link": link,
                "label": label,
                "article_text": title + "\n\n" + " ".join(translated_content),
                "embeddings": embeddings,
                "flags": flags,
                "split": None
            })
        print(f"Article {i} complete")
        i += 1

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

    training_set = []
    validating_set = []
    testing_set = []

    contains_fake_article = False
    while contains_fake_article == False:
        new_final_data = final_data
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

        for individual_article in training_set:
            if individual_article['label'] == 0:
                valid_training_set = True
            individual_article['split'] = "train"
        for individual_article in validating_set:
            if individual_article['label'] == 0:
                valid_validating_set = True
            individual_article['split'] = "validate"
        for individual_article in testing_set:
            if individual_article['label'] == 0:
                valid_testing_set = True
            individual_article['split'] = "test"
        
        contains_fake_article = valid_training_set and valid_validating_set and valid_testing_set

        valid_training_set = False
        valid_validating_set = False
        valid_testing_set = False

    CURRENT_LOCATION = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_LOCATION, "..", ".."))
    DATA_LOCATION = os.path.join(PROJECT_ROOT, "data", "clean")
    os.makedirs(DATA_LOCATION, exist_ok=True)

    np.save(os.path.join(DATA_LOCATION, "training_dataset.npy"), training_set)
    np.save(os.path.join(DATA_LOCATION, "validating_dataset.npy"), validating_set)
    np.save(os.path.join(DATA_LOCATION, "testing_dataset.npy"), testing_set)

if __name__ == "__main__":
    main()