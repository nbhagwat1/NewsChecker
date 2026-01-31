import numpy as np
import pandas as pd
from backend.preprocessing.text_extraction import get_content, analyze_language, create_embeddings

def main():
    news_data = pd.read_csv("data/original/FakeNewsNet.csv")

    news_data = news_data.sample(frac=1, random_state=42).reset_index(drop=True)
    test_data = news_data.iloc[500:600]

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

            # print("Method returned something")

            embeddings, flags = create_embeddings(translated_content)

            # print("Another method returned something")

            final_data.append({
                "link": link,
                "label": label,
                "article_text": title + "\n\n" + " ".join(translated_content),
                "embeddings": embeddings,
                "flags": flags
            })
        print(f"Article {i} complete")
        i += 1

    print(f"Length of valid articles: {len(final_data)}")
    print(f"Length of invalid articles: {len(failed_data)}")

    fake_article_count = 0
    for valid_article in final_data:
        if valid_article['label'] == 0:
            fake_article_count += 1
    
    print("\n")
    print(f"Fake articles: {fake_article_count}")
    print(f"Real articles: {len(final_data) - fake_article_count}")
    
if __name__ == "__main__":
    main()