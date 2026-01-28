import numpy as np
import pandas as pd
from backend.preprocessing.text_extraction import get_content, analyze_language, create_embeddings

def main():
    news_data = pd.read_csv("data/original/FakeNewsNet.csv")

    article_links = news_data['news_url'].tolist()
    article_labels = news_data['real'].tolist()

    final_data = []
    failed_data = []

    for link, label in zip(article_links, article_labels):
        content, title, list, additional_information, reason = get_content(link)
        if content is None:
            failed_data.append({
                "link": link,
                "reason": reason
            })
        else:
            translated_content = analyze_language(list)
            embeddings, flags = create_embeddings(translated_content)
            final_data.append({
                "link": link,
                "label": label,
                "article_text": title + "\n\n" + " ".join(translated_content),
                "embeddings": embeddings,
                "flags": flags
            })
    
    print(f"Length of valid articles: {len(final_data)}")
    print(f"Length of invalid articles: {len(failed_data)}")
    
if __name__ == "__main__":
    main()