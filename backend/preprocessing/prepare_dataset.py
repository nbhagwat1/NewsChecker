import numpy as np
import pandas as pd
from backend.preprocessing.text_extraction import get_content, analyze_language, create_embeddings

def main():
    news_data = pd.read_csv("data/original/FakeNewsNet.csv")

    article_links = news_data['news_url'].tolist()
    article_labels = news_data['real'].tolist()

    final_data = []
    failed_data = []

    for link in article_links:
        content, title, list, additional_information, reason = get_content(link)
        if content is None:
            failed_data.append({
                "link": link,
                "reason": reason
            })
        else:
            translated_content = analyze_language(content)
            embeddings = create_embeddings