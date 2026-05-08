import pandas as pd
import random
from backend.preprocessing.text_extraction import get_content

def main():
    news_data = pd.read_csv("data/original/FakeNewsNet.csv")
    news_data = news_data.sample(frac=1, random_state=42).reset_index(drop=True)
    test_data = news_data.iloc[:]

    article_links = test_data['news_url'].tolist()
    final_list = []

    while len(final_list) < 60:
        random_link = random.choice(article_links)

        content, title, text_list, additional_information, reason = get_content(random_link)
        if not content is None:
            final_list.append(content)
            print(f"Articles processed: {len(final_list)}")
        
        article_links.remove(random_link)
    
    for i, content in enumerate(final_list):
        print(f"Article {i + 1}:")
        print(content)
        print(" ")

if __name__ == "__main__":
    main()