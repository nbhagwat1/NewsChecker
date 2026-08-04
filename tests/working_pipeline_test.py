import pandas as pd
import random
from backend.preprocessing.text_extraction import get_content

def main():
    """
    Samples successfully processed articles from the training pipeline.

    This function repeatedly selects random article URLs from the original
    dataset until 60 articles have been successfully processed by the
    content extraction pipeline. It then prints the extracted article text
    for each successful article, allowing the extraction results to be
    inspected manually.

    Args:
        None

    Returns:
        None
    """
    
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