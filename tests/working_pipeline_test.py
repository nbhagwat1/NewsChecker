import pandas as pd
import random
from backend.preprocessing.text_extraction import get_content

def main():
    """
    Examines the results of the training pipeline successfully extracting the main
    article content from many article URLs.

    This method finds 60 article URLs that the training pipeline can successfully
    extract the main article content from. Once it successfully extracts the main
    article content from those 60 article URLs, it prints the extracted article
    content.

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