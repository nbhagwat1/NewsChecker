import numpy as np
import pandas as pd
from backend.preprocessing.text_extraction import examine_link, get_content, analyze_language, create_embeddings

def main():
    news_data = pd.read_csv("data/original/FakeNewsNet.csv")

    article_links = news_data['news_url'].tolist()