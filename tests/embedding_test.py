from backend.model.train import get_content, create_embeddings

def test_sentence_formation():
    article_link = "https://www.cnn.com/2026/01/05/politics/jd-vance-ohio-residence"
    a, b, c, d, e, f, g, h = get_content(article_link)
    '''
    print("Article text:\n")
    print(d)
    print("\n")
    '''

    i = 0
    print("List of paragraphs:")
    for text in e:
        print(f"Sentence {i + 1}: {text}")
        i += 1

def test_word_count():
    article_link = "https://www.cnn.com/2026/01/05/politics/jd-vance-ohio-residence"
    a, b, c, d, e, f, g, h = get_content(article_link)
    '''
    print("Article text:\n")
    print(d)
    print("\n")
    '''

    print(create_embeddings(d, e))

def main():
    test_sentence_formation()
    test_word_count()

if __name__ == "__main__":
    main()