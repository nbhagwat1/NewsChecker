from backend.model.train import get_content, create_embeddings, analyze_language

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

    sample_list = ["I love yogurt. Yogurt is my favorite thing ever. If I didn't love yogurt, I don't know what else I would love. Oh, yeah, I really love broccoli. Broccoli is so nice as a food and as a vegetable. I love video games. In Mario Party 9, my favorite game is Toad Road. There are no unfair twists that make you lose half of your mini stars. Unlike in Boo's Horror Castle, which has like 8 boos, all of which will make you lose half of your mini stars. And also in Magma Mine, where you could lose your mini stars as many times as possible because you could hit the lava. I love yogurt. Yogurt is my favorite thing ever. If I didn't love yogurt, I don't know what else I would love. Oh, yeah, I really love broccoli. Broccoli is so nice as a food and as a vegetable. I love video games. In Mario Party 9, my favorite game is Toad Road. There are no unfair twists that make you lose half of your mini stars. Unlike in Boo's Horror Castle, which has like 8 boos, all of which will make you lose half of your mini stars. And also in Magma Mine, where you could lose your mini stars as many times as possible because you could hit the lava. I love yogurt. Yogurt is my favorite thing ever. If I didn't love yogurt, I don't know what else I would love. Oh, yeah, I really love broccoli. Broccoli is so nice as a food and as a vegetable. I love video games. In Mario Party 9, my favorite game is Toad Road. There are no unfair twists that make you lose half of your mini stars. Unlike in Boo's Horror Castle, which has like 8 boos, all of which will make you lose half of your mini stars. And also in Magma Mine, where you could lose your mini stars as many times as possible because you could hit the lava.", "SMG4 was an amazing YouTuber. Every day, he would make me laugh. His departure is something that no one would have ever expected. He will be missed."]
    create_embeddings(sample_list)

def test_translation_separation():
    article_link = "https://www.cnn.com/2026/01/05/politics/jd-vance-ohio-residence"
    a, b, c, d, e, f, g, h = get_content(article_link)
    analyze_language(d)

def main():
    test_sentence_formation()
    # test_word_count()
    test_translation_separation()

if __name__ == "__main__":
    main()