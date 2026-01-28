from backend.preprocessing.text_extraction import get_content, create_embeddings, analyze_language

def test_sentence_formation(list):
    '''
    print("Article text:\n")
    print(d)
    print("\n")
    '''

    i = 0
    print("List of paragraphs:")
    for text in list:
        print(f"Sentence {i + 1}: {text}")
        i += 1

def test_word_count():
    '''
    print("Article text:\n")
    print(d)
    print("\n")
    '''

    sample_list = ["I love yogurt. Yogurt is my favorite thing ever. If I didn't love yogurt, I don't know what else I would love. Oh, yeah, I really love broccoli. Broccoli is so nice as a food and as a vegetable. I love video games. In Mario Party 9, my favorite game is Toad Road. There are no unfair twists that make you lose half of your mini stars. Unlike in Boo's Horror Castle, which has like 8 boos, all of which will make you lose half of your mini stars. And also in Magma Mine, where you could lose your mini stars as many times as possible because you could hit the lava. I love yogurt. Yogurt is my favorite thing ever. If I didn't love yogurt, I don't know what else I would love. Oh, yeah, I really love broccoli. Broccoli is so nice as a food and as a vegetable. I love video games. In Mario Party 9, my favorite game is Toad Road. There are no unfair twists that make you lose half of your mini stars. Unlike in Boo's Horror Castle, which has like 8 boos, all of which will make you lose half of your mini stars. And also in Magma Mine, where you could lose your mini stars as many times as possible because you could hit the lava. I love yogurt. Yogurt is my favorite thing ever. If I didn't love yogurt, I don't know what else I would love. Oh, yeah, I really love broccoli. Broccoli is so nice as a food and as a vegetable. I love video games. In Mario Party 9, my favorite game is Toad Road. There are no unfair twists that make you lose half of your mini stars. Unlike in Boo's Horror Castle, which has like 8 boos, all of which will make you lose half of your mini stars. And also in Magma Mine, where you could lose your mini stars as many times as possible because you could hit the lava.", "SMG4 was an amazing YouTuber. Every day, he would make me laugh. His departure is something that no one would have ever expected. He will be missed."]
    x = test_translation_separation(sample_list)
    print(f"Sample list: {sample_list}\n")
    for i, j in enumerate(x):
        print(f"Index {i}: {j}")
    print("\n")
    y, z = create_embeddings(sample_list)
    print("Embeddings:")
    print(y)
    print("\n")
    print(f"Dictionary: {z}\n")

def test_translation_separation(text):
    final_text = analyze_language(text)
    return final_text

def get_embeddings(list):
    embeddings, b = create_embeddings(list)
    print(b)
    # print(embeddings)

def main():
    article_link = "https://www.npr.org/2026/01/05/nx-s1-5667078/maduro-indictment-hearing-underway"
    text, list, f, g, h = get_content(article_link)

    # test_sentence_formation(list)
    test_word_count()
    '''
    j = test_translation_separation(f)
    print(text)
    print("\n")
    for i, sentence in enumerate(j):
        print(f"Index {i + 1}: {sentence}")
    # get_embeddings(list)
    '''

    # print(test_translation_separation(["Hola. Me llamo Nikhil"]))

if __name__ == "__main__":
    main()