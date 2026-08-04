from backend.preprocessing.text_extraction import get_content, create_embeddings, segment_text_and_detect_language

def print_segments_chronologically(segments):
    """
    Prints each text segment in the order it appears in the list.

    This function iterates through the input list of text segments and
    prints each segment together with its position in the list. It is
    primarily intended as a debugging utility for inspecting article
    content after preprocessing.

    Args:
        segments (list[str]): A list of text segments in the order they
            appear in the article.

    Returns:
        None
    """

    i = 0
    print("List of segments:")
    for text in segments:
        print(f"Segment {i + 1}: {text}")
        i += 1

def examine_embedding_generation():
    """
    Demonstrates the text segmentation and embedding generation pipeline.

    This function creates sample article text, splits it into segments of
    up to 300 words, prints the resulting segments, generates a semantic
    embedding for the article, and prints both the embedding and the
    embedding quality information. It is intended as a debugging utility
    for verifying that text segmentation and embedding generation behave
    as expected.

    Args:
        None

    Returns:
        None
    """

    sample_list = [
        "I love yogurt. Yogurt is my favorite thing ever. If I didn't love yogurt, I don't know what else I would love. Oh, yeah, I really love broccoli. Broccoli is so nice as a food and as a vegetable. I love video games. In Mario Party 9, my favorite game is Toad Road. There are no unfair twists that make you lose half of your mini stars. Unlike in Boo's Horror Castle, which has like 8 boos, all of which will make you lose half of your mini stars. And also in Magma Mine, where you could lose your mini stars as many times as possible because you could hit the lava. I love yogurt. Yogurt is my favorite thing ever. If I didn't love yogurt, I don't know what else I would love. Oh, yeah, I really love broccoli. Broccoli is so nice as a food and as a vegetable. I love video games. In Mario Party 9, my favorite game is Toad Road. There are no unfair twists that make you lose half of your mini stars. Unlike in Boo's Horror Castle, which has like 8 boos, all of which will make you lose half of your mini stars. And also in Magma Mine, where you could lose your mini stars as many times as possible because you could hit the lava. I love yogurt. Yogurt is my favorite thing ever. If I didn't love yogurt, I don't know what else I would love. Oh, yeah, I really love broccoli. Broccoli is so nice as a food and as a vegetable. I love video games. In Mario Party 9, my favorite game is Toad Road. There are no unfair twists that make you lose half of your mini stars. Unlike in Boo's Horror Castle, which has like 8 boos, all of which will make you lose half of your mini stars. And also in Magma Mine, where you could lose your mini stars as many times as possible because you could hit the lava.", "SMG4 was an amazing YouTuber. Every day, he would make me laugh. His departure is something that no one would have ever expected. He will be missed."
    ]
    
    x = examine_text_segmentation_and_language_detection(sample_list)
    print(f"Sample list: {sample_list}\n")
    for i, j in enumerate(x):
        print(f"Index {i}: {j}")
    print("\n")
    y, z = create_embeddings(sample_list)
    print("Embeddings:")
    print(y)
    print("\n")
    print(f"Dictionary: {z}\n")

def examine_text_segmentation_and_language_detection(paragraph_list):
    """
    Segments article text and detects its language.

    This function processes a list of article paragraphs by splitting the
    text into segments of up to 300 words and detecting the language of the
    article content. It returns both the generated text segments and the
    detected language.

    Args:
        paragraph_list (list[str]): A list of article paragraphs to be
            segmented and analyzed.

    Returns:
        list[str]: A list of text segments containing the article content,
            with each segment limited to approximately 300 words.
        str: The detected language code of the article text.
    """

    segment_list, language = segment_text_and_detect_language(paragraph_list)
    return segment_list, language

def create_and_print_embeddings(segment_list):
    """
    Tests and displays the article embedding generation process.

    This function takes a list of article text segments, generates a single
    semantic embedding that represents the article content, and prints the
    generated embedding along with a dictionary containing checks for
    suspicious embedding characteristics, such as insufficient content,
    invalid values, unusual segment lengths, or low variation.

    Args:
        segment_list (list[str]): A list of text segments that together make
            up the article content.

    Returns:
        None
    """

    embeddings, b = create_embeddings(segment_list)
    print(embeddings)
    print(b)

def main():
    """
    Tests the article content extraction pipeline on a sample article URL.

    This function passes a predefined article URL to the content extraction
    pipeline and prints the extracted main article content. It is intended
    as a debugging utility for verifying that article extraction works
    correctly.

    Args:
        None

    Returns:
        None
    """

    article_link = "https://www.npr.org/2026/01/05/nx-s1-5667078/maduro-indictment-hearing-underway"
    text, list, f, g, h = get_content(article_link)

    print(text)

if __name__ == "__main__":
    main()