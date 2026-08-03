from backend.preprocessing.text_extraction import get_content, create_embeddings, segment_text_and_detect_language

def print_paragraphs_chronologically(list):
    """
    Prints each of the strings in the list in chronological order.

    The parameter represents a list of paragraphs in the order that they occur in the main content of
    an article. This method simply prints each paragraph in the list in chronological order, doing so
    in a structured format.

    Args:
        list (list[str]): A list of strings, with each string being a paragraph in the main content of
        an article.
    
    Returns:
        None
    """

    i = 0
    print("List of paragraphs:")
    for text in list:
        print(f"Sentence {i + 1}: {text}")
        i += 1

def examine_embedding_generation():
    """
    Prints a very long string, the same string when split into segments that are up to 300 words long, an embedding 
    that captures the meaning of the very long string, and important information about the embedding.

    This method takes a very long string that is over 300 words long and splits it into segments that are up to 300
    words long, adding all of those segments to a list. The method then prints each of these segments in the order
    that they occur in the original string. Then, the method converts the list of segments to a singular embedding
    that captures the meaning of the original string. Finally, the method prints this singular embedding as well as
    a dictionary that reveals important statistics about the embedding.

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
    segment_list, language = segment_text_and_detect_language(paragraph_list)
    return segment_list

def create_and_print_embeddings(list):
    embeddings, b = create_embeddings(list)
    print(b)

def main():
    """
    Tests the main article content extraction method on a random article URL.

    This method takes an article URL and extracts and prints its main article content.

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