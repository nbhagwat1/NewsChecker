from backend.preprocessing.text_extraction import get_content, create_embedding, segment_text_and_detect_language
from transformers import AutoTokenizer, AutoModel
import torch

def print_segments_chronologically(segments):
    """
    Prints text segments in the order they appear in the input list.

    Each segment is printed with its position in the list. This function
    is intended as a debugging utility for inspecting the text produced
    during preprocessing.

    Args:
        segments (list[str]): Text segments to display.

    Returns:
        None
    """

    segment_index = 0
    print("List of segments:")
    for segment_text in segments:
        print(f"Segment {segment_index + 1}: {segment_text}")
        segment_index += 1

def examine_embedding_generation():
    """
    Tests the text segmentation and embedding generation pipeline.

    This function creates sample article text, loads the
    all-MiniLM-L6-v2 model, segments the sample text, and generates
    an embedding representing the article. The resulting segments
    and embedding are printed for inspection.

    This function is intended as a debugging utility for verifying
    that text segmentation and embedding generation work as expected.

    Args:
        None

    Returns:
        None
    """

    sample_list = [
        "I love yogurt. Yogurt is my favorite thing ever. If I didn't love yogurt, I don't know what else I would love. Oh, yeah, I really love broccoli. Broccoli is so nice as a food and as a vegetable. I love video games. In Mario Party 9, my favorite game is Toad Road. There are no unfair twists that make you lose half of your mini stars. Unlike in Boo's Horror Castle, which has like 8 boos, all of which will make you lose half of your mini stars. And also in Magma Mine, where you could lose your mini stars as many times as possible because you could hit the lava. I love yogurt. Yogurt is my favorite thing ever. If I didn't love yogurt, I don't know what else I would love. Oh, yeah, I really love broccoli. Broccoli is so nice as a food and as a vegetable. I love video games. In Mario Party 9, my favorite game is Toad Road. There are no unfair twists that make you lose half of your mini stars. Unlike in Boo's Horror Castle, which has like 8 boos, all of which will make you lose half of your mini stars. And also in Magma Mine, where you could lose your mini stars as many times as possible because you could hit the lava. I love yogurt. Yogurt is my favorite thing ever. If I didn't love yogurt, I don't know what else I would love. Oh, yeah, I really love broccoli. Broccoli is so nice as a food and as a vegetable. I love video games. In Mario Party 9, my favorite game is Toad Road. There are no unfair twists that make you lose half of your mini stars. Unlike in Boo's Horror Castle, which has like 8 boos, all of which will make you lose half of your mini stars. And also in Magma Mine, where you could lose your mini stars as many times as possible because you could hit the lava.", "SMG4 was an amazing YouTuber. Every day, he would make me laugh. His departure is something that no one would have ever expected. He will be missed."
    ]

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    # Load the tokenizer and model used to turn the sample text
    # into numerical representations of its meaning.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16
    )

    model.eval()

    # Split the sample article into smaller text segments before
    # generating its embedding.
    segments = examine_text_segmentation_and_language_detection(sample_list)

    print(f"Sample list: {sample_list}\n")

    for segment_index, segment in enumerate(segments):
        print(f"Index {segment_index}: {segment}")

    print("\n")

    # Generate one embedding representing the entire article from
    # the individual text segments.
    embeddings = create_embedding(segments, tokenizer, model)

    print("Embeddings:")
    print(embeddings)

def examine_text_segmentation_and_language_detection(paragraph_list):
    """
    Tests the article text segmentation process.

    This function passes article paragraphs through the text
    segmentation and language detection pipeline and returns the
    resulting text segments.

    Args:
        paragraph_list (list[str]): Article paragraphs to process.

    Returns:
        list[str]: Text segments produced from the input paragraphs.
    """

    segment_list, _, _ = segment_text_and_detect_language(paragraph_list, None)
    return segment_list

def create_and_print_embeddings(segment_list):
    """
    Generates and prints a single embedding representing an article.

    This function takes a list of article text segments, generates a
    semantic embedding representing the article content, and prints
    the resulting embedding.

    Args:
        segment_list (list[str]): Text segments that make up the article.

    Returns:
        None
    """

    embedding = create_embedding(segment_list)
    print(embedding)

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

    # Use a real article to test whether the content extraction
    # pipeline can successfully retrieve its main text.
    article_link = "https://www.npr.org/2026/01/05/nx-s1-5667078/maduro-indictment-hearing-underway"
    
    article_text, _, _, _, _ = get_content(article_link)

    print(article_text)

if __name__ == "__main__":
    main()