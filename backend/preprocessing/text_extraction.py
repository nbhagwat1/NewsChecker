import requests
import re
from bs4 import BeautifulSoup, Tag
from sentence_transformers import SentenceTransformer
import fasttext
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
import math
import time
import gc

def examine_link(link):
    """
    Checks if the specified parameter is a valid link.

    This method checks if the specified parameter contains exactly
    one '.' character and no empty spaces. If the specified parameter
    does, then it is a valid link, and the method does nothing. 
    Otherwise, the specified parameter is not a valid link, and the
    running program terminates.

    Args:
        link (str): The link that will be checked for validity

    Returns:
        None
    """
    link = link.strip().lower()

    empty_count = 0
    dot_count = 0
    for char in link:
        if char == " ":
            empty_count += 1
        elif char == ".":
            dot_count += 1
    if empty_count > 0 or dot_count == 0:
        print("Not a link")
        exit(0)

def get_content(link):
    """
    Extracts the main text content from a news article URL.

    Retrieves the webpage HTML through an HTTP request, uses BeautifulSoup
    to process the page structure, and removes unnecessary content such as
    scripts, navigation menus, and advertisements to extract the main article text.

    Args:
        link (str): The URL of the news article whose main content will be extracted.
    
    Returns:
        str: The main article text of the news article that the URL leads to.
        str: The title of the news article that the URL leads to.
        list[str]: List of text segments that together make up the main article content.
        dict: Information removed from the main article content that may still be useful
        for assessing the article's trustworthiness.
        str: An error message that explains why the method failed to extract the main
        text content from the inputted news article URL (if necessary).
    """

    if (link is None) or (isinstance(link, float) and math.isnan(link)):
        return None, None, None, None, "Link is missing"

    if not link.startswith("http://") and not link.startswith("https://"):
        link = "https://" + link

    headers = {
        "User-Agent": "NewsChecker/1.0 (learning project)"
    }

    response = None

    for attempt in range(3):
        try:    
            response = requests.get(link, headers=headers, timeout=20)
            response.raise_for_status()
            break
        except requests.exceptions.ReadTimeout:
            time.sleep(2)
        except requests.exceptions.RequestException as e:
            code = 0
            if e.response:
                code = e.response.status_code
            reason = f"Error code {code}: "

            if code == 403:
                reason += "Website does not provide permission to get HTML"
            elif code == 404:
                reason += "URL is not recognized"
            elif code == 401:
                reason += "Client must authenticate itself to get HTML"
            else:
                reason += "Complicated"
            
            return None, None, None, None, "HTTP request failed"

    if response is None:
        return None, None, None, None, "HTTP request failed"
    if (response.status_code != 200):
        code = response.status_code
        return None, None, None, None, "HTTP request failed"
    

    website_content = response.text
    website_code = BeautifulSoup(website_content, 'html.parser')

    heading = website_code.find("h1") # CHANGE - May not be safe to use later
    if heading:
        website_title = heading.get_text(strip=True)
        heading.decompose()
    else:
        if website_code.title and website_code.title.string:
            website_title = website_code.title.string
        else:
            website_title = ""

    website_text = ""
    time_list = []
    emphasized_text_list = []
    footer_information = []
    recommended_list = []
    menu_list = []
    publish_list = []
    source_list = []
    structure_list = []
    distracting_words = ["click here", "learn more", "check out", "this article originally appeared", "subscribe", "premium", "originally published"]

    for tag in website_code(["script", "style", "noscript", "meta", "header", "footer", "img", "nav", "aside", "style", "figcaption", "button"]):
        tag.decompose()
    for tag in website_code("a"):
        for word in distracting_words:
            if word in tag.get_text(strip=True).lower():
                tag.decompose()
    for tag in website_code("time"):
        time_list.append(tag.get_text(strip=True))
        tag.decompose()
    for tag in website_code(["em", "i", "strong", "b", "u", "mark"]):
        emphasized_text_list.append(tag.get_text(strip=True))
        for word in distracting_words:
            if word in tag.get_text(strip=True).lower():
                tag.decompose()
                continue
            
            tag_text = tag.get_text(strip=True).lower()
            if ("more" in tag_text or "related" in tag_text) and "news" in tag_text and ":" in tag_text:
                tag.decompose()
    for tag in website_code("li"):
        if not tag.find_all(["p", "span"]):
            tag.decompose()
    for tag in website_code(["p", "li"]):
        tag_children = []
        for child in tag.contents:
            if str(child).strip():
                tag_children.append(child)
        tag_children = [c for c in tag_children if c != ' HTML_TAG_START ' and c != ' HTML_TAG_END ']

        if len(tag_children) == 1:
            only_child = tag_children[0]
            if only_child.name == "a":
                tag.decompose()
            elif isinstance(only_child, Tag) and (len(list(only_child.descendants)) >= 2 and list(only_child.descendants)[len(list(only_child.descendants)) - 2].name == "a"):
                tag.decompose()
    for tag in website_code.find_all(True):
        if tag.name == "html" or tag.name == "body":
            continue
        if isinstance(tag, Tag) == False:
            continue
        if not tag.attrs:
            continue

        style = tag.get("style", "")
        if "visibility:hidden" in style:
            tag.decompose()
            continue
        if tag.get("aria-hidden") == "true":
            tag.decompose()
            continue

        class_list = tag.get('class', [])
        data = tag.get("data-testid")
        important_words = ["metadata", "social-link", "social-share", "social-bookmark", "follow-topics", "footnote", "caption", "byline", "subscribe", "newsletter", "footer", "headline", "promotion", "prism-card", "recommended", "licensing", "button", "description", "infobox", "menu", "publish", "boilerplate", "source"]
        decomposed = False

        if data or class_list:
            if data:
                for word in important_words:
                    if word in data.lower():
                        if word == "footer":
                            footer_information.append(tag.get_text(strip=True))
                        if word == "recommended":
                            if tag.name != "div" and tag.name != "section":
                                break
                            else:
                                recommended_list.append(tag.get_text(strip=True))
                        if word == "menu":
                            if tag.find(['p', 'span']):
                                menu_list.append(tag.get_text(strip=True))
                        if word == "publish":
                            publish_list.append(tag.get_text(strip=True))
                        if word == "source":
                            source_list.append(tag.get_text(strip=True))
                        tag.decompose()
                        decomposed = True
                        break
            if not decomposed and class_list: 
                for class_name in class_list:
                    for word in important_words:
                        if word in class_name.lower():
                            if word == "footer":
                                footer_information.append(tag.get_text(strip=True))
                            if word == "recommended":
                                if tag.name != "div" and tag.name != "section":
                                    break
                                else:
                                    recommended_list.append(tag.get_text(strip=True))
                            if word == "menu":
                                if tag.find(['p', 'span']):
                                    menu_list.append(tag.get_text(strip=True))
                            if word == "publish":
                                publish_list.append(tag.get_text(strip=True))
                            if word == "source":
                                source_list.append(tag.get_text(strip=True))
                            tag.decompose()
                            decomposed = True
                            break
                    if decomposed:
                        break

    paragraph_list = []
    if (bool(website_code.find("article"))):
        article = website_code.find("article")
        for paragraph in article.find_all(["p", "li"]):
            if paragraph.name == "p":
                if paragraph.find_parent("li") is None:
                    paragraph_list.append(paragraph.get_text(" ", strip=True))
                    structure_list.append(paragraph.get_text(" ", strip=True))
            elif paragraph.name == "li":
                if paragraph.find_parent("p") is None:
                    paragraph_list.append(paragraph.get_text(" ", strip=True))
                    structure_list.append(paragraph.get_text(" ", strip=True))
    elif (bool(website_code.find("main"))):
        main = website_code.find("main")
        for paragraph in main.find_all(["p", "li"]):
            if paragraph.name == "p":
                if paragraph.find_parent("li") is None:
                    paragraph_list.append(paragraph.get_text(" ", strip=True))
                    structure_list.append(paragraph.get_text(" ", strip=True))
            elif paragraph.name == "li":
                if paragraph.find_parent("p") is None:
                    paragraph_list.append(paragraph.get_text(" ", strip=True))
                    structure_list.append(paragraph.get_text(" ", strip=True))
    else:
        website_list = website_code.find_all(["p", "li"])
        for paragraph in website_list:
            if paragraph.name == "p":
                if paragraph.find_parent("li") is None:
                    paragraph_list.append(paragraph.get_text(" ", strip=True))
                    structure_list.append(paragraph.get_text(" ", strip=True))
            elif paragraph.name == "li":
                if paragraph.find_parent("p") is None:
                    paragraph_list.append(paragraph.get_text(" ", strip=True))
                    structure_list.append(paragraph.get_text(" ", strip=True))
    website_text = " ".join(paragraph_list)

    website_text = re.sub(r'\s+([.,!?;:])', r'\1', website_text) # removes any unnecessary spaces between punctuation and other words
    website_text = re.sub(r'\s+', ' ', website_text) # replaces any sequence of 2+ spaces with a single space
    website_text = re.sub(r'\n+', '\n', website_text) # replaces any sequence of 2+ newline characters with a single newline character
    website_text = website_text.strip() # removes any whitespace from the text
    cleaned_text = website_text

    for text in structure_list:
        text = re.sub(r'\s+([.,!?;:])', r'\1', text) # removes any unnecessary spaces between punctuation and other words
        text = re.sub(r'\s+', ' ', text) # replaces any sequence of 2+ spaces with a single space
        text = re.sub(r'\n+', '\n', text) # replaces any sequence of 2+ newline characters with a single newline character
        text = text.strip() # removes any whitespace from the text

    if len(website_text) == 0:
        return None, None, None, None, "Text cleanup function got rid of everything"

    additional_information = {
        "time_list": time_list,
        "emphasized_text_list": emphasized_text_list,
        "footer_information": footer_information,
        "recommended_list": recommended_list,
        "menu_list": menu_list,
        "publish_list": publish_list,
        "source_list": source_list
    }

    return cleaned_text, website_title, structure_list, additional_information, None

def segment_text_and_detect_language(segment_list, detection_model):
    """
    Combines the extracted article text into segments of up to 300 words and detects the 
    language of the article.

    The input text segments are combined in their original order into segments of up to 
    300 words, which are stored in a list. The method also detects the article's language.

    Args:
        segment_list (list[str]): List of text segments that together make up the main article content.
        detection_model: Loaded fastText language detection model.

    Returns:
        list[str]: List of text segments of up to 300 words that together make up the main article content.
        str: Error message that explains why the method failed to execute (if necessary)
        str: The language of the article text
    """
    
    # Use FastText to determine the text's language

    for tokenizer in ["punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{tokenizer}")
        except LookupError:
            nltk.download('punkt')
            nltk.download('punkt_tab')

    initial_list = []
    for paragraph in segment_list:
        word_count = len(paragraph.split())
        if word_count < 300:
            initial_list.append(paragraph)
        else:
            sentences = nltk.sent_tokenize(paragraph)
            new_paragraph = ""
            paragraph_words = 0
            for i, sentence in enumerate(sentences):
                paragraph_words += len(sentence.split())
                if (paragraph_words < 300):
                    new_paragraph += (sentence + " ")
                    if i == len(sentences) - 1:
                        initial_list.append(new_paragraph.strip())
                else:
                    initial_list.append(new_paragraph.strip())
                    if i < len(sentences) - 1:
                        new_paragraph = sentence + " "
                        paragraph_words = len(sentence.split())
                    else:
                        initial_list.append(sentence.strip())

    initial_list_copy = initial_list[:]
    final_list = []

    segment = initial_list[0]
    segment_copy = initial_list_copy[0]
    clean_segment = segment_copy.replace("\n", " ")
    language_tuple = detection_model.predict(clean_segment)
    language = language_tuple[0][0][9:12]

    return initial_list, None, language

def create_embeddings(paragraph_list, embedding_model):
    """
    Generates a semantic embedding for an article and checks for suspicious characteristics.

    The method groups the extracted article text into segments of up to 300 words. The method
    then uses the inputted embedding model to encode each of these segments into a high-dimensional 
    semantic embedding. All of these embeddings are then averaged to create a single embedding for 
    the entire article. Finally, the method checks the generated embedding for potential quality issues, 
    such as insufficient content, invalid values, unusual segment lengths, or low variation.

    Args:
        paragraph_list (list[str]): List of text segments that together make up the main article content.
        embedding_model: SentenceTransformer model used to generate semantic embeddings.
    
    Returns:
        np.ndarray: A fixed-length embedding that captures the meaning of the entire article.
        dict: A dictionary that contains details about the suspicious characteristics of the embedding.
    """

    # model: SentenceTransformers - all-mpnet-base-v2
    # Use SentenceTransformers to convert text into an embedding
    
    for tokenizer in ["punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{tokenizer}")
        except LookupError:
            nltk.download('punkt')
            nltk.download('punkt_tab')

    initial_list = []
    for paragraph in paragraph_list:
        word_count = len(paragraph.split())
        if word_count < 300:
            initial_list.append(paragraph)
        else:
            sentences = nltk.sent_tokenize(paragraph)
            new_paragraph = ""
            paragraph_words = 0
            for i, sentence in enumerate(sentences):
                paragraph_words += len(sentence.split())
                if (paragraph_words < 300):
                    new_paragraph += (sentence + " ")
                    if i == len(sentences) - 1:
                        initial_list.append(new_paragraph.strip())
                else:
                    initial_list.append(new_paragraph.strip())
                    if i < len(sentences) - 1:
                        new_paragraph = sentence + " "
                        paragraph_words = len(sentence.split())
                    else:
                        initial_list.append(sentence.strip())

    segment_count = len(initial_list)
    if segment_count > 2000:
        sampled_segments = []
        stride = segment_count / 2000

        for i in range(2000):
            index = int(stride * i)
            sampled_segments.append(initial_list[index])
        
        initial_list = sampled_segments
    
    embeddings = embedding_model.encode(initial_list, batch_size=64, show_progress_bar=False)
    average_embedding = np.mean(embeddings, axis=0)

    del embeddings
    del initial_list
    gc.collect()

    suspicious_factors = {}

    total_word_count = 0
    for segment in initial_list:
        total_word_count += len(segment.split())
    average_word_count = total_word_count / len(initial_list)

    suspicious_factors = {
        "too_short": False,
        "all_zero": False,
        "extreme_segment_length": False,
        "low_variance": False
    }

    suspicious_factors["too_short"] = embeddings.shape[0] < 3
    suspicious_factors["all_zero"] = np.all(embeddings == 0)
    suspicious_factors["extreme_segment_length"] = not (30 <= average_word_count <= 200)
    suspicious_factors["low_variance"] = np.var(embeddings, axis=0).mean() <= 0.001

    return average_embedding, suspicious_factors
