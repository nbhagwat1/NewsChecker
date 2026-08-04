import requests
import re
from bs4 import BeautifulSoup, Tag
from sentence_transformers import SentenceTransformer
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
    Validates that a provided string follows a basic URL format.

    This function checks that the input contains at least one period
    character and does not contain whitespace. If the input does not
    meet these conditions, the program exits.

    Args:
        link (str): The string to validate as a URL.

    Returns:
        None
    
    Raises:
        SystemExit: If the input does not meet the basic URL requirements.    
    """

    link = link.strip().lower()

    # Perform a lightweight URL check. Requiring at least one dot and no spaces
    # filters out inputs that are unlikely to be valid web addresses.

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
    Extracts and cleans the main article content from a news webpage.

    This function retrieves the HTML content of a news article URL,
    processes the webpage structure using BeautifulSoup, removes
    irrelevant elements such as advertisements and navigation content,
    and extracts the main article text. It also extracts the article
    title and preserves removed webpage information that may be useful
    for future credibility analysis.

    The function automatically handles missing URLs, unsupported webpages,
    failed HTTP requests, and webpages where usable article text cannot
    be extracted.

    Args:
        link (str): URL of the news article to scrape and process.
    
    Returns:
        str: Cleaned article text extracted from the webpage.
        str: Title of the news article.
        list[str]: List of extracted article sections in their original document order.
        dict: Additional webpage information removed during extraction, including metadata, 
            source information, and other potentially useful text.
        str: Error message describing why extraction failed, or None if extraction was
            successful.
    """

    # Skip entries with missing URLs. Missing values may appear as either
    # None or NaN depending on how the dataset was loaded.
    if (link is None) or (isinstance(link, float) and math.isnan(link)):
        return None, None, None, None, "Link is missing"

    # Add a default HTTPS protocol to URLs that don't already specify one.
    # This allows the scraper to handle inputs such as "www.example.com".
    if not link.startswith("http://") and not link.startswith("https://"):
        link = "https://" + link

    # Some websites reject requests that do not include a User-Agent header.
    # Providing one improves the scraper's ability to retrieve article content.
    headers = {
        "User-Agent": "NewsChecker/1.0 (learning project)"
    }

    response = None

    # Retry the request up to three times since temporary network issues or
    # slow server responses may succeed on a subsequent attempt.
    for attempt in range(3):
        try:    
            # Use a 20-second timeout to prevent requests from waiting indefinitely
            # while still giving slower websites enough time to respond.
            response = requests.get(link, headers=headers, timeout=20)

            response.raise_for_status()
            break
        except requests.exceptions.ReadTimeout:
            # If the request times out, wait 2 seconds before retrying to avoid
            # sending repeated requests to the website in rapid succession.
            time.sleep(2)
        except requests.exceptions.RequestException as e:
            # Handle request failures that cannot be recovered through retries.

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

    # If no successful HTTP response was obtained after all retry attempts,
    # stop processing since the webpage content could not be retrieved.
    if response is None:
        return None, None, None, None, "HTTP request failed"

    # Only continue processing if the server returned a successful HTTP
    # status code (200 OK).
    if (response.status_code != 200):
        code = response.status_code
        return None, None, None, None, "HTTP request failed"

    # A successful HTTP response was received, so the webpage content can
    # now be processed.
    
    website_content = response.text
    website_code = BeautifulSoup(website_content, 'html.parser')

    # Extract the article title from the main heading when available.
    # The <h1> tag typically contains the article's actual headline.
    heading = website_code.find("h1")

    if heading:
        website_title = heading.get_text(strip=True)
        heading.decompose()
    else:
        if website_code.title and website_code.title.string:
            # Some webpages do not include an <h1> tag, so use the HTML title
            # as a fallback source for the article title.
            website_title = website_code.title.string
        else:
            # Continue processing even if no title can be extracted. The article
            # content is still sufficient for generating a prediction.
            website_title = ""

    # Filter webpage content by removing sections that are unlikely to be part
    # of the main article while preserving remaining content for article extraction.

    website_text = ""

    # Store non-article HTML elements separately. These elements are not part
    # of the main article content but may contain useful information for analysis.
    time_list = []
    emphasized_text_list = []
    footer_information = []
    recommended_list = []
    menu_list = []
    publish_list = []
    source_list = []
    structure_list = []

    # Common phrases associated with promotional or navigational content.
    # If these phrases appear in hyperlinks or emphasized text, they are
    # treated as unlikely to be part of the article body.
    distracting_words = ["click here", "learn more", "check out", "this article originally appeared", "subscribe", "premium", "originally published"]

    # Removing tags that are irrelevant to the main content of the article. If a certain
    # tag has the possibility of being part of the main content of the article but also
    # not, then it is removed if its text is likely to be not part of the main content.

    for tag in website_code(["script", "style", "noscript", "meta", "header", "footer", "img", "nav", "aside", "style", "figcaption", "button"]):
        # Remove HTML elements that do not contribute to the article's main content.
        tag.decompose()

    for tag in website_code("a"):
        for word in distracting_words:
            if word in tag.get_text(strip=True).lower():
                # Remove hyperlinks containing phrases commonly associated with
                # promotional or navigational content.
                tag.decompose()

    for tag in website_code("time"):
        # Preserve publication time as metadata while removing it from the
        # article body.
        time_list.append(tag.get_text(strip=True))
        tag.decompose()

    for tag in website_code(["em", "i", "strong", "b", "u", "mark"]):
        # Preserve emphasized text separately since it may contain useful
        # information outside the article body.
        emphasized_text_list.append(tag.get_text(strip=True))

        for word in distracting_words:
            if word in tag.get_text(strip=True).lower():
                # Remove emphasized text that matches common promotional phrases.
                tag.decompose()
                continue
            
            tag_text = tag.get_text(strip=True).lower()
            if ("more" in tag_text or "related" in tag_text) and "news" in tag_text and ":" in tag_text:
                # Remove emphasized headings that introduce related-article sections.
                tag.decompose()

    for tag in website_code("li"):
        if not tag.find_all(["p", "span"]):
            # Remove list items that do not contain meaningful article text.
            tag.decompose()

    for tag in website_code(["p", "li"]):
        tag_children = []
        for child in tag.contents:
            if str(child).strip():
                tag_children.append(child)
        tag_children = [c for c in tag_children if c != ' HTML_TAG_START ' and c != ' HTML_TAG_END ']

        if len(tag_children) == 1:
            only_child = tag_children[0]

            # Remove paragraphs and list items whose entire visible content is
            # effectively just a hyperlink, since they are usually navigation or
            # promotional elements rather than part of the article body.
            if only_child.name == "a":
                tag.decompose()
            elif isinstance(only_child, Tag) and (len(list(only_child.descendants)) >= 2 and list(only_child.descendants)[len(list(only_child.descendants)) - 2].name == "a"):
                tag.decompose()

    # Handle edge cases when filtering HTML content. This includes preventing
    # removal of essential page structure, removing hidden elements, and
    # identifying tags that may contain unrelated content using class names
    # and data attributes.
    
    for tag in website_code.find_all(True):
        if tag.name == "html" or tag.name == "body":
            # Preserve the root HTML structure to avoid accidentally removing
            # the entire webpage content.
            continue
        if isinstance(tag, Tag) == False:
            # Skip non-HTML elements because attribute checks and removal only apply
            # to BeautifulSoup Tag objects.
            continue
        if not tag.attrs:
            # Skip tags without attributes since class names and data attributes
            # are used to identify irrelevant content.
            continue

        style = tag.get("style", "")
        if "visibility:hidden" in style:
            # Remove elements hidden with CSS since they are not part of the
            # article content visible to the reader.
            tag.decompose()
            continue
        if tag.get("aria-hidden") == "true":
            # Remove elements marked with aria-hidden since they are typically
            # excluded from the visible article content.
            tag.decompose()
            continue

        class_list = tag.get('class', [])
        data = tag.get("data-testid")

        # Keywords commonly found in class names and data attributes of HTML
        # elements that contain content unrelated to the main article, such as
        # navigation, social links, recommendations, and promotional sections.
        important_words = ["metadata", "social-link", "social-share", "social-bookmark", "follow-topics", "footnote", "caption", "byline", "subscribe", "newsletter", "footer", "headline", "promotion", "prism-card", "recommended", "licensing", "button", "description", "infobox", "menu", "publish", "boilerplate", "source"]

        decomposed = False

        if data or class_list:
            if data:
                for word in important_words:
                    if word in data.lower():
                        if word == "footer":
                            # Preserve footer information separately before removing it from the
                            # article text since it may contain useful metadata.
                            footer_information.append(tag.get_text(strip=True))
                        if word == "recommended":
                            if tag.name != "div" and tag.name != "section":
                                # Avoid removing tags where "recommended" may refer to normal
                                # article text instead of a separate recommendation section.
                                break
                            else:
                                # Save recommendation sections separately before removing them
                                # from the article content.
                                recommended_list.append(tag.get_text(strip=True))
                        if word == "menu":
                            if tag.find(['p', 'span']):
                                # Store menu text separately because navigation content should not be
                                # included in the extracted article text.
                                menu_list.append(tag.get_text(strip=True))
                        if word == "publish":
                            # Store publishing information separately since it is metadata about
                            # the article rather than part of the article's main content.
                            publish_list.append(tag.get_text(strip=True))
                        if word == "source":
                            # Store source information separately since it provides attribution
                            # rather than contributing to the article's main content.
                            source_list.append(tag.get_text(strip=True))

                        # Remove this element from the article content after preserving any
                        # useful information separately above.
                        tag.decompose()
                        decomposed = True
                        break

            # If the element was not already removed using its data attributes,
            # check its class names for keywords that indicate unrelated content.
            if not decomposed and class_list: 
                for class_name in class_list:
                    for word in important_words:
                        if word in class_name.lower():
                            if word == "footer":
                                # Preserve footer information separately before removing it from the
                                # article text since it may contain useful metadata.
                                footer_information.append(tag.get_text(strip=True))
                            if word == "recommended":
                                if tag.name != "div" and tag.name != "section":
                                    # Avoid removing tags where "recommended" may refer to normal
                                    # article text instead of a separate recommendation section.
                                    break
                                else:
                                    # Save recommendation sections separately before removing them
                                    # from the article content.
                                    recommended_list.append(tag.get_text(strip=True))
                            if word == "menu":
                                if tag.find(['p', 'span']):
                                    # Store menu text separately because navigation content should not be
                                    # included in the extracted article text.
                                    menu_list.append(tag.get_text(strip=True))
                            if word == "publish":
                                # Store publishing information separately since it is metadata about
                                # the article rather than part of the article's main content.
                                publish_list.append(tag.get_text(strip=True))
                            if word == "source":
                                # Store source information separately since it provides attribution
                                # rather than contributing to the article's main content.
                                source_list.append(tag.get_text(strip=True))

                            # Remove this element from the article content after preserving any
                            # useful information separately above.
                            tag.decompose()
                            decomposed = True
                            break
                    if decomposed:
                        # Stop checking other class names once the tag has already been
                        # removed since additional matches are no longer needed.
                        break

    # Extract the main article content from the remaining HTML. If an
    # <article> or <main> tag is available, use it as the primary content
    # source; otherwise, build the article by collecting the remaining
    # <p> and <li> elements in document order.

    paragraph_list = []
    if (bool(website_code.find("article"))):
        # Prefer the <article> tag when available since it is intended to
        # contain the webpage's primary article content.
        article = website_code.find("article")

        for paragraph in article.find_all(["p", "li"]):
            # Extract paragraph and list item text since these tags typically
            # contain the main written content of an article.

            if paragraph.name == "p":
                if paragraph.find_parent("li") is None:
                    # Avoid extracting duplicate text when a paragraph is nested inside
                    # a list item.

                    paragraph_list.append(paragraph.get_text(" ", strip=True))
                    structure_list.append(paragraph.get_text(" ", strip=True))
            elif paragraph.name == "li":
                if paragraph.find_parent("p") is None:
                    # Avoid extracting duplicate text when a list item is nested inside
                    # a paragraph.

                    paragraph_list.append(paragraph.get_text(" ", strip=True))
                    structure_list.append(paragraph.get_text(" ", strip=True))
    elif (bool(website_code.find("main"))):
        # If no <article> tag is available, use the <main> tag as the next
        # best source for the webpage's primary content.
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
        # As a final fallback, search the entire webpage for paragraph and
        # list item text since some websites do not use <article> or <main>
        # tags to identify their primary content.
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

    # Combine the extracted paragraphs and list items into a single string
    # representing the article's main content.
    website_text = " ".join(paragraph_list)

    # Clean the extracted article content by removing unnecessary
    # whitespace and spacing issues caused by HTML extraction.

    website_text = re.sub(r'\s+([.,!?;:])', r'\1', website_text) # Remove spaces that appear immediately before punctuation.
    website_text = re.sub(r'\s+', ' ', website_text) # Replace consecutive whitespace characters with a single space.
    website_text = re.sub(r'\n+', '\n', website_text) # Collapse consecutive newline characters into a single newline.
    website_text = website_text.strip() # Remove leading and trailing whitespace.
    cleaned_text = website_text

    for text in structure_list:
        text = re.sub(r'\s+([.,!?;:])', r'\1', text) # Remove spaces that appear immediately before punctuation.
        text = re.sub(r'\s+', ' ', text) # Replace consecutive whitespace characters with a single space.
        text = re.sub(r'\n+', '\n', text) # Collapse consecutive newline characters into a single newline.
        text = text.strip() # Remove leading and trailing whitespace.

    # If no article text remains after HTML extraction and cleanup, stop
    # processing since there is no usable content to analyze.
    if len(website_text) == 0:
        return None, None, None, None, "Text cleanup function got rid of everything"

    # Return the extracted article content, title, and additional information
    # that was removed from the main text but may still be useful for future
    # processing, analysis, or displaying results.

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
    Splits article text into smaller segments and detects the article language.

    This function combines and divides extracted article content into segments
    of up to 300 words while preserving sentence boundaries. The segmentation
    keeps text inputs within the size limitations of the embedding model.
    The function also uses a language detection model to identify the language
    of the article.

    Args:
        segment_list (list[str]): List of text segments that together make up
            the extracted article content.
        detection_model: Loaded fastText language detection model used to
            identify the article language.

    Returns:
        list[str]: List of article text segments, each containing up to
            300 words.
        str: Error message describing why processing failed, or None if 
            the function completed successfully.
        str: Three-letter language code detected from the article text.
    """
    
    # Download the resources required by NLTK's sentence tokenizer.
    # These resources allow paragraphs to be split into individual sentences.
    for tokenizer in ["punkt", "punkt_tab"]:
        try:
            # Check whether the sentence tokenization resource is already
            # available before attempting to download it.
            nltk.data.find(f"tokenizers/{tokenizer}")
        except LookupError:
            # Download missing sentence tokenization resources so text can be
            # separated into individual sentences.
            nltk.download('punkt')
            nltk.download('punkt_tab')

    # Transform the structure of the list of paragraphs
    # so that each item in the list is up to 300 words long

    initial_list = []
    for paragraph in segment_list:
        word_count = len(paragraph.split())

        # Split article text into segments of up to 300 words to keep each
        # segment within the input size limitations of the embedding model.
        if word_count < 300:
            initial_list.append(paragraph)
        else:
            # Split the paragraph into sentences so segments can be created without
            # breaking sentences apart.
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

    # Detect the article's language using FastText on a single cleaned segment
    # instead of the entire article to reduce processing time while providing
    # enough text for language detection.
    initial_list_copy = initial_list[:]
    segment = initial_list[0]
    segment_copy = initial_list_copy[0]
    clean_segment = segment_copy.replace("\n", " ")
    language_tuple = detection_model.predict(clean_segment)
    language = language_tuple[0][0][9:12]

    return initial_list, None, language

def create_embeddings(segment_list, embedding_model):
    """
    Generates an article embedding and evaluates embedding quality.

    This function converts article text segments into semantic embeddings
    using a SentenceTransformer model. For extremely large articles, the
    number of segments is reduced using sampling to limit processing time.
    The generated segment embeddings are averaged to create a single
    fixed-length representation of the article.

    The function also performs quality checks on the generated embeddings
    and extracted article content, including checks for insufficient
    content, failed embedding generation, unusual segment lengths, and
    low embedding variation.

    Args:
        segment_list (list[str]): List of article text segments used to
            generate embeddings.
        embedding_model: SentenceTransformer model used to convert text
            segments into semantic embeddings.

    Returns:
        np.ndarray: A fixed-length embedding representing the overall
            semantic content of the article.
        dict: Dictionary containing flags indicating potential quality
            issues detected during embedding generation.
    """
    
    initial_list = segment_list

    # If the article contains more than 2000 segments, reduce the number of
    # segments before generating embeddings. A stride-based sampling approach
    # is used to evenly select 2000 segments from the original list while
    # preserving coverage of the article's content.
    segment_count = len(initial_list)
    if segment_count > 2000:
        sampled_segments = []

        # Calculate the interval between selected segments so exactly 2000
        # segments are sampled from the original list.
        stride = segment_count / 2000

        # A maximum of 2000 segments was chosen as a practical limit to balance
        # embedding generation time and retaining enough article content.
        for i in range(2000):
            index = int(stride * i)
            sampled_segments.append(initial_list[index])
        
        initial_list = sampled_segments

    # Generate embeddings for each article segment and combine them into a
    # single embedding by averaging the segment embeddings. This creates one
    # representation of the article's overall content..
    
    # Generate embeddings for each article segment. Segments are processed
    # in batches of 64 to balance processing speed and memory usage.
    embeddings = embedding_model.encode(
        initial_list, 
        batch_size=64, 
        show_progress_bar=False
    )

    # Average the segment embeddings to create a single embedding that
    # represents the article for model training.
    average_embedding = np.mean(embeddings, axis=0)

    # Free memory before processing the next article to reduce memory usage
    # during large-scale training.
    del embeddings
    del initial_list
    gc.collect()

    # Perform quality checks on the generated embeddings to identify
    # potential problems with the extracted article content.

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

    # Articles with fewer than three segment embeddings may not provide
    # enough information for reliable analysis.
    suspicious_factors["too_short"] = embeddings.shape[0] < 3

    # Check whether every embedding value is zero, which may indicate that
    # embedding generation failed.
    suspicious_factors["all_zero"] = np.all(embeddings == 0)

    # Flag unusually short or long average segment lengths since they may
    # indicate problems with article extraction.
    suspicious_factors["extreme_segment_length"] = not (30 <= average_word_count <= 200)

    # Check whether the embeddings vary very little across segments, which
    # may indicate that the extracted content lacks meaningful variation.
    suspicious_factors["low_variance"] = np.var(embeddings, axis=0).mean() <= 0.001

    return average_embedding, suspicious_factors
