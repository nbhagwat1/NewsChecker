import requests
import re
from bs4 import BeautifulSoup, Tag
from sentence_transformers import SentenceTransformer
import fasttext
from huggingface_hub import hf_hub_download
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize

def examine_link(link):
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
    headers = {
        "User-Agent": "NewsChecker/1.0 (learning project)"
    }

    response = requests.get(link, headers=headers, timeout=10)
    if (response.status_code != 200):
        print(response.status_code)
        print("HTTP request failed")
        exit(0)
    
    website_content = response.text
    # print(website_content)
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
    text_list = []
    time_list = []
    emphasized_text_list = []
    footer_information = []
    recommended_list = []
    tag_list = []
    menu_list = []
    publish_list = []
    external_list = []
    source_list = []
    structure_list = []
    distracting_words = ["click here", "learn more", "check out", "this article originally appeared", "subscribe", "premium", "originally published"]

    for tag in website_code(["script", "style", "noscript", "meta", "header", "footer", "img", "nav", "aside", "style", "figcaption", "button"]):
        tag.decompose()
        # a = 1
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
        # important_words = ["abcdefghijk"]
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

    original_text = website_text
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
        print("Text cleanup failed")
        exit(0)

    return website_text, website_title, original_text, cleaned_text, structure_list, time_list, footer_information, emphasized_text_list

def analyze_language(text):
    # Use FastText to determine the text's language
    # Use HuggingFace / NLLB to translate the text

    language_model = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
    detection_model = fasttext.load_model(language_model)

    language_tuple = detection_model.predict(text)
    language = language_tuple[0][0][9:12]

    final_text = text
    if (language.lower() != "eng"):
        translation_tool = pipeline("translation", model="facebook/nllb-200-distilled-600M")
        nltk.download('punkt')
        nltk.download('punkt_tab')

        line_list = nltk.sent_tokenize(text)
        
        paragraph = ""
        total_words = 0
        paragraph_list = []
        for line in line_list:
            total_words += len(line.split())
            if total_words > 250:
                paragraph_list.append(paragraph.strip())
                paragraph = line + " "
                total_words = 0
            else:
                paragraph += line + " "
        paragraph_list.append(paragraph.strip())

        translated_text = ""
        for paragraph in paragraph_list:
            translated_paragraph = translation_tool(paragraph, src_lang=language_tuple[0][0][9:len(language_tuple[0][0])], tgt_lang="eng_Latn")
            translated_text += translated_paragraph[0]['translation_text']
            translated_text += " "
        final_text = translated_text.strip()
    
    return final_text

def create_embeddings(paragraph_list):
    # model: SentenceTransformers - all-mpnet-base-v2
    # Use SentenceTransformers to convert text into an embedding

    embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
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
    
    embeddings = embedding_model.encode(initial_list)
    return embeddings

def main():
    create_embeddings("Hi")

if __name__ == "__main__":
    main()