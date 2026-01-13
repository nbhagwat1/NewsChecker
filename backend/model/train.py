import requests
import re
from bs4 import BeautifulSoup, Tag
from sentence_transformers import SentenceTransformer
import fasttext
from huggingface_hub import hf_hub_download
from transformers import pipeline

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
    for tag in website_code(["script", "meta", "header", "footer", "img", "nav", "aside", "style", "figcaption", "button"]):
        tag.decompose()
        # a = 1
    for tag in website_code("time"):
        time_list.append(tag.get_text(strip=True))
        tag.decompose()
    for tag in website_code("em"):
        emphasized_text_list.append(tag.get_text(strip=True))
    for tag in website_code("li"):
        if not tag.find_all(["p", "span"]):
            tag.decompose()  
    for tag in website_code.find_all(True):
        if isinstance(tag, Tag) == False:
            continue
        if not tag.attrs:
            continue
        class_list = tag.get('class', [])
        data = tag.get("data-testid")
        important_words = ["metadata", "social-link", "social-share", "follow-topics", "footnote", "caption", "byline", "subscribe", "newsletter", "footer", "headline", "promotion", "prism-card", "recommended", "licensing", "button", "description", "infobox"]
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
                            tag.decompose()
                            decomposed = True
                            break
                    if decomposed:
                        break

    paragraph_list = []
    if (bool(website_code.find_all("article"))):
        website_list = website_code.find_all("article")
        for article in website_list:
            for paragraph in article.find_all(["p", "ul", "ol"]):
                paragraph_list.append(paragraph.get_text(" ", strip=True))
    elif (bool(website_code.find_all("main"))):
        website_list = website_code.find_all("main")
        for main in website_list:
            for paragraph in main.find_all(["p", "ul", "ol"]):
                paragraph_list.append(paragraph.get_text(" ", strip=True))
    else:
        website_list = website_code.find_all(["p", "ul", "ol"])
        for paragraph in website_list:
            paragraph_list.append(paragraph.get_text(" ", strip=True))
    website_text = " ".join(paragraph_list)

    original_text = website_text
    website_text = re.sub(r'\s+', ' ', website_text) # replaces any sequence of 2+ spaces with a single space
    website_text = re.sub(r'\n+', '\n', website_text) # replaces any sequence of 2+ newline characters with a single newline character
    website_text = website_text.strip() # removes any whitespace from the text
    cleaned_text = website_text

    if len(website_text) == 0:
        print("Text cleanup failed")
        exit(0)

    return website_text, website_title, original_text, cleaned_text, time_list, footer_information, emphasized_text_list

def analyze_language(text):
    # Use FastText to determine the text's language
    # Use HuggingFace / NLLB to translate the text

    new_text = ""
    for char in text:
        if char != "\n":
            new_text += char
        else: 
            new_text += " "

    language_model = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
    detection_model = fasttext.load_model(language_model)

    language_tuple = detection_model.predict(new_text)
    language = language_tuple[0][0][9:12]

    final_text = new_text
    if (language.lower() != "eng"):
        translation_tool = pipeline("translation", model="facebook/nllb-200-distilled-600M")
    
        line_list = []
        for line in text.split("\n"):
            line_list.append(line)
        
        paragraph = ""
        total_words = 0
        paragraph_list = []
        for line in line_list:
            total_words += len(line.split())
            if total_words > 250:
                paragraph_list.append(paragraph)
                paragraph = line + " "
                total_words = 0
            else:
                paragraph += line + " "
        paragraph_list.append(paragraph)

        translated_text = ""
        for paragraph in paragraph_list:
            new_paragraph = ""
            for char in paragraph:
                if char != "\n":
                    new_paragraph += char
                else:
                    new_paragraph += " "
            translated_paragraph = translation_tool(new_paragraph, src_lang=language_tuple[0][0][9:len(language_tuple[0][0])], tgt_lang="eng_Latn")
            translated_text += translated_paragraph[0]['translation_text']
            translated_text += " "
        final_text = translated_text
    
    return final_text

def create_embeddings(text):
    # model: SentenceTransformers - all-mpnet-base-v2
    # Use SentenceTransformers to convert text into an embedding

    tone_model = SentenceTransformer("all-mpnet-base-v2")
    
    return text

def main():
    print("Hi")

if __name__ == "__main__":
    main()