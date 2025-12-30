import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import fasttext
from huggingface_hub import hf_hub_download
from transformers import pipeline

article_title = ""

def get_link(link):
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
    response = requests.get(link)
    if (response.status_code != 200):
        print("HTTP request failed")
        exit(0)
    
    website_content = response.text
    website_code = BeautifulSoup(website_content, 'html.parser')
    website_title = website_code.title.string
    article_title = website_title

    website_text = ""
    text_list = []
    for tag in website_code(["script", "meta", "header", "footer", "img"]):
        tag.decompose()
    

    if (bool(website_code.find_all("article"))):
        website_list = website_code.find_all("article")
        for element in website_list:
            text_list.append(element.get_text(strip=True, separator="\n"))
        website_text = "\n".join(text_list)
    elif (bool(website_code.find_all("main"))):
        website_list = website_code.find_all("main")
        for element in website_list:
            text_list.append(element.get_text(strip=True, separator="\n"))
        website_text = "\n".join(text_list)
    else:
        website_text = website_code.get_text()
    
    return website_text 

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

def analyze_tone(text):
    # model: SentenceTransformers - all-mpnet-base-v2
    # Use SentenceTransformers to convert text into an embedding
    # Use 
    # This article is written in a way that is less careful about knowledge

    tone_model = SentenceTransformer("all-mpnet-base-v2")
    
    return text

def main():
    analyze_language("Hola a todos. Me llamo Nikhil y los quiero. Mi cosa favorita para hacer es jugar videojuegos y comer espaguetis.")
    analyze_tone("Hi")

if __name__ == "__main__":
    main()