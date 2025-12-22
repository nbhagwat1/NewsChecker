import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import fasttext
from huggingface_hub import hf_hub_download

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
    for tag in website_code(["script", "meta", "header", "footer"]):
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

    language_model = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
    detection_model = fasttext.load_model(language_model)
    language = detection_model.predict(text)
    
    return language


def analyze_tone(text):
    # model: SentenceTransformers - all-mpnet-base-v2
    
    return text

def main():
    text = get_content("https://www.today.com/style/see-people-s-choice-awards-red-carpet-looks-t141832")
    print(analyze_language(text))

if __name__ == "__main__":
    main()