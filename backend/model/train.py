import requests
from bs4 import BeautifulSoup

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
    print(website_code)

def main():
    try:
        get_content("https://www.today.com/style/see-people-s-choice-awards-red-carpet-looks-t141832")
    except Exception as e:
        print("Exception occurred: ", e, flush=True)

if __name__ == "__main__":
    main()