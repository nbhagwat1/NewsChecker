from backend.model.train import get_content

def main():
    link_list = ["https://www.cnn.com/2026/01/05/politics/jd-vance-ohio-residence", "https://www.bbc.com/news/articles/cdre0008l4ko", "https://www.npr.org/2026/01/05/nx-s1-5667078/maduro-indictment-hearing-underway", "https://www.nbcnews.com/world/venezuela/delcy-rodriguez-venezuelas-interim-leader-capture-nicolas-maduro-rcna252322", "https://www.cbsnews.com/minnesota/news/tim-walz-drop-out-of-gubernatorial-race-2026/", "https://abcnews.go.com/Health/new-federal-screening-guidance-expands-cervical-cancer-testing/story?id=128891185", "https://www.socialmediatoday.com/news/instagram-chief-says-longer-captions-dont-impact-post-reach/758462/?utm_source=chatgpt.com"]

    i = 1
    for link in link_list:
        print("Link ", i)
        a, b, c, d, e, f = get_content(link)
        print(d)
        print("\n")
        i += 1

if __name__ == "__main__":
    main()