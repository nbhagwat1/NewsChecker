from backend.model.train import get_content

def main():
    link_list = ["https://www.cnn.com/2026/01/05/politics/jd-vance-ohio-residence", "https://www.bbc.com/news/articles/cdre0008l4ko", "https://www.npr.org/2026/01/05/nx-s1-5667078/maduro-indictment-hearing-underway", "https://www.nbcnews.com/world/venezuela/delcy-rodriguez-venezuelas-interim-leader-capture-nicolas-maduro-rcna252322", "https://www.cbsnews.com/minnesota/news/tim-walz-drop-out-of-gubernatorial-race-2026/", "https://abcnews.go.com/Health/new-federal-screening-guidance-expands-cervical-cancer-testing/story?id=128891185", "https://www.socialmediatoday.com/news/instagram-chief-says-longer-captions-dont-impact-post-reach/758462/?utm_source=chatgpt.com"]
    link_list_two = ["https://apnews.com/article/mexico-us-sheinbaum-trump-cartels-3b90e4a7efaf26f8f481dedf5e6423f4", "https://www.reuters.com/business/media-telecom/uk-regulator-launches-investigation-into-x-over-grok-sexualised-imagery-2026-01-12/", "https://www.theguardian.com/books/2026/jan/12/chimamanda-ngozi-adichie-accuses-lagos-hospital-negligence-son-death", "https://www.politico.com/news/2026/01/11/minneapolis-republicans-democrats-dig-in-heels-00721471", "https://www.msn.com/en-us/news/crime/ex-fbi-agent-reveals-potential-motive-in-murder-of-dentist-wife/ar-AA1U3AUq?uxmode=ruby&ocid=edgntpruby&pc=DCTS&cvid=69656b6329ed4ae3bd0c34eee2fa707b&ei=8", "https://news.yahoo.com/articles/two-brewing-snow-storms-puzzle-173201877.html", "https://www.usatoday.com/story/news/politics/2026/01/12/mark-kelly-sues-pete-hegseth-military-censure/88147119007/"]

    i = 1
    for link in link_list_two:
        print("Link ", i)
        a, b, c, d, e, f, g = get_content(link)
        print(d)
        print("\n")
        i += 1

if __name__ == "__main__":
    main()