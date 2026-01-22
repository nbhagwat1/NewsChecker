from backend.model.text_extraction import get_content

def main():
    link_list = ["https://www.cnn.com/2026/01/05/politics/jd-vance-ohio-residence", "https://www.bbc.com/news/articles/cdre0008l4ko", "https://www.npr.org/2026/01/05/nx-s1-5667078/maduro-indictment-hearing-underway", "https://www.nbcnews.com/world/venezuela/delcy-rodriguez-venezuelas-interim-leader-capture-nicolas-maduro-rcna252322", "https://www.cbsnews.com/minnesota/news/tim-walz-drop-out-of-gubernatorial-race-2026/", "https://abcnews.go.com/Health/new-federal-screening-guidance-expands-cervical-cancer-testing/story?id=128891185", "https://www.socialmediatoday.com/news/instagram-chief-says-longer-captions-dont-impact-post-reach/758462/?utm_source=chatgpt.com"]
    link_list_two = ["https://apnews.com/article/mexico-us-sheinbaum-trump-cartels-3b90e4a7efaf26f8f481dedf5e6423f4", "https://en.wikinews.org/wiki/Kyiv_street_to_be_named_for_former_Ukrainian_parliament_chairman_Andriy_Parubiy,_killed_in_Lviv", "https://allthatsinteresting.com/thailand-flat-headed-cats", "https://www.foxnews.com/entertainment/timothy-busfield-turns-himself-police-promises-fight-child-sex-abuse-accusations", "https://news.yahoo.com/articles/two-brewing-snow-storms-puzzle-173201877.html", "https://www.usatoday.com/story/news/politics/2026/01/12/mark-kelly-sues-pete-hegseth-military-censure/88147119007/"]
    link_list_three = ["https://news.yahoo.com/articles/two-brewing-snow-storms-puzzle-173201877.html", "https://www.yahoo.com/news/articles/east-tennessee-school-closures-delays-020404502.html", "https://www.yahoo.com/finance/news/mcdonalds-bets-ai-2026-fix-193300993.html", "https://www.yahoo.com/news/articles/snow-florida-arctic-blast-stun-164720507.html", "https://www.yahoo.com/news/articles/she-terrified-details-emerge-ohio-010134244.html"]
    currently_testing = ["https://www.yahoo.com/news/articles/east-tennessee-school-closures-delays-020404502.html"]

    i = 1
    for link in link_list:
        print("Link ", i)
        a, b, c, d, e, f, g, h = get_content(link)
        print(d)
        print("\n")
        i += 1

if __name__ == "__main__":
    main()