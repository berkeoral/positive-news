import newspaper

class Crawler:
    def __init__(self):
        self.papers = []

    def __init_papers(self):
        source_urls = self.__get_sources()
        i = 0
        for source_url in source_urls:
            if i == 5:
                break
            i += 1
            paper = newspaper.build(source_url,
                                    memoize_articles=True,
                                    keep_article_html=True,
                                    fetch_images=False,
                                    language='en')
            self.papers.append(paper)

    def __get_sources(self):
        sources = newspaper.popular_urls()
        return sources

    def crawl(self):
        self.__init_papers()
        self.__start_crawl()

    #TODO separate non-english articles
    def __start_crawl(self):
        print(self.papers[0].size())
        for paper in self.papers:
            for article in paper.articles:
                article.build()
                print(article.meta_lang)
                print(article.url)
                print(article.title)
                print(article.summary)
                print("----------------------------------------")