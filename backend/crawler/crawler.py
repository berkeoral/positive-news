import newspaper

from backend.crawler.Utils import Utils
from backend.crawler.txtops import TextOps


class Crawler:
    def __init__(self):
        self.papers = []
        self.textops = TextOps()

    def __get_sources(self):
        sources = newspaper.popular_urls()
        return sources

    def crawl(self):
        self.__init_papers()
        self.__start_crawl()

    def __init_papers(self):
        source_urls = self.__get_sources()
        i = 0
        for source_url in source_urls:
            if i == 1:
                break
            i += 1
            print("Initialising paper: " + source_url)
            paper = newspaper.build(source_url,
                                    memoize_articles=False,
                                    keep_article_html=True,
                                    fetch_images=False)
            # Category already downloaded, bad solution
            #   > sometimes prefix contains country code
            #   > solution is by changing newspaper's source class
            #paper.categories = [category for category in paper.categories
            #                    if Utils().is_eng_suffix(None, category.url)]
            self.papers.append(paper)

    # TODO separate non-english articles
    def __start_crawl(self):
        print("Initialized \" " + str(self.papers.__len__()) + " \" articles, starting to crawl articles")
        for paper in self.papers:
            paper.articles = [article for article in paper.articles
                                if Utils().is_eng_suffix(None, article.url)]
            paper.articles = [article for article in paper.articles
                                if Utils().is_eng_suffix(None, article.source_url)]
            for article in paper.articles:
                article.build()
                self.textops.append_file(article.url, article.title, article.text)
                print(article.meta_lang)
                print(article.url)
                print(article.title)
                print(article.summary)
                print("----------------------------------------")

