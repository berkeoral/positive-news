import newspaper

import tldextract

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
        for source_url in source_urls:
            print("Initialising paper: " + source_url)
            paper = newspaper.build(source_url,
                                    memoize_articles=True,
                                    keep_article_html=True,
                                    fetch_images=False)
            # Category already downloaded, bad solution
            #   > solution is by changing newspaper's source class
            paper.categories = [category for category in paper.categories
                           if Utils().is_eng_tld(None, category.url)]
            self.papers.append(paper)

    # TODO separate non-english articles
    def __start_crawl(self):
        print(self.papers[0].size())
        for paper in self.papers:
            for article in paper.articles:
                article.build()
                if article.config._language is not "en":
                    continue
                self.textops.append_file(article.url, article.title, article.text)
                print(article.meta_lang)
                print(article.url)
                print(article.title)
                print(article.summary)
                print("----------------------------------------")


class Utils:
    @staticmethod
    def is_eng_tld(self, url):
        result = tldextract.extract(url)
        if result.suffix == "com":
            return True
        elif result.suffix == "us":
            return True
        elif result.suffix == "uk":
            return True
        elif result.suffix == "co.uk":
            return True
        elif result.suffix == "au":
            return True
        elif result.suffix == "com.au":
            return True
        elif result.suffix == "ca":
            return True
        elif result.suffix == "com.ca":
            return True
        return False
