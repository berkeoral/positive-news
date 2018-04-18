import sys

import newspaper

from langdetect import detect
from tqdm import tqdm

from backend.utils.utils import Utils
from backend.crawler.news_sources.source import Sources
from backend.utils.txtops import TextOps



class Crawler:
    def __init__(self):
        self.papers = []
        self.textops = TextOps()

    def crawl(self):
        self.__init_papers()
        self.__start_crawl()

    def __init_papers(self):
        source_urls = Sources().get_sources()
        for source_url in tqdm(source_urls, file=sys.stdout):
            try:
                tqdm.write("Initialising paper: " + source_url)
                paper = newspaper.build(source_url,
                                        memoize_articles=True,
                                        keep_article_html=True,
                                        fetch_images=False)
                # Downloads non-english categories, bad solution
                #   > sometimes prefix contains country code
                paper.categories = [category for category in paper.categories
                                    if Utils().is_eng_suffix(None, category.url)]
                #       > not conclusive
                self.papers.append(paper)
            except:
                print('An error occurred.')

    def __start_crawl(self):
        tqdm.write("Initialized \" " + str(self.papers.__len__()) + " \" papers, starting to crawl articles")
        for paper in tqdm(self.papers, file=sys.stdout):
            paper.articles = [article for article in paper.articles
                              if Utils().is_eng_suffix(None, article.url)
                              and Utils().is_eng_suffix(None, article.source_url)]
            for article in paper.articles:
                try:
                    article.build()
                    if detect(article.text) == "en":
                        self.textops.append_records(article.url, article.title, article.text)
                        tqdm.write(article.title)
                except:
                    pass


