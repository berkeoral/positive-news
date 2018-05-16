import sys
import time

import newspaper

from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from tqdm import tqdm

from backend.utils.utils import Utils
from backend.crawler.news_sources.source import Sources
from backend.utils.txtops import TextOps

max_interval = 10*60.

class Crawler:
    def __init__(self, max_interval = 10*60.):
        self.papers = []
        self.textops = TextOps()
        self.max_interval = max_interval
        self.start_time = time.time()

    def crawl(self, debug_mode=-1):
        self._init_papers(debug_mode)
        return self._start_crawl()

    def _init_papers(self, debug_mode):
        source_urls = Sources().get_sources()
        for i, source_url in tqdm(enumerate(source_urls), file=sys.stdout,
                                  total=(debug_mode if debug_mode > 0 else len(source_urls))):
            try:
                if i == debug_mode:
                    return
                tqdm.write("Initialising paper: " + source_url, file=sys.stdout)
                paper = newspaper.build(source_url,
                                        memoize_articles=True,
                                        keep_article_html=True,
                                        fetch_images=True)
                # Downloads non-english categories, bad solution
                #   > sometimes prefix contains country code
                paper.categories = [category for category in paper.categories
                                    if Utils().is_eng_suffix(None, category.url)]
                #       > not conclusive
                self.papers.append(paper)
                if time.time() - self.start_time > self.max_interval:
                    break
            except:
                print('An error occurred.')

    def _start_crawl(self):
        articles = []
        tqdm.write("Initialized \" " + str(self.papers.__len__()) + " \" papers, starting to crawl articles")
        self.start_time = time.time()
        for paper in tqdm(self.papers, file=sys.stdout):
            paper.articles = [article for article in paper.articles
                              if Utils().is_eng_suffix(None, article.url)
                              and Utils().is_eng_suffix(None, article.source_url)]
            for article in paper.articles:
                try:
                    article.build()
                    if detect(article.text) == "en":
                        tqdm.write(article.title, file=sys.stdout)
                        articles.append([article.url, article.top_image, article.title, article.text])
                        self.textops.append_records(article.url, article.title, article.text)
                except LangDetectException:
                    continue
                except:
                    continue
                if time.time() - self.start_time > self.max_interval:
                    break

        return articles
