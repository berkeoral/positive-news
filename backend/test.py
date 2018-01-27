import time
import threading

import newspaper

from backend.crawler.crawler import Crawler
from backend.crawler.news_sources.source import Sources
from backend.crawler.txtops import TextOps
from backend.nlp.sentiment_analysis.SIF import *

CRAWLER_CALL_INTERVAL = 5
CRAWLER_INIT_PADDING = 5
CRAWLER_THREAD_NAME = "non_deamon"

NLP_PRECOMPUTED_WORD_EMBEDINGS_PF = "/home/berke/Desktop/Workspace/positive-news/backend/nlp/glove.6B/glove.6B.50d.txt"



def __start_crawler():
    print("Starting crawler")
    arr = TextOps().records_as_list()
    crawler_thread = threading.Thread(target=Crawler().crawl)
    crawler_thread.start()
    crawler_thread.join()
    __start_crawler()


def __word2vec_sentiment_analysis():
    print("Starting sentiment analysis")
    arr = TextOps().records_as_list()
    sentimentAnalysis = SIF(NLP_PRECOMPUTED_WORD_EMBEDINGS_PF)
    for article in arr:
        sentimentAnalysis.debug_master(article[2])
    return


def main():
    __word2vec_sentiment_analysis()
    #__start_crawler()


if __name__ == "__main__":
    main()
