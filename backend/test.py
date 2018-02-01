import threading
import timeit

import time

from backend.crawler.crawler import Crawler
from backend.crawler.txtops import TextOps
from backend.nlp.sentiment_analysis.SIF.SentenceEmbedding import SentenceEmbedding

CRAWLER_CALL_INTERVAL = 5
CRAWLER_INIT_PADDING = 5
CRAWLER_THREAD_NAME = "non_deamon"

WORD_EMBEDDINGS_FOLDER = "/home/berke/Desktop/Workspace/positive-news/backend/word_embeddings/"
WORD_EMBEDDING_FILE = "glove.6B.50d.txt"
WORD_FREQUENCIES = "/home/berke/Desktop/Workspace/positive-news/backend/nlp/sentiment_analysis" \
                   "/SIF/enwiki_vocab_min200.txt"


def __start_crawler():
    print("Starting crawler")
    arr = TextOps().records_as_list(TextOps().filename)
    crawler_thread = threading.Thread(target=Crawler().crawl)
    crawler_thread.start()
    crawler_thread.join()
    __start_crawler()


def __word2vec_sentiment_analysis():
    print("Advance stuff about to happen")
    start = time.time()
    sentence_embeder = SentenceEmbedding(WORD_EMBEDDINGS_FOLDER + WORD_EMBEDDING_FILE, WORD_FREQUENCIES)
    end_of_load = time.time()
    print("Time elapsed " + str(end_of_load - start))
    return


def main():
    __word2vec_sentiment_analysis()
    #__start_crawler()
    #TextOps().tag_papers_()

if __name__ == "__main__":
    main()
