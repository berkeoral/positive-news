import datetime
import sys
import time
import requests

import numpy as np
from tqdm import tqdm

from backend.crawler.crawler import Crawler
from backend.nlp.basics.preprocessing import Preprocessor
from backend.nlp.final_algorithms.sentiment.positive_news import PositiveNews
from backend.nlp.final_algorithms.summary.lexrank import LexRank

service_url = "https://posnews.herokuapp.com/news/send"

wemb_path = "/home/berke/Desktop/workspace/bitirme/positive-news/backend/word_embeddings/glove.6B.200d.txt"
sent_name_space = "SentimentModel"
category_name_space = "CategoryModel"
positivenews_name_space = "PositiveNews"

sent_tb_path = "/home/berke/Desktop/workspace/bitirme/positive-news/backend/nlp/final_algorithms/sentiment/sent_tb"
category_tb_path = "/home/berke/Desktop/workspace/bitirme/positive-news/backend/nlp/final_algorithms/sentiment/category_tb"
positive_tb_path = "/home/berke/Desktop/workspace/bitirme/positive-news/backend/nlp/final_algorithms/sentiment/positive_news_tb"

nlpdb_path = "/home/berke/Desktop/workspace/bitirme/positive-news/backend/nlpdb.txt"
positive_path = "/home/berke/Desktop/workspace/bitirme/positive-news/backend/uplifting_news.tsv"
cnn_dailymail_path = "/home/berke/Desktop/workspace/bitirme/positive-news/backend/CNN_Daily Mail_Dataset"
tfidf_path = "/home/berke/Desktop/workspace/bitirme/positive-news/backend/tfidf.txt"

max_vocab_size = 200000
inter_crawl_time = 300.  # seconds


class Controller(object):
    def __init__(self):
        self.summarizer = LexRank(data_dir=cnn_dailymail_path, tfidf_file=tfidf_path)
        self.positive_model = PositiveNews(wembs_path=wemb_path, name_space=positivenews_name_space,
                                           data_path=positive_path, tb_path=positive_tb_path,
                                           filter_most_frequent_words=max_vocab_size)
        self.sentiment_evaluator = self.positive_model.evaluator()
        self.preprocessor = Preprocessor()
        self.max_batch_size = 128

    def start(self):
        prog_bar = tqdm(total=-1, file=sys.stdout)

        def sleep_inter_crawl_time(end, start):
            if end - start < inter_crawl_time:
                time.sleep(inter_crawl_time - (end - start))

        while True:
            start_time = time.time()
            articles = self._crawler(debug_mode=-1)
            if len(articles) < 2:  # dim in tf.argmax - TODO fix
                tqdm.write("No article been fetched", file=sys.stdout)
                sleep_inter_crawl_time(time.time(), start_time)
                continue
            articles = self._summarize(articles)
            labels = self._classifier(articles)
            for label, article in zip(labels, articles):
                article.append(label)
            self._communicator(articles)
            prog_bar.update(1)
            end_time = time.time()
            sleep_inter_crawl_time(end_time, start_time)

    # Article [0]:image_url, [1]:title, [2]:text
    @staticmethod
    def _crawler(debug_mode):
        articles = Crawler().crawl(debug_mode)
        return articles

    def _classifier(self, articles):
        labels = []
        if articles is None:
            return None
        ids, seq_lens = self._preprocess(articles)
        for i in range(0, len(articles), self.max_batch_size):
            labels += self.send_routine(ids[i:i + self.max_batch_size], seq_lens[i:i + self.max_batch_size])
        return labels

    def _summarize(self, articles):
        for article in tqdm(articles, file=sys.stdout, desc="Summarizing Articles"):
            article.append(self.summarizer.summarize(article[3], summary_length=3))
        return articles

    def _communicator(self, articles):
        tqdm.write("Sending %s articles" % len(articles))
        if len(articles) == 0:
            return
        request_body = []
        for article in articles:
            request_body.append(
                {
                    "news_url": article[0],
                    "image_url": article[1],
                    "news_title": article[2],
                    "news_text": article[3],
                    "news_date": str(datetime.datetime.now()),
                    "news_summary": article[4],
                    "sentiment": article[5]
                }
            )
        requests.post(service_url, json=request_body)

        tqdm.write("Sending found articles", file=sys.stdout)
        for article in articles:
            if article[4] in ["Uplifting", "Positive"]:
                tqdm.write("Title: %s\t Sentiment:%s" % (article[1], article[4]), file=sys.stdout)
        tqdm.write("#"*50, file=sys.stdout)

    def _preprocess(self, articles):
        _prp_articles = [self.preprocessor.default_preprocess(article[2], raw=True, lemmatizer=False)
                         for article in articles]
        article_ids = []
        seq_lens = []
        for article in _prp_articles:
            _a_ids = [self.positive_model.embeddings.word_to_ind_dict[word.lower()] for word in article if
                      word.lower() in self.positive_model.embeddings.word_to_ind_dict]
            seq_lens.append(len(_a_ids) if len(_a_ids) < self.positive_model.hparams.max_seq_len
                            else self.positive_model.hparams.max_seq_len)
            if len(_a_ids) > self.positive_model.hparams.max_seq_len:
                _a_ids = _a_ids[:self.positive_model.hparams.max_seq_len]
            else:
                # reflect padding
                while len(_a_ids) != self.positive_model.hparams.max_seq_len:
                    _a_ids += _a_ids[:self.positive_model.hparams.max_seq_len - len(_a_ids)]
            article_ids.append(_a_ids)

        article_ids = np.stack(article_ids)
        seq_len = np.stack(seq_lens)
        return article_ids, seq_len

    # Not working
    def _remove_dubs(self, articles):
        title_set = set()
        for i, article in enumerate(articles):
            if article[1] not in title_set:
                title_set.add(article[1])
                articles[i] = None
            else:
                continue
        return [article for article in articles if article is not None]

    def send_routine(self, article_ids, article_seq_lens):
        next(self.sentiment_evaluator)
        labels = self.sentiment_evaluator.send((article_ids, article_seq_lens))
        return labels


controller = Controller()
controller.start()
