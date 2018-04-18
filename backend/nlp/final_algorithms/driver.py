import sys

import numpy as np

from tqdm import tqdm

from backend.nlp.basics.preprocessing import Preprocessor
from backend.nlp.final_algorithms.sentiment.category_model import CategoryModel
from backend.nlp.final_algorithms.sentiment.positive_news import PositiveNews
from backend.nlp.final_algorithms.sentiment.sentiment_model import SentimentModel
from backend.utils.txtops import TextOps

wemb_path = "/home/berke/Desktop/workspace/bitirme/positive-news/backend/word_embeddings/glove.6B.200d.txt"
sent_name_space = "SentimentModel"
category_name_space = "CategoryModel"
positivenews_name_space = "PositiveNews"

sent_tb_path = "/home/berke/Desktop/workspace/bitirme/positive-news/backend/nlp/final_algorithms/sentiment/sent_tb"
category_tb_path = "/home/berke/Desktop/workspace/bitirme/positive-news/backend/nlp/final_algorithms/sentiment/category_tb"
positive_tb_path = "/home/berke/Desktop/workspace/bitirme/positive-news/backend/nlp/final_algorithms/sentiment/positive_news_tb"

aclimdb_path = "/home/berke/Desktop/workspace/bitirme/positive-news/backend/aclImdb/"
nlpdb_path = "/home/berke/Desktop/workspace/bitirme/positive-news/backend/nlpdb.txt"
positive_path = "/home/berke/Desktop/workspace/bitirme/positive-news/backend/uplifting_news.tsv"

max_vocab_size = 200000

positive_model = PositiveNews(wembs_path=wemb_path, name_space=positivenews_name_space, data_path=positive_path,
                              tb_path=positive_tb_path, filter_most_frequent_words=max_vocab_size)

#positive_model.train()

txt_ops = TextOps()
nlpdb = txt_ops.records_as_list(nlpdb_path)

preprocessor = Preprocessor()


def preprocess(article):
    article = preprocessor.default_preprocess(article, raw=True, lemmatizer=False)
    article_ids = [positive_model.embeddings.word_to_ind_dict[word.lower()] for word in article if
                   word.lower() in positive_model.embeddings.word_to_ind_dict]
    seq_len = len(article_ids) if len(
        article_ids) < positive_model.hparams.max_seq_len else positive_model.hparams.max_seq_len
    if len(article_ids) > positive_model.hparams.max_seq_len:
        article_ids = article_ids[:positive_model.hparams.max_seq_len]
    else:
        # reflect padding
        while len(article_ids) != positive_model.hparams.max_seq_len:
            article_ids = article_ids + article_ids[:positive_model.hparams.max_seq_len - len(article_ids)]

        # article_ids = article_ids + [0 for _ in range(category_model.hparams.max_seq_len - len(article_ids))]
    article_ids = np.array(article_ids, dtype=np.int32, ndmin=2)
    seq_len = np.array(seq_len, dtype=np.int32, ndmin=1)
    return article_ids, seq_len


def send_routine(generator, article_ids, seq_len):
    next(generator)
    label = generator.send((article_ids, seq_len))
    return label


positivenews_evaluator = positive_model.evaluator()

evaluated = []

for i in tqdm(range(5000, 6000), file=sys.stdout):
    article = nlpdb[i]
    article_ids, article_seq_len = preprocess(article[2])
    sentiment = send_routine(positivenews_evaluator, article_ids, article_seq_len)

    article += [sentiment]
    evaluated.append(article)
    tqdm.write("Title: %s\n Sentiment: %s\n" % (article[1], sentiment))
    tqdm.write("_-*-" * 10)

evaluated_path = "/home/berke/Desktop/workspace/bitirme/positive-news/backend/evaluated.csv"
picked_path = "/home/berke/Desktop/workspace/bitirme/positive-news/backend/picked.csv"


def export(path, data):
    with open(path, "w", encoding="utf_8") as file:
        for article in data:
            line = "\t".join([article[1], article[-1]])
            print(line, file=file)

picked = [article for article in evaluated if article[-1] in ["Positive", "Uplifting"]]

export(evaluated_path, evaluated)
export(picked_path, picked)


