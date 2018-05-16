import threading

import time

from backend.crawler.crawler import Crawler
from backend.nlp.basics.embedding_ops import Embeddings
from backend.nlp.sentiment_analysis.dffn_classifier.dnn_classifier import DFNClassifier
from backend.nlp.sentiment_analysis.rnn_classifier.dynamic_rnn import DynamicRNN
from backend.nlp.sentiment_analysis.rnn_classifier.rnn_with_attention import RNNWithAttention
from backend.nlp.summary.lexrank import LexRank
from backend.nlp.summary.pagerank_with_bow import PagerankWithBOW
from backend.utils.txtops import TextOps

CRAWLER_CALL_INTERVAL = 5
CRAWLER_INIT_PADDING = 5
CRAWLER_THREAD_NAME = "non_deamon"

WORD_EMBEDDINGS_FOLDER = "/home/berke/Desktop/workspace/bitirme/positive-news/backend/word_embeddings/"
WORD_EMBEDDING_FILE = "glove.6B.300d.txt"
WORD_FREQUENCIES = "/home/berke/Desktop/workspace/bitirme/positive-news/backend/nlp/basics/enwiki_vocab_min200.txt"
ACMIMDB_PATH = "/home/berke/Desktop/workspace/bitirme/positive-news/backend/aclImdb/"
ACMIMDB_TRAINING_POS_FOLDER = "train/pos/*.txt"
ACMIMDB_TRAINING_NEG_FOLDER = "train/neg/*.txt"
ACMIMDB_TEST_POS_FOLDER = "test/pos/*.txt"
ACMIMDB_TEST_NEG_FOLDER = "test/neg/*.txt"
NLPDB_FILE = "/home/berke/Desktop/workspace/bitirme/positive-news/backend/nlpdb.txt"
SUMMARY_FILE = "/home/berke/Desktop/workspace/bitirme/positive-news/backend/news_summary.csv"


TB_LSTM = "/home/berke/Desktop/workspace/bitirme/positive-news/backend/nlp/sentiment_analysis/rnn_classifier/tb_lstm"
TB_DYNAMIC_RNN = "/home/berke/Desktop/workspace/bitirme/positive-news/backend/nlp/sentiment_analysis/rnn_classifier" \
                 "/tb_dynamic_rnn"
TB_RNN_WITH_ATTENTION = "/home/berke/Desktop/workspace/bitirme/positive-news/backend/nlp/sentiment_analysis" \
                        "/rnn_classifier/tb_rnn_with_attention"
TB_DFN = "/home/berke/Desktop/workspace/bitirme/positive-news/backend/nlp/sentiment_analysis/dffn_classifier/tb_l.sgs"


def _start_crawler():
    print("Starting crawler")
    arr = TextOps().records_as_list(TextOps().filename)



def _model_test():
    start = time.time()
    print("Loading embeddings")
    embeddings = Embeddings(WORD_EMBEDDINGS_FOLDER + WORD_EMBEDDING_FILE,
                            WORD_FREQUENCIES)
    load = time.time()
    print("Time elapsed " + str(load - start))
    print("Initializing model")
    model = RNNWithAttention(ACMIMDB_PATH, embeddings, TB_RNN_WITH_ATTENTION, debug=-1)
    init = time.time()
    print("Time elapsed " + str(init - load))
    print("Classify")
    model.classify()
    end_of_execution = time.time()
    print("Time elapsed " + str(end_of_execution - start))


def _summary_test():
    start = time.time()
    print("Loading embeddings")
    embeddings = Embeddings(WORD_EMBEDDINGS_FOLDER + WORD_EMBEDDING_FILE, WORD_FREQUENCIES)
    load = time.time()
    print("Time elapsed " + str(load - start))
    print("Initializing object")
    model = PagerankWithBOW(SUMMARY_FILE, embeddings, debug=-1)
    init = time.time()
    print("Time elapsed " + str(init - load))
    print("Summarize")
    model.summarize()
    end_of_execution = time.time()
    print("Time elapsed " + str(end_of_execution - start))


def main():
    #_summary_test()
    #_model_test()
    _start_crawler()
    #TextOps().tag_papers_()


if __name__ == "__main__":
    main()
