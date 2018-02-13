import threading

import time

from backend.crawler.crawler import Crawler
from backend.nlp.basics.embedding_ops import Embeddings
from backend.nlp.sentiment_analysis.dnn_classifier.dnn_classifier import DNNClassifier
from backend.nlp.sentiment_analysis.lstm_classifier.dynamic_rnn import DynamicRNN
from backend.nlp.sentiment_analysis.lstm_classifier.lstm_classifier import LSTMClassifier
from backend.nlp.summary.lexrank import LexRank
from backend.nlp.summary.pagerank_with_bow import PagerankWithBOW
from backend.utils.txtops import TextOps

CRAWLER_CALL_INTERVAL = 5
CRAWLER_INIT_PADDING = 5
CRAWLER_THREAD_NAME = "non_deamon"

WORD_EMBEDDINGS_FOLDER = "/home/berke/Desktop/Workspace/positive-news/backend/word_embeddings/"
WORD_EMBEDDING_FILE = "glove.6B.50d.txt"
WORD_FREQUENCIES = "/home/berke/Desktop/Workspace/positive-news/backend/nlp/basics/enwiki_vocab_min200.txt"
ACMIMDB_PATH = "/home/berke/Desktop/Workspace/positive-news/backend/aclImdb/"
ACMIMDB_TRAINING_POS_FOLDER = "train/pos/*.txt"
ACMIMDB_TRAINING_NEG_FOLDER = "train/neg/*.txt"
ACMIMDB_TEST_POS_FOLDER = "test/pos/*.txt"
ACMIMDB_TEST_NEG_FOLDER = "test/neg/*.txt"
NLPDB_FILE = "/home/berke/Desktop/Workspace/positive-news/backend/nlpdb.txt"

TB_LSTM = "/home/berke/Desktop/Workspace/positive-news/backend/nlp/sentiment_analysis/lstm_classifier/tb_lstm"
TB_DYNAMIC_RNN = "/home/berke/Desktop/Workspace/positive-news/backend/nlp/sentiment_analysis/lstm_classifier" \
                 "/tb_dynamic_rnn"
TB_DNN = "/home/berke/Desktop/Workspace/positive-news/backend/nlp/sentiment_analysis/dnn_classifier/tb_logs"


def __start_crawler():
    print("Starting crawler")
    arr = TextOps().records_as_list(TextOps().filename)
    crawler_thread = threading.Thread(target=Crawler().crawl)
    crawler_thread.start()
    crawler_thread.join()
    __start_crawler()


def __model_test():
    start = time.time()
    print("Loading embeddings")
    embeddings = Embeddings(WORD_EMBEDDINGS_FOLDER + WORD_EMBEDDING_FILE,
                            WORD_FREQUENCIES)
    load = time.time()
    print("Time elapsed " + str(load - start))
    print("Initializing model")
    model = DynamicRNN(ACMIMDB_PATH, embeddings, TB_DNN, debug=100)
    init = time.time()
    print("Time elapsed " + str(init - load))
    print("Classify")
    model.classify()
    end_of_execution = time.time()
    print("Time elapsed " + str(end_of_execution - start))


def __summary_test():
    start = time.time()
    print("Loading embeddings")
    embeddings = Embeddings(WORD_EMBEDDINGS_FOLDER + WORD_EMBEDDING_FILE,
                            WORD_FREQUENCIES)
    load = time.time()
    print("Time elapsed " + str(load - start))
    print("Initializing object")
    model = LexRank(NLPDB_FILE, debug=10)
    init = time.time()
    print("Time elapsed " + str(init - load))
    print("Summarize")
    model.summarize()
    end_of_execution = time.time()
    print("Time elapsed " + str(end_of_execution - start))


def main():
    __summary_test()
    #__model_test()
    #__start_crawler()
    #TextOps().tag_papers_()


if __name__ == "__main__":
    main()
