import threading

import time

from backend.crawler.crawler import Crawler
from backend.nlp.sentiment_analysis.basic_ltsm import BasicLSTMClassifier
from backend.nlp.sentiment_analysis.dnn_classifier.dnn_classifier import DNNClassifier
from backend.nlp.sentiment_analysis.lstm_classifier.lstm_classifier import LSTMClassifier
from backend.utils.txtops import TextOps
from backend.nlp.sentiment_analysis.SIF.SentenceEmbedding import SentenceEmbedding

CRAWLER_CALL_INTERVAL = 5
CRAWLER_INIT_PADDING = 5
CRAWLER_THREAD_NAME = "non_deamon"

WORD_EMBEDDINGS_FOLDER = "/home/berke/Desktop/Workspace/positive-news/backend/word_embeddings/"
WORD_EMBEDDING_FILE = "glove.6B.300d.txt"
WORD_FREQUENCIES = "/home/berke/Desktop/Workspace/positive-news/backend/nlp/sentiment_analysis" \
                   "/SIF/enwiki_vocab_min200.txt"
ACMIMDB_PATH = "/home/berke/Desktop/Workspace/positive-news/backend/aclImdb/"
ACMIMDB_TRAINING_POS_FOLDER = "train/pos/*.txt"
ACMIMDB_TRAINING_NEG_FOLDER = "train/neg/*.txt"
ACMIMDB_TEST_POS_FOLDER = "test/pos/*.txt"
ACMIMDB_TEST_NEG_FOLDER = "test/neg/*.txt"

LSTM_LOGDIR = "/home/berke/Desktop/Workspace/positive-news/backend/nlp/sentiment_analysis/lstm_classifier/tb_logs"
DNN_LOGDIR = "/home/berke/Desktop/Workspace/positive-news/backend/nlp/sentiment_analysis/dnn_classifier/tb_logs"


def __start_crawler():
    print("Starting crawler")
    arr = TextOps().records_as_list(TextOps().filename)
    crawler_thread = threading.Thread(target=Crawler().crawl)
    crawler_thread.start()
    crawler_thread.join()
    __start_crawler()


def __basic_sentiment_analysis_test():
    start = time.time()
    print("Advance stuff about to happen")
    training_set = TextOps().acmimdb_as_list(ACMIMDB_PATH)
    end_of_train = time.time()
    print("Time elapsed " + str(end_of_train - start))
    print("Training set loaded")
    sentence_embeder = SentenceEmbedding(WORD_EMBEDDINGS_FOLDER + WORD_EMBEDDING_FILE, WORD_FREQUENCIES)
    end_of_load = time.time()
    print("Time elapsed " + str(end_of_load - start))
    arr = TextOps().records_as_list(TextOps().filename)
    for article in arr:
        if len(article) != 3:
            continue
        sentences = article[2].split(".")
        for sentence in sentences:
            emb = sentence_embeder.calc_sentence_embedding(sentence, 1)
    end_of_execution = time.time()
    print("Time elapsed " + str(end_of_execution - start))
    return

def __model_test():
    start = time.time()
    print("Calculating sentence embeddings, document embeddings")
    model = DNNClassifier(ACMIMDB_PATH, WORD_EMBEDDINGS_FOLDER + WORD_EMBEDDING_FILE, WORD_FREQUENCIES
                                     , DNN_LOGDIR)
    init = time.time()
    print("Time elapsed " + str(init - start))
    print("Classify")
    model.classify()
    end_of_execution = time.time()
    print("Time elapsed " + str(end_of_execution - start))

def main():
    __model_test()
    #__start_crawler()
    #TextOps().tag_papers_()

if __name__ == "__main__":
    main()
