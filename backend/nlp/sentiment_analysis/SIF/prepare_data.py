"""
List of data processing functions
"""


import numpy as np
import tensorflow as tf  # TF 1.1.0rc1

from backend.nlp.sentiment_analysis.SIF.SentenceEmbedding import SentenceEmbedding
from backend.nlp.sentiment_analysis.SIF.doc2vec import doc2vec
from backend.utils.txtops import TextOps
import backend.nlp.sentiment_analysis.lstm_classifier.classifier

tf.logging.set_verbosity(tf.logging.ERROR)
import matplotlib.pyplot as plt

class PrepareData:
    def __init__(self):
        self.doc2vec = doc2vec()

    # Not working, 50% acc
    def mean_document_embeding(self, sentence_vectors, score):
        """
        self.data[0].append(dat)
        self.data[1].append(label)
        """
        if int(score) < 5:
            score = 0
        else:
            score = 1
        return np.array(self.doc2vec.coordinate_mean(sentence_vectors)), np.array(score)


    def sentence_time(self, sentence_vectors, score):
        """
        for i in range(len(dat)):
            self.data[0].append(dat[i])
            self.data[1].append(label[i])
        """
        if int(score) < 5:
            score = 0
        else:
            score = 1
        score = [np.array(score, dtype=int) for i in range(len(sentence_vectors))]
        return sentence_vectors, score