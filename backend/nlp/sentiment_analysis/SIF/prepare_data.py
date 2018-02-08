"""
List of data processing functions
"""


import numpy as np
import tensorflow as tf

from backend.nlp.sentiment_analysis.SIF.SentenceEmbedding import SentenceEmbedding
from backend.nlp.sentiment_analysis.SIF.doc2vec import doc2vec

tf.logging.set_verbosity(tf.logging.ERROR)

class PrepareData:
    def __init__(self):
        self.doc2vec = doc2vec()

    # Not working, 50% acc
    def mean_document_embedding(self, sentence_embeddings, score):
        """
        self.data[0].append(dat)
        self.data[1].append(label)
        """
        if int(score) < 5:
            score = 0
        else:
            score = 1
        return np.array(self.doc2vec.coordinate_mean(sentence_embeddings)), np.array(score)

    # Not working
    # maybe issue with sentence embeddings
    @staticmethod
    def sentence_time(sentence_embeddings, score):
        """
        for i in range(len(dat)):
            self.data[0].append(dat[i])
            self.data[1].append(label[i])
        """
        if int(score) < 5:
            score = 0
        else:
            score = 1
        score = [np.array(score, dtype=int) for i in range(len(sentence_embeddings))]
        return sentence_embeddings, score

    # Faster than mean_document_embedding, same acc
    @staticmethod
    def sum_document_embedding(sentence_embeddings, score):
        if int(score) < 5:
            score = 0
        else:
            score = 1
        semb = np.sum(sentence_embeddings, axis=0)
        return semb, np.array(score)

    # Faster than mean_document_embedding, same acc
    @staticmethod
    def weighted_mean_document_embedding(sentence_embeddings, score):
        if int(score) < 5:
            score = 0
        else:
            score = 1
        semb = doc2vec.weighted_coordinate_mean(sentence_embeddings)
        return semb, np.array(score)

    @staticmethod
    def sif_document_embedding(documents, scores, _sentence_embedder):
        _score = []
        for score in scores:
            if int(score) < 5:
                _score.append(0)
            else:
                _score.append(1)
        document_embeddings = _sentence_embedder.calc_sentence_embedding(documents, npc=1)
        return document_embeddings, np.array(_score)







