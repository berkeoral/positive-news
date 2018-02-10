"""
List of data processing functions
"""

import numpy as np


class PrepareData:
    def __init__(self):
        dummy = 1

    @staticmethod
    def mean_document_embedding(sentence_embeddings, score):
        """
        self.data[0].append(dat)
        self.data[1].append(label)
        """
        if int(score) < 5:
            score = 0
        else:
            score = 1
        return np.array(np.mean(sentence_embeddings, axis=0)), np.array(score)

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

    @staticmethod
    def sif_document_embedding(documents, scores, _sentence_embedder):
        _score = []
        for score in scores:
            if int(score) < 5:
                _score.append(0)
            else:
                _score.append(1)
        document_embeddings = _sentence_embedder.weighted_bow(documents, npc=1)
        return document_embeddings, np.array(_score)







