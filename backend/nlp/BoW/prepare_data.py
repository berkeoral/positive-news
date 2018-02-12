"""
List of data processing functions
"""

import numpy as np


class PrepareData:
    def __init__(self):
        self.dummy = 0

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

    @staticmethod
    def irregular_sequence(sentence_embeddings, score):
        if int(score) < 5:
            score = 0
        else:
            score = 1

        return sentence_embeddings, np.array(score)

    @staticmethod
    def just_words(article, score, sentence_embedder):
        if int(score) < 5:
            score = 0
        else:
            score = 1
        article = article.split()
        article_words = [word.lower() for word in article
                         if word.lower() in sentence_embedder.embedding_dictionary]
        return article_words, np.array(score)

    # for cnn & rnn
    # if greater than seq_len cut it
    # if lesser fill with 0 vectors
    @staticmethod
    def padded_sequence(sentence_embeddings, score, seq_len=100):
        if int(score) < 5:
            score = 0
        else:
            score = 1
        input_length = len(sentence_embeddings)
        if input_length < seq_len:
            fill = seq_len - input_length
            dims = len(sentence_embeddings[0])
            sentence_embeddings = np.concatenate((sentence_embeddings, np.zeros((fill, dims), dtype=float)))
        else:
            sentence_embeddings = sentence_embeddings[:seq_len]
        return sentence_embeddings, np.array(score)

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







