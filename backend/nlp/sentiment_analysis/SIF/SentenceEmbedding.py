"""
@article{
    arora2017asimple,
	author = {Sanjeev Arora and Yingyu Liang and Tengyu Ma},
	title = {A Simple but Tough-to-Beat Baseline for Sentence Embeddings},
	booktitle = {International Conference on Learning Representations},
	year = {2017}
}
"""

from backend.nlp.utils import Utils

import tensorflow as tf
import numpy as np
from sklearn.decomposition import TruncatedSVD


class SentenceEmbedding:
    def __init__(self, word_embeddings_path, word_frequencies_path):
        self.word_embeddings_path = word_embeddings_path
        self.glove_vocab = []
        self.embedding_dictionary = {}
        self.get_embedings()
        self.word_frequencies = {}  # Glove vocabulary contain word_frequencies vocabulary
        self.word_frequencies_path = word_frequencies_path
        self.get_frequencies()
        self.glove_vocab_size = len(self.glove_vocab)
        self.glove_embedding_dim = len(self.embedding_dictionary[self.glove_vocab[0]])

    def __weighted_sentence_average(self, sentence):
        sentence = list(set(sentence))  # remove duplicate words from sentence
        embedding = np.zeros(1, self.glove_embedding_dim)
        word_vectors = np.array(len(sentence), self.glove_embedding_dim)
        word_weights = np.array(1, len(sentence))
        for i in range(len(sentence)):
            word_vectors[i] = self.embedding_dictionary[sentence[i]]
            word_weights[0][i] = self.word_frequencies[sentence[i]]
        embedding[0] = word_weights[1, :].dot(word_vectors[:, :]) / np.count_nonzero(word_weights[:])
        return embedding

    def __calc_pc(self, embedding, npc=1):
        svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
        svd.fit(embedding)
        return svd.components_

    def __rm_pc(self, embedding, npc=1):
        pc = self.__calc_pc(embedding, npc)
        if npc == 1:
            n_embedding = embedding - embedding.dot(pc.transpose())*pc
        else:
            n_embedding = embedding - embedding.dot(pc.transpose()).dot(pc)
        return n_embedding

    def calc_sentence_embedding(self, sentence, npc):
        embedding = self.__weighted_sentence_average(sentence)
        if npc > 0:
            embedding = self.__rm_pc(embedding, npc)
        return embedding


    def get_embedings(self):
        file = open(self.word_embeddings_path, 'r', encoding='UTF-8')
        for line in file.readlines():
            row = line.strip().split(' ') # Space is default separator unnecessary
            self.glove_vocab.append(row[0])
            embed_vector = [float(i) for i in row[1:]]
            self.embedding_dictionary[row[0]] = embed_vector
        file.close()
        print("Glove loaded")

    def get_frequencies(self):
        file = open(self.word_frequencies_path, 'r', encoding='UTF-8')
        for line in file.readlines():
            row = line.split(' ')
            self.word_frequencies[row[0]]=row[1]
        file.close()
        print("Word frequencies loaded")

