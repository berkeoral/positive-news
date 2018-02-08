"""
Adopted from:
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
from nltk.stem.snowball import SnowballStemmer
from hunspell import HunSpell

class SentenceEmbedding:
    def __init__(self, word_embeddings_path, word_frequencies_path):
        self.word_embeddings_path = word_embeddings_path
        self.glove_vocab = []
        self.embedding_dictionary = {}
        self.get_embeddings()
        self.word_weights = {}  # Glove vocabulary contains word_frequencies vocabulary
        self.word_frequencies_path = word_frequencies_path
        self.get_weights()
        self.glove_vocab_size = len(self.glove_vocab)
        self.glove_embedding_dim = len(self.embedding_dictionary[self.glove_vocab[0]])
        self.snowball_stemmer = SnowballStemmer("english")

    def __preprocess_sentence(self, sentence):
        sentence = (list(set(sentence.split())))
        sentence = [(word).lower() for word in sentence]
        sentence = [word for word in sentence if word in self.embedding_dictionary and word in self.word_weights]
        return sentence

    def __weighted_sentence_average(self, sentences):
        sentences = [self.__preprocess_sentence(sentence) for sentence in sentences]
        sentences = [sentence for sentence in sentences if len(sentence) != 0]
        if len(sentences) == 0:
            return None
        emb = np.zeros((len(sentences), self.glove_embedding_dim), dtype=float)
        for sentence, i in zip(sentences, range(len(sentences))):
            word_vectors = np.empty([len(sentence), self.glove_embedding_dim], dtype=float)
            word_weights = np.empty(len(sentence), dtype=float)
            for j in range(len(sentence)):
                word_vectors[j] = self.embedding_dictionary[sentence[j]]
                word_weights[j] = self.word_weights[sentence[j]]
            emb[i, :] = word_weights[:].dot(word_vectors[:, :]) / np.count_nonzero(word_weights[:])
        return emb

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

    def get_embeddings(self):
        file = open(self.word_embeddings_path, 'r', encoding='UTF-8')
        for line in file.readlines():
            row = line.strip().split(' ') # Space is default separator unnecessary
            self.glove_vocab.append(row[0])
            embed_vector = [float(i) for i in row[1:]]
            self.embedding_dictionary[row[0]] = embed_vector
        file.close()
        print("Glove loaded")

    def get_weights(self, weight_param=1e-3):
        file = open(self.word_frequencies_path, 'r', encoding='UTF-8')
        count = 0
        for line in file.readlines():
            row = line.split(' ')
            self.word_weights[row[0]] = float(row[1])
            count += float(row[1])
        for word, value in self.word_weights.items():
            self.word_weights[word] = weight_param / (weight_param + value / count)
        file.close()
        print("Word weights loaded")

    def calc_sentence_embedding(self, sentences, npc):
        embedding = self.__weighted_sentence_average(sentences)
        if embedding is None:
            return None
        if npc > 0:
            embedding = self.__rm_pc(embedding, npc)
        return embedding

