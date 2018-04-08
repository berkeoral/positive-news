"""
Core embeddings object
"""
import sys

import tensorflow as tf
import numpy as np
import csv

from tqdm import tqdm


class Embeddings:
    def __init__(self, word_embeddings_path, word_frequencies_path):
        self.word_embeddings_path = word_embeddings_path
        self.embedding_dictionary = {}
        self.__get_embeddings()
        self.word_weights = {}  # Glove vocabulary contains word_frequencies vocabulary
        self.word_frequencies_path = word_frequencies_path
        self.__get_weights()
        self.glove_vocab_size = len(self.embedding_dictionary)
        self.glove_embedding_dim = len(self.embedding_dictionary["this"])

    def __get_embeddings(self):
        file = open(self.word_embeddings_path, 'r', encoding='UTF-8')
        for line in file.readlines():
            row = line.strip().split(' ')  # Space is default separator unnecessary
            embed_vector = [float(i) for i in row[1:]]
            self.embedding_dictionary[row[0]] = embed_vector
        file.close()
        print("Glove loaded")

    def __get_weights(self, weight_param=1e-3):
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


class EmbeddingsV2:
    def __init__(self, word_embeddings_path, special_chars=None, filter_most_frequent_words=-1):
        self.word_embeddings_path = word_embeddings_path
        self.special_chars = special_chars
        self.most_frequent = filter_most_frequent_words
        self.vocab_size = self.embedding_dim = 0
        self.word_embeddings = None
        self.word_to_ind_dict = {}
        self.ind_to_word_dict = {}
        self.__get_embeddings()

    def __get_embeddings(self):
        with open(self.word_embeddings_path, 'r', encoding='UTF-8') as file:
            embed_vectors = []
            self.embedding_dim = len(file.readlines(1)[0].strip().split()) - 1
            file.seek(0)

            # Push special characters : <UNK>, <begin> etc
            num_of_special_characters = 0 if self.special_chars is None else len(self.special_chars)
            for i in range(num_of_special_characters):
                embed_vectors.append([0. for _i in range(self.embedding_dim)])
                self.word_to_ind_dict[self.special_chars[i]] = i
                self.ind_to_word_dict[i] = self.special_chars[i]

            for i, line in tqdm(enumerate(file.readlines()), file=sys.stdout, total=self.most_frequent):
                row = line.strip().split()
                embed_vectors.append([float(_i) for _i in row[1:]])
                self.word_to_ind_dict[row[0]] = i + num_of_special_characters
                self.ind_to_word_dict[i + num_of_special_characters] = row[0]
                if i == self.most_frequent:
                    break

        self.vocab_size = len(embed_vectors)
        self.word_embeddings = embed_vectors
        print("Embeddings loaded")
