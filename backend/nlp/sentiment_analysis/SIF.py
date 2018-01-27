from backend.nlp.utils import Utils

import tensorflow as tf
import numpy as np

class SIF:
    def __init__(self, word_embedings_path):
        self.word_embedings_path = word_embedings_path
        self.glove_vocab = []
        self.embeding_dictionary = {}
        self.get_embedings()
        self.glove_vocab_size = len(self.glove_vocab)
        self.glove_embeding_dim = len(self.embeding_dictionary[self.glove_vocab[0]])
        return

    def calc_sentence_embeding(self, sentence):
        sentence_embeding = np.zeros(len(sentence), len(self.embeding_dictionary[self.glove_vocab[0]]))
        for i in range(0,len(sentence)):
            sentence_embeding = self.glove_vocab[0] # NOT FINISHED
        return

    def calc_article_embeding(self, article):
        sentences = article.split(".")
        for sentence in sentences:
            final_sum = self.calc_sentence_embeding(sentence)
        if(final_sum > 0):
            print("POSITIVE : " + article[1])
        else:
            print("NEGATIVE : " + article[1])

    def debug_master(self, article):
        self.calc_article_embeding(article)

    def get_embedings(self):
        file = open(self.word_embedings_path, 'r', encoding='UTF-8')
        for line in file.readlines():
            row = line.strip().split(' ')
            self.glove_vocab.append(row[0])
            embed_vector = [float(i) for i in row[1:]]
            self.embeding_dictionary[row[0]] = embed_vector
        file.close()
        print("Glove loaded")
