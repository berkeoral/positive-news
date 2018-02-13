import abc

from backend.nlp.basics.bow import BoW
from backend.nlp.basics.prepare_data import PrepareData
from backend.utils.txtops import TextOps
from backend.nlp.basics.embedding_ops import Embeddings

import numpy as np


class BaseModel:
    def __init__(self, data_dir, embeddings, tb_logdir, debug=-1):
        self.tb_logdir = tb_logdir
        self.text_ops = TextOps()
        self.raw_data = self.text_ops.acmimdb_as_list(data_dir)
        self.embeddings = embeddings
        self.sentence_embedder = BoW(embeddings)
        np.random.shuffle(self.raw_data)
        self.prepare_data = PrepareData()
        if debug != -1:
            self.raw_data = self.raw_data[:debug]
        for i in range(len(self.raw_data)):
            sentences = self.raw_data[i][2].split('.')
            sentence_embeddings = self.sentence_embedder.weighted_bow(sentences, npc=0)
            self.raw_data[i].append(sentence_embeddings)
        self.__prepare_data()

    def __prepare_data(self):
        print("Preparing data")
        np.random.shuffle(self.raw_data)
        self.data = [[], []]
        for i in range(len(self.raw_data)):
            dat, label = self.prepare_data.just_words(self.raw_data[i][2], self.raw_data[i][1]
                                                      , self.embeddings)
            self.data[0].append(dat)
            self.data[1].append(label)

    def separate_data(self, ratio, sequential=False):
        # if sequential arr dimentions are different
        if not sequential:
            x_data = np.stack(self.data[0][i] for i in range(len(self.data[0])))
        else:
            x_data = self.data[0]
        y_data = np.stack(self.data[1][i] for i in range(len(self.data[1])))
        n = len(x_data)
        ratio = [int(rat * n) for rat in ratio]

        x_train = x_data[0:ratio[0]]
        y_train = y_data[0:ratio[0]]

        x_val = x_data[ratio[0]:ratio[1]]
        y_val = y_data[ratio[0]:ratio[1]]

        x_test = x_data[ratio[1]:]
        y_test = y_data[ratio[1]:]

        return x_train, x_val, x_test, y_train, y_val, y_test

    @abc.abstractmethod
    def classify(self):
        return
