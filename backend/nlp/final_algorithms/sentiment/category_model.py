import tensorflow as tf
import numpy as np

from nltk.corpus import brown
from sklearn.utils import shuffle

from backend.nlp.final_algorithms.sentiment.base_model import BaseModel


class CategoryModel(BaseModel):
    def __init__(self, wembs_path, name_space, tb_path, filter_most_frequent_words=-1):
        super().__init__(wembs_path, name_space, tb_path, filter_most_frequent_words)
        self.data_func()
        self.train_generator = self.batch_generator(self.hparams.batch_size)
        self.graph()

    def data_func(self):
        self.data = [[], [], []]
        categories = list(brown.categories())
        self.hparams.n_classes = len(categories)
        for category in brown.categories():
            self.label_dict[len(self.label_dict)] = category
            for document in brown.words(categories=category)._pieces:
                article_ids = [self.embeddings.word_to_ind_dict[word.lower()] for word in list(document) if
                               word.lower() in self.embeddings.word_to_ind_dict]
                seq_len = len(article_ids) if len(article_ids) < self.hparams.max_seq_len else self.hparams.max_seq_len
                label = [1 if i == categories.index(category) else 0 for i in range(self.hparams.n_classes)]

                if len(article_ids) > self.hparams.max_seq_len:
                    article_ids = article_ids[:self.hparams.max_seq_len]
                else:
                    article_ids = article_ids + [-1 for _ in range(self.hparams.max_seq_len - len(article_ids))]

                self.data[0].append(article_ids)
                self.data[1].append(label)
                self.data[2].append(seq_len)
        self.data[0] = np.stack(self.data[0])
        self.data[1] = np.stack(self.data[1])
        self.data[2] = np.stack(self.data[2])