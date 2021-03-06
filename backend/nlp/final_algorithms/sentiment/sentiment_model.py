import sys
import numpy as np

from tqdm import tqdm
from sklearn.utils import shuffle

from backend.nlp.final_algorithms.sentiment.base_model import BaseModel
from backend.utils.txtops import TextOps


class SentimentModel(BaseModel):
    def __init__(self, wembs_path, name_space, data_path, tb_path, filter_most_frequent_words=-1):
        super().__init__(wembs_path, name_space, tb_path, filter_most_frequent_words)
        self.data_path = data_path
        self.data_func()
        self.train_generator = self.batch_generator(self.hparams.batch_size)
        self.graph()

    def data_func(self):
        self.hparams.n_classes = 2
        self.label_dict = {0: "Negative", 1: "Positive"}
        if self.data_path is None:
            return
        text_ops = TextOps()
        raw_data = text_ops.acmimdb_as_list(self.data_path)
        data = []
        self.data = []
        for article in tqdm(raw_data, file=sys.stdout, unit="article", total=50000):
            article_ids = [self.embeddings.word_to_ind_dict[word.lower()] for word in article[2].split() if
                           word.lower() in self.embeddings.word_to_ind_dict]
            article_len = len(article_ids) if len(article_ids) < self.hparams.max_seq_len else self.hparams.max_seq_len
            article_label = [1., 0.] if int(article[1]) < 5 else [0., 1.]

            if len(article_ids) > self.hparams.max_seq_len:
                article_ids = article_ids[:self.hparams.max_seq_len]
            else:
                article_ids = article_ids + [-1 for _ in range(self.hparams.max_seq_len - len(article_ids))]

            data.append([article_ids, article_label, article_len])
        self.data.append(np.stack([data[i][0] for i in range(len(data))]).astype(np.int32))
        self.data.append(np.stack([data[i][1] for i in range(len(data))], axis=0))
        self.data.append(np.stack([data[i][2] for i in range(len(data))]).astype(np.int32))
