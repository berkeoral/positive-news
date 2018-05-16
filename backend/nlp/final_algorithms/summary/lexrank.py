"""
Implementation of original Lexrank algorithm
"""

import math
import os
import sys

from tqdm import tqdm

from backend.nlp.summary.eval import *

from backend.nlp.basics.preprocessing import Preprocessor
from backend.utils.txtops import TextOps

from nltk.tokenize import sent_tokenize


class LexRank:
    def __init__(self, tfidf_file=None, data_dir=None):
        self.text_ops = TextOps()
        # self.raw_data = self.text_ops.indian_news_summary_as_list(data_dir)
        self.preprocessor = Preprocessor()
        self.idf = {}
        self.tf_idf = {}
        self.tf = {}
        assert tfidf_file is not None or data_dir is not None, "tf-idf file or data directory should given"
        if tfidf_file is not None:
            self._restore_tf_idf(tfidf_file)
        else:
            self._train_tf_idf(data_dir)
            cwd = os.getcwd()
            self._save_tf_idf(os.path.join(cwd, "tfidf.txt"))

    def _prepare_article(self, article):
        article = self.preprocessor.sentences_of_words(article)
        article = [self.preprocessor.default_preprocess(sentence, lemmatizer=True, raw=False)
                   for sentence in article]
        article = [sentence for sentence in article if len(sentence) != 0]
        return article

    def _restore_tf_idf(self, tfidf_file):
        with open(file=tfidf_file, mode="r", encoding="utf_8") as file:
            for line in tqdm(file.readlines(), file=sys.stdout, desc="Restoring TF-IDF From Disk"):
                line = (line.strip()).split()
                self.tf_idf[line[0]] = float(line[1])

    def _save_tf_idf(self, tfidf_file):
        with open(file=tfidf_file, mode="w", encoding="utf_8") as file:
            for key in self.tf_idf.keys():
                print(" ".join([key, str(self.tf_idf[key])]), file=file)

    def _train_tf_idf(self, data_dir):
        data = self.text_ops.cnn_dailymail_as_list(data_dir)
        # Debug
        data = data[:25000]
        #
        data = [article[0] for article in data]

        number_of_document = len(data)
        number_of_words = 0
        flag = set()
        for article in tqdm(data, file=sys.stdout, desc="Calculating TF-IDF"):
            article = self.preprocessor.default_preprocess(article, raw=True, lemmatizer=True)
            number_of_words += len(article)
            for word in article:
                self.tf[word] = self.tf.get(word, 0) + 1
                if word not in flag:
                    self.idf[word] = self.idf.get(word, 0) + 1
                    flag.add(word)
            flag.clear()
        del flag
        for key in self.idf.keys():
            # tf
            self.tf[key] = self.tf[key] / number_of_words
            # idf
            self.idf[key] = math.log2(number_of_document / self.idf[key])
            # tf-idf
            self.tf_idf[key] = self.tf[key] * self.idf[key]

    def _idf_modified_cosine(self, s1, s2):
        words_set = set(s1 + s2)

        def tf(word, sentence):
            count = 0
            for _w in sentence:
                if _w == word:
                    count += 1
            return count

        divident = 0.
        for word in words_set:
            divident += tf(word, s1) * tf(word, s2) * (self.tf_idf.get(word, 0) ** 2)
        divisor_x = 0.
        divisor_y = 0.
        for word in s1:
            divisor_x += (tf(word, s1) * self.tf_idf.get(word, 0)) ** 2
        for word in s2:
            divisor_y += (tf(word, s2) * self.tf_idf.get(word, 0)) ** 2
        divisor = math.sqrt(divisor_x) * math.sqrt(divisor_y)
        if divisor == 0:
            return 0
        return divident / divisor

    def _form_graph(self, document, threshold=0.1):
        dim = len(document)
        graph = np.zeros((dim, dim), dtype=float)
        for i in range(dim):
            for j in range(i + 1):
                distance = self._idf_modified_cosine(document[i], document[j])
                if distance > threshold:
                    graph[j][i] = graph[i][j] = distance
                else:
                    graph[j][i] = graph[i][j] = 0
        return graph

    def _rank_sentences(self, graph, damping=0.85):
        dim = graph.shape[0]
        scores = np.zeros(dim)
        for i in range(dim):
            sum = 0.
            for j in range(dim):
                if graph[i][j] != 0:
                    divident = graph[i][j]
                    divider = np.sum(graph[j])
                    sum += divident / divider
            scores[i] = damping / dim + (1 - damping) * sum
        return scores

    def summarize(self, article, summary_length=3):
        p_article = self._prepare_article(article)
        graph = self._form_graph(p_article)
        scores = self._rank_sentences(graph)
        summary_ind = np.argsort(-scores)
        sentences = sent_tokenize(article)
        summary = " ".join([sentences[i] for i in summary_ind
                            if i < len(sentences) and i < summary_length])
        return summary
