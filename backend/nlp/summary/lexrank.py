"""
Implementation of original Lexrank algorithm
"""

import math
import numpy as np

from backend.nlp.basics.preprocessing import Preprocessor
from backend.utils.txtops import TextOps

from nltk.tokenize import sent_tokenize


class LexRank:
    def __init__(self, data_dir, debug=-1):
        self.text_ops = TextOps()
        self.raw_data = self.text_ops.records_as_list(data_dir)
        self.preprocessor = Preprocessor()
        self.idf = {}
        self.tf = []
        self.document_word_counts = []
        if debug > 0:
            self.raw_data = self.raw_data[:debug]
        self.data = [[]]
        self.__prepare_data()
        self.__train_tf_idf()
        print("debug")

    def __prepare_data(self):
        for _data in self.raw_data:
            article = self.preprocessor.sentences_of_words(_data[2])
            article = [self.preprocessor.default_preprocess(sentence, lemmatizing=True)
                       for sentence in article]
            article = [sentence for sentence in article if len(sentence) != 0]
            self.data[0].append(article)

    # tf calculation is redundant
    def __train_tf_idf(self):
        number_of_document = 0
        flag = set()
        for document in self.data[0]:
            number_of_document += 1
            self.document_word_counts.append(0)
            self.tf.append({})
            for sentence in document:
                for word in sentence:
                    self.document_word_counts[-1] += 1
                    self.tf[-1][word] = self.tf[-1].get(word, 0) + 1
                    if word not in flag:
                        self.idf[word] = self.idf.get(word, 0) + 1
                        flag.add(word)
            flag.clear()
        del flag
        for key in self.idf:
            self.idf[key] = math.log2(number_of_document/self.idf[key])

    def __idf_modified_cosine(self, s1, s2):
        words_set = set(s1 + s2)
        def tf(word, sentence):
            count = 0
            for _w in sentence:
                if _w == word:
                    count += 1
            return count
        divident = 0.
        for word in words_set:
            divident += tf(word, s1)*tf(word, s2)*(self.idf.get(word, 0)**2)
        divisor_x = 0.
        divisor_y = 0.
        for word in s1:
            divisor_x += (tf(word, s1)*self.idf.get(word, 0)) ** 2
        for word in s2:
            divisor_y += (tf(word, s2)*self.idf.get(word, 0)) ** 2
        divisor = math.sqrt(divisor_x) * math.sqrt(divisor_y)
        return divident/divisor

    def __form_graph(self, document, threshold=0.1):
        dim = len(document)
        graph = np.zeros((dim, dim), dtype=float)
        for i in range(dim):
            for j in range(i + 1):
                distance = self.__idf_modified_cosine(document[i], document[j])
                if distance > threshold:
                    graph[j][i] = graph[i][j] = distance
                else:
                    graph[j][i] = graph[i][j] = 0
        return graph

    def __rank_sentences(self, graph, damping=0.85):
        dim = graph.shape[0]
        scores = np.zeros(dim)
        for i in range(dim):
            sum = 0.
            for j in range(dim):
                if graph[i][j] != 0:
                    divident = graph[i][j]
                    divider = np.sum(graph[j])
                    sum += divident/divider
            scores[i] = damping/dim + (1-damping)*sum
        return scores

    def summarize(self, summary_length=3):
        summaries = []
        originals_debug = []
        lexrank_scores = []
        for i in range(len(self.raw_data)):
            graph = self.__form_graph(self.data[0][i])
            lexrank_scores.append(self.__rank_sentences(graph))
            summary_ind = np.argsort(-lexrank_scores[i])
            originals_debug.append(self.raw_data[i][2])
            sentences = sent_tokenize(self.raw_data[i][2])
            summaries.append([sentences[_ind] for _ind in summary_ind])
        for i in range(len(summaries)):
            print("*****************************")
            for j in range(summary_length):
                if j < len(summaries[i]):
                    print(summaries[i][j])
            print("-----------------------------")
            print(originals_debug[i])
            print("*****************************")
        return summaries  # for debug
