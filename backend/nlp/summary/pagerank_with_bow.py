"""
Naive Pagerank implementation with sentence embeddings
"""

from backend.nlp.BoW.bow import BoW
from backend.nlp.BoW.prepare_data import PrepareData
from backend.utils.txtops import TextOps

import numpy as np
from scipy import spatial


class PagerankWithBOW:
    def __init__(self, data_dir, we_path, wf_path, debug = -1):
        self.text_ops = TextOps()
        self.raw_data = self.text_ops.records_as_list(data_dir)
        self.sentence_embedder = BoW(we_path, wf_path)
        np.random.shuffle(self.raw_data)
        self.prepare_data = PrepareData()
        if debug != -1:
            self.raw_data = self.raw_data[:debug]
        for i in range(len(self.raw_data)):
            sentences = self.raw_data[i][2].split('.')
            sentence_embeddings = self.sentence_embedder.weighted_bow(sentences, npc=0)
            self.raw_data[i].append(sentence_embeddings)

    def __form_graph(self, sentence_embeddings, threshold=0.2):
        dim = len(sentence_embeddings)
        graph = np.zeros((dim, dim), dtype=float)
        degree = [-1] * dim  # double diagonal elements
        for i in range(dim):
            for j in range(i+1):
                distance = 1 - spatial.distance.cosine(sentence_embeddings[i], sentence_embeddings[j])
                if distance < threshold:
                    graph[j][i] = graph[i][j] = 0
                else:
                    graph[j][i] = graph[i][j] = distance
                    degree[i] += 1
                    degree[j] += 1
        for i in range(dim):
            graph[i] = graph[i]/degree[i]
        return graph, degree

    def __power_method(self, graph, degree, d=0.8):
        """
        ranks = np.zeros(graph.shape[0],dtype=float)
        for i in range(len(graph.shape[0])):
            ranks[0] = (1/degree[i]) + (1-d)*np.sum(graph[i])
        """
        ranks = np.array([((1 - d)/degree[i]) + d*np.sum(graph[i]) for i in range(graph.shape[0])],
                         dtype=float)  # :)
        return ranks

    def summarize(self, summary_length= 3, max_character = 50, threshold=0.7):
        summaries = []
        originals_debug = []
        lexrank_scores = []
        for i in range(len(self.raw_data)):
            graph, degree = self.__form_graph(self.raw_data[i][3], threshold)
            lexrank_scores.append(self.__power_method(graph, degree, d=0.8))
            summary_ind = np.argsort(-lexrank_scores[i])
            originals_debug.append(self.raw_data[i][2])
            sentences = self.raw_data[i][2].split('.')
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
