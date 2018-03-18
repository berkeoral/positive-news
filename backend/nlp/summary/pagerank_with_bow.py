from tqdm import tqdm

from backend.nlp.basics.bow import BoW
from backend.nlp.basics.prepare_data import PrepareData
from backend.nlp.summary.eval import rogue_n
from backend.utils.txtops import TextOps

from nltk.tokenize import sent_tokenize

import numpy as np
from scipy import spatial


class PagerankWithBOW:
    def __init__(self, data_dir, embeddings, debug=-1):
        self.text_ops = TextOps()
        self.raw_data = self.text_ops.news_summary_as_list(data_dir)
        self.embeddings = embeddings
        self.sentence_embedder = BoW(embeddings)
        #np.random.shuffle(self.raw_data)
        self.prepare_data = PrepareData()
        if debug != -1:
            self.raw_data = self.raw_data[:debug]
        for i in range(len(self.raw_data)):
            sentences = sent_tokenize(self.raw_data[i][2])
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
        return graph

    def __power_method(self, graph, damping=0.2):
        N = graph.shape[0]
        ranks = np.array([(damping) / N + (1-damping)*np.sum(graph[i]) for i in range(graph.shape[0])],
                         dtype=float)
        return ranks

    def summarize(self, summary_length=5, threshold=0.8):
        summaries = []
        originals_debug = []
        lexrank_scores = []
        refference = []
        for i in tqdm(range(len(self.raw_data))):
            if self.raw_data[i][4] is None or len(self.raw_data[i][4]) < 1:
                continue
            if i == 131:
                print("fuck me")
            graph = self.__form_graph(self.raw_data[i][4], threshold)
            lexrank_scores.append(self.__power_method(graph, damping=0.8))
            summary_ind = np.argsort(-lexrank_scores[-1])
            originals_debug.append(self.raw_data[i][2])
            sentences = sent_tokenize(self.raw_data[i][2])
            summaries.append([sentences[_ind] for _ind in summary_ind])
            refference.append(self.raw_data[i][3])
        precision = recall = f1 = 0.
        for i in range(len(summaries)):
            summ = [summaries[i][j] for j in range(summary_length) if j < len(summaries[i])]
            summ = ' '.join(summ)
            reff = refference[i]
            _prec, _recall, _f1 = rogue_n(summary=summ, reference=reff, n=4)
            precision += _prec
            recall += _recall
            f1 += _f1
            print("Precision {0},\tRecall {1},\tf1 {2}".format(str(_prec), str(_recall), str(_f1)))
            print("*****************************")
        print("*--------------------------")
        print("Precision {0},\tRecall {1},\tf1 {2}".format(str(precision / len(summaries)),
                                                               str(recall / len(summaries)),
                                                               str(f1 / len(summaries))))
        """
        for i in range(len(summaries)):
            print("*****************************")
            for j in range(summary_length):
                if j < len(summaries[i]):
                    print(summaries[i][j])
            print("-----------------------------")
            print(originals_debug[i])
            print("*****************************")
        """

