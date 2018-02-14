import numpy as np
from sklearn.decomposition import TruncatedSVD
from nltk.stem.snowball import SnowballStemmer

from backend.nlp.basics.preprocessing import Preprocessor
from backend.nlp.basics.embedding_ops import Embeddings


class BoW:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.snowball_stemmer = SnowballStemmer("english")
        self.preprocessor = Preprocessor()

    def __preprocess_sentence(self, sentence):
        sentence = (list(set(sentence.split())))
        sentence = self.preprocessor.default_preprocess(sentence, lemmatizing=False)
        sentence = [word for word in sentence
                    if word in self.embeddings.embedding_dictionary and word in self.embeddings.word_weights]
        return sentence

    def __weighted_sentence_average(self, sentences):
        sentences = [self.__preprocess_sentence(sentence) for sentence in sentences]
        sentences = [sentence for sentence in sentences if len(sentence) != 0]
        if len(sentences) == 0:
            return None
        emb = np.zeros((len(sentences), self.embeddings.glove_embedding_dim), dtype=float)
        for sentence, i in zip(sentences, range(len(sentences))):
            word_vectors = np.empty([len(sentence), self.embeddings.glove_embedding_dim], dtype=float)
            word_weights = np.empty(len(sentence), dtype=float)
            for j in range(len(sentence)):
                word_vectors[j] = self.embeddings.embedding_dictionary[sentence[j]]
                word_weights[j] = self.embeddings.word_weights[sentence[j]]
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

    def weighted_bow(self, sentences, npc=0):
        embedding = self.__weighted_sentence_average(sentences)
        if embedding is None:
            return None
        if npc > 0:
            embedding = self.__rm_pc(embedding, npc)
        return embedding

