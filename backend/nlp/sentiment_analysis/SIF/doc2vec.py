"""
Creates document embedding from SIF embedding
"""

import numpy as np


class doc2vec:
    def __init__(self):
        self.dummy = 1

    @staticmethod
    def coordinate_mean(sif_embeddings):
        if sif_embeddings is None:
            return None
        return np.mean(sif_embeddings, axis=0)

    @staticmethod
    def weighted_coordinate_mean(embs):
        if embs is None:
            return None
        weights = [np.math.exp(-i) for i in range(len(embs))]
        return np.average(embs, axis=0, weights=weights)

    @staticmethod
    def coordinate_max(sif_embeddings):
        if sif_embeddings is None:
            return None
        return np.max(sif_embeddings, axis=0)

    @staticmethod
    def coordinate_min(sif_embeddings):
        if sif_embeddings is None:
            return None
        return np.min(sif_embeddings, axis=0)

