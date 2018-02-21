import numpy as np
import itertools
from collections import deque
from backend.nlp.basics.preprocessing import Preprocessor


def rogue_n(summary, reference, n=1):
    # Preprocess
    summary = summary.split()
    reference = reference.split()

    preprocessor = Preprocessor()
    summary = preprocessor.default_preprocess(summary, lemmatizing=False)
    reference = preprocessor.default_preprocess(reference, lemmatizing=False)

    def calc_n_gram(doc):
        ret = []
        queue = deque()
        for word in doc:
            if len(queue) < n:
                queue.append(word)
            else:
                queue.append(word)  # Adds to right side
                ret.append(list(queue))
                queue.popleft()  # Removes from left side
        return ret

    n_summary = calc_n_gram(summary)
    n_gold_standard = calc_n_gram(reference)

    intersection = 0.
    # duplicates are needed: set doesnt work
    for n_gram in n_summary:
        if n_gram in n_gold_standard:
            intersection += 1
    if len(n_gold_standard) == 0:
        precession = 0
    else:
        precession = intersection/float(len(n_gold_standard))
    if len(n_summary) == 0:
        recall = 0
    else:
        recall = intersection/float(len(n_summary))
    if precession + recall == 0:
        f1 = 0.
    else:
        f1 = 2*((precession * recall) / (precession + recall))
    return precession, recall, f1


def rogue_n_precession(summary, gold_standard, n=1):
    return rogue_n(summary, gold_standard, n)[0]


def rogue_n_recall(summary, gold_standard, n=1):
    return rogue_n(summary, gold_standard, n)[1]


def rogue_n_f1(summary, gold_standard, n=1):
    return rogue_n(summary, gold_standard, n)[2]
