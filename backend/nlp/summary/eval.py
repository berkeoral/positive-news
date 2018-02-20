import numpy as np
import itertools
from collections import deque


def rogue_n(summary, gold_standard, n=1):
    summary = list(itertools.chain.from_iterable(summary))
    gold_standard = list(itertools.chain.from_iterable(gold_standard))

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
    n_gold_standard = calc_n_gram(gold_standard)

    intersection = 0.
    # duplicates are needed: set doesnt work
    for n_gram in n_summary:
        if n_gram in n_gold_standard:
            intersection += 1

    precession = intersection/float(len(n_gold_standard))
    recall = intersection/float(len(n_summary))
    f1 = 2*((precession * recall) / (precession + recall))

    return precession, recall, f1


def rogue_n_precession(summary, gold_standard, n=1):
    return rogue_n(summary, gold_standard, n)[0]


def rogue_n_recall(summary, gold_standard, n=1):
    return rogue_n(summary, gold_standard, n)[1]


def rogue_n_f1(summary, gold_standard, n=1):
    return rogue_n(summary, gold_standard, n)[2]
