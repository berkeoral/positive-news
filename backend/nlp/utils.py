from collections import Counter
from nltk.corpus import cmudict


class Utils:
    @staticmethod
    def numof_words(self, text):
        words = text.split(" ")
        return words.len()

    # May require preprocessor for words
    @staticmethod
    def numof_diff_words(self, text):
        return Counter(text.split(" "))

    @staticmethod
    def numof_sentences(self, text):
        sentences = text.split(".")
        return sentences.len()

    @staticmethod
    def numof_syllables(self, text):
        words = text.split(" ")
        d = cmudict.dict()
        counter = 0
        for word in words:
            counter += [len(list(y for y in x if y[-1].isdigit()))
                        for x in d[word.lower()]].__len__()
        return counter

