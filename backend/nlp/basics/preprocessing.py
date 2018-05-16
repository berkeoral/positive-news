import string
from itertools import chain

import numpy as np
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize


class Preprocessor:
    def __init__(self):
        self.stopwords = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.porter = PorterStemmer()

    # TODO BUG: not removing " and '
    def default_preprocess(self, input, raw=True, remove_digit=True, remove_stopwords=False, lemmatizer=True,
                           stemming=False):
        if raw:
            input = input.split()

        input = [((word.lower()).translate(str.maketrans('', '', string.punctuation))).rstrip()
                 for word in input if word != "."]
        input = [word for word in input if word != ""]
        if remove_digit:
            input = [word for word in input if not any(char.isdigit() for char in word)]
        if remove_stopwords:
            input = [word for word in input if word not in self.stopwords]
        if lemmatizer:
            input = [self.lemmatizer.lemmatize(word) for word in input]
        if stemming:
            input = [self.porter.stem(word) for word in input]
        return input

    @staticmethod
    def sentences_of_words(input):
        input = sent_tokenize(input)
        input = [sentence.split() for sentence in input]
        return input

    @staticmethod
    def merge_sentences(input):
        return "".join(input)

    @staticmethod
    def to_lower(input):
        return [word.lower() for word in input]

    @staticmethod
    def remove_stopwords(input):
        return [word for word in input if word not in stopwords.words('english')]

    @staticmethod
    def remove_punctuation(input):
        return [_input.translate(None, string.punctuation) for _input in input]

    @staticmethod
    def remove_number(input):
        return [_input for _input in input if _input.replace('.', '', 1).isdigit()]

    @staticmethod
    def strip(input):
        return [_input.rstrip() for _input in input]
