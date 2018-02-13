import string

import numpy as np
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer


class Preprocessor:
    def __init__(self):
        self.stopwords = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.porter = PorterStemmer()

    def default_preprocess(self, input, stopwords=True, lemmatizing=True, stemming=False):
        input = [((word.lower()).translate(None, string.punctuation)).rstrip() for word in input]
        # input = [_input.rstrip() for _input in input]
        input = [_input for _input in input if _input.replace('.', '', 1).isdigit()]
        if stopwords:
            input = [word for word in input if word not in self.stopwords]
        if lemmatizing:
            input = [self.lemmatizer.lemmatize(word) for word in input]
        if stemming:
            input = [self.porter.stem(word) for word in input]
        return input

    @staticmethod
    def sentences_of_words(input):
        return None

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