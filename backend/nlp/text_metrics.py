import math
from backend.nlp.utils import Utils


class TextMetrics:
    # Outputs minute
    @staticmethod
    def read_time(self, text):
        words_per_min = 215
        return math.ceil(Utils.numof_words(None, text) / words_per_min)

    @staticmethod
    def us_level_flesh_kincaid_readability(self, text):
        red_lvl = self.flesh_kincaid_readability(None, text)
        if red_lvl > 90:
            return "5th Grade"
        elif red_lvl > 80:
            return "6th Grade"
        elif red_lvl > 70:
            return "7th Grade"
        elif red_lvl > 60:
            return "8th Grade"
        elif red_lvl > 50:
            return "10th Grade"
        elif red_lvl > 30:
            return "College"
        else:
            return "College Graduate"

    @staticmethod
    def flesh_kincaid_readability(self, text):
        numof_sent = Utils.numof_sentences(None, text)
        numof_words = Utils.numof_words(None, text)
        numof_syll = Utils.numof_syllables(None, text)
        return (206.835 - 1.015 * (numof_words/numof_sent)) - 84.6 * (numof_syll / numof_sent)
