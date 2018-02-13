"""
Core embeddings object
"""

class Embeddings:
    def __init__(self, word_embeddings_path, word_frequencies_path):
        self.word_embeddings_path = word_embeddings_path
        self.embedding_dictionary = {}
        self.__get_embeddings()
        self.word_weights = {}  # Glove vocabulary contains word_frequencies vocabulary
        self.word_frequencies_path = word_frequencies_path
        self.__get_weights()
        self.glove_vocab_size = len(self.embedding_dictionary)
        self.glove_embedding_dim = len(self.embedding_dictionary["this"])

    def __get_embeddings(self):
        file = open(self.word_embeddings_path, 'r', encoding='UTF-8')
        for line in file.readlines():
            row = line.strip().split(' ') # Space is default separator unnecessary
            embed_vector = [float(i) for i in row[1:]]
            self.embedding_dictionary[row[0]] = embed_vector
        file.close()
        print("Glove loaded")

    def __get_weights(self, weight_param=1e-3):
        file = open(self.word_frequencies_path, 'r', encoding='UTF-8')
        count = 0
        for line in file.readlines():
            row = line.split(' ')
            self.word_weights[row[0]] = float(row[1])
            count += float(row[1])
        for word, value in self.word_weights.items():
            self.word_weights[word] = weight_param / (weight_param + value / count)
        file.close()
        print("Word weights loaded")