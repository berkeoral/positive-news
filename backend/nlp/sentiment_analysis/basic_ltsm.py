import tensorflow as tf

from backend.nlp.sentiment_analysis.SIF.SentenceEmbedding import SentenceEmbedding
from backend.utils.txtops import TextOps


class BasicLSTMClassifier:
    def __init__(self, data_dir, we_path, wf_path):
        self.text_ops = TextOps()
        self.data = self.text_ops.acmimdb_as_list(data_dir)
        self.sentence_embedder = SentenceEmbedding(we_path, wf_path)
        for i in range(len(self.data)):
            sentences = self.data[i][2].split('.')
            sentence_embeddings = [self.sentence_embedder.calc_sentence_embedding(sentence, npc=1)
                                   for sentence in sentences]
            self.data[i].append(sentence_embeddings)
        """Hyperparamaters"""
        self.batch_size = 30
        self.max_iterations = 3000
        self.dropout = 0.8
        self.config = {'num_layers': 3,  # number of layers of stacked RNN's
                  'hidden_size': 120,  # memory cells in a layer
                  'max_grad_norm': 5,  # maximum gradient norm during training
                  'batch_size': self.batch_size,
                  'learning_rate': .005,
                  'sl': self.sentence_embedder.glove_embedding_dim,
                  'num_classes': 2} # either positive or negative
        



    def debug(self):
        lets_say_three = 3
        ltsm_cell = tf.contrib.rnn.BasicLSTMCell(lets_say_three)




