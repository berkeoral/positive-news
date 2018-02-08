"""
LSTM for time series classification
This model takes in time series and class labels.
The LSTM models the time series. A fully-connected layer
generates an output to be classified with Softmax
"""

import numpy as np
import tensorflow as tf  # TF 1.1.0rc1

from backend.nlp.sentiment_analysis.SIF.SentenceEmbedding import SentenceEmbedding
from backend.nlp.sentiment_analysis.SIF.doc2vec import doc2vec
from backend.nlp.sentiment_analysis.SIF.prepare_data import PrepareData
from backend.utils.txtops import TextOps

tf.logging.set_verbosity(tf.logging.ERROR)
import matplotlib.pyplot as plt

from backend.nlp.sentiment_analysis.lstm_classifier.model import Model, sample_batch, load_data

class LSTMClassifier:
    def __init__(self, data_dir, we_path, wf_path, tb_logdir):
        self.tb_logdir = tb_logdir
        self.text_ops = TextOps()
        self.raw_data = self.text_ops.acmimdb_as_list(data_dir)
        # DEBUG
        np.random.shuffle(self.raw_data)
        self.raw_data = self.raw_data[0:10000]
        # END DEBUG
        self.sentence_embedder = SentenceEmbedding(we_path, wf_path)
        sentences = [r_d[2].split('.') for r_d in self.raw_data]
        for i in range(len(self.raw_data)):
            sentences = self.raw_data[i][2].split('.')
            sentence_embeddings = self.sentence_embedder.calc_sentence_embedding(sentences, npc=1)
            # sentence_embeddings = np.stack(emb[0] for emb in sentence_embeddings if emb is not None) #  is this actually usefull
            self.raw_data[i].append(sentence_embeddings)  # Hopefully sentence embeddings is not none
        self.__prepare_data()

    def __prepare_data(self):
        np.random.shuffle(self.raw_data)
        self.doc2vec = doc2vec()
        self.data = [[], []]
        prepare_data = PrepareData()
        for i in range(len(self.raw_data)):
            dat, label = prepare_data.sentence_time(self.raw_data[i][3], self.raw_data[i][1])
            for i in range(len(dat)):
                self.data[0].append(dat[i])
                self.data[1].append(label[i])
        """
        for i in range(len(self.raw_data)):
            self.raw_data[i].append(self.doc2vec.weighted_coordinate_mean(self.raw_data[i][3]))
            if int(self.raw_data[i][1]) < 5:
                self.raw_data[i].append(0)
            else:
                self.raw_data[i].append(1)
        """

    def __separate_data(self, ratio):
        # Do not call shuffle

        x_data = np.stack(self.data[0][i] for i in range(len(self.data[0])))
        y_data = np.stack(self.data[1][i] for i in range(len(self.data[1])))
        n = len(x_data)
        ratio = [int(rat*n) for rat in ratio]

        x_train = x_data[0:ratio[0]]
        y_train = y_data[0:ratio[0]]

        x_val = x_data[ratio[0]+1:ratio[1]]
        y_val = y_data[ratio[0]+1:ratio[1]]

        x_test = x_data[ratio[1]+1:]
        y_test = y_data[ratio[1]+1:]

        return x_train, x_val, x_test, y_train, y_val, y_test

    def classify(self):
        # Set these directorie
        # direc = '/home/rob/Dropbox/ml_projects/LSTM/UCR_TS_Archive_2015'
        # summaries_dir = '/home/berke/Desktop/Workspace/positive-news/backend/nlp/sentiment_analysis/lstm_classifier'
        # lets try this..

        """Load the data"""
        ratio = [0.8, 0.9]  # Ratios where to split the training and validation set
        x_train, x_val, x_test, y_train, y_val, y_test = self.__separate_data(ratio)
        N, sl = x_train.shape
        num_classes = 2 # Positive or negative

        """Hyperparamaters"""
        batch_size = 30
        max_iterations = 3000
        dropout = 0.8
        config = {'num_layers': 3,  # number of layers of stacked RNN's
                  'hidden_size': 120,  # memory cells in a layer
                  'max_grad_norm': 5,  # maximum gradient norm during training
                  'batch_size': batch_size,
                  'learning_rate': .005,
                  'sl': sl,
                  'num_classes': num_classes}

        epochs = np.floor(batch_size * max_iterations / N)
        print('Train %.0f samples in approximately %d epochs' % (N, epochs))

        # Instantiate a model
        model = Model(config)

        """Session time"""
        sess = tf.Session()  # Depending on your use, do not forget to close the session
        writer = tf.summary.FileWriter(self.tb_logdir, sess.graph)  # writer for Tensorboard
        sess.run(model.init_op)

        cost_train_ma = -np.log(1 / float(num_classes) + 1e-9)  # Moving average training cost
        acc_train_ma = 0.0
        try:
            for i in range(max_iterations):
                X_batch, y_batch = sample_batch(x_train, y_train, batch_size)

                # Next line does the actual training
                cost_train, acc_train, _ = sess.run([model.cost, model.accuracy, model.train_op],
                                                    feed_dict={model.input: X_batch,
                                                               model.labels: y_batch,
                                                               model.keep_prob: dropout})
                cost_train_ma = cost_train_ma * 0.99 + cost_train * 0.01
                acc_train_ma = acc_train_ma * 0.99 + acc_train * 0.01
                if i % 100 == 1:
                    # Evaluate validation performance
                    X_batch, y_batch = sample_batch(x_val, y_val, batch_size)
                    cost_val, summ, acc_val = sess.run([model.cost, model.merged, model.accuracy],
                                                       feed_dict={model.input: X_batch, model.labels: y_batch,
                                                                  model.keep_prob: 1.0})
                    print('At %5.0f/%5.0f: COST %5.3f/%5.3f(%5.3f) -- Acc %5.3f/%5.3f(%5.3f)' % (
                    i, max_iterations, cost_train, cost_val, cost_train_ma, acc_train, acc_val, acc_train_ma))
                    # Write information to TensorBoard
                    writer.add_summary(summ, i)
                    writer.flush()
        except KeyboardInterrupt:
            pass

        epoch = float(i) * batch_size / N
        print('Trained %.1f epochs, accuracy is %5.3f and cost is %5.3f' % (epoch, acc_val, cost_val))

        # now run in your terminal:
        # $ tensorboard --logdir = <summaries_dir>
        # Replace <summaries_dir> with your own dir

