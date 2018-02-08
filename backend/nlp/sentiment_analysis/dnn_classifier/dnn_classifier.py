"""
TODO FIX CODE DUPLICATIONS
Base classifier to evaluate models
"""
import feedfinder2
import tensorflow as tf
import numpy as np
from tensorflow.python.estimator.inputs import numpy_io

from backend.nlp.sentiment_analysis.SIF.SentenceEmbedding import SentenceEmbedding
from backend.nlp.sentiment_analysis.SIF.doc2vec import doc2vec
from backend.nlp.sentiment_analysis.SIF.prepare_data import PrepareData
from backend.utils.txtops import TextOps


class DNNClassifier:
    def __init__(self, data_dir, we_path, wf_path, tb_logdir):
        self.tb_logdir = tb_logdir
        self.text_ops = TextOps()
        self.prepare_data = PrepareData()
        self.raw_data = self.text_ops.acmimdb_as_list(data_dir)
        # DEBUG
        np.random.shuffle(self.raw_data)
        #self.raw_data = self.raw_data[0:1000]
        # END DEBUG
        self.sentence_embedder = SentenceEmbedding(we_path, wf_path)
        sentences = [r_d[2].split('.') for r_d in self.raw_data]
        _skip = False # TODO FIX THIS.
        for i in range(len(self.raw_data)):
            if _skip:
                break
            sentences = self.raw_data[i][2].split('.')
            sentence_embeddings = self.sentence_embedder.calc_sentence_embedding(sentences, npc=0)
            self.raw_data[i].append(sentence_embeddings)
        self.__prepare_data()

    def __prepare_data(self):
        print("Preparing data")
        np.random.shuffle(self.raw_data)
        self.doc2vec = doc2vec()
        self.data = [[], []]
        for i in range(len(self.raw_data)):
            dat, label = self.prepare_data.mean_document_embedding(self.raw_data[i][3], self.raw_data[i][1])
            self.data[0].append(dat)
            self.data[1].append(label)
        """
        for i in range(len(self.raw_data)):
            documents = [self.raw_data[i][2] for i in range(len(self.raw_data))]
            labels = [self.raw_data[i][1] for i in range(len(self.raw_data))]
            dat, label = self.prepare_data.sif_document_embedding(documents, labels
                                                                  , self.sentence_embedder)
            self.data[0] = [dat[i] for i in range(dat.shape[0])]
            self.data[1] = [label[i] for i in range(label.shape[0])]
        """
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
        ratio = [int(rat * n) for rat in ratio]

        x_train = x_data[0:ratio[0]]
        y_train = y_data[0:ratio[0]]

        x_val = x_data[ratio[0] + 1:ratio[1]]
        y_val = y_data[ratio[0] + 1:ratio[1]]

        x_test = x_data[ratio[1] + 1:]
        y_test = y_data[ratio[1] + 1:]

        return x_train, x_val, x_test, y_train, y_val, y_test

    def input_fn(x_data, y_data, batch_size):
        """ Function to sample a batch for training"""
        N, data_len = x_data.shape
        ind_N = np.random.choice(N, batch_size, replace=True)  # Already shuffled
        X_batch = x_data[ind_N]
        y_batch = y_data[ind_N]
        return X_batch, y_batch

    def classify(self, reset=True):
        """Load the data"""
        if reset:
            tf.reset_default_graph()
        ratio = [0.8, 0.9]  # Ratios where to split the training and validation set
        x_train, x_val, x_test, y_train, y_val, y_test = self.__separate_data(ratio)
        N, sl = x_train.shape

        batch_size = 100
        train_steps = 10000
        repeat = 8

        feature_columns = [tf.feature_column.numeric_column(key="x", dtype=tf.float64, shape=sl)]

        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": x_train},
            y=y_train,
            batch_size=batch_size,
            num_epochs=None,
            shuffle=True)
        test_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": x_test},
            y=y_test,
            batch_size=batch_size,
            num_epochs=None,
            shuffle=True)

        # Build 2 hidden layer DNN with 10, 10 units respectively.
        classifier = tf.estimator.DNNClassifier(
            model_dir=self.tb_logdir,
            feature_columns=feature_columns,
            # Two hidden layers of 10 nodes each.
            hidden_units=[50, 40, 60, 40, 20],
            # The model must choose between 2 classes.
            n_classes=2
        )

        # Train the Model.
        classifier.train(
            input_fn=train_input_fn,
            steps=train_steps)

        # Evaluate the model.
        eval_result = classifier.evaluate(
            input_fn=test_input_fn,
            steps=train_steps)
        print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
