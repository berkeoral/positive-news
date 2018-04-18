import abc
import os
import sys

import tensorflow as tf
import numpy as np
import tensorflow.contrib as contrib

from sklearn.datasets import fetch_20newsgroups
from sklearn.utils import shuffle
from tqdm import tqdm

from backend.nlp.basics.embedding_ops import EmbeddingsV2
from backend.utils.txtops import TextOps


class BaseModel(object):
    def __init__(self, wembs_path, name_space, tb_path, filter_most_frequent_words=-1):
        self.name_space = name_space
        self.data = None
        self.train_generator = None
        self.label_dict = {}
        self.tb_path = tb_path

        self.hparams = contrib.training.HParams(
            batch_size=16,
            epochs=3,  # models quickly overfits

            keep_prob=0.8,
            max_seq_len=300,

            n_classes=-1,
            n_hidden=128,
            attention_size=50,
            learning_rate=0.01,
            display_step=25,
        )
        with tf.device("cpu:0"):
            with tf.variable_scope("WordEmbeddings") as scope:
                self.embeddings = EmbeddingsV2(wembs_path, filter_most_frequent_words=filter_most_frequent_words)
                try:
                    self.word_embeddings = tf.get_variable("word_embeddings",
                                                           initializer=self.embeddings.word_embeddings,
                                                           trainable=False)
                    print("Initialized word embeddings")
                except ValueError:
                    scope.reuse_variables()
                    self.word_embeddings = tf.get_variable("word_embeddings",
                                                           initializer=self.embeddings.word_embeddings,
                                                           trainable=False)
                    print("Reusing initialized word embeddings")
        del self.embeddings.word_embeddings

    def graph(self):
        with tf.variable_scope(self.name_space):
            with tf.variable_scope("Inputs"):
                self.input_word_ids = tf.placeholder(tf.int32, [None, self.hparams.max_seq_len], name="input_ids")
                input_word_embeddings = tf.nn.embedding_lookup(self.word_embeddings, self.input_word_ids)
                self.label = tf.placeholder(tf.float32, [None, self.hparams.n_classes], name="labels")
                self.sequence_length = tf.placeholder(tf.int32, [None], name="sequence_length")
                self.keep_probability = tf.placeholder(tf.float32, name='keep_probability')

            with tf.variable_scope("RNN"):
                rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(contrib.rnn.GRUCell(self.hparams.n_hidden),
                                                                 contrib.rnn.GRUCell(self.hparams.n_hidden),
                                                                 inputs=input_word_embeddings,
                                                                 sequence_length=self.sequence_length,
                                                                 dtype=tf.float32)
                tf.summary.histogram('RNN_outputs', rnn_outputs)

            with tf.variable_scope("Attention"):
                attention_output, alphas = self.attention(rnn_outputs, self.hparams.attention_size, return_alphas=True)
                drop = tf.nn.dropout(attention_output, self.keep_probability)
                tf.summary.histogram('alphas', alphas)

            with tf.variable_scope("FullyConnected"):
                W = tf.Variable(tf.truncated_normal([self.hparams.n_hidden * 2, self.hparams.n_classes],
                                                    stddev=0.1))
                b = tf.Variable(tf.constant(0., shape=[self.hparams.n_classes]))
                self.label_predicted = tf.squeeze(tf.nn.xw_plus_b(drop, W, b))
                tf.summary.histogram('W', W)

            with tf.variable_scope("Metrics"):
                self.loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=self.label_predicted, labels=self.label))
                self.prediction_index = tf.argmax(tf.sigmoid(self.label_predicted), output_type=tf.int32)
                tf.summary.scalar('loss', self.loss)
                self.optimiser = tf.train.AdamOptimizer(learning_rate=self.hparams.learning_rate).minimize(self.loss)
                self.accuracy = tf.reduce_mean(
                    tf.cast(tf.equal(tf.round(tf.sigmoid(self.label_predicted)), self.label), tf.float32))
                tf.summary.scalar('accuracy', self.accuracy)

            with tf.variable_scope("tensorboard"):
                self.merged = tf.summary.merge_all(scope=self.name_space)
                self.writer = tf.summary.FileWriter(self.tb_path, self.accuracy.graph)
                self.saver = tf.train.Saver()

            print("Graph initialized")

    def train(self):
        assert self.data is not None, "Data not initialized"
        assert self.saver is not None, "Graph is not initialized"
        assert self.train_generator is not None, "Input generator is not initialized"

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            checkpoint_file = os.path.join(self.tb_path, "model.ckpt")
            try:
                self.saver.restore(sess, save_path=checkpoint_file)
            except Exception:
                print("Failed to restore checkpoint %s" % checkpoint_file)

            total_batch_step = int(self.hparams.epochs * len(self.data[0]) / self.hparams.batch_size)
            _loss = _acc = 0.

            for batch in tqdm(range(total_batch_step), file=sys.stdout, unit="batch"):
                x_batch, y_batch, sql_batch = next(self.train_generator)
                acc_tr, loss_tr, summary_tr, opt_tr = sess.run([self.accuracy, self.loss, self.merged, self.optimiser],
                                                               feed_dict={self.input_word_ids: x_batch,
                                                                          self.label: y_batch,
                                                                          self.sequence_length: sql_batch,
                                                                          self.keep_probability: self.hparams.keep_prob}
                                                               )
                _loss += loss_tr
                _acc += acc_tr
                if batch % self.hparams.display_step == 0:
                    self.writer.add_summary(summary_tr, batch * self.hparams.batch_size)
                    self.saver.save(sess, checkpoint_file)
                    tqdm.write("Step: {0},\tLoss: {1},\tAccuracy: {2}".format(str(batch * self.hparams.batch_size),
                                                                              str(_loss / self.hparams.display_step),
                                                                              str(_acc / self.hparams.display_step)))
                    _loss = _acc = 0.
            sess.close()

    def evaluator(self):
        assert self.saver is not None, "Graph is not initialized"
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        with tf.Session(graph=self.prediction_index.graph,
                                   config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(tf.global_variables_initializer())
            checkpoint_file = os.path.join(self.tb_path, "model.ckpt")
            try:
                self.saver.restore(sess, save_path=checkpoint_file)
            except Exception:
                print("Failed to restore checkpoint %s" % checkpoint_file)
                return
            while True:
                article_ids, seq_len = yield
                _pred_label = sess.run([self.prediction_index], feed_dict={self.input_word_ids: article_ids,
                                                                           self.sequence_length: seq_len,
                                                                           self.keep_probability: self.hparams.keep_prob})
                label = self.label_dict[_pred_label[0]]
                yield label

    # Taken from: https://github.com/ilivans/tf-rnn-attention/blob/master/attention.py
    def attention(self, inputs, attention_size, time_major=False, return_alphas=False):
        if isinstance(inputs, tuple):
            inputs = tf.concat(inputs, 2)
        if time_major:
            inputs = tf.array_ops.transpose(inputs, [1, 0, 2])
        hidden_size = inputs.shape[2].value

        w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

        with tf.name_scope('v'):
            v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

        vu = tf.tensordot(v, u_omega, axes=1, name='vu')
        alphas = tf.nn.softmax(vu, name='alphas')

        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

        if not return_alphas:
            return output
        else:
            return output, alphas

    # sets self.data and hparams.n_classes
    @abc.abstractmethod
    def data_func(self):
        pass

    def batch_generator(self, batch_size):
        self.data[0], self.data[1], self.data[2] = shuffle(self.data[0], self.data[1], self.data[2])
        i = 0
        while True:
            if i < len(self.data[0]):
                _x = self.data[0][i:i + batch_size]
                _y = self.data[1][i:i + batch_size]
                _x_sql = self.data[2][i:i + batch_size]
                yield _x, _y, _x_sql
                i += batch_size
            else:
                i = 0
                self.data[0], self.data[1], self.data[2] = shuffle(self.data[0], self.data[1], self.data[2])

