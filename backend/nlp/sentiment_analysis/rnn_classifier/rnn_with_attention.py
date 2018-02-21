import tensorflow as tf
import numpy as np
import sys
from tensorflow.python.ops.rnn_cell_impl import GRUCell
from tqdm import tqdm

from backend.nlp.sentiment_analysis.base_model import BaseModel
from backend.nlp.sentiment_analysis.rnn_classifier.attention import attention


class RNNWithAttention(BaseModel):
    def __init__(self, data_dir, embeddings, tb_logdir, debug=-1):
        super().__init__(data_dir, embeddings, tb_logdir, debug)

    """
    def input_fn(self, x, y, seq_len, batch_size):
        ind = np.random.choice(range(len(x)), batch_size)
        y_ret = y[ind]
        x_ret = x[ind]
        batch_seq_len = seq_len[ind]
        return x_ret, y_ret, batch_seq_len

    # input x: sequence words of every document
    # output x: 0 padded word embeddings for every document
    # output seq_len: 0 padded word embeddings for every document
    def padded_word_emb(self, x, max_seq_len):
        if len(x) < 1:
            return None, None
        for i in range(len(x)):
            x[i] = [self.embeddings.embedding_dictionary[_word] for _word in x[i]]
        seq_len = np.stack(len(_x) if len(_x) < max_seq_len else max_seq_len for _x in x)
        x = np.stack([self.prepare_data.padded_sequence(x_, 0, seq_len=max_seq_len)[0] for x_ in x])
        return x, seq_len
    """


    def input_fn(self, x, y, batch_size, max_seq_len):
        ind = np.random.choice(range(len(x)), batch_size)
        y_ret = y[ind]
        x_ret = [x[i] for i in ind]
        for i in range(len(x_ret)):
            x_ret[i] = [self.embeddings.embedding_dictionary[_word] for _word in x_ret[i]]
        batch_seq_len = [len(_x) if len(_x) < max_seq_len else max_seq_len for _x in x_ret]
        x_ret = np.stack([self.prepare_data.padded_sequence(x_, 0, seq_len=max_seq_len)[0] for x_ in x_ret])
        return x_ret, y_ret, batch_seq_len

    def __one_hot_label(self, y_train, y_val, y_test):
        y_train = np.stack(np.array([1, 0]) if y_ == 1 else np.array([0, 1]) for y_ in y_train)
        if len(y_val) != 0:
            y_val = np.stack(np.array([1, 0]) if y_ == 1 else np.array([0, 1]) for y_ in y_val)
        if len(y_test) != 0:
            y_test = np.stack(np.array([1, 0]) if y_ == 1 else np.array([0, 1]) for y_ in y_test)
        return y_train, y_val, y_test

    # Modified version of: https://github.com/ilivans/tf-rnn-attention/blob/master/train.py
    def classify(self, config=None):
        # config
        ratio = [0.9, 0.9]  # Ratios where to split the training and validation set
        batch_size = 128
        training_steps = 750  # 750*128=max_step=96000
        keep_prob = 0.8
        max_seq_len = 300
        n_classes = 2
        n_hidden = 128
        attention_size = 50
        learning_rate = 0.01
        display_step = 50

        # prepare data
        x_train, x_val, x_test, y_train, y_val, y_test = super().separate_data(ratio, sequential=True)
        emb_dim = self.embeddings.glove_embedding_dim
        y_train, y_val, y_test = self.__one_hot_label(y_train=y_train, y_val=y_val, y_test=y_test)

        """ Tensors """
        # Input
        with tf.name_scope('Inputs'):
            input = tf.placeholder(tf.float32, [None, max_seq_len, emb_dim], name="input")
            label = tf.placeholder(tf.float32, [None, n_classes], name="label")
            sequence_length = tf.placeholder(tf.int32, [None], name="sequence_length")
            keep_probability = tf.placeholder(tf.float32, name='keep_probability')

        # RNN layer
        with tf.name_scope('Sentence'):
            rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(GRUCell(n_hidden), GRUCell(n_hidden),
                                    inputs=input, sequence_length=sequence_length, dtype=tf.float32)
            tf.summary.histogram('RNN_outputs', rnn_outputs)

        # Attention layer
        with tf.name_scope('Attention'):
            attention_output, alphas = attention(rnn_outputs, attention_size, return_alphas=True)
            tf.summary.histogram('alphas', alphas)

        drop = tf.nn.dropout(attention_output, keep_probability)

        # Fully connected layer
        with tf.name_scope('Fully_connected_layer'):
            W = tf.Variable(
                tf.truncated_normal([n_hidden * 2, n_classes], stddev=0.1))  # truncated_normal cuts tails of normal dist
            b = tf.Variable(tf.constant(0., shape=[n_classes]))
            label_predicted = tf.nn.xw_plus_b(drop, W, b)
            label_predicted = tf.squeeze(label_predicted)
            tf.summary.histogram('W', W)

        with tf.name_scope('Metrics'):
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=label_predicted, labels=label))
            tf.summary.scalar('loss', loss)
            optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(tf.sigmoid(label_predicted)), label), tf.float32))
            tf.summary.scalar('accuracy', accuracy)

        # saver = tf.train.Saver()  # Not save to avoid over training
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(self.tb_logdir, accuracy.graph)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print("Start training")
            _loss = _acc = 0
            for step in tqdm(range(training_steps), file=sys.stdout,  unit="steps"):
                x_batch, y_batch, sql_batch = self.input_fn(x_train, y_train, batch_size, max_seq_len)
                acc_tr, loss_tr, summary_tr, opt_tr = sess.run([accuracy, loss, merged, optimiser],
                                                               feed_dict={input: x_batch,
                                                                          label: y_batch,
                                                                          sequence_length: sql_batch,
                                                                          keep_probability: keep_prob})
                _loss += loss_tr
                _acc += acc_tr
                if step % display_step == 0:
                    writer.add_summary(summary_tr, step * batch_size)
                    tqdm.write("Step: {0},\tLoss: {1},\tAccuracy: {2}".format(str(step*batch_size),
                                                                              str(_loss/display_step),
                                                                              str(_acc/display_step)))
                    _loss = _acc = 0

            # Testing
            x_batch, y_batch, sql_batch = self.input_fn(x_test, y_test, batch_size, max_seq_len)
            print("Testing Accuracy:",
                  sess.run(accuracy, feed_dict={input: x_batch,
                                                label: y_batch,
                                                sequence_length: sql_batch,
                                                keep_probability: keep_prob}))











