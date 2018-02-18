import tensorflow as tf
import numpy as np
import time

from backend.nlp.sentiment_analysis.base_model import BaseModel
from backend.nlp.basics.embedding_ops import Embeddings


class DynamicRNN(BaseModel):
    def __init__(self, data_dir, embeddings, tb_logdir, debug=-1):
        super().__init__(data_dir, embeddings, tb_logdir, debug=debug)

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

    def classify(self, config=None):
        ratio = [0.9, 0.9]  # Ratios where to split the training and validation set
        probe = time.time()
        x_train, x_val, x_test, y_train, y_val, y_test = super().separate_data(ratio, sequential=True)
        y_train, y_val, y_test = self.__one_hot_label(y_train=y_train, y_val=y_val, y_test=y_test)
        emb_dim = self.embeddings.glove_embedding_dim
        print("separate_data: ", time.time() - probe)

        batch_size = 128
        training_steps = 5000
        max_seq_len = 250
        n_classes = 2
        n_hidden = 128
        learning_rate = 0.01
        display_step = 50

        probe = time.time()
        x = tf.placeholder("float", [None, max_seq_len, emb_dim], name="inputs")
        y = tf.placeholder("float", [None, n_classes], name="labels")
        # A placeholder for indicating each sequence length
        tensor_sequence_length = tf.placeholder(tf.int32, [None], name="sequence_length")

        # Random initial state weights
        with tf.name_scope("softmax"):
            with tf.variable_scope("softmax_params"):
                weights = {
                    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]), name="softmax_weight")
                }
                biases = {
                    'out': tf.Variable(tf.random_normal([n_classes]), name="softmax_bias")
                }
        print("Internal: ", time.time() - probe)
        def dynamicRNN(x, seq_len, weights, biases):
            probe = time.time()
            x = tf.unstack(x, seq_len, 1)
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)

            outputs, states = tf.nn.static_rnn(lstm_cell,
                                               x,
                                               dtype=tf.float32,
                                               sequence_length=tensor_sequence_length)
            outputs = tf.stack(outputs)
            outputs = tf.transpose(outputs, [1, 0, 2]) # dont get this

            # def not get this
            batch_size = tf.shape(outputs)[0]
            index = tf.range(0, batch_size) * seq_len + (tensor_sequence_length - 1)
            outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)
            print("dynamicRNN: ", time.time() - probe)
            return tf.matmul(outputs, weights['out']) + biases['out']

        pred = dynamicRNN(x, max_seq_len, weights, biases)
        # Define loss and optimizer
        probe = time.time()
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        # Initialize the variables (i.e. assign their default value)
        # Tensorboard
        with tf.name_scope('summaries'):
            tf.summary.scalar('acc', accuracy)
            tf.summary.scalar('loss', cost)
        merged = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        print("Tensors: ", time.time() - probe)
        # Start training
        with tf.Session() as sess:
            writer = tf.summary.FileWriter(self.tb_logdir, sess.graph)
            sess.run(init)
            for step in range(training_steps):
                batch_x, batch_y, batch_seqlen = self.input_fn(x_train, y_train, batch_size, max_seq_len)

                # Run optimization op (backprop)
                sess.run(optimizer, feed_dict={x: batch_x,
                                               y: batch_y,
                                               tensor_sequence_length: batch_seqlen})
                if step % display_step == 0 or step == 0:
                    with tf.name_scope("training") as scope:
                        acc, loss, summ = sess.run([accuracy, cost, merged], feed_dict={x: batch_x,
                                                                                        y: batch_y,
                                                                                        tensor_sequence_length: batch_seqlen})
                        print(str(step) + ":Step " + str(step * batch_size) + ", Minibatch Loss= " + \
                              "{:.6f}".format(loss) + ", Training Accuracy= " + \
                              "{:.5f}".format(acc))
                        writer.add_summary(summ, step*batch_size)

            print("Optimization Finished!")
            # Calculate accuracy
            x_test, y_test, batch_seqlen = self.input_fn(x_test, y_test, len(x_test), max_seq_len)
            print("Testing Accuracy:",
                  sess.run(accuracy, feed_dict={x: x_test,
                                                y: y_test,
                                                tensor_sequence_length: batch_seqlen}))

