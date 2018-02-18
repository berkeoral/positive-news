"""
LSTM for time series classification
This model takes in time series and class labels.
The LSTM models the time series. A fully-connected layer
generates an output to be classified with Softmax
"""

import numpy as np
import tensorflow as tf

from backend.nlp.sentiment_analysis.base_model import BaseModel

tf.logging.set_verbosity(tf.logging.ERROR)

from backend.nlp.sentiment_analysis.rnn_classifier.model import Model, sample_batch, load_data


class LSTMClassifier(BaseModel):
    def __init__(self, data_dir, embeddings, tb_logdir):
        super().__init__(data_dir, embeddings, tb_logdir)

    def classify(self):
        """Load the data"""
        ratio = [0.8, 0.9]  # Ratios where to split the training and validation set
        x_train, x_val, x_test, y_train, y_val, y_test = super().separate_data(ratio)
        N, sl = x_train.shape
        num_classes = 2  # Positive or negative

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
