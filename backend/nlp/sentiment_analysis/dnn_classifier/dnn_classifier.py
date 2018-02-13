import tensorflow as tf

from backend.nlp.sentiment_analysis.base_model import BaseModel
from backend.nlp.basics.embedding_ops import Embeddings


class DNNClassifier(BaseModel):
    def __init__(self, data_dir, embeddings, tb_logdir, debug):
        super().__init__(data_dir, embeddings, tb_logdir, debug)

    def classify(self):
        """Load the data"""
        ratio = [0.8, 0.9]  # Ratios where to split the training and validation set
        x_train, x_val, x_test, y_train, y_val, y_test = super().separate_data(ratio)
        n, sl = x_train.shape

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

        classifier = tf.estimator.DNNClassifier(
            model_dir=self.tb_logdir,
            feature_columns=feature_columns,
            hidden_units=[20, 40, 60, 40, 20],
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
