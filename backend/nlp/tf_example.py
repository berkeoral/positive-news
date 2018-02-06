import tensorflow as tf
import tensorboard as tb
import numpy as np

def basics():
    N = 5
    batch_size = 1
    dx = 3

    b = tf.Variable(tf.zeros((N,)))
    W = tf.Variable(tf.random_uniform((dx, N), -1, 1))
    x = tf.placeholder(tf.float32, (N, dx))
    h = tf.nn.softmax(tf.matmul(x, W) + b, axis=0)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    result = session.run(h, feed_dict={x: np.random.random((N, dx))})

    print(result)

def novice():
    print("hi there idiot")



basics()