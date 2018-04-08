import sys
from collections import OrderedDict

import tensorflow as tf
import numpy as np
import tensorflow.contrib as contrib

from tensorflow.python.layers.core import Dense
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from backend.nlp.basics.embedding_ops import EmbeddingsV2
from backend.nlp.basics.preprocessing import Preprocessor
from backend.utils.txtops import TextOps

embeddings_path = "/home/berke/Desktop/workspace/bitirme/positive-news/backend/word_embeddings/glove.6B.50d.txt"
cnn_daily_base_path = "/home/berke/Desktop/workspace/bitirme/positive-news/backend/CNN_Daily Mail_Dataset"
tb_path = "/home/berke/Desktop/workspace/bitirme/positive-news/backend/nlp/summary/abstractive/tb"
save_path = tb_path + "/model.ckpt"

hparams = contrib.training.HParams(
    max_input_length=200,
    max_target_length=50,

    max_vocab_size=150000,
    test_size=0.1,

    batch_size=8,
    display_step=25,
    epochs=3,

    learning_rate=0.01,
    max_gradient_norm=5.0,  # from tf seq2seq article: "max gradient norm, is often set to a value like 5 or 1"

    num_layers=3,
    n_hidden=64,
    attention_size=64,
    beam_width=10,

    tgt_eos_id=0,
    tgt_sos_id=1,
    unk_word_id=2,
)

special_chars = OrderedDict()
special_chars["<END>"] = hparams.tgt_eos_id
special_chars["<GO>"] = hparams.tgt_sos_id
special_chars["<UNK>"] = hparams.unk_word_id

# Embeddings Layer
with tf.variable_scope("embedding"):
    with tf.device("/cpu:0"):
        embeddings = EmbeddingsV2(word_embeddings_path=embeddings_path,
                                  special_chars=[key for key, value in special_chars.items()],
                                  filter_most_frequent_words=hparams.max_vocab_size)

        word_embeddings = tf.Variable(tf.constant(0.0, shape=[embeddings.vocab_size, embeddings.embedding_dim]),
                                      trainable=False, name="word_embeddings")
        embedding_placeholder = tf.placeholder(tf.float32, [embeddings.vocab_size, embeddings.embedding_dim])
        embedding_init = word_embeddings.assign(embedding_placeholder)
    """
    word_embeddings = tf.get_variable("word_embeddings", shape=[embeddings.vocab_size, embeddings.embedding_dim],
                                      initializer=tf.constant_initializer(embeddings.word_embeddings),
                                      trainable=False, dtype=tf.float32)
    """

# Input Layer
with tf.variable_scope("input"):
    with tf.device("/cpu:0"):
        input_word_indices = tf.placeholder(shape=[hparams.batch_size, hparams.max_input_length], dtype=tf.int32)
        input_sequence_length = tf.placeholder(shape=[hparams.batch_size], dtype=tf.int32)
        input_word_embeddings = tf.nn.embedding_lookup(word_embeddings, input_word_indices)

        target_word_indices = tf.placeholder(shape=[hparams.batch_size, hparams.max_target_length], dtype=tf.int32)
        target_sequence_length = tf.placeholder(shape=hparams.batch_size, dtype=tf.int32)
        target_word_embeddings = tf.nn.embedding_lookup(word_embeddings, target_word_indices)


def create_cell():
    return tf.nn.rnn_cell.BasicLSTMCell(hparams.n_hidden)


""" Encoding """
# TODO benchmark with tf.contrib.cudnn_rnn cells
with tf.variable_scope("encode"):
    encoder_cell = contrib.rnn.MultiRNNCell([create_cell() for _ in range(hparams.num_layers)])

    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell,
                                                       inputs=input_word_embeddings,
                                                       sequence_length=input_sequence_length,
                                                       dtype=tf.float32,
                                                       time_major=False)
""" Decoding """
with tf.variable_scope("decode"):
    with tf.device("/cpu:0"):
        train_helper = contrib.seq2seq.TrainingHelper(target_word_embeddings, target_sequence_length)

        # Decoder
        base_decoder_cell = contrib.rnn.MultiRNNCell([create_cell() for _ in range(hparams.num_layers)])

        attention_mechanism = contrib.seq2seq.BahdanauAttention(num_units=hparams.n_hidden,
                                                                memory=encoder_outputs,
                                                                memory_sequence_length=input_sequence_length)

        decoder_cell = contrib.seq2seq.AttentionWrapper(cell=base_decoder_cell,
                                                        attention_mechanism=attention_mechanism,
                                                        attention_layer_size=hparams.attention_size)

        initial_state = decoder_cell.zero_state(hparams.batch_size, tf.float32).clone(cell_state=encoder_state)

        """
            decoder_cell_with_attention = contrib.rnn.OutputProjectionWrapper(cell=attention_cell,
                                                                              output_size=embeddings.vocab_size)
        """
        projection_layer = Dense(embeddings.vocab_size,
                                 use_bias=False,
                                 name="DEBUGprojection_layer")

        decoder = contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                                               helper=train_helper,
                                               initial_state=initial_state,
                                               output_layer=projection_layer)

        """output_cell.zero_state(
                                                   dtype=tf.float32,
                                                   batch_size=hparams.batch_size)"""

        final_outputs, final_state, final_sequence_lengths = contrib.seq2seq.dynamic_decode(decoder=decoder,
                                                                                            output_time_major=True,
                                                                                            impute_finished=True,
                                                                                            maximum_iterations=hparams.max_target_length)

with tf.variable_scope("metrics"):
    logits = final_outputs.rnn_output

    # Labels are indices, not one hot vectors
    # target_labels = tf.placeholder(shape=[hparams.batch_size, hparams.max_target_length], dtype=tf.int32)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.transpose(target_word_indices), logits=logits,
                                                          name="loss")
    tf.summary.tensor_summary(loss.op.name, tf.squeeze(loss))

    global_step = tf.Variable(0, name="global_step", trainable=False)

    params = tf.trainable_variables()
    gradients = tf.gradients(loss, params)
    clipped_gradients, _ = tf.clip_by_global_norm(
        gradients, hparams.max_gradient_norm)

    optimizer = tf.train.AdamOptimizer(learning_rate=hparams.learning_rate)
    train_op = optimizer.apply_gradients(zip(clipped_gradients, params), global_step)

"""Inference"""
with tf.variable_scope("decode", reuse=True):
    with tf.device("/cpu:0"):
        inference_helper = contrib.seq2seq.GreedyEmbeddingHelper(word_embeddings,
                                                                 tf.fill([hparams.batch_size], hparams.tgt_sos_id),
                                                                 hparams.tgt_eos_id)

        inference_decoder = contrib.seq2seq.BasicDecoder(decoder_cell,
                                                         inference_helper,
                                                         initial_state=initial_state,
                                                         output_layer=projection_layer)

        inference_outputs, inference_states, inference_lengths = contrib.seq2seq.dynamic_decode(
            decoder=inference_decoder, maximum_iterations=tf.reduce_max(target_sequence_length), impute_finished=True)
        # pred_summaries = inference_outputs.sample_id

        pred_summaries = inference_outputs.sample_id


"""
with tf.variable_scope("decode", reuse=True):
    # Replicate encoder infos beam_width times
    decoder_initial_state = tf.contrib.seq2seq.tile_batch(
        initial_state, multiplier=hparams.beam_width)

    # Define a beam-search decoder
    inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
        cell=,
        embedding=word_embeddings,
        start_tokens=tf.fill([hparams.batch_size], hparams.tgt_sos_id),
        end_token=hparams.tgt_eos_id,
        initial_state=decoder_initial_state,
        beam_width=hparams.beam_width,
        output_layer=projection_layer,
        length_penalty_weight=0.0)
    inference_outputs, inference_states, inference_lengths = contrib.seq2seq.dynamic_decode(
        decoder=inference_decoder, maximum_iterations=tf.reduce_max(target_sequence_length), impute_finished=True)
    pred_summaries = inference_outputs.sample_id
"""
"""Summary and Checkpoints"""
# saver = tf.train.Saver()  # Not save to avoid over training
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(tb_path, train_op.graph)

saver = tf.train.Saver()


def prepare_data(path):
    txtops = TextOps()
    preprocessor = Preprocessor()
    data = txtops.cnn_dailymail_as_list(path)

    data = data[:]

    article_ids = []
    highlight_ids = []
    article_lengths = []
    highlight_lengths = []

    for i in tqdm(range(len(data)), file=sys.stdout):
        data[i][0] = preprocessor.default_preprocess(data[i][0],
                                                     raw=True,
                                                     lemmatizer=False,
                                                     remove_digit=False)
        data[i][1] = preprocessor.default_preprocess(preprocessor.merge_sentences(data[i][1]),
                                                     raw=True,
                                                     lemmatizer=False,
                                                     remove_digit=False)

        article_ids.append([hparams.tgt_sos_id] + [embeddings.word_to_ind_dict.get(word, hparams.unk_word_id)
                                                   for word in data[i][0]])

        highlight_ids.append([hparams.tgt_sos_id] + [embeddings.word_to_ind_dict.get(word, hparams.unk_word_id)
                                                     for word in data[i][1]])

        # sequence lengths

        article_lengths.append(len(article_ids[-1]))
        highlight_lengths.append(len(highlight_ids[-1]))

    return article_ids, highlight_ids, article_lengths, highlight_lengths


article_ids, highlight_ids, article_lengths, highlight_lengths = prepare_data(cnn_daily_base_path)

article_ids = pad_sequences(article_ids, maxlen=hparams.max_input_length, dtype=np.int32, padding="post",
                            truncating="post",
                            value=hparams.tgt_eos_id)
article_ids[:, -1] = hparams.tgt_eos_id

highlight_ids = pad_sequences(highlight_ids, maxlen=hparams.max_target_length, dtype=np.int32, padding="post",
                              truncating="post",
                              value=hparams.tgt_eos_id)
highlight_ids[:, -1] = hparams.tgt_eos_id

article_lengths = np.stack([article_length if article_length < hparams.max_input_length else hparams.max_input_length
                            for article_length in article_lengths])

highlight_lengths = np.stack(
    [highlight_length if highlight_length < hparams.max_target_length else hparams.max_target_length
     for highlight_length in highlight_lengths])

train_article, test_article, train_highlight, test_highlight, train_article_sql, \
test_article_sql, train_highlight_sql, test_highlight_sql = train_test_split(article_ids,
                                                                             highlight_ids,
                                                                             article_lengths,
                                                                             highlight_lengths,
                                                                             test_size=hparams.test_size,
                                                                             shuffle=False)


def batch_generator(article, highlight, batch_size):
    i = 0
    size = article[0].shape[0]
    assert isinstance(article, tuple) and isinstance(highlight, tuple)
    _article_ids, _article_lengths = article
    _highlight_ids, _highlight_lengths = highlight

    while True:
        if i + batch_size < size:
            _x = _article_ids[i:i + batch_size]
            _x_sql = _article_lengths[i:i + batch_size]

            _y = _highlight_ids[i:i + batch_size]
            _y_sql = _highlight_lengths[i:i + batch_size]

            yield _x, _x_sql, _y, _y_sql
            i += batch_size
        else:
            i = 0


train_generator = batch_generator((train_article, train_article_sql), (train_highlight, train_highlight_sql),
                                  hparams.batch_size)
test_generator = batch_generator((test_article, test_article_sql), (test_highlight, test_highlight_sql),
                                 hparams.batch_size)

# options
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(embedding_init, feed_dict={embedding_placeholder: embeddings.word_embeddings})
    # sess.run(embedding_init, feed_dict={word_embeddings: embeddings.word_embeddings})

    try:
        saver.restore(sess, save_path)
    except Exception:
        print("Failed to restore checkpoint %s" % save_path)

    total_batch_step = int(hparams.epochs * train_highlight.shape[0] / hparams.batch_size)
    _loss_total = 0.

    for i in tqdm(range(total_batch_step), file=sys.stdout, unit="batch"):
        x_batch, x_sql_batch, y_batch, y_sql_batch = next(train_generator)
        _loss, _train_op, _summ = sess.run([loss, train_op, merged],
                                           feed_dict={input_word_indices: x_batch,
                                                      input_sequence_length: x_sql_batch,
                                                      target_word_indices: y_batch,
                                                      target_sequence_length: y_sql_batch},
                                           options=run_options)

        _loss_total += np.average(_loss)

        if i % hparams.display_step == 0:
            tqdm.write(str(_loss_total / hparams.display_step))
            _loss_total = 0

            writer.add_summary(_summ, i * hparams.batch_size)
            saver.save(sess, save_path=save_path)

            x_batch, x_sql_batch, y_batch, y_sql_batch = next(test_generator)
            _inds = sess.run([pred_summaries],
                             feed_dict={input_word_indices: x_batch,
                                        input_sequence_length: x_sql_batch,
                                        target_word_indices: y_batch,
                                        target_sequence_length: y_sql_batch})

            to_inspect = _inds[0][0]
            summary_words = [embeddings.ind_to_word_dict[i] for i in to_inspect]
            generated_summary = " ".join(summary_words)
            tqdm.write(generated_summary)
