import tensorflow as tf

import pandas as pd
import numpy as np
import logging
from models.unet import unet
import exp.util as util
import PIL.Image as Image

def input_fn(params, is_training):
    batch_size = params['batch_size']
    if is_training:
        data = pd.read_csv(params['data_set'] + '/train.csv')
    else:
        data = pd.read_csv(params['data_set'] + '/test.csv')
    labels = data['description'][:]
    files = data['image_name'][:]
    word_index = params['word_index']
    image_dir = params['data_set'] + '/images/'
    end_token = word_index['<end>']
    def _input_fn():
        def _generator():
            for i, f in enumerate(files):
                text = labels[i]
                tokens = util.tokenize(word_index, text)
                x = Image.open(image_dir+f)
                x = x.resize((320,320))
                x = np.asarray(x,np.float32)/127.5-1
                x = np.reshape(x,(320,320,1))
                tokens.append(end_token)
                # logging.info('Tokens: {}'.format(len(tokens)))
                yield (x, np.array(tokens, dtype=np.int32))

        ds = tf.data.Dataset.from_generator(_generator, (tf.float32, tf.int32),
                                            (tf.TensorShape([320,320,1]), tf.TensorShape([None])))

        def _features_labels(x, y):
            return x, y

        ds = ds.map(_features_labels)
        if is_training:
            ds = ds.apply(tf.contrib.data.shuffle_and_repeat(100))
        ds = ds.padded_batch(batch_size, padded_shapes=([320, 320, 1], [None]),
                             padding_values=(0.0, np.int32(end_token)))
        return ds

    return _input_fn


def model_fn(features, labels, mode, params=None, config=None, model_dir=None):
    embedding_dim = 256
    if mode == tf.estimator.ModeKeys.PREDICT:
        x = features['images']
    else:
        x = features
    x = unet(x,embedding_dim,32,0.5,4,mode == tf.estimator.ModeKeys.TRAIN)
    logging.info('Unet {}'.format(x.shape))
    word_index = params['word_index']
    x = tf.nn.relu(x)
    _, l1,l2, _ = tf.unstack(tf.shape(x))
    x = tf.reshape(x,[params['batch_size'],l1*l2,embedding_dim])
    features_length = tf.zeros((params['batch_size']), dtype=tf.int64) + tf.cast(l1*l2, tf.int64)
    with tf.variable_scope('embedding'):
        embedding = tf.get_variable('embedding_op',
                                    [len(params['word_index']), embedding_dim],
                                    initializer=tf.random_uniform_initializer(-1, 1),
                                    trainable=True,
                                    dtype=tf.float32)
    start_tokens = tf.zeros([params['batch_size']], dtype=tf.int32) + word_index['<start>']
    if mode == tf.estimator.ModeKeys.PREDICT:
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding, start_tokens=tf.to_int32(start_tokens), end_token=word_index['<end>'])
    else:
        train_output = tf.concat([tf.expand_dims(start_tokens, 1), labels], 1)
        output_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(train_output, word_index['<end>'])), 1)
        output_embed = tf.nn.embedding_lookup(embedding, train_output)
        if mode == tf.estimator.ModeKeys.EVAL:
            helper = tf.contrib.seq2seq.TrainingHelper(output_embed, output_lengths)
        else:
            helper = tf.contrib.seq2seq.TrainingHelper(output_embed, output_lengths)
            #helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(output_embed, output_lengths, embedding, 0.5)

    cells = [tf.contrib.rnn.GRUCell(params['hidden_size']) for _ in range(params['num_layers'])]
    mrnn = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
        num_units=params['hidden_size'], memory=x, memory_sequence_length=features_length)
    attn_cell = tf.contrib.seq2seq.AttentionWrapper(
        mrnn, attention_mechanism, attention_layer_size=params['hidden_size'],alignment_history=True)
    output_layer = tf.layers.Dense(len(word_index))
    initial_state = attn_cell.zero_state(dtype=tf.float32, batch_size=params['batch_size'])
    decoder = tf.contrib.seq2seq.BasicDecoder(attn_cell, helper, initial_state, output_layer=output_layer)
    outputs = tf.contrib.seq2seq.dynamic_decode(
        decoder,
        output_time_major=False,
        impute_finished=True,
        maximum_iterations=params['max_target_seq_length'])

    train_op = None
    if mode == tf.estimator.ModeKeys.PREDICT:
        loss = None
        predictions = outputs[0].sample_id
        alligments = outputs[1].alignment_history.stack()
        export_outputs = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
                {'labels':predictions,'attentions':alligments})}
    else:
        predictions = None
        export_outputs = None
        train_outputs = outputs[0]
        logging.info('Output: {}'.format(train_outputs.rnn_output))
        weights = tf.to_float(tf.not_equal(train_output[:, :-1], word_index['<end>']))
        loss = tf.contrib.seq2seq.sequence_loss(train_outputs.rnn_output, labels, weights=weights)
        if mode == tf.estimator.ModeKeys.TRAIN:
            opt = tf.train.AdamOptimizer(params['learning_rate'])
            if params['grad_clip'] is None:
                train_op = opt.minimize(loss, global_step=tf.train.get_or_create_global_step())
            else:
                gradients, variables = zip(*opt.compute_gradients(loss))
                gradients, _ = tf.clip_by_global_norm(gradients, params['grad_clip'])
                train_op = opt.apply_gradients([(gradients[i], v) for i, v in enumerate(variables)],
                                               global_step=tf.train.get_or_create_global_step())
    return tf.estimator.EstimatorSpec(
        mode=mode,
        eval_metric_ops=None,
        predictions=predictions,
        loss=loss,
        export_outputs=export_outputs,
        train_op=train_op)

