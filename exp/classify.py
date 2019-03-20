import tensorflow as tf

slim = tf.contrib.slim
import pandas as pd
import numpy as np
import logging
from tensorflow.python.training import session_run_hook
import exp.util as util
import PIL.Image as Image
from tensorflow.contrib.slim.nets import inception
from tensorflow.contrib.framework.python.ops import arg_scope
BATCH_NORM_DECAY = 0.996
BATCH_NORM_EPSILON = 1e-3

def input_fn(params, is_training):
    batch_size = params['batch_size']
    if is_training:
        data = pd.read_csv(params['data_set'] + '/train.csv')
    else:
        data = pd.read_csv(params['data_set'] + '/test.csv')
    labels = data['description'][:]
    files = data['image_name'][:]
    word_index = params['word_index']
    num_classes = util.dictionary_size(word_index)
    image_dir = params['data_set'] + '/images/'
    def _input_fn():
        def _generator():
            for i, f in enumerate(files):
                text = labels[i]
                tokens = util.labels(word_index, text)
                x = Image.open(image_dir+f)
                x = x.resize((299,299))
                x = np.asarray(x,np.float32)/127.5-1
                x = np.reshape(x,(299,299,1))
                # logging.info('Tokens: {}'.format(len(tokens)))
                yield (x, tokens)

        ds = tf.data.Dataset.from_generator(_generator, (tf.float32, tf.float32),
                                            (tf.TensorShape([299,299,1]), tf.TensorShape([num_classes])))

        def _features_labels(x, y):
            return x, y

        ds = ds.map(_features_labels)
        if is_training:
            ds = ds.apply(tf.contrib.data.shuffle_and_repeat(100))
        ds = ds.batch(batch_size)
        return ds

    return _input_fn


def model_fn(features, labels, mode, params=None, config=None, model_dir=None):
    if mode == tf.estimator.ModeKeys.PREDICT:
        x = features['images']
    else:
        x = features
    logging.info('x:{}'.format(x.shape))
    with arg_scope(inception.inception_v3_arg_scope(
            batch_norm_decay=BATCH_NORM_DECAY,
            batch_norm_epsilon=BATCH_NORM_EPSILON)):
        logits, end_points = inception.inception_v3(
            x,
            num_classes=util.dictionary_size(params['word_index']),
            is_training=(mode == tf.estimator.ModeKeys.TRAIN),
            scope='InceptionV3')

    train_op = None
    if mode == tf.estimator.ModeKeys.PREDICT:
        loss = None
        predictions = tf.nn.sigmoid(logits)
        export_outputs = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
                {'labels':predictions})}
    else:
        predictions = None
        export_outputs = None
        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_mean(cross_entropy)
        if mode == tf.estimator.ModeKeys.TRAIN:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            opt = tf.train.AdamOptimizer(params['learning_rate'])
            with tf.control_dependencies(update_ops):
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
        training_hooks=[IniInceptionHook(params['inception_checkpoint'])],
        train_op=train_op)


class IniInceptionHook(session_run_hook.SessionRunHook):
    def __init__(self, model_path):
        self._model_path = model_path
        self._ops = None

    def begin(self):
        if self._model_path is not None:
            inception_variables_dict = {
                var.op.name: var
                for var in slim.get_model_variables('InceptionV3')
            }
            self._init_fn_inception = slim.assign_from_checkpoint_fn(self._model_path, inception_variables_dict)

    def after_create_session(self, session, coord):
        if self._model_path is not None:
            logging.info('Do  Init Inception')
            self._init_fn_inception(session)

    def before_run(self, run_context):  # pylint: disable=unused-argument
        return None

    def after_run(self, run_context, run_values):
        None
