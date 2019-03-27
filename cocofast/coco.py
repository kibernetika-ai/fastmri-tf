import tensorflow as tf
import os
from models.unet import unet
import numpy as np
import logging
import json
from kibernetika.rpt import MlBoardReporter
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.training.session_run_hook import SessionRunArgs
import math

import cv2


def data_fn(params, training):
    data_set = params['data_set']
    with open(data_set + '/annotations/instances_train2017.json') as f:
        data = json.load(f)
    tmp = []
    for a in data['annotations']:
        if a['category_id'] == 1 and a['iscrowd'] == 0:
            fname = data_set + '/train2017/{:012d}.jpg'.format(a['image_id'])
            segmentation = a['segmentation']
            area = a['area']
            if os.path.exists(fname):
                if len(segmentation) < 4 and area > 900:
                    tmp.append((a['segmentation'], fname))
            else:
                if params['limit'] < 0:
                    logging.info('Can\'t find image {:012d}.jpg'.format(a['image_id']))
            if params['limit'] > 0 and len(tmp) >= params['limit']:
                break
    data = tmp
    resolution = params['resolution']

    def _input_fn():
        def _generator():
            for i in data:
                img = cv2.imread(i[1], cv2.IMREAD_COLOR)[:, :,
                      ::-1]
                m = np.zeros((img.shape[0], img.shape[1]), np.float32)
                for s in i[0]:
                    p = np.array(s, np.int32)
                    p = np.reshape(p, (1, int(p.shape[0] / 2), 2))
                    m = cv2.fillPoly(m, p, color=(255, 255, 255))
                img = cv2.resize(img, (resolution, resolution))
                img = img.astype(np.float32) / 127.5 - 1
                m = cv2.resize(m, (resolution, resolution)) / 127.5 - 1
                m = np.reshape(m, (resolution, resolution, 1))
                yield (img, m)

        ds = tf.data.Dataset.from_generator(_generator, (tf.float32, tf.float32),
                                            (tf.TensorShape([resolution, resolution, 3]),
                                             tf.TensorShape([resolution, resolution, 1])))
        if training:
            ds = ds.shuffle(params['batch_size'] * 2, reshuffle_each_iteration=True)
        ds = ds.apply(tf.contrib.data.batch_and_drop_remainder(params['batch_size']))
        if training:
            ds = ds.repeat(params['num_epochs']).prefetch(params['batch_size'])
        return ds

    return len(data) // params['batch_size'], _input_fn


def _unet_model_fn(features, labels, mode, params=None, config=None, model_dir=None):
    features = tf.reshape(features, [params['batch_size'], params['resolution'], params['resolution'], 3])
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    result = unet(features, 1, params['num_chans'], params['drop_prob'], params['num_pools'], training)
    loss = None
    train_op = None
    hooks = []
    export_outputs = None
    eval_hooks = []
    chief_hooks = []
    metrics = {}
    if mode != tf.estimator.ModeKeys.PREDICT:
        learning_rate_var = tf.Variable(float(params['lr']), trainable=False, name='lr',
                                        collections=[tf.GraphKeys.LOCAL_VARIABLES])
        loss = tf.losses.absolute_difference(labels, result)
        mse = tf.losses.mean_squared_error(labels, result)
        nmse = tf.norm(labels - result) ** 2 / tf.norm(labels) ** 2

        global_step = tf.train.get_or_create_global_step()
        epoch = global_step // params['epoch_len']
        if training:
            tf.summary.scalar('lr', learning_rate_var)
            tf.summary.scalar('mse', mse)
            tf.summary.scalar('nmse', nmse)
            board_hook = MlBoardReporter({
                "_step": global_step,
                "_epoch": epoch,
                "_train_loss": loss,
                '_train_lr': learning_rate_var,
                '_train_mse': mse,
                '_train_nmse': nmse}, every_steps=params['save_summary_steps'])
            chief_hooks = [board_hook]
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                opt = tf.train.RMSPropOptimizer(learning_rate_var, params['weight_decay'])
                train_op = opt.minimize(loss, global_step=global_step)

        tf.summary.image('Src', features, 3)
        rimage = (result - tf.reduce_min(result))
        rimage = rimage / tf.reduce_max(rimage)
        tf.summary.image('Reconstruction', rimage, 3)
        limage = (labels - tf.reduce_min(labels))
        limage = limage / tf.reduce_max(limage)
        tf.summary.image('Original', limage, 3)
        hooks = [TrainingLearningRateHook(
            params['epoch_len'],
            learning_rate_var,
            float(params['lr']),
            int(params['lr_step_size']),
            float(params['lr_gamma']))]
        if not training:
            metrics['mse'] = tf.metrics.mean(mse)
            metrics['nmse'] = tf.metrics.mean(nmse)
    else:
        export_outputs = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
                result)}

    return tf.estimator.EstimatorSpec(
        mode=mode,
        eval_metric_ops=metrics,
        predictions=result,
        training_chief_hooks=chief_hooks,
        loss=loss,
        training_hooks=hooks,
        export_outputs=export_outputs,
        evaluation_hooks=eval_hooks,
        train_op=train_op)


class TrainingLearningRateHook(session_run_hook.SessionRunHook):
    def __init__(self, epoch_len, learning_rate_var, initial_learning_rate, lr_step_size, lr_gamma):
        self._learning_rate_var = learning_rate_var
        self._lr_step_size = lr_step_size
        self._lr_gamma = lr_gamma
        self._epoch_len = epoch_len
        self._prev_learning_rate = 0
        self._initial_learning_rate = initial_learning_rate
        self._epoch = -1

    def begin(self):
        self._global_step_tensor = training_util._get_or_create_global_step_read()  # pylint: disable=protected-access
        if self._global_step_tensor is None:
            raise RuntimeError("Global step should be created to use LearningRateHook.")
        self._args = [self._global_step_tensor, self._learning_rate_var]
        self._learning_rate_ph = tf.placeholder(tf.float32, name='learning_rate_ph')
        self._learning_rate_op = self._learning_rate_var.assign(self._learning_rate_ph)

    def after_create_session(self, session, coord):
        session.run(self._learning_rate_op, {self._learning_rate_ph: self._initial_learning_rate})

    def before_run(self, run_context):  # pylint: disable=unused-argument
        return SessionRunArgs(self._args)

    def after_run(self, run_context, run_values):
        result = run_values.results
        global_step = result[0]
        learning_rate = result[1]
        if learning_rate != self._prev_learning_rate:
            logging.warning('Set Learning rate to {} at global step {}'.format(learning_rate, global_step))
            self._prev_learning_rate = learning_rate
        epoch = global_step // self._epoch_len

        if self._epoch != epoch:
            logging.info('Start epoch {}'.format(epoch))
            self._epoch = epoch

        lr_step = epoch // self._lr_step_size
        if lr_step > 0:
            desired_learning_rate = self._initial_learning_rate * math.pow(self._lr_gamma, lr_step)
        else:
            desired_learning_rate = self._initial_learning_rate

        if self._prev_learning_rate != desired_learning_rate:
            run_context.session.run(self._learning_rate_op, {self._learning_rate_ph: desired_learning_rate})


class CocoUnet(tf.estimator.Estimator):
    def __init__(
            self,
            params=None,
            model_dir=None,
            config=None,
            warm_start_from=None
    ):
        def _model_fn(features, labels, mode, params, config):
            return _unet_model_fn(
                features=features,
                labels=labels,
                mode=mode,
                params=params,
                config=config,
                model_dir=model_dir,
            )

        super(CocoUnet, self).__init__(
            model_fn=_model_fn,
            model_dir=model_dir,
            config=config,
            params=params,
            warm_start_from=warm_start_from
        )
