import tensorflow as tf
import os
from models.unet import unet
import logging
from kibernetika.rpt import MlBoardReporter
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.training.session_run_hook import SessionRunArgs
import math
import glob


def data_fn(params, training):
    data_set = params['data_set']
    files = glob.glob(data_set+'/masks/*.jpg')
    for i in range(len(files)):
        mask = files[i]
        img = os.path.basename(mask)
        img = data_set+'/images/'+img
        files[i] = [img,mask]

    resolution = params['resolution']

    def _input_fn():
        ds = tf.data.Dataset.from_tensor_slices(files)
        def _read_images(a):
            img = tf.read_file(a[0])
            img = tf.image.decode_jpeg(img)
            img = tf.reshape(img,[resolution,resolution,3])
            mask = tf.read_file(a[1])
            mask = tf.image.decode_jpeg(mask)
            mask = mask[:,:,0]
            mask = tf.reshape(mask,[resolution,resolution,1])
            img = tf.cast(img,dtype=tf.float32)/127.5-1
            mask = tf.cast(mask,dtype=tf.int32)/255
            return img,mask
        ds = ds.map(_read_images)
        if training:
            ds = ds.shuffle(params['batch_size'] * 2, reshuffle_each_iteration=True)
        ds = ds.apply(tf.contrib.data.batch_and_drop_remainder(params['batch_size']))
        if training:
            ds = ds.repeat(params['num_epochs']).prefetch(params['batch_size'])
        return ds

    return len(files) // params['batch_size'], _input_fn


def _unet_model_fn(features, labels, mode, params=None, config=None, model_dir=None):
    if mode == tf.estimator.ModeKeys.PREDICT:
        features = features['image']
    else:
        features = tf.reshape(features, [params['batch_size'], params['resolution'], params['resolution'], 3])
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    out_chans = 2 if params['loss']=='entropy' else 1
    logits = unet(features, out_chans, params['num_chans'], params['drop_prob'], params['num_pools'], training=training,unpool_layer=params['unpool'])
    if params['loss']=='entropy':
        mask = tf.cast(tf.argmax(logits, axis=3),tf.float32)
        logging.info('Mask shape1: {}'.format(mask.shape))
        mask = tf.expand_dims(mask,-1)
        logging.info('Mask shape2: {}'.format(mask.shape))
    else:
        mask = tf.sigmoid(logits)
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

        flabels = tf.cast(labels,tf.float32)
        if params['loss']=='entropy':
            llabels = tf.cast(labels, tf.int32)
            logits = tf.reshape(logits, [tf.shape(logits)[0], -1, 2])
            llabels = tf.reshape(llabels, [tf.shape(llabels)[0], -1])
            loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=llabels)
        else:
            loss = tf.losses.absolute_difference(flabels, mask)
        mse = tf.losses.mean_squared_error(flabels, mask)
        nmse = tf.norm(flabels - mask) ** 2 / tf.norm(flabels) ** 2

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
                if params['optimizer']=='AdamOptimizer':
                    opt = tf.train.AdamOptimizer(learning_rate_var)
                else:
                    opt = tf.train.RMSPropOptimizer(learning_rate_var, params['weight_decay'])
                train_op = opt.minimize(loss, global_step=global_step)

        tf.summary.image('Src', features, 3)
        rimage = (mask - tf.reduce_min(mask))
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
                mask)}

    return tf.estimator.EstimatorSpec(
        mode=mode,
        eval_metric_ops=metrics,
        predictions=mask,
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
