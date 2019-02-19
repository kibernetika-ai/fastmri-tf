import tensorflow as tf
from models.unet import unet
import logging
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.training.session_run_hook import SessionRunArgs
import math
from kibernetika.rpt import MlBoardReporter


def _unet_model_fn(features, labels, mode, params=None, config=None, model_dir=None):
    features = tf.reshape(features, [params['batch_size'], params['resolution'], params['resolution'], 1])
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    result = unet(features, 1, params['num_chans'], params['drop_prob'], params['num_pools'], training)
    loss = None
    train_op = None
    hooks = []
    export_outputs = None
    eval_hooks = []
    chief_hooks = []
    if mode != tf.estimator.ModeKeys.PREDICT:
        learning_rate_var = tf.Variable(float(params['lr']), trainable=False, name='lr',
                                        collections=[tf.GraphKeys.LOCAL_VARIABLES])
        loss = tf.losses.absolute_difference(labels, result)
        mse = tf.losses.mean_squared_error(labels, result)
        nmse = tf.norm(labels - result) ** 2 / tf.norm(labels) ** 2

        global_step = tf.train.get_or_create_global_step()
        epoch = global_step // params['epoch_len']
        if training:
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
        tf.summary.scalar('lr', learning_rate_var)
        tf.summary.scalar('mse', mse)
        tf.summary.scalar('nmse', nmse)
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
            tf.summary.scalar('loss', loss)
            board_hook = MlBoardReporter({
                "_eval_loss": loss,
                '_eval_mse': mse,
                '_eval_nmse': nmse}, every_steps=1)
            eval_hooks = [tf.train.SummarySaverHook(
                save_steps=1,
                output_dir=model_dir + "/test",
                summary_op=tf.summary.merge_all()), board_hook]
    else:
        export_outputs = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
                result)}

    return tf.estimator.EstimatorSpec(
        mode=mode,
        eval_metric_ops={},
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

        lr_step = epoch//self._lr_step_size
        if lr_step > 0:
            desired_learning_rate = self._initial_learning_rate*math.pow(self._lr_gamma, lr_step)
        else:
            desired_learning_rate = self._initial_learning_rate

        if self._prev_learning_rate != desired_learning_rate:
            run_context.session.run(self._learning_rate_op, {self._learning_rate_ph: desired_learning_rate})


class UNet(tf.estimator.Estimator):
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

        super(UNet, self).__init__(
            model_fn=_model_fn,
            model_dir=model_dir,
            config=config,
            params=params,
            warm_start_from=warm_start_from
        )
