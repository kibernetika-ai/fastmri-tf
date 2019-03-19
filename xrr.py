import tensorflow as tf
import argparse
import os
import logging
import configparser
import exp.attention as attention
import json
from mlboardclient.api import client
import exp.util as util
mlboard = client.Client()


def parse_args():
    conf_parser = argparse.ArgumentParser(
        add_help=False
    )
    conf_parser.add_argument(
        '--checkpoint_dir',
        default=os.environ.get('TRAINING_DIR', 'training') + '/' + os.environ.get('BUILD_ID', '1'),
        help='Directory to save checkpoints and logs',
    )
    args, remaining_argv = conf_parser.parse_known_args()
    parser = argparse.ArgumentParser(
        parents=[conf_parser],
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    checkpoint_dir = args.checkpoint_dir
    logging.getLogger().setLevel('INFO')
    tf.logging.set_verbosity(tf.logging.INFO)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=2,
        help='Batch size.',
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=2e-4,
        help='Recommended learning_rate is 2e-4',
    )
    parser.add_argument(
        '--num_layers',
        type=int,
        default=1,
        help='Number RNN layers.',
    )
    parser.add_argument(
        '--max_target_seq_length',
        type=int,
        default=100,
        help='Maximum number of letters we allow in a single training (or test) example output',
    )
    parser.add_argument(
        '--hidden_size',
        type=int,
        default=16,
        help='LSTM hidden size.',
    )
    parser.add_argument(
        '--grad_clip',
        type=float,
        default=1,
        help='Norm for gradients clipping.',
    )
    parser.add_argument(
        '--save_summary_steps',
        type=int,
        default=1,
        help="Log summary every 'save_summary_steps' steps",
    )
    parser.add_argument(
        '--save_checkpoints_secs',
        type=int,
        default=600,
        help="Save checkpoints every 'save_checkpoints_secs' secs.",
    )
    parser.add_argument(
        '--save_checkpoints_steps',
        type=int,
        default=100,
        help="Save checkpoints every 'save_checkpoints_steps' steps",
    )
    parser.add_argument(
        '--keep_checkpoint_max',
        type=int,
        default=2,
        help='The maximum number of recent checkpoint files to keep.',
    )
    parser.add_argument(
        '--log_step_count_steps',
        type=int,
        default=1,
        help='The frequency, in number of global steps, that the global step/sec and the loss will be logged during training.',
    )
    parser.add_argument(
        '--data_set',
        default='./test',
        help='Location of training files or evaluation files',
    )
    parser.add_argument(
        '--inception_checkpoint',
        default=None,
        help='inception_checkpoint',
    )
    parser.add_argument(
        '--warm_start_from',
        type=str,
        default=None,
        help='Warm start',
    )
    parser.add_argument(
        '--dictionary',
        type=str,
        default='./test/dictionary.csv',
        help='Warm start',
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.set_defaults(worker=False)
    group.set_defaults(evaluator=False)
    group.set_defaults(test=False)
    group.add_argument('--worker', dest='worker', action='store_true',
                       help='Start in Worker(training) mode.')
    group.add_argument('--evaluator', dest='evaluator', action='store_true',
                       help='Start in evaluation mode')
    group.add_argument('--test', dest='test', action='store_true',
                       help='Test mode')
    group.add_argument('--export', dest='export', action='store_true',
                       help='Export model')
    p_file = os.path.join(checkpoint_dir, 'parameters.ini')
    if tf.gfile.Exists(p_file):
        parameters = configparser.ConfigParser(allow_no_value=True)
        parameters.read(p_file)
        parser.set_defaults(**dict(parameters.items("PARAMETERS", raw=True)))
    args = parser.parse_args(remaining_argv)
    print('\n*************************\n')
    print(args)
    print('\n*************************\n')
    return checkpoint_dir, args



def export(checkpoint_dir,params):
    conf = tf.estimator.RunConfig(
        model_dir=checkpoint_dir,
    )
    feature_placeholders = {
        'images': tf.placeholder(tf.float32, [params['batch_size'],299,299,3], name='images'),
    }
    receiver = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_placeholders,default_batch_size=params['batch_size'])
    net = attention.Model(
        params=params,
        model_dir=checkpoint_dir,
        config=conf,
    )
    export_path = net.export_savedmodel(
        checkpoint_dir,
        receiver,
    )
    export_path = export_path.decode("utf-8")
    client.update_task_info({'model_path': export_path})


def train(mode, checkpoint_dir, params):
    logging.info("start build  model")
    logging.info("TF: {}".format(tf.__version__))

    save_summary_steps = params['save_summary_steps']
    save_checkpoints_secs = params['save_checkpoints_secs'] if params['save_checkpoints_steps'] is None else None
    save_checkpoints_steps = params['save_checkpoints_steps']

    conf = tf.estimator.RunConfig(
        model_dir=checkpoint_dir,
        save_summary_steps=save_summary_steps,
        save_checkpoints_secs=save_checkpoints_secs,
        save_checkpoints_steps=save_checkpoints_steps,
        keep_checkpoint_max=params['keep_checkpoint_max'],
        log_step_count_steps=params['log_step_count_steps'],
    )

    net = attention.Model(
        params=params,
        model_dir=checkpoint_dir,
        config=conf,
        warm_start_from=params['warm_start_from']
    )
    logging.info("Start %s mode", mode)
    if mode == 'train':
        input_fn = attention.input_fn(params, True)
        net.train(input_fn=input_fn)
    elif mode == 'eval':
        train_fn = attention.null_dataset()
        train_spec = tf.estimator.TrainSpec(input_fn=train_fn)
        eval_fn = attention.input_fn(params,False)
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_fn, steps=1, start_delay_secs=10, throttle_secs=10)
        tf.estimator.train_and_evaluate(net, train_spec, eval_spec)
    else:
        logging.info("Not implemented")


def main():
    checkpoint_dir, args = parse_args()
    logging.info('------------------')
    logging.info('TF VERSION: {}'.format(tf.__version__))
    logging.info('ARGS: {}'.format(args))
    logging.info('------------------')
    if args.worker:
        mode = 'train'
    elif args.test:
        mode = 'test'
    else:
        mode = 'eval'
        cluster = {'chief': ['fake_worker1:2222'],
                   'ps': ['fake_ps:2222'],
                   'worker': ['fake_worker2:2222']}
        os.environ['TF_CONFIG'] = json.dumps(
            {
                'cluster': cluster,
                'task': {'type': 'evaluator', 'index': 0}
            })

    word_index = util.dictionary(args.dictionary)
    logging.info("NumClasses: {}".format(len(word_index)))
    params = {
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'save_summary_steps': args.save_summary_steps,
        'save_checkpoints_steps': args.save_checkpoints_steps,
        'save_checkpoints_secs': args.save_checkpoints_secs,
        'keep_checkpoint_max': args.keep_checkpoint_max,
        'log_step_count_steps': args.log_step_count_steps,
        'data_set': args.data_set,
        'max_target_seq_length':args.max_target_seq_length,
        'word_index':word_index,
        'grad_clip':args.grad_clip,
        'hidden_size':args.hidden_size,
        'num_layers':args.num_layers,
        'warm_start_from': args.warm_start_from,
        'inception_checkpoint': args.inception_checkpoint,
        'dictionary': args.dictionary,
    }

    if not tf.gfile.Exists(checkpoint_dir):
        tf.gfile.MakeDirs(checkpoint_dir)

    if args.export:
        export(checkpoint_dir,params)
        return
    train(mode, checkpoint_dir, params)



if __name__ == '__main__':
    main()