from common.args import Args
import pathlib
import random
import numpy as np
import tensorflow as tf
import logging
from models.model import UNet
from models.data import data_fn

def train(mode, checkpoint_dir, params):
    logging.info("start build  model")

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
    epoch_len,fn = data_fn(params)
    params['epoch_len'] = epoch_len
    net = UNet(
        params=params,
        model_dir=checkpoint_dir,
        config=conf,
        warm_start_from=params['warm_start_from']

    )
    logging.info("Start %s mode", mode)
    if mode == 'train':
        net.train(input_fn=fn)
    else:
        logging.info("Not implemented")

def main(args):
    params = {
        'num_pools': args.num_pools,
        'num_epochs':args.num_epochs,
        'drop_prob': args.drop_prob,
        'num_chans': args.num_chans,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'lr_step_size': args.lr_step_size,
        'lr_gamma': args.lr_gamma,
        'weight_decay': args.weight_decay,
        'checkpoint': str(args.exp_dir),
        'seed':args.seed,
        'resolution':args.resolution,
        'challenge':args.challenge,
        'data_path':args.data_path,
        'accelerations':args.accelerations,
        'center_fractions':args.center_fractions,
        'save_summary_steps':args.save_summary_steps,
        'save_checkpoints_secs':args.save_checkpoints_secs,
        'save_checkpoints_steps':args.save_checkpoints_steps,
        'keep_checkpoint_max':args.keep_checkpoint_max,
        'log_step_count_steps':args.log_step_count_steps,
        'warm_start_from':args.warm_start_from,
        'use_seed': False,
    }
    if not tf.gfile.Exists(args.exp_dir):
        tf.gfile.MakeDirs(args.exp_dir)
    train('train',args.exp_dir,params)

def create_arg_parser():
    parser = Args()
    parser.add_argument('--num-pools', type=int, default=4, help='Number of U-Net pooling layers')
    parser.add_argument('--drop-prob', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--num-chans', type=int, default=32, help='Number of U-Net channels')

    parser.add_argument('--batch-size', default=4, type=int, help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr-step-size', type=int, default=40,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='Strength of weight decay regularization')

    parser.add_argument('--exp-dir', type=str, default='checkpoints',
                        help='Path where model and results should be saved')
    parser.add_argument(
        '--save_summary_steps',
        type=int,
        default=10,
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
        default=1000,
        help="Save checkpoints every 'save_checkpoints_steps' steps",
    )
    parser.add_argument(
        '--keep_checkpoint_max',
        type=int,
        default=5,
        help='The maximum number of recent checkpoint files to keep.',
    )
    parser.add_argument(
        '--log_step_count_steps',
        type=int,
        default=1,
        help='The frequency, in number of global steps, that the global step/sec and the loss will be logged during training.',
    )
    parser.add_argument(
        '--warm_start_from',
        type=str,
        default=None,
        help='Warm start from',
    )
    return parser


if __name__ == '__main__':
    logging.getLogger().setLevel('INFO')
    args = create_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    main(args)
