import tensorflow as tf

slim = tf.contrib.slim
from tensorflow.contrib.slim.python.slim.nets import inception_v3
import numpy as np
import argparse
import os
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


def _inception_v3_arg_scope(is_training=True,
                            weight_decay=0.00004,
                            stddev=0.1,
                            batch_norm_var_collection='moving_vars'):
    """Defines the default InceptionV3 arg scope.
    Args:
      is_training: Whether or not we're training the model.
      weight_decay: The weight decay to use for regularizing the model.
      stddev: The standard deviation of the trunctated normal weight initializer.
      batch_norm_var_collection: The name of the collection for the batch norm
        variables.
    Returns:
      An `arg_scope` to use for the inception v3 model.
    """
    batch_norm_params = {
        'is_training': is_training,
        # Decay for the moving averages.
        'decay': 0.9997,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # collection containing the moving mean and moving variance.
        'variables_collections': {
            'beta': None,
            'gamma': None,
            'moving_mean': [batch_norm_var_collection],
            'moving_variance': [batch_norm_var_collection],
        }
    }
    normalizer_fn = slim.batch_norm

    # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope(
                [slim.conv2d],
                weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                activation_fn=tf.nn.relu6,
                normalizer_fn=normalizer_fn,
                normalizer_params=batch_norm_params) as sc:
            return sc


def inception_classify_num_classes(inputs,num_classes,is_training=False):
    with slim.arg_scope(_inception_v3_arg_scope(is_training=is_training)):
        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected, slim.batch_norm],
                trainable=True):
            net, _ = inception_v3.inception_v3(inputs,num_classes,scope='InceptionV3',is_training=is_training)
            return net
def inception(inputs,is_training=False):
    with slim.arg_scope(_inception_v3_arg_scope(is_training=is_training)):
        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected, slim.batch_norm],
                trainable=True):
            with slim.arg_scope(
                    [slim.batch_norm, slim.dropout], is_training=is_training):
                net, _ = inception_v3.inception_v3_base(
                    inputs,
                    scope='InceptionV3')
                return net


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_set', type=str, default='./data', help='Dataset directory')
    parser.add_argument('--to_dir', type=str, default='./features', help='Destinition directory')
    parser.add_argument('--inception', type=str, default='./inception_v3.ckpt', help='Inception checkpoint')
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    data = pd.read_csv(args.data_set + '/descriptions.csv')
    labels = data['norm_description'][:]
    files = data['image_name'][:]

    vectorizer = CountVectorizer()
    x = vectorizer.fit_transform(labels)
    words = vectorizer.get_feature_names()
    counts = x.toarray().sum(axis=0)
    words = vectorizer.get_feature_names()
    index = 1
    with open(args.to_dir + '/dictionary.csv', 'w') as f:
        words = vectorizer.get_feature_names()
        for i, w in enumerate(words):
            if counts[i] > 2:
                if w.isdigit():
                    continue
                f.write('{},{}\n'.format(index, w))
                index += 1
    file = tf.placeholder(tf.string, shape=None, name='file')
    image_data = tf.read_file(file)
    img = tf.image.decode_png(image_data, channels=3)
    img = tf.image.resize_bilinear([img], (299, 299))
    img = tf.cast(img,tf.float32)/127.5-1
    net = inception(img)
    inception_variables_dict = {var.op.name: var for var in slim.get_model_variables('InceptionV3')}
    init_fn_inception = slim.assign_from_checkpoint_fn(args.inception, inception_variables_dict)
    if not tf.gfile.Exists(args.to_dir + '/images'):
        tf.gfile.MakeDirs(args.to_dir + '/images')
    with tf.Session() as sess:
        init_fn_inception(sess)
        for f in files:
            res = sess.run([net], {file: args.data_set+'/images/'+f})
            name = os.path.basename(f)
            np.save(args.to_dir + '/images/' + name, res[0][0])
