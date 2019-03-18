import tensorflow as tf
slim = tf.contrib.slim
from tensorflow.contrib.slim.python.slim.nets import inception_v3
import glob
import pandas as pd


def input_fn(params):
    batch_size = params['batch_size']
    checkpoint_dir = params['checkpoint_dir']
    data = pd.read_csv(params['data_set']+'/descriptions.csv')
    labels = data['norm_description'][:]
    files = data['image_name'][:]

    def _input_fn():
