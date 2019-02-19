import tensorflow as tf
import glob
import h5py
import numpy as np
from models.subsample import MaskFunc
import logging

def null_dataset():
    def _input_fn():
        return None
    return _input_fn

def real_tensor(data):
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)
    return data


def apply_mask(data, mask_func, seed=None):
    shape = np.array(data.shape)
    shape[:-3] = 1
    mask = mask_func(shape, seed)
    return data * mask, mask


def fftshift(x, dim=None):
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = tf.shape(x)[dim] // 2
    else:
        shift = [tf.shape(x)[i] // 2 for i in dim]
    return tf.manip.roll(x, shift, dim)

def ifftshift(x, dim=None):
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (tf.shape(x)[dim] + 1) // 2
    else:
        shift = [(tf.shape(x)[i] + 1) // 2 for i in dim]
    return tf.manip.roll(x,shift,dim)

def ifft2(data):
    data = ifftshift(data, dim=(-2, -1))
    data = tf.ifft2d(data)
    data = fftshift(data, dim=(-2, -1))
    return data

def center_crop(data, shape):
    data_shape = tf.shape(data)
    w_from = (data_shape[0] - shape[0]) // 2
    h_from = (data_shape[1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[w_from:w_to, h_from:h_to]

def normalize(data, mean, stddev, eps=0.):
    return (data - mean) / (stddev + eps)

def data_fn(params,training):
    examples = []
    recons_key = 'reconstruction_esc' if params['challenge'] == 'singlecoil' \
        else 'reconstruction_rss'
    for fname in glob.glob(params['data_path'] + '/*.h5'):
        kspace = h5py.File(fname, 'r')['kspace']
        num_slices = kspace.shape[0]
        examples += [(fname, slice) for slice in range(num_slices)]
    mask_func = MaskFunc(params['center_fractions'], params['accelerations'])

    def _input_fn():
        def _generator():
            for fname, slice in examples:
                with h5py.File(fname, 'r') as data:
                    kspace = data['kspace'][slice]
                    if recons_key not in data:
                        continue
                    target = data[recons_key][slice]
                    seed = None if not params['use_seed'] else tuple(map(ord, fname))
                    masked_kspace, mask = apply_mask(kspace, mask_func, seed)
                yield (masked_kspace, target)

        ds = tf.data.Dataset.from_generator(_generator, (tf.complex64, tf.float32),
                                            (tf.TensorShape([None, None]), tf.TensorShape([None, None])))

        def _transform(kspace, target):
            kspace = ifft2(kspace)
            # Crop input image
            kspace = center_crop(kspace, (params['resolution'], params['resolution']))
            # Absolute value
            kspace = tf.abs(kspace)
            # Normalize input
            mean,variance = tf.nn.moments(kspace,axes=[0,1])
            kspace = normalize(kspace,mean,variance,1e-11)
            kspace = tf.clip_by_value(kspace,-6,6)

            # Normalize target
            target = normalize(target, mean, variance, eps=1e-11)
            target = tf.clip_by_value(target,-6,6)
            kspace = tf.expand_dims(kspace,2)
            target = tf.expand_dims(target,2)
            return kspace,target

        if training:
            ds = ds.shuffle(params['batch_size']*2,reshuffle_each_iteration=True)
        ds = ds.map(_transform)
        if training:
            ds = ds.repeat(params['num_epochs'])
        ds = ds.apply(tf.contrib.data.batch_and_drop_remainder(params['batch_size']))
        return ds

    return len(examples)//params['batch_size'], _input_fn
