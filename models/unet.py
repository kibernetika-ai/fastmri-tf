import tensorflow as tf
import logging

def conv_block(input, out_chans, drop_prob, name, pooling, training):
    with tf.variable_scope("layer_{}".format(name)):
        out = input
        for j in range(2):
            out = tf.layers.conv2d(out, out_chans, kernel_size=3, padding='same', name="conv_{}".format(j + 1))
            out = tf.layers.batch_normalization(out, training=training, name="bn_{}".format(j + 1))
            out = tf.nn.relu(out, name="relu_{}".format(j + 1))
            if training:
                out = tf.layers.dropout(out, drop_prob, name="dropout_{}".format(j + 1))

        if not pooling:
            return out

        pool, pool_arg = tf.nn.max_pool_with_argmax(out,
                                                    ksize=[1, 2, 2, 1],
                                                    strides=[1, 2, 2, 1],
                                                    padding='SAME', name='pool')
        return out, (pool, pool_arg)


def unpool(pool, ind, ksize=[1, 2, 2, 1], name=None):
    with tf.variable_scope(name) as scope:
        mask = tf.cast(ind, tf.int32)
        input_shape = tf.shape(pool, out_type=tf.int32)
        output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
        one_like_mask = tf.ones_like(mask, dtype=tf.int32)
        batch_shape = tf.concat([[input_shape[0]], [1], [1], [1]], axis=0)
        batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int32), shape=batch_shape)
        b = one_like_mask * batch_range
        y = mask // (output_shape[2] * output_shape[3]) % output_shape[1]
        x = (mask // output_shape[3]) % output_shape[2]
        feature_range = tf.range(output_shape[3], dtype=tf.int32)
        f = one_like_mask * feature_range
        updates_size = tf.size(pool)
        indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
        values = tf.reshape(pool, [updates_size])
        ret = tf.scatter_nd(indices, values, output_shape)
        return ret

def unet(inputs, out_chans, chans, drop_prob, num_pool_layers, training):
    output, pull = conv_block(inputs, chans, drop_prob, 'down_1', True, training)
    logging.info('Down_1 - {}'.format(output.shape))
    down_sample_layers = [output]
    pull_args = [pull[1]]
    ch = chans
    for i in range(num_pool_layers - 1):
        ch *= 2
        output, pull = conv_block(pull[0], ch, drop_prob, 'down_{}'.format(i + 2), True, training)
        logging.info('Down_{} - {}'.format(i+2,output.shape))
        down_sample_layers += [output]
        pull_args += [pull[1]]
    i+=1

    output = conv_block(pull[0], ch, drop_prob, 'down_{}'.format(i + 2), False, training)
    logging.info('Down_{} - {}'.format(i+2,output.shape))

    for i in range(num_pool_layers):
        down = down_sample_layers.pop()
        output = unpool(output, pull_args.pop(),name='unpool_{}'.format(i + 1))
        _,w,h,_ = down.shape
        output = output[:,:w,:h,:]
        output = tf.reshape(output, tf.shape(down))
        output = tf.concat([output, down], 3)
        logging.info('Up_{} - {}'.format(i+1,output.shape))
        if i < (num_pool_layers-1):
            ch //= 2
        output = conv_block(output, ch, drop_prob, 'up_{}'.format(i + 1), False, training)

    i+=1
    #output = conv_block(output, ch, drop_prob, 'up_{}'.format(i + 1), False, training)

    output = tf.layers.conv2d(output, ch, kernel_size=1, padding='same', name="conv_1")
    output = tf.layers.conv2d(output, out_chans, kernel_size=1, padding='same', name="conv_2")
    output = tf.layers.conv2d(output, out_chans, kernel_size=1, padding='same', name="final")
    return output


def unet_wrong(inputs, out_chans, chans, drop_prob, num_pool_layers, training):
    output, pull = conv_block(inputs, chans, drop_prob, 'down_1', True, training)
    logging.info('Down_1 - {}'.format(output.shape))
    down_sample_layers = [output]
    pull_args = [pull[1]]
    ch = chans
    for i in range(num_pool_layers - 1):
        ch *= 2
        output, pull = conv_block(pull[0], ch, drop_prob, 'down_{}'.format(i + 2), True, training)
        logging.info('Down_{} - {}'.format(i+2,output.shape))
        down_sample_layers += [output]
        pull_args += [pull[1]]
    i+=1

    output = conv_block(pull[0], ch, drop_prob, 'down_{}'.format(i + 2), False, training)
    logging.info('Down_{} - {}'.format(i+2,output.shape))

    for i in range(num_pool_layers):
        down = down_sample_layers.pop()
        output = unpool(output, pull_args.pop(),name='unpool_{}'.format(i + 1))
        output = tf.reshape(output, tf.shape(down))
        output = tf.concat([output, down], 3)
        logging.info('Up_{} - {}'.format(i+1,output.shape))
        ch //= 2
        output = conv_block(output, ch, drop_prob, 'up_{}'.format(i + 1), False, training)
    ch *= 2
    i+=1
    #output = conv_block(output, ch, drop_prob, 'up_{}'.format(i + 1), False, training)

    output = tf.layers.conv2d(output, ch, kernel_size=1, padding='same', name="conv_1")
    output = tf.layers.conv2d(output, out_chans, kernel_size=1, padding='same', name="conv_2")
    output = tf.layers.conv2d(output, out_chans, kernel_size=1, padding='same', name="final")
    return output
