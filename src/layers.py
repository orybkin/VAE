import tensorflow as tf
import numpy as np

slim = tf.contrib.slim


def conv_factory(x, hidden_num, kernel_size, stride, is_train, pure=False, reuse=False):
    vs = tf.get_variable_scope()
    in_channels = x.get_shape()[3]
    W = tf.get_variable('weights', [kernel_size, kernel_size, in_channels, hidden_num],
                        initializer=tf.contrib.layers.variance_scaling_initializer())
    b = tf.get_variable('biases', [1, 1, 1, hidden_num],
                        initializer=tf.constant_initializer(0.0))

    x = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')
    if not pure:
        # x = slim.batch_norm(x, is_training=is_train, reuse=reuse, scale=True,
        #                     fused=True, scope=vs, updates_collections=None)
        x = batch_norm(x, is_train=is_train)
        x = tf.nn.relu(x)
    #  x = tf.nn.sigmoid(x)
    return x


def deconv_factory(batch_input, out_channels, is_train, pure=False, reuse=False):
    vs = tf.get_variable_scope()
    batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
    filter = tf.get_variable("weights", [3, 3, out_channels, in_channels], dtype=tf.float32,
                             initializer=tf.random_normal_initializer(0, 0.02))
    # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
    #     => [batch, out_height, out_width, out_channels]
    x = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * 2, in_width * 2, out_channels], [1, 2, 2, 1],
                               padding="SAME")
    if not pure:
        # x = slim.batch_norm(x, is_training=is_train, reuse=reuse, scale=True,
        #                     fused=True, scope=vs, updates_collections=None)
        x = batch_norm(x, is_train=is_train)
        x = tf.nn.relu(x)
    return x


def fc_factory(x, hidden_num, is_train, pure=False, reuse=False):
    vs = tf.get_variable_scope()
    in_channels = x.get_shape()[1]
    W = tf.get_variable('weights', [in_channels, hidden_num],
                        initializer=tf.contrib.layers.variance_scaling_initializer())
    b = tf.get_variable('biases', [1, hidden_num],
                        initializer=tf.constant_initializer(0.0))

    x = tf.matmul(x, W)
    #  x = slim.batch_norm(x, is_training=is_train, reuse=reuse, scale=True,
    #        fused=True, scope=vs, updates_collections=None)
    if not pure:
        # x = slim.batch_norm(x, is_training=is_train, reuse=reuse, scale=True,
        #                     fused=True, scope=vs, updates_collections=None)
        x = batch_norm(x, is_train=is_train)
        x = tf.nn.relu(x)
    #  x = tf.nn.sigmoid(x)
    return x


def leaky_relu(x):
    alpha = 0.2
    pos = tf.nn.relu(x)
    neg = alpha * (x - abs(x)) * 0.5
    return pos + neg


def batch_norm(x, is_train=True, decay=0.99, epsilon=0.001):
    shape_x = x.get_shape().as_list()
    beta = tf.get_variable('beta', shape_x[-1], initializer=tf.constant_initializer(0.0))
    gamma = tf.get_variable('gamma', shape_x[-1], initializer=tf.constant_initializer(1.0))
    moving_mean = tf.get_variable('moving_mean', shape_x[-1],
                                  initializer=tf.constant_initializer(0.0), trainable=False)
    moving_var = tf.get_variable('moving_var', shape_x[-1],
                                 initializer=tf.constant_initializer(1.0), trainable=False)

    if is_train:
        mean, var = tf.nn.moments(x, np.arange(len(shape_x) - 1), keep_dims=True)
        mean = tf.reshape(mean, [mean.shape.as_list()[-1]])
        var = tf.reshape(var, [var.shape.as_list()[-1]])

        update_moving_mean = tf.assign(moving_mean, moving_mean * decay + mean * (1 - decay))
        update_moving_var = tf.assign(moving_var,
                                      moving_var * decay + shape_x[0] / (shape_x[0] - 1) * var * (1 - decay))
        update_ops = [update_moving_mean, update_moving_var]

        with tf.control_dependencies(update_ops):
            return tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon)

    else:
        mean = moving_mean
        var = moving_var
        return tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon)


def encoder(x, ksize, pool_size, pool, hidden_num, is_train, reuse, additional_layers=False):
    with tf.variable_scope('first', reuse=reuse):
        x = conv_factory(x, hidden_num, ksize, 2, is_train, reuse=reuse)
        # x = pool(x, ksize=[1, pool_size, pool_size, 1], strides=[1, 2, 2, 1], padding='VALID')

    if additional_layers:
        with tf.variable_scope('conv11', reuse=reuse):
            x = conv_factory(x, hidden_num, ksize, 1, is_train, reuse=reuse)
        with tf.variable_scope('conv12', reuse=reuse):
            x = conv_factory(x, hidden_num, ksize, 1, is_train, reuse=reuse)

    # conv2
    with tf.variable_scope('conv2', reuse=reuse):
        hidden_num = hidden_num * 2
        x = conv_factory(x, hidden_num, ksize, 2, is_train, reuse=reuse)
        # x = pool(x, ksize=[1, pool_size, pool_size, 1], strides=[1, 2, 2, 1], padding='VALID')

    if additional_layers:
        with tf.variable_scope('conv21', reuse=reuse):
            x = conv_factory(x, hidden_num, ksize, 1, is_train, reuse=reuse)
        with tf.variable_scope('conv22', reuse=reuse):
            x = conv_factory(x, hidden_num, ksize, 1, is_train, reuse=reuse)

    # conv3
    with tf.variable_scope('conv3', reuse=reuse):
        hidden_num = 2 * hidden_num
        x = conv_factory(x, hidden_num, ksize, 2, is_train, reuse=reuse)
        # x = pool(x, ksize=[1, pool_size, pool_size, 1], strides=[1, 2, 2, 1], padding='VALID')

    if additional_layers:
        with tf.variable_scope('conv31', reuse=reuse):
            x = conv_factory(x, hidden_num, ksize, 1, is_train, reuse=reuse)
        with tf.variable_scope('conv32', reuse=reuse):
            x = conv_factory(x, hidden_num, ksize, 1, is_train, reuse=reuse)


    with tf.variable_scope('conv4', reuse=reuse):
        hidden_num = 2 * hidden_num
        x = conv_factory(x, hidden_num, ksize, 2, is_train, reuse=reuse)
        # x = pool(x, ksize=[1, pool_size, pool_size, 1], strides=[1, 2, 2, 1], padding='VALID')
        print(x)

    if additional_layers:
        with tf.variable_scope('conv41', reuse=reuse):
            x = conv_factory(x, hidden_num, ksize, 1, is_train, reuse=reuse)
        with tf.variable_scope('conv42', reuse=reuse):
            x = conv_factory(x, hidden_num, ksize, 1, is_train, reuse=reuse)


    with tf.variable_scope('conv5', reuse=reuse):
        # hidden_num = 2 * hidden_num
        x = conv_factory(x, hidden_num, ksize, 2, is_train, reuse=reuse)
        # x = pool(x, ksize=[1, pool_size, pool_size, 1], strides=[1, 2, 2, 1], padding='VALID')
        print(x)

    # if additional_layers:
    #     with tf.variable_scope('conv51', reuse=reuse):
    #         x = conv_factory(x, hidden_num, ksize, 1, is_train, reuse=reuse)
    #     with tf.variable_scope('conv52', reuse=reuse):
    #         x = conv_factory(x, hidden_num, ksize, 1, is_train, reuse=reuse)

    with tf.variable_scope('conv6', reuse=reuse):
        # hidden_num = 2 * hidden_num
        x = conv_factory(x, hidden_num, ksize, 2, is_train, reuse=reuse)
        # x = pool(x, ksize=[1, pool_size, pool_size, 1], strides=[1, 2, 2, 1], padding='VALID')
        print(x)

    return x


def decoder(x, ksize, hidden_num, is_train, reuse, additional_layers=False):

    def deconv_factory1(x, out_channels, is_train, pure=False, reuse=False):
        x = tf.image.resize_nearest_neighbor(x, size=(x.get_shape()[1]*2, x.get_shape()[2]*2))
        x = conv_factory(x, out_channels, 3, 1, is_train, pure, reuse)
        return x

    with tf.variable_scope('deconv1', reuse=reuse):
        hidden_num = int(hidden_num / 2)
        x = deconv_factory(x, hidden_num, is_train, reuse=reuse)

    if additional_layers:
        with tf.variable_scope('deconv11', reuse=reuse):
            x = conv_factory(x, hidden_num, 3, 1, is_train, reuse=reuse)
        with tf.variable_scope('deconv12', reuse=reuse):
            x = conv_factory(x, hidden_num, 3, 1, is_train, reuse=reuse)

    with tf.variable_scope('deconv2', reuse=reuse):
        hidden_num = int(hidden_num / 2)
        x = deconv_factory(x, hidden_num, is_train, reuse=reuse)


    if additional_layers:
        with tf.variable_scope('deconv21', reuse=reuse):
            x = conv_factory(x, hidden_num, 3, 1, is_train, reuse=reuse)
        with tf.variable_scope('deconv22', reuse=reuse):
            x = conv_factory(x, hidden_num, 3, 1, is_train, reuse=reuse)

    with tf.variable_scope('deconv3', reuse=reuse):
        hidden_num = int(hidden_num / 2)
        x = deconv_factory(x, hidden_num, is_train, reuse=reuse)

    if additional_layers:
        with tf.variable_scope('deconv31', reuse=reuse):
            x = conv_factory(x, hidden_num, 3, 1, is_train, reuse=reuse)
        with tf.variable_scope('deconv32', reuse=reuse):
            x = conv_factory(x, hidden_num, 3, 1, is_train, reuse=reuse)

    with tf.variable_scope('deconv4', reuse=reuse):
        hidden_num = int(hidden_num / 2)
        x = deconv_factory(x, hidden_num, is_train, reuse=reuse)

    if additional_layers:
        with tf.variable_scope('deconv41', reuse=reuse):
            x = conv_factory(x, hidden_num, 3, 1, is_train, reuse=reuse)
        with tf.variable_scope('deconv42', reuse=reuse):
            x = conv_factory(x, hidden_num, 3, 1, is_train, reuse=reuse)

    with tf.variable_scope('last', reuse=reuse):
        hidden_num = int(hidden_num / 2)
        x = deconv_factory(x, 1, is_train, pure=True, reuse=reuse)
        # x = tf.sigmoid(x)

    return x
