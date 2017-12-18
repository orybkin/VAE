import tensorflow as tf
from layers import *


def ConvNet(x, labels, c_num, batch_size, is_train, reuse):
    with tf.variable_scope('C', reuse=reuse) as vs:
        pool_size = 2
        pool = tf.nn.max_pool
        # conv1
        with tf.variable_scope('first', reuse=reuse):
            hidden_num = 32
            x = conv_factory(x, hidden_num, 5, 1, is_train, reuse)
            x = pool(x, ksize=[1, pool_size, pool_size, 1], strides=[1, 2, 2, 1], padding='VALID')

        # conv2
        with tf.variable_scope('conv2', reuse=reuse):
            hidden_num = hidden_num * 2
            x = conv_factory(x, hidden_num, 5, 1, is_train, reuse)
            x = pool(x, ksize=[1, pool_size, pool_size, 1], strides=[1, 2, 2, 1], padding='VALID')

        # conv3
        with tf.variable_scope('conv3', reuse=reuse):
            hidden_num = 2 * hidden_num
            x = conv_factory(x, hidden_num, 5, 1, is_train, reuse)
            x = pool(x, ksize=[1, pool_size, pool_size, 1], strides=[1, 2, 2, 1], padding='VALID')

        with tf.variable_scope('conv4', reuse=reuse):
            hidden_num = 2 * hidden_num
            x = conv_factory(x, hidden_num, 5, 1, is_train, reuse)
            x = pool(x, ksize=[1, pool_size, pool_size, 1], strides=[1, 2, 2, 1], padding='VALID')

        with tf.variable_scope('last', reuse=reuse):
            hidden_num = 2 * hidden_num
            x = conv_factory(x, hidden_num, 3, 1, is_train, reuse)
            x = pool(x, ksize=[1, pool_size, pool_size, 1], strides=[1, 2, 2, 1], padding='VALID')
            x = tf.reshape(x, [batch_size, -1])


        feat = x

        # dropout
        #    if is_train:
        #      x = tf.nn.dropout(x, keep_prob=0.5)

        # local5
        with tf.variable_scope('fc6', reuse=reuse):
            W = tf.get_variable('weights', [hidden_num, c_num],
                                initializer=tf.contrib.layers.variance_scaling_initializer())
            x = tf.matmul(x, W)

        # Softmax
        with tf.variable_scope('sm', reuse=reuse):
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=tf.one_hot(labels, c_num))
            accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(x, axis=1), labels)))

    variables = tf.contrib.framework.get_variables(vs)
    return loss, feat, accuracy, variables


def AE(x, labels, c_num, batch_size, is_train, reuse):
    #x=tf.constant(1,dtype=tf.float32, shape=[batch_size,64,64,1])
    with tf.variable_scope('C', reuse=reuse) as vs:
        pool_size = 2
        pool = tf.nn.max_pool
        enc_ksize = 3
        n_latent=16

        x_input=x
        print('input',x)

        # conv1

        #with tf.variable_scope('encoder', reuse=reuse):

        filter_num = 4
        x=encoder(x, enc_ksize, pool_size, pool, filter_num, is_train, reuse)
        x_enc_sz=x.get_shape()
        filter_num=x_enc_sz[3]

        # with tf.variable_scope('fc1', reuse=reuse):
        #     x = tf.reshape(x, [batch_size, -1])
        #     x = fc_factory(x, n_latent, is_train, reuse=reuse)

        latent = x
        print('latent',x)
        # if is_train:
        #     x=tf.nn.dropout(x,0.5)

        # with tf.variable_scope('fc2', reuse=reuse):
        #     x = fc_factory(x, x_enc_sz[1] * x_enc_sz[2] * x_enc_sz[3], is_train, reuse)
        #     x = tf.reshape(x, [batch_size, x_enc_sz[1], x_enc_sz[2], x_enc_sz[3]])

        x=decoder(x, None, filter_num, is_train, reuse)

        print('output',x)
        with tf.variable_scope('loss', reuse=reuse):
            loss_rec = tf.reduce_mean(tf.pow(x_input-x,2))


    variables = tf.contrib.framework.get_variables(vs)
    return loss_rec, latent, loss_rec, variables, x


def VAE(x, labels, c_num, batch_size, is_train, reuse):
    with tf.variable_scope('C', reuse=reuse) as vs:
        pool_size = 2
        pool = tf.nn.max_pool
        enc_ksize = 3
        n_latent=512

        x_input=x

        # conv1

        #with tf.variable_scope('encoder', reuse=reuse):

        filter_num = 32
        x=encoder(x, enc_ksize, pool_size, pool, filter_num, is_train, reuse, additional_layers=True)
        x_enc_sz=x.get_shape()
        filter_num=x_enc_sz[3]

        with tf.variable_scope('latent', reuse=reuse):
            x=tf.reshape(x, [batch_size, -1])
            with tf.variable_scope('mean', reuse=reuse):
                mean = fc_factory(x, n_latent, is_train, pure=True, reuse=reuse)
            with tf.variable_scope('std', reuse=reuse):
                std = fc_factory(x, n_latent, is_train, pure=True, reuse=reuse)

            # mean=x[:,:,:,:32]
            # std=x[:,:,:,32:]

            with tf.variable_scope('sample', reuse=reuse):
                # x = tf.random_normal([batch_size, n_latent])*tf.sqrt(tf.exp(std))+mean
                x = tf.random_normal(std.get_shape())*(tf.exp(std))+mean
                # x = tf.random_normal([batch_size, 1])*(tf.exp(std))+mean
                latent=x
                # loss_var = -0.5 * tf.reduce_mean(1 + std - tf.square(mean) - tf.exp(std))
                loss_KL = -0.5 * tf.reduce_mean(1 + (((std))) - tf.square(mean) - tf.exp(std))

            with tf.variable_scope('fc_latent', reuse=reuse):
                x = fc_factory(x, tf.Dimension(2*2)*x_enc_sz[1]*x_enc_sz[2]*x_enc_sz[3], is_train, pure=True, reuse=reuse)
                x = tf.reshape(x,[batch_size, tf.Dimension(2)*x_enc_sz[1], tf.Dimension(2)*x_enc_sz[2], -1]) #x_enc_sz[3]]) #16]) #

        x=decoder(x, None, filter_num, is_train, reuse, additional_layers=True)

        with tf.variable_scope('loss', reuse=reuse):
            # x_input = tf.clip_by_value(x_input, 1e-10, 1- 1e-10)
            # x = tf.clip_by_value(x, 1e-10, 1 - 1e-10)
            # loss_rec = -tf.reduce_mean(x_input * tf.log(1e-10 + x) + (1 - x_input) * tf.log(1e-10+ 1 - x))
            loss_rec = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=x_input, logits=x))
            # loss_rec = tf.reduce_mean(tf.pow(x_input-x,2))


    loss=tf.reduce_mean(loss_rec+loss_KL/10)

    variables = tf.contrib.framework.get_variables(vs)
    return loss, latent, loss_rec, variables, x
