import tensorflow as tf


class Network(object):
    def __init__(self):
        pass

    def create_variables(self, shape, initializer, weight_decay=0.00004, trainable=True, use_regularizer=False, name=None):
        # different weight decay for fully connected layer and conv layer
        if trainable and use_regularizer:
            assert weight_decay >= 0
            regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)
        else:
            regularizer = None

        var = tf.get_variable(name, shape=shape, initializer=initializer, trainable=trainable, regularizer=regularizer)
        # tf.summary.histogram(name + '/value', var)
        return var

    def conv2d(self, input_, num_filters, ksize, stride, weight_decay, stddev=1.0, padding="SAME", trainable=True, use_regularizer=True,name='conv2d'):
        in_channels = input_.get_shape().as_list()[-1]
        shape = [ksize, ksize, in_channels, num_filters]
        strides = [1, stride, stride, 1]
        with tf.variable_scope(name):
            weight = self.create_variables(shape=shape,
                                            weight_decay=weight_decay,
                                            initializer=tf.truncated_normal_initializer(stddev=stddev),
                                            trainable=trainable,
                                            use_regularizer=use_regularizer,
                                            name="weight")
            return tf.nn.conv2d(input_, weight, strides=strides, padding=padding, name=name)

    def deconv2d(self, input_, num_filters, ksize, stride, weight_decay, stddev=1.0, output_shape=None, padding="SAME", trainable=True, use_regularizer=True, name="deconv2d"):
        in_channels = input_.get_shape().as_list()[3]
        shape = [ksize, ksize, num_filters, in_channels]
        strides = [1, stride, stride, 1]

        if output_shape is None:
            output_shape = input_.get_shape().as_list()
            output_shape[1] *= stride
            output_shape[2] *=stride
            output_shape[3] = num_filters

        with tf.variable_scope(name):
            weight = self.create_variables(shape=shape,
                                           weight_decay=weight_decay,
                                           initializer=tf.truncated_normal_initializer(stddev=stddev),
                                           trainable=trainable,
                                           use_regularizer=use_regularizer,
                                           name="weight")
            return tf.nn.conv2d_transpose(input_, weight, output_shape=output_shape, strides=strides, padding=padding, name=name)

    def fc(self, input_, num_filters, weight_decay=0.00004, stddev=0.01, trainable=True, use_regularizer=True, name="fc"):
        assert len(input_.get_shape()) == 2
        in_channels = input_.get_shape().as_list()[-1]
        initializer = tf.truncated_normal_initializer(stddev=stddev)

        with tf.variable_scope(name):
            weight = self.create_variables(shape=[in_channels, num_filters],
                                           weight_decay=weight_decay,
                                           initializer=initializer,
                                           trainable=trainable,
                                           use_regularizer=use_regularizer,
                                           name='weight')
            bias = self.create_variables(shape=[num_filters],
                                         initializer=tf.zeros_initializer(),
                                         trainable=trainable,
                                         use_regularizer=use_regularizer,
                                         name='biases')

            output_ = tf.nn.xw_plus_b(input_, weight, bias, name=name)
        return output_

    def batch_norm(self, input_, momentum=0.99, epsilon=1e-5, training=None, name="batch_norm"):
        # When train, set training=True, and add the following dependencies to update moving_mean and moving_variance
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        #     train_op = optimizer.minimize(loss)
        # By tf data format NHWC, axis is 1 by default.
        bn_train = tf.layers.batch_normalization(input_, axis=-1, momentum=momentum, epsilon=epsilon, center=True, scale=True,
                                             beta_initializer=tf.zeros_initializer(),
                                             gamma_initializer=tf.ones_initializer(),
                                             moving_mean_initializer=tf.zeros_initializer(),
                                             moving_variance_initializer=tf.ones_initializer(),
                                             training=True,
                                             reuse=None,
                                             name=name)
        bn_inference = tf.layers.batch_normalization(input_, axis=-1, momentum=momentum, epsilon=epsilon, center=True,
                                                 scale=True,
                                                 beta_initializer=tf.zeros_initializer(),
                                                 gamma_initializer=tf.ones_initializer(),
                                                 moving_mean_initializer=tf.zeros_initializer(),
                                                 moving_variance_initializer=tf.ones_initializer(),
                                                 training=False,
                                                 reuse=True,
                                                 name=name)
        bn_output = tf.cond(training, lambda: bn_train, lambda: bn_inference)
        return bn_output

    def instance_norm(self, input_, epsilon=1e-5, name="instance_norm"):
        # usually used in GAN
        depth = input_.get_shape().as_list()[-1]
        with tf.variable_scope(name):
            scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
            offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
            mean, variance = tf.nn.moments(input_, [1, 2], keep_dims=True)
            inv = tf.rsqrt(variance + epsilon)
            normalized = (input_-mean) * inv
            return scale * normalized + offset

    def pooling(self, input_, ksize, stride, padding="SAME", pool_type="max_pool", name="pooling"):
        if pool_type == "max_pool":
            return tf.nn.max_pool(input_,
                                  ksize=[1, ksize, ksize, 1],
                                  strides=[1, stride, stride, 1],
                                  padding=padding,
                                  name=name)
        else:
            return tf.nn.avg_pool(input_,
                                  ksize=[1, ksize, ksize, 1],
                                  strides=[1, stride, stride, 1],
                                  padding=padding,
                                  name=name)

    def relu(self, input_, name="relu"):
        return tf.nn.relu(input_, name=name)

    def lrelu(self, input_, alpha=0.2, name="lrelu"):
        return tf.nn.leaky_relu(input_, alpha=alpha, name=name)

    def dropout(self, input_, keep_prob, name="dropout"):
        return tf.nn.dropout(input_, keep_prob=keep_prob, name=name)

    def concat(self, input_, axis=-1, name="concat"):
        return tf.concat(input_, axis=axis, name=name)

    def loss(self, logits, labels):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        mean_cross_entropy = tf.reduce_mean(cross_entropy)
        regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss_ = tf.add_n([mean_cross_entropy] + regularization_loss)
        tf.summary.scalar(loss_)
        return loss_


if __name__ == "__main__":
    network = Network()
    with tf.variable_scope("my_variable"):
        var1 = network.create_variables([1, 3, 3, 1], weight_decay=0.5, initializer=tf.random_normal_initializer(), name="var1")
        var2 = network.create_variables([1, 3, 3, 1], weight_decay=0.5, initializer=tf.random_normal_initializer(), name="var2")
    with tf.variable_scope("my_variableA"):
        var1 = network.create_variables([1, 3, 3, 1], weight_decay=0.5, initializer=tf.random_normal_initializer(),
                                        name="var1")
        var2 = network.create_variables([1, 3, 3, 1], weight_decay=0.5, initializer=tf.random_normal_initializer(),
                                    name="var2")

    logger = tf.summary.FileWriter('./logs')
    merged_summary = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary = sess.run(merged_summary)
        logger.add_graph(sess.graph)
        logger.add_summary(summary)






