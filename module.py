import tensorflow as tf
from nn import Network

class Modules(Network):
    def __init__(self, logger, is_print=False):
        super(Modules, self).__init__()
        self.logger = logger
        self.is_print = is_print

    def G_Unet(self, input_, gf_dim, z_dim, keep_prob, weight_decay=0.00004, is_training=True, reuse=False, name="generator"):
        keep_prob = tf.cond(is_training, lambda: keep_prob, lambda: 1.0)
        with tf.variable_scope(name, reuse=reuse):
            # 32*32*64
            e1 = self.instance_norm(self.conv2d(input_, gf_dim, ksize=4, stride=2, weight_decay=weight_decay, stddev=0.02, name="g_e1_conv"), name="g_bn_e1")
            # 16*16*128
            e2 = self.instance_norm(self.conv2d(self.lrelu(e1), gf_dim * 2, ksize=4, stride=2, weight_decay=weight_decay, stddev=0.02, name="g_e2_conv"), name="g_bn_e2")
            # 8*8*256
            e3 = self.instance_norm(self.conv2d(self.lrelu(e2), gf_dim * 4, ksize=4, stride=2, weight_decay=weight_decay, stddev=0.02, name="g_e3_conv"), name="g_bn_e3")
            # 4*4*512
            e4 = self.instance_norm(self.conv2d(self.lrelu(e3), gf_dim * 8, ksize=4, stride=2, weight_decay=weight_decay, stddev=0.02, name="g_e4_conv"), name="g_bn_e4")
            tz = e4.get_shape().as_list()[0:3] + [z_dim]
            z = tf.random_uniform(shape=tz, minval=-1, maxval=1, dtype=tf.float32, name="noise")
            ez = tf.concat([e4, z], axis=-1)

            en = self.instance_norm(self.conv2d(self.lrelu(ez), gf_dim * 8, ksize=1, stride=1, weight_decay=weight_decay, stddev=0.02, name="g_e5_conv"), name="g_bn_e5")

            d1 = self.deconv2d(self.relu(en), gf_dim * 8, ksize=4, stride=2, weight_decay=weight_decay, stddev=0.02, name="g_d1")
            d1 = self.dropout(d1, keep_prob=keep_prob)
            d1 = self.concat([self.instance_norm(d1, name="g_bn_d1"), e3])

            d2 = self.deconv2d(self.relu(d1), gf_dim * 4, ksize=4, stride=2, weight_decay=weight_decay, stddev=0.02,name="g_d2")
            d2 = self.concat([self.instance_norm(d2, name="g_bn_d2"), e2])

            d3 = self.deconv2d(self.relu(d2), gf_dim * 2, ksize=4, stride=2, weight_decay=weight_decay, stddev=0.02,name="g_d3")
            d3 = self.concat([self.instance_norm(d3, name="g_bn_d3"), e1])

            d4 = self.deconv2d(self.relu(d3), 1, ksize=4, stride=2, weight_decay=weight_decay, stddev=0.02, name="g_d4")
            d4 = tf.nn.tanh(d4)
            return d4

    def G_Unet_Cyc(self, input_, gf_dim, z_dim, keep_prob, weight_decay=0.00004, is_training=True, reuse=False, name="generator"):
        keep_prob = tf.cond(is_training, lambda: keep_prob, lambda: 1.0)
        with tf.variable_scope(name, reuse=reuse):
            # 32*32*64
            e1 = self.instance_norm(self.conv2d(input_, gf_dim, ksize=4, stride=2, weight_decay=weight_decay, stddev=0.02, name="g_e1_conv"), name="g_bn_e1")
            # 16*16*128
            e2 = self.instance_norm(self.conv2d(self.lrelu(e1), gf_dim * 2, ksize=4, stride=2, weight_decay=weight_decay, stddev=0.02, name="g_e2_conv"), name="g_bn_e2")
            # 8*8*256
            e3 = self.instance_norm(self.conv2d(self.lrelu(e2), gf_dim * 4, ksize=4, stride=2, weight_decay=weight_decay, stddev=0.02, name="g_e3_conv"), name="g_bn_e3")
            # 4*4*512
            e4 = self.instance_norm(self.conv2d(self.lrelu(e3), gf_dim * 8, ksize=4, stride=2, weight_decay=weight_decay, stddev=0.02, name="g_e4_conv"), name="g_bn_e4")
            e5 = tf.contrib.layers.flatten(e4, 'flatten')

            tz = e4.get_shape().as_list()[0:3] + [z_dim]
            z = tf.random_uniform(shape=tz, minval=-1, maxval=1, dtype=tf.float32, name="noise")
            ez = tf.concat([e4, z], axis=-1)

            en = self.instance_norm(self.conv2d(self.lrelu(ez), gf_dim * 8, ksize=1, stride=1, weight_decay=weight_decay, stddev=0.02, name="g_e5_conv"), name="g_bn_e5")

            d1 = self.deconv2d(self.relu(en), gf_dim * 8, ksize=4, stride=2, weight_decay=weight_decay, stddev=0.02, name="g_d1")
            d1 = self.dropout(d1, keep_prob=keep_prob)
            d1 = self.concat([self.instance_norm(d1, name="g_bn_d1"), e3])

            d2 = self.deconv2d(self.relu(d1), gf_dim * 4, ksize=4, stride=2, weight_decay=weight_decay, stddev=0.02,name="g_d2")
            d2 = self.concat([self.instance_norm(d2, name="g_bn_d2"), e2])

            d3 = self.deconv2d(self.relu(d2), gf_dim * 2, ksize=4, stride=2, weight_decay=weight_decay, stddev=0.02,name="g_d3")
            d3 = self.concat([self.instance_norm(d3, name="g_bn_d3"), e1])

            d4 = self.deconv2d(self.relu(d3), 1, ksize=4, stride=2, weight_decay=weight_decay, stddev=0.02, name="g_d4")
            d4 = tf.nn.tanh(d4)
            return e5, d4

    def Discriminator(self, input_, df_dim, weight_decay=0.00004, reuse=False, name="discriminator"):
        with tf.variable_scope(name, reuse=reuse):
            # 32*32*df_dim, 4
            h0 = self.lrelu(self.conv2d(input_, df_dim, ksize=4, stride=2, weight_decay=weight_decay, stddev=0.02, padding="SAME", name="d_h0_conv"))
            # 16*16*df_dim*2, 10
            h1 = self.lrelu(self.instance_norm(self.conv2d(h0, df_dim * 2, ksize=4, stride=2, weight_decay=weight_decay, stddev=0.02, padding="SAME", name="d_h1_conv"), name="d_bn1"))
            # 8*8*df_dim*4, 22
            h2 = self.lrelu(self.instance_norm(self.conv2d(h1, df_dim * 4, ksize=4, stride=2, weight_decay=weight_decay, stddev=0.02, padding="SAME", name="d_h2_conv"), name="d_bn2"))
            # 8*8*df_dim*8, 46
            h3 = self.lrelu(self.instance_norm(self.conv2d(h2, df_dim * 8, ksize=4, stride=1, weight_decay=weight_decay, stddev=0.02, padding="SAME", name="d_h3_conv"), name="d_bn3"))
            # 8*8*1, 70
            h4 = self.conv2d(h3, 1, ksize=4, stride=1, weight_decay=weight_decay, stddev=0.02, padding="SAME", name="d_h4_pred")
            return h4

    def PoseClassifier(self, input_, pf_dim, weight_decay=0.00004, reuse=False, training=True, name="pose_classifier"):
        with tf.variable_scope(name, reuse=reuse):
            # 32*32
            c0 = self.relu(self.conv2d(input_, pf_dim, ksize=4, stride=2, weight_decay=weight_decay, stddev=0.02, padding="SAME", name="p_c0_conv"))
            # 16*16
            c1 = self.lrelu(self.batch_norm(self.conv2d(c0, pf_dim*2, ksize=4, stride=2, weight_decay=weight_decay, stddev=0.02, padding="SAME", name="p_c1_conv"), training=training, name="p_bn1"))
            # 8*8
            c2 = self.lrelu(self.batch_norm(self.conv2d(c1, pf_dim*4, ksize=4, stride=2, weight_decay=weight_decay, stddev=0.02, padding="SAME", name="p_c2_conv"), training=training, name="p_bn2"))
            # 4*4
            c3 = self.lrelu(self.batch_norm(self.conv2d(c2, pf_dim*8, ksize=4, stride=2, weight_decay=weight_decay, stddev=0.02, padding="SAME", name="p_c3_conv"), training=training, name="p_bn3"))
            # 1*1
            c4 = self.conv2d(c3, 1, ksize=4, stride=4, weight_decay=weight_decay, stddev=0.02, padding="SAME", name="p_c4_conv")
            return tf.squeeze(c4, axis=[1, 2, 3])

    def IdentityClassifier(self, input_, if_dim, num_class, weight_decay=0.00004, reuse=False, training=True, name="identity_classifier"):
        with tf.variable_scope(name, reuse=reuse):
            # 32*32
            c0 = self.relu(self.conv2d(input_, if_dim, ksize=4, stride=2, weight_decay=weight_decay, stddev=0.02, padding="SAME", name="id_c0_conv"))
            # 16*16
            c1 = self.lrelu(self.batch_norm(self.conv2d(c0, if_dim * 2, ksize=4, stride=2, weight_decay=weight_decay, stddev=0.02, padding="SAME", name="id_c1_conv"), training=training, name="id_bn1"))
            # 8*8
            c2 = self.lrelu(self.batch_norm(self.conv2d(c1, if_dim * 4, ksize=4, stride=2, weight_decay=weight_decay, stddev=0.02, padding="SAME", name="id_c2_conv"), training=training, name="id_bn2"))
            # 4*4
            c3 = self.lrelu(self.batch_norm(self.conv2d(c2, if_dim * 8, ksize=4, stride=2, weight_decay=weight_decay, stddev=0.02, padding="SAME", name="id_c3_conv"), training=training, name="id_bn3"))
            # 1*1
            c4 = self.conv2d(c3, 128, ksize=4, stride=4, weight_decay=weight_decay, stddev=0.02, padding="SAME",name="id_c4_conv")

            # fc
            fc = self.fc(tf.squeeze(c4, axis=[1, 2]), num_class, weight_decay=weight_decay, name="fc")
            # normalized embedding
            embedding = tf.nn.l2_normalize(tf.squeeze(c4, axis=[1, 2]), dim=1, name='norm_embedding')
            return embedding, fc

    def sce_loss(self, logits, labels):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

    def mae_loss(self, in_, target):
        return tf.reduce_mean((in_-target)**2)

    def abs_loss(self, in_, target):
        return tf.reduce_mean(tf.abs(in_ - target))

    def ssce_loss(self, logits, labels):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

    def contrastive_loss(self, fake, real, margin, labels):
        # return tf.contrib.losses.metric_learning.contrastive_loss(labels, fake, real, margin)
        #labels = tf.cast(tf.argmax(labels, axis=1), dtype=tf.float32)
        distance = tf.reduce_sum(tf.square(fake - real), 1)
        sqrt_dist = tf.sqrt(distance)
        loss = (1. - tf.to_float(labels)) * tf.square(tf.maximum(0., margin - sqrt_dist)) + tf.to_float(labels) * distance
        return 0.5 * tf.reduce_mean(loss)




