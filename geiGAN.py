import sys
sys.path.append('../')
import tensorflow as tf
from module import Modules
from datetime import datetime
import time
from utils import *
from dataset import casiaTrainGenerator, casiaValGenerator, casiaTestGenerator


class GeiGAN(object):
    def __init__(self, sess, args, logger):
        self.sess = sess
        self.logger = logger
        # define input placeholders
        self.real = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, args.img_height, args.img_width, 2], name="real_images")
        self.pair = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, args.img_height, args.img_width, 2], name="anchor_images")
        self.id_labels = tf.placeholder(dtype=tf.int32, shape=[args.batch_size], name="labels")
        self.p_labels = tf.placeholder(dtype=tf.float32, shape=[args.batch_size], name='p_labels')
        self.is_train = tf.placeholder(dtype=tf.bool, shape=[], name="is_train")
        self.stddev = tf.placeholder(dtype=tf.float32, shape=[], name="stddev")
        # initialize blocks
        self.model = Modules(self.logger, is_print=args.is_print)
        self.G = self.model.G_Unet
        self.D = self.model.Discriminator
        self.P = self.model.PoseClassifier
        self.ID = self.model.IdentityClassifier
        self.build_contrastive_loss(args)

    def build_contrastive_loss(self, args):
        self.logger.add_log("Building the graph...", is_print=args.is_print)
        self.real_src = self.real[:, :, :, :1]
        self.real_dst = self.real[:, :, :, 1:]

        self.pair_src = self.pair[:, :, :, :1]
        self.pair_dst = self.pair[:, :, :, 1:]

        # ############################### Stage 1####################################
        # G1
        self.logger.add_log("Generating FAKE Target from Source...", is_print=args.is_print)
        self.fake_dst = self.G(self.real_src, args.gf_dim, args.z_dim, keep_prob=args.keep_prob, weight_decay=0.00004,
                               is_training=self.is_train, reuse=False, name="generator_1")
        self.fake_dst_pair = self.G(self.pair_src, args.gf_dim, args.z_dim, keep_prob=args.keep_prob, weight_decay=0.00004,
                               is_training=self.is_train, reuse=True, name="generator_1")

        # D1--fake
        self.logger.add_log("Building discriminator for generated fake image...", is_print=args.is_print)
        self.DB_fake = self.D(self.fake_dst + tf.truncated_normal(shape=self.fake_dst.shape, stddev=self.stddev),
                              args.df_dim, weight_decay=0.00004, reuse=False, name="discriminator_1")

        # D1--real
        self.logger.add_log("Building discriminator for real target image...", is_print=args.is_print)
        self.DB_real = self.D(self.real_dst + tf.truncated_normal(shape=self.real_dst.shape, stddev=self.stddev),
                              args.df_dim, weight_decay=0.00004, reuse=True, name="discriminator_1")

        # # P --[fake, src] 0
        # fake_src = tf.concat([self.fake_dst, self.real_src], axis=-1, name="fake_src")
        # self.PB_fake_src = self.P(fake_src, args.pf_dim, weight_decay=0.00004, reuse=False, training=self.is_train, name="pose_classifier")
        # # P --[fake, dst] 1
        # fake_dst = tf.concat([self.fake_dst, self.real_dst], axis=-1, name="fake_dst")
        # self.PB_fake_dst = self.P(fake_dst, args.pf_dim, weight_decay=0.00004, reuse=True, training=self.is_train, name="pose_classifier")
        # # P --[dst, src] 0
        # dst_src = tf.concat([self.real_dst, self.real_src], axis=-1, name="dst_src")
        # self.PB_dst_src = self.P(dst_src, args.pf_dim, weight_decay=0.00004, reuse=True, training=self.is_train, name="pose_classifier")

        # Pose
        self.PB_fake = self.P(self.fake_dst, args.pf_dim, weight_decay=0.00004, reuse=False, training=self.is_train, name="pose_classifier")
        self.PB_src = self.P(self.real_src, args.pf_dim, weight_decay=0.00004, reuse=True, training=self.is_train, name="pose_classifier")
        self.PB_dst = self.P(self.real_dst, args.pf_dim, weight_decay=0.00004, reuse=True, training=self.is_train, name="pose_classifier")
        # G1 losses
        self.g1_loss = self.model.sce_loss(self.DB_fake, labels=tf.ones_like(self.DB_fake))

        # D1 loss
        self.d1_loss_real = self.model.sce_loss(self.DB_real, labels=tf.ones_like(self.DB_real) - tf.truncated_normal(shape=self.DB_real.shape, mean=0, stddev=args.label_smooth))
        self.d1_loss_fake = self.model.sce_loss(self.DB_fake, labels=tf.zeros_like(self.DB_fake) + tf.abs(tf.truncated_normal(shape=self.DB_fake.shape, mean=0, stddev=args.label_smooth)))

        # l1 loss for G1
        self.g1_l1_loss = self.model.abs_loss(self.real_dst, self.fake_dst)

        # P losses
        # self.p_loss_fake_dst = self.model.sce_loss(self.PB_fake_dst, labels=tf.ones(shape=[args.batch_size]))
        # self.p_loss_fake_src = self.model.sce_loss(self.PB_fake_src, labels=tf.zeros(shape=[args.batch_size]))
        # self.p_loss_dst_src  = self.model.sce_loss(self.PB_dst_src, labels=tf.zeros(shape=[args.batch_size]))
        self.p_loss_fake = self.model.sce_loss(self.PB_fake, labels=tf.ones(shape=[args.batch_size]))
        # self.p_loss_src  = self.model.sce_loss(self.PB_src, labels=tf.zeros(shape=[args.batch_size]))
        self.p_loss_src  = self.model.sce_loss(self.PB_src, labels=self.p_labels)
        self.p_loss_dst  = self.model.sce_loss(self.PB_dst, labels=tf.ones(shape=[args.batch_size]))

        # regularization loss
        reg_loss_g1 = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope="generator_1")
        reg_loss_d1 = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope="discriminator_1")
        reg_loss_p1 = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope="pose_classifier")

        # G1 los
        self.gen_1_loss = tf.add_n([self.g1_loss + args.alpha * self.g1_l1_loss + args.beta*self.p_loss_fake]+reg_loss_g1)
        # D1 loss
        self.dis_1_loss = tf.add_n([self.d1_loss_real+self.d1_loss_fake]+reg_loss_d1)
        # P loss
        # self.pos_loss = tf.add_n([self.p_loss_src+self.p_loss_dst+self.p_loss_fake]+reg_loss_p1)
        self.pos_loss = tf.add_n([self.p_loss_src + self.p_loss_dst] + reg_loss_p1)

        #  ################################ Stage 2####################################
        # When performing stage II, stop the gradient to stage 1
        self.logger.add_log("Stopping gradient...", is_print=args.is_print)
        self.fake_dst_ = tf.stop_gradient(self.fake_dst, name="barrier")
        self.fake_dst_pair_ = tf.stop_gradient(self.fake_dst_pair, name="barrier1")
        self.g2_input = tf.concat([self.real_src, self.fake_dst_], axis=-1)
        self.g2_input_pair = tf.concat([self.pair_src, self.fake_dst_pair_], axis=-1)

        # G2
        self.logger.add_log("Refining FAKE target from stage 1...", is_print=args.is_print)
        # noise
        self.fake_dst_diff = self.G(self.g2_input, args.z_dim, args.gf_dim, keep_prob=args.keep_prob, weight_decay=0.00004,
                                      is_training=self.is_train, reuse=False, name="generator_2")
        self.fake_dst_pair_diff = self.G(self.g2_input_pair, args.z_dim, args.gf_dim, keep_prob=args.keep_prob, weight_decay=0.00004,
                                      is_training=self.is_train, reuse=True, name="generator_2")
        self.fake_dst_refine = self.fake_dst_ + self.fake_dst_diff
        self.fake_dst_pair_refine = self.fake_dst_pair_ + self.fake_dst_pair_diff
        # D2--fake
        self.logger.add_log("Building discriminator for generated fake image...", is_print=args.is_print)
        fake_pair = tf.concat([self.real_src, self.fake_dst_refine], axis=-1)
        self.g2_DB_fake = self.D(fake_pair + tf.truncated_normal(shape=fake_pair.shape, stddev=self.stddev),
                                 args.df_dim, weight_decay=0.00004, reuse=False, name="discriminator_2")

        # D2--real
        self.logger.add_log("Building discriminator for real image...", is_print=args.is_print)
        real_pair = tf.concat([self.real_src, self.real_dst], axis=-1)
        self.g2_DB_real = self.D(real_pair + tf.truncated_normal(shape=real_pair.shape, stddev=self.stddev),
                                 args.df_dim, weight_decay=0.00004, reuse=True, name="discriminator_2")

        # ID -- fake
        self.logger.add_log("Building identity classifier for fake image...", is_print=args.is_print)
        self.id_fea, self.id_logic_fake = self.ID(self.fake_dst_refine, args.if_dim, args.num_class, weight_decay=0.00004,
                                     reuse=False, training=self.is_train, name="id_classifier")
        self.id_fea_pair, self.id_logic_fake_pair = self.ID(self.fake_dst_pair_refine, args.if_dim, args.num_class, weight_decay=0.00004,
                                     reuse=True, training=self.is_train, name="id_classifier")

        # # ID -- real
        # self.logger.add_log("Building identity classifier for real image...", is_print=args.is_print)
        # _, self.id_logic_real = self.ID(self.real_dst, args.if_dim, args.num_class, weight_decay=0.00004,
        #                              reuse=True, training=self.is_train, name="id_classifier")

        # G2 loss
        self.g2_loss = self.model.sce_loss(self.g2_DB_fake, labels=tf.ones_like(self.g2_DB_fake))

        # D2 loss
        self.d2_loss_real = self.model.sce_loss(self.g2_DB_real, labels=tf.ones_like(self.g2_DB_real) - tf.truncated_normal(shape=self.g2_DB_real.shape, mean=0, stddev=args.label_smooth))
        self.d2_loss_fake = self.model.sce_loss(self.g2_DB_fake, labels=tf.zeros_like(self.g2_DB_fake) + tf.abs(tf.truncated_normal(shape=self.g2_DB_fake.shape, mean=0, stddev=args.label_smooth)))

        # ID loss
        # self.id_loss_real = self.model.ssce_loss(self.id_logic_real, labels=self.labels)
        # self.id_loss_fake = self.model.ssce_loss(self.id_logic_fake, labels=self.labels)
        self.id_loss_cons = self.model.contrastive_loss(self.id_fea, self.id_fea_pair, margin=args.margin, labels=self.id_labels)
        # l1 loss
        self.g2_l1_loss = self.model.abs_loss(real_pair, fake_pair)

        # regularization loss
        reg_loss_g2 = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope="generator_2")
        reg_loss_d2 = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope="discriminator_2")
        reg_loss_id = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope="id_classifier")

        # G2 losses
        self.gen_2_loss = tf.add_n([self.g2_loss + args.alpha * self.g2_l1_loss + args.gamma * self.id_loss_cons] + reg_loss_g2)
        # D2 losses
        self.dis_2_loss = tf.add_n([self.d2_loss_real + self.d2_loss_fake] + reg_loss_d2)
        # ID losses
        # self.id_loss = tf.add_n([self.id_loss_fake + self.id_loss_real] + reg_loss_id)
        self.id_loss = tf.add_n([self.id_loss_cons] + reg_loss_id)

        # Summary for stage 1
        g1_loss_summary = tf.summary.scalar('g1_loss', self.g1_loss)
        d1_loss_fake_summary = tf.summary.scalar('d1_loss_fake', self.d1_loss_fake)
        d1_loss_real_summary = tf.summary.scalar('d1_loss_real', self.d1_loss_real)
        g1_l1_loss_summary = tf.summary.scalar('g1_l1_loss', self.g1_l1_loss)
        gen_1_loss_summary = tf.summary.scalar('gen_1_loss', self.gen_1_loss)
        dis_1_loss_summary = tf.summary.scalar('dis_1_loss', self.dis_1_loss)
        p_loss_fake_dst_summary = tf.summary.scalar('p_loss_fake', self.p_loss_fake)
        p_loss_fake_src_summary = tf.summary.scalar('p_loss_src', self.p_loss_src)
        p_loss_dst_src_summary = tf.summary.scalar('p_loss_dst', self.p_loss_dst)
        pos_loss_summary = tf.summary.scalar('pos_loss', self.pos_loss)

        self.g1_loss_summary_op = tf.summary.merge([gen_1_loss_summary, g1_loss_summary, g1_l1_loss_summary])
        self.d1_loss_summary_op = tf.summary.merge([dis_1_loss_summary, d1_loss_real_summary, d1_loss_fake_summary])
        self.pos_loss_summary_op = tf.summary.merge([pos_loss_summary, p_loss_dst_src_summary, p_loss_fake_dst_summary, p_loss_fake_src_summary])

        # Summary for Stage 2
        g2_loss_summary = tf.summary.scalar('g2_loss', self.g1_loss)
        d2_loss_fake_summary = tf.summary.scalar('d2_loss_fake', self.d2_loss_fake)
        d2_loss_real_summary = tf.summary.scalar('d2_loss_real', self.d2_loss_real)
        g2_l1_loss_summary = tf.summary.scalar('g2_l1_loss', self.g2_l1_loss)
        gen_2_loss_summary = tf.summary.scalar('gen_2_loss', self.gen_2_loss)
        dis_2_loss_summary = tf.summary.scalar('dis_2_loss', self.dis_2_loss)
        # id_loss_real_summary = tf.summary.scalar('id_loss_real', self.id_loss_real)
        # id_loss_fake_summary = tf.summary.scalar('id_loss_fake', self.id_loss_fake)
        id_loss_summary= tf.summary.scalar('id_loss', self.id_loss)

        self.g2_loss_summary_op = tf.summary.merge([gen_2_loss_summary, g2_loss_summary, g2_l1_loss_summary])
        self.d2_loss_summary_op = tf.summary.merge([dis_2_loss_summary, d2_loss_real_summary, d2_loss_fake_summary])
        self.id_loss_summary_op = tf.summary.merge([id_loss_summary])

        # Training Variables
        t_vars = tf.trainable_variables()
        self.d1_vars = [var for var in t_vars if "discriminator_1" in var.name]
        self.g1_vars = [var for var in t_vars if "generator_1" in var.name]
        self.p_vars = [var for var in t_vars if "pose_classifier" in var.name]

        self.d2_vars = [var for var in t_vars if "discriminator_2" in var.name]
        self.g2_vars = [var for var in t_vars if "generator_2" in var.name]
        self.id_vars = [var for var in t_vars if "id_classifier" in var.name]

    def sample_model(self, args, epoch, step, stage):
        dataGenerator = casiaValGenerator(args.dataset_dir, batch_size=args.batch_size, img_size=(args.img_height, args.img_width))

        batch_imgs = dataGenerator.get_batch()
        if stage == 1:
            real_src, real_dst, fake_dst = self.sess.run([self.real_src, self.real_dst, self.fake_dst],
                                                         feed_dict={self.real:batch_imgs, self.is_train:False})
            sample_name = os.path.join(args.sample_dir, 'stage_1', 'sample_{:04d}_{:05d}.jpg'.format(epoch, step))
        else:
            real_src, real_dst, fake_dst = self.sess.run([self.real_src, self.real_dst, self.fake_dst_refine],
                                                         feed_dict={self.real: batch_imgs, self.is_train: False})
            sample_name = os.path.join(args.sample_dir, 'stage_2', 'sample_{:04d}_{:05d}.jpg'.format(epoch, step))
        save_imgs(sample_name, real_src, real_dst, fake_dst, args.sample_per_column)

    def train(self, args):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op_d = tf.train.AdamOptimizer(args.lr, beta1=args.beta1).minimize(self.dis_1_loss, var_list=self.d1_vars)
            train_op_g = tf.train.AdamOptimizer(args.lr, beta1=args.beta1).minimize(self.gen_1_loss, var_list=self.g1_vars)
            train_op_p = tf.train.AdamOptimizer(args.lr, beta1=args.beta1).minimize(self.pos_loss, var_list=self.p_vars)

            train_op_d2 = tf.train.AdamOptimizer(args.lr, beta1=args.beta1).minimize(self.dis_2_loss, var_list=self.d2_vars)
            train_op_g2 = tf.train.AdamOptimizer(args.lr, beta1=args.beta1).minimize(self.gen_2_loss, var_list=self.g2_vars)
            train_op_id = tf.train.AdamOptimizer(args.lr, beta1=args.beta1).minimize(self.id_loss, var_list=self.id_vars)

        init_op = tf.global_variables_initializer()

        self.writer = tf.summary.FileWriter(args.logs)
        self.saver = tf.train.Saver(max_to_keep=10)

        # initialize all variables
        self.sess.run(init_op)
        # if exist checkpoint, load it
        start_global_step = 1
        start_epoch=0
        ckpt = tf.train.get_checkpoint_state(args.checkpoint_dir)

        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            self.logger.add_log("Loading the checkpoint successfully...", is_print=True)
            start_epoch = int(os.path.split(ckpt.model_checkpoint_path)[1].split('_')[2].split('.ckpt')[0])
            start_global_step = int(os.path.split(ckpt.model_checkpoint_path)[1].split('-')[1])
            self.logger.add_log("Starting from epoch {:04d} global step {}".format(start_epoch+1, start_global_step))

        self.writer.add_graph(self.sess.graph)

        dataGenerator = casiaTrainGenerator(args.dataset_dir, batch_size=args.batch_size, img_size=(args.img_height, args.img_width))

        self.logger.add_log("{} Starting training G1...".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), is_print=True)
        counter = start_global_step

        if (start_epoch + 1) < 2*args.epoch:
            # ##########################################################################################################
            #                                             Stage 1                                                      #
            # ##########################################################################################################
            for epoch in range(start_epoch+1, args.epoch):
                stddev = args.stddev if epoch < 2 else args.stddev * (args.epoch-epoch)/(args.epoch-2)
                for step in range(args.batch_per_epoch):
                    batch_imgs, pair_imgs, labels,batch_ag = dataGenerator.get_batch_stage1()
                    start_time = time.clock()
                    # running discriminator
                    _, dis_1_loss, d1_loss_summary = self.sess.run([train_op_d, self.dis_1_loss, self.d1_loss_summary_op],
                                                                   feed_dict={self.real:batch_imgs,self.pair:pair_imgs, self.p_labels:batch_ag, self.is_train:True, self.stddev: stddev})
                    _, pos_loss, pos_loss_summary = self.sess.run([train_op_p, self.pos_loss, self.pos_loss_summary_op],
                                                                  feed_dict={self.real:batch_imgs, self.pair:pair_imgs, self.p_labels:batch_ag, self.is_train:True, self.stddev:stddev})
                    if (step + 1) % args.num_reverse == 0:
                        _, gen_1_loss, g1_loss_summary = self.sess.run([train_op_g, self.gen_1_loss, self.g1_loss_summary_op],
                                                                   feed_dict={self.real:batch_imgs, self.pair:pair_imgs, self.p_labels:batch_ag, self.is_train:True, self.stddev: stddev})
                        elapse = time.clock() - start_time
                        self.writer.add_summary(g1_loss_summary, counter)
                        self.logger.add_log(
                            "{} epoch: {:04d} batch:{:05d} gen_1_loss: {:.6f} dis_1_loss: {:.6f} elapse: {:.4f}"
                            .format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch, step, gen_1_loss, dis_1_loss,
                                    elapse), is_print=True)

                    self.writer.add_summary(d1_loss_summary, counter)
                    self.writer.add_summary(pos_loss_summary, counter)

                    if np.mod(counter, args.display_freq) == 0:
                        self.sample_model(args, epoch, step, 1)
                    counter += 1
                checkpoint_name = os.path.join(args.checkpoint_dir, 'model_epoch_' + str(epoch) + '.ckpt')
                self.saver.save(self.sess, checkpoint_name, global_step=counter-1)

                start_epoch = epoch
            self.logger.add_log("{} Training of G1 is done.".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), is_print=True)

            # #########################################################################################################
            #                                             Stage 2                                                     #
            # #########################################################################################################
            self.logger.add_log("{} Starting training G2...".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), is_print=True)
            for epoch in range(start_epoch+1, args.epoch*2):
                stddev = args.stddev if epoch < start_epoch + 2 else args.stddev * (args.epoch * 2 - epoch) / (args.epoch *2 - 2 - start_epoch)
                for step in range(args.batch_per_epoch):
                    batch_imgs, pair_imgs, labels, batch_ag = dataGenerator.get_batch_stage1()
                    start_time = time.clock()
                    #running discriminator
                    _, dis_2_loss, d2_loss_summary = self.sess.run([train_op_d2, self.dis_2_loss, self.d2_loss_summary_op],
                        feed_dict={self.real: batch_imgs, self.pair:pair_imgs, self.id_labels: labels, self.p_labels:batch_ag, self.is_train: True, self.stddev: stddev})

                    _, id_loss, id_loss_summary = self.sess.run([train_op_id, self.id_loss, self.id_loss_summary_op],
                                                                feed_dict={self.real: batch_imgs, self.pair: pair_imgs, self.id_labels:labels, self.p_labels:batch_ag, self.is_train: True, self.stddev: stddev})
                    if (step + 1)% args.num_reverse == 0:
                        _, gen_2_loss, g2_loss_summary = self.sess.run([train_op_g2, self.gen_2_loss, self.g2_loss_summary_op],
                                                                       feed_dict={self.real: batch_imgs, self.pair:pair_imgs, self.id_labels:labels, self.p_labels:batch_ag,self.is_train: True, self.stddev: stddev})
                        elapse = time.clock() - start_time
                        self.writer.add_summary(g2_loss_summary, counter)
                        self.logger.add_log(
                            "{} epoch: {:04d} batch:{:05d} gen_2_loss: {:.6f} dis_2_loss: {:.6f} elapse: {:.4f}"
                                .format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch, step, gen_2_loss,
                                        dis_2_loss,elapse), is_print=True)
                    self.writer.add_summary(d2_loss_summary, counter)
                    self.writer.add_summary(id_loss_summary, counter)
                    if np.mod(counter, args.display_freq) == 0:
                        self.sample_model(args, epoch, step, 2)
                    counter += 1
                checkpoint_name = os.path.join(args.checkpoint_dir,'model_epoch_' + str(epoch) + '.ckpt')
                self.saver.save(self.sess, checkpoint_name, global_step=counter-1)
            self.logger.add_log("{} Training of G2 is done.".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), is_print=True)


    def test(self, args):
        self.saver = tf.train.Saver(max_to_keep=5)
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        # ckpt = tf.train.get_checkpoint_state(args.checkpoint_dir)
        # if ckpt and ckpt.model_checkpoint_path:
        #     self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, './checkpoint/20180918/model_epoch_8.ckpt-8000')
        dataGenerator = casiaTestGenerator(args.dataset_dir, img_size=(args.img_height, args.img_width))
        for id in range(124):
            for cond in range(10):
                for ag in range(11):
                    batch = dataGenerator.get_batch(id+1, cond, ag)
                    if type(batch) is not tuple:
                        continue
                    batch_imgs, batch_names = batch
                    fake_dst = self.sess.run(self.fake_dst, feed_dict={self.real: batch_imgs, self.is_train: False})
                    fake_dst2, id_fea = self.sess.run([self.fake_dst_refine, self.id_fea], feed_dict={self.real:batch_imgs, self.is_train: False})
                    fake_dst_norm = inverse_transform(fake_dst)
                    fake_dst_norm2 = inverse_transform(fake_dst2)
                    num_spl = fake_dst_norm.shape[0]
                    for i in range(num_spl):
                        save_name = os.path.join(args.generated_test_dir,'stage_1', batch_names[i])
                        save_name2 = os.path.join(args.generated_test_dir, 'stage_2', batch_names[i])
                        save_name3 = os.path.join(args.generated_test_dir,'id_fea', batch_names[i][:-4]+'.npy')
                        imsave(save_name, np.squeeze(fake_dst_norm[i, :, :, :]))
                        imsave(save_name2, np.squeeze(fake_dst_norm2[i, :, :, :]))
                        np.save(save_name3, id_fea)

    def generate_sample(self, args):
        self.saver = tf.train.Saver(max_to_keep=5)
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.saver.restore(self.sess, './checkpoint/20180918/model_epoch_8.ckpt-8000')
        dataGenerator = casiaTestGenerator(args.dataset_dir, img_size=(args.img_height, args.img_width))
        id = 0
        cond = 0
        ag = 2
        batch = dataGenerator.get_batch(id + 1, cond, ag)
        batch_imgs, batch_names = batch
        fake_dst = self.sess.run(self.fake_dst, feed_dict={self.real: batch_imgs, self.is_train: False})
        fake_dst2, fake_dst2_diff = self.sess.run([self.fake_dst_refine, self.fake_dst_diff],
                                          feed_dict={self.real: batch_imgs, self.is_train: False})
        fake_dst_norm = inverse_transform(fake_dst)
        fake_dst_norm2 = inverse_transform(fake_dst2)
        fake_dst_diff = inverse_transform(fake_dst2_diff)
        save_name = os.path.join('./stage_1.jpg')
        save_name_ = os.path.join('./stage_2.jpg')
        save_name_diff = os.path.join('./stage_diff.jpg')
        imsave(save_name, np.squeeze(fake_dst_norm))
        imsave(save_name_, np.squeeze(fake_dst_norm2))
        imsave(save_name_diff, np.squeeze(fake_dst_diff))


