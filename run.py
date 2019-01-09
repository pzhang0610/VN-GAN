import argparse
import os
import tensorflow as tf
from geiGAN import GeiGAN
from mylog import myLog

os.environ["CUDA_VISIBLE_DEVICES"]="0"
parser = argparse.ArgumentParser(description="")

parser.add_argument('--dataset_dir', dest="dataset_dir", default="../data/gei", help="path to dataset")
parser.add_argument('--img_height', dest="img_height", default=64, help="image height")
parser.add_argument('--img_width', dest="img_width", default=64, help="img width")
parser.add_argument('--num_class', dest="num_class", default=62, help="number of classes in stage 2")

parser.add_argument('--batch_size', dest="batch_size", default=100, help="number of images in a batch")
parser.add_argument('--epoch', dest="epoch", type=int, default=5, help="number of epochs")
parser.add_argument('--batch_per_epoch', dest="batch_per_epoch", default=1000, help="number of batches in a epoch")
parser.add_argument('--num_reverse', dest="num_reverse", default=1, help="Training number of discriminator for each generator")

# Trade-off parameters
parser.add_argument('--alpha', dest="alpha", type=float, default=5, help="trade-off parameter for l1 norm")
parser.add_argument('--beta', dest="beta", type=float, default=5, help="trade-off parameter for pose loss")
parser.add_argument('--gamma', dest="gamma", type=float, default=10, help="trade-off parameter for classification loss in stage 2")

parser.add_argument('--lr', dest='lr', default=0.0002, help="learning rate")
parser.add_argument('--beta1', dest="beta1", default=0.5, help="beta1 for the AdamOptimizer suggested by DCGAN")
parser.add_argument('--keep_prob', dest="keep_prob", default=0.5, help="keep probability for dropout")
parser.add_argument('--margin', dest='margin', default=1.0, help="margin for the contrastive loss")
# label smooth and noise input
parser.add_argument('--label_smooth', dest="label_smooth", type=float, default=0, help="stddev for label smooth, default is 0.15")
parser.add_argument('--stddev', dest='stddev', type=float, default=0.5, help="stddev for noise input")

parser.add_argument('--gf_dim', dest='gf_dim', type=int, default=64, help="channels for generator")
parser.add_argument("--df_dim", dest="df_dim", type=int, default=64, help="channels for discriminator")
parser.add_argument("--pf_dim", dest="pf_dim", type=int, default=64, help="channels for pose classifier")
parser.add_argument("--if_dim", dest="if_dim", type=int, default=64, help="channels for identity classifier")
parser.add_argument("--z_dim", dest="z_dim", type=int, default=64, help="channels for noise input")

parser.add_argument("--is_train", dest="is_train", action=store_true, help="if True: running train, else: running test")
# paths to store logs and models
parser.add_argument("--logs", dest="logs", default="./logs/20180920", help="path to store summaries")
parser.add_argument("--checkpoint_dir", dest="checkpoint_dir", default="./checkpoint/20180920", help="path to store the model")
parser.add_argument("--mylog", dest="mylog", default="./mylog/20180920", help="path to store running logs")
parser.add_argument("--sample_dir", dest="sample_dir", default="./sample/20180920", help="path to store generated samples")
parser.add_argument("--generated_test_dir", dest="generated_test_dir", default="./test_imgs/20180920", help="path to store generated samples")
parser.add_argument("--is_print", dest="is_print", type=bool, default=False, help="print the running logs or not.")
parser.add_argument("--display_freq", dest="display_freq", type=int, default=100, help="frequency to sample")

parser.add_argument("--sample_per_column", dest="sample_per_column", type=int, default=10, help="columns to store sample")

args = parser.parse_args()


def main(_):
    print(args.is_train)
    # make dirs if not exist
    if not os.path.exists(args.logs):
        os.makedirs(args.logs)
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.mylog):
        os.makedirs(args.mylog)
    if not os.path.exists(args.sample_dir):
        os.makedirs(os.path.join(args.sample_dir, 'stage_1'))
        os.makedirs(os.path.join(args.sample_dir, 'stage_2'))



    if not os.path.exists(args.generated_test_dir):
        os.makedirs(os.path.join(args.generated_test_dir, 'stage_1'))
        os.makedirs(os.path.join(args.generated_test_dir, 'stage_2'))
        os.makedirs(os.path.join(args.generated_test_dir, 'id_fea'))

    if args.is_train:
        logger = myLog(log_base_dir=os.path.join(args.mylog, "train"))
        logger.add_log(args, is_print=True)
    else:
        logger = myLog(log_base_dir=os.path.join(args.mylog, "test"))
        logger.add_log(args, is_print=True)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        model = GeiGAN(sess, args, logger)
        if args.is_train:
            model.train(args)
        else:
            print('--------------------testing---------------')
            model.test(args)


if __name__ == "__main__":
    tf.app.run()

