import matplotlib.pyplot as plt
import PIL.Image as Image
from utils import *
import random

def read_one_image(img_path, fine_size):
    # color: "RGB", "L"
    img = Image.open(img_path).convert('L')
    img = np.asarray(img.resize(fine_size), dtype="float")
    img = transform(img)
    if len(img.shape) == 2:
        img = np.expand_dims(img, -1)
    return img


class casiaTrainGenerator(object):
    def __init__(self, dataset_dir, batch_size, img_size):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.states = ['nm-01', 'nm-02', 'nm-03', 'nm-04', 'nm-05', 'nm-06', 'bg-01', 'bg-02', 'cl-01', 'cl-02']
        # self.states = ['nm-01', 'nm-02', 'nm-03', 'nm-04', 'nm-05', 'nm-06']
        self.angles = ['000', '018', '036', '054', '072', '090', '108', '126', '144', '162', '180']
        self.n_id = 62

        self.n_st = len(self.states)
        self.n_ag = len(self.angles)

    # get batch size samples for stage 1 training
    def get_batch_stage1(self):
        batch_imgs = []
        batch_ancs = []
        batch_names = []
        batch_ag = []
        pos_spl_index = random.sample(range(0, self.batch_size), self.batch_size // 2)
        for n in range(self.batch_size):
            # source sample
            while True:
                src_id = np.random.randint(0, self.n_id, 1).item() + 1
                src_st = np.random.randint(0, self.n_st, 1).item()
                src_ag = np.random.randint(0, self.n_ag, 1).item()
                src_spl = '%03d' % src_id + '-' + self.states[src_st] + '-' + self.angles[src_ag] + '.png'
                src_dir = os.path.join(self.dataset_dir, '%03d' % src_id, self.states[src_st], src_spl)
                if os.path.exists(src_dir):
                    break

            # target sample
            while True:
                dst_id = src_id
                dst_st = np.random.randint(0, self.n_st - 4, 1).item()
                dst_spl = '%03d' % dst_id + '-' + self.states[dst_st] + '-090.png'
                dst_dir = os.path.join(self.dataset_dir, '%03d' % dst_id, self.states[dst_st], dst_spl)
                if os.path.exists(dst_dir):
                    break

            # anchor sample
            while True:
                if n in pos_spl_index:
                    anc_id = src_id
                    label = 1
                else:
                    anc_id = np.random.randint(0, self.n_id, 1).item() + 1
                    if anc_id == src_id:
                        continue
                    label = 0
                anc_st = np.random.randint(0, self.n_st, 1).item()
                anc_ag = np.random.randint(0, self.n_ag, 1).item()
                anc_spl = '%03d' % anc_id + '-' + self.states[anc_st] + '-' + self.angles[anc_ag] + '.png'
                anc_dir = os.path.join(self.dataset_dir, '%03d' % anc_id, self.states[anc_st], anc_spl)
                if os.path.exists(anc_dir) and anc_spl != src_spl:
                    break

            # anchor target
            while True:
                neg_id = anc_id
                neg_st = np.random.randint(0, self.n_st, 1).item()
                neg_ag = np.random.randint(0, self.n_ag, 1).item()
                neg_spl = '%03d' % neg_id + '-' + self.states[neg_st] + '-' + self.angles[neg_ag] + '.png'
                neg_dir = os.path.join(self.dataset_dir, '%03d' % neg_id, self.states[neg_st], neg_spl)
                if os.path.exists(neg_dir):
                    break

            src_img = read_one_image(src_dir, self.img_size)
            dst_img = read_one_image(dst_dir, self.img_size)
            anc_img = read_one_image(anc_dir, self.img_size)
            neg_img = read_one_image(neg_dir, self.img_size)

            img_merge_src = np.concatenate([src_img, dst_img], axis=-1)
            img_merge_anc = np.concatenate([anc_img, neg_img], axis=-1)
            batch_imgs.append(img_merge_src)
            batch_ancs.append(img_merge_anc)
            batch_names.append(label)
            batch_ag.append(src_ag == self.angles[src_ag])

        return np.stack(batch_imgs), np.stack(batch_ancs), batch_names, np.asarray(batch_ag, dtype=np.float32)

    # get batch size samples for stage 1 training
    def get_batch_stage2(self):
        batch_imgs = []
        batch_ancs = []
        batch_names = []
        batch_ag = []
        pos_spl_index = random.sample(range(0, self.batch_size), self.batch_size//2)
        for n in range(self.batch_size):
            # source sample
            while True:
                src_id = np.random.randint(0, self.n_id, 1).item() + 1
                src_st = np.random.randint(0, self.n_st, 1).item()
                src_ag = np.random.randint(0, self.n_ag, 1).item()
                src_spl = '%03d'%src_id + '-' + self.states[src_st] + '-' + self.angles[src_ag] + '.png'
                src_dir = os.path.join(self.dataset_dir, '%03d'%src_id, self.states[src_st], src_spl)
                if os.path.exists(src_dir):
                    break

            # target sample
            while True:
                dst_id = src_id
                if self.states[src_st][:2] == "nm":
                    dst_st = np.random.randint(0, self.n_st-4, 1).item()
                elif self.states[src_st][:2] == "cl":
                    dst_st = np.random.randint(0, 2, 1).item() + 6
                else:
                    dst_st = np.random.randint(0, 2, 1).item() + 8
                dst_spl = '%03d'%dst_id + '-' + self.states[dst_st] + '-090.png'
                dst_dir = os.path.join(self.dataset_dir, '%03d'%dst_id, self.states[dst_st], dst_spl)
                if os.path.exists(dst_dir):
                    break

            # anchor sample
            while True:
                if n in pos_spl_index:
                    anc_id = src_id
                    label = 1.
                else:
                    anc_id = np.random.randint(0, self.n_id, 1).item() + 1
                    if anc_id == src_id:
                        continue
                    label = 0.
                anc_st = np.random.randint(0, self.n_st, 1).item()
                anc_ag = np.random.randint(0, self.n_ag, 1).item()
                anc_spl = '%03d' % anc_id + '-' + self.states[anc_st] + '-' + self.angles[anc_ag] + '.png'
                anc_dir = os.path.join(self.dataset_dir, '%03d' % anc_id, self.states[anc_st], anc_spl)
                if os.path.exists(anc_dir) and anc_spl != src_spl:
                    break

            # anchor target
            while True:
                neg_id = anc_id
                neg_st = np.random.randint(0, self.n_st, 1).item()
                neg_ag = np.random.randint(0, self.n_ag, 1).item()
                neg_spl = '%03d' % neg_id + '-' + self.states[neg_st] + '-' + self.angles[neg_ag] + '.png'
                neg_dir = os.path.join(self.dataset_dir, '%03d' % neg_id, self.states[neg_st], neg_spl)
                if os.path.exists(neg_dir):
                    break

            src_img = read_one_image(src_dir, self.img_size)
            dst_img = read_one_image(dst_dir, self.img_size)
            anc_img = read_one_image(anc_dir, self.img_size)
            neg_img = read_one_image(neg_dir, self.img_size)

            img_merge_src = np.concatenate([src_img, dst_img], axis=-1)
            img_merge_anc = np.concatenate([anc_img, neg_img], axis=-1)
            batch_imgs.append(img_merge_src)
            batch_ancs.append(img_merge_anc)
            batch_names.append(label)
            batch_ag.append(src_ag == self.angles[src_ag])

        return np.stack(batch_imgs), np.stack(batch_ancs), batch_names, np.asarray(batch_ag, dtype=np.float32)


class casiaValGenerator(object):
    def __init__(self, dataset_dir, batch_size, img_size):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.states = ['nm-01', 'nm-02', 'nm-03', 'nm-04', 'nm-05', 'nm-06', 'bg-01', 'bg-02', 'cl-01', 'cl-02']
        # self.states = ['nm-01', 'nm-02', 'nm-03', 'nm-04', 'nm-05', 'nm-06']
        self.angles = ['000', '018', '036', '054', '072', '090', '108', '126', '144', '162', '180']
        self.n_id = 62

        self.n_st = len(self.states)
        self.n_ag = len(self.angles)

    def get_batch(self):
        batch_imgs = []
        for n in range(self.batch_size):
            # source sample
            while True:
                src_id = np.random.randint(0, self.n_id, 1).item() + self.n_id + 1
                src_st = np.random.randint(0, self.n_st, 1).item()
                src_ag = np.random.randint(0, self.n_ag, 1).item()
                src_spl = '%03d'%src_id + '-' + self.states[src_st] + '-' + self.angles[src_ag] + '.png'
                src_dir = os.path.join(self.dataset_dir, '%03d'%src_id, self.states[src_st], src_spl)
                if os.path.exists(src_dir):
                    break

            # target sample
            while True:
                dst_id = src_id
                dst_st = np.random.randint(0, self.n_st-4, 1).item()
                dst_spl = '%03d'%dst_id + '-' + self.states[dst_st] + '-090.png'
                dst_dir = os.path.join(self.dataset_dir, '%03d'%dst_id, self.states[dst_st], dst_spl)
                if os.path.exists(dst_dir):
                    break

            src_img = read_one_image(src_dir, self.img_size)
            dst_img = read_one_image(dst_dir, self.img_size)

            img_merge = np.concatenate([src_img, dst_img], axis=-1)
            batch_imgs.append(img_merge)

        return np.stack(batch_imgs)


class casiaTestGenerator(object):
    def __init__(self, dataset_dir, img_size):
        self.dataset_dir = dataset_dir
        self.img_size = img_size
        self.states = ['nm-01', 'nm-02', 'nm-03', 'nm-04', 'nm-05', 'nm-06', 'bg-01', 'bg-02', 'cl-01', 'cl-02']
        # self.states = ['nm-01', 'nm-02', 'nm-03', 'nm-04', 'nm-05', 'nm-06']
        self.angles = ['000', '018', '036', '054', '072', '090', '108', '126', '144', '162', '180']
        self.n_id = 124

        self.n_st = len(self.states)
        self.n_ag = len(self.angles)

    def get_batch(self, id, cond, angle):
        src_id = id
        src_st = cond
        dst_ag = '090'
        src_ag = self.angles[angle]

        dst_spl = '%03d'%id + '-' + self.states[src_st] + '-' + dst_ag + '.png'
        dst_dir = os.path.join(self.dataset_dir, '%03d'%src_id, self.states[src_st], dst_spl)
        dst_img = read_one_image(dst_dir, self.img_size)

        batch_imgs = []
        batch_names = []

        src_spl = '%03d'%src_id + '-' + self.states[src_st] + '-' + src_ag + '.png'
        src_dir = os.path.join(self.dataset_dir, '%03d'%src_id, self.states[src_st], src_spl)
        if not os.path.exists(src_dir):
            return -1
        src_img = read_one_image(src_dir, self.img_size)
        img_merge = np.concatenate([src_img, dst_img], axis=-1)
        batch_imgs.append(img_merge)
        batch_names.append(src_spl)
        return np.stack(batch_imgs, axis=0), batch_names


class visualizeBatch(object):
    def __init__(self, cols_to_display):
        self.cols_to_display = cols_to_display

    def plot_batch(self, batch, idx):
        batch_size = batch.shape[0]
        rows = int(batch_size//self.cols_to_display) + 1
        for i in range(batch_size):
            plt.suptitle('Batch: ' + str(idx))
            plt.subplots_adjust(hspace=0.8, wspace=0.1, top=0.8)
            plt.subplot(rows, self.cols_to_display*2, 2*i+1)
            plt.axis('off')
            plt.imshow(inverse_transform(np.squeeze(batch[i, :, :, :1])), cmap=plt.cm.gray)
            plt.subplot(rows, self.cols_to_display * 2, 2 * i + 2)
            plt.axis('off')
            plt.imshow(inverse_transform(np.squeeze(batch[i, :, :, 1:])), cmap=plt.cm.gray)
        plt.show()

    def save_batch(self, batch):
        pass


# visualize batch of images

if __name__ == "__main__":
    dataset_dir = '../gaitGAN/gei'
    batch_size = 30
    img_size = 64
    batches_per_epoch = 100
    dataGenerator = casiaTrainGenerator(dataset_dir, batch_size, img_size)
    visHandler = visualizeBatch(cols_to_display=5)

    for i in range(batches_per_epoch):
        print(i)
        batch = dataGenerator.get_batch_stage1()
        visHandler.plot_batch(batch, i)





