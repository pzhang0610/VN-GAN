import numpy as np
import os
from scipy.misc import imsave


def transform(img):
    return np.array(img)/127.5 - 1.


def inverse_transform(img_data):
    return (img_data + 1.0)/2.0


def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def save_imgs(img_name, real_src, real_dst, fake_dst, sample_per_column):
    real_src_norm = inverse_transform(real_src)
    real_dst_norm = inverse_transform(real_dst)
    fake_dst_norm = inverse_transform(fake_dst)
    N, H, W, C= real_src_norm.shape
    rows = int(np.ceil(N/sample_per_column))
    img = np.zeros((rows*H, 3*sample_per_column*W))
    for i in range(N):
        row = i//sample_per_column
        col = np.mod(i, sample_per_column)
        img[row*H:(row+1)*H, 3*col*W:(3*col+1)*W] = np.squeeze(real_src_norm[i, :, :])
        img[row*H:(row+1)*H, (3*col+1)*W:(3*col+2)*W] = np.squeeze(real_dst_norm[i, :, :])
        img[row*H:(row+1)*H, (3*col+2)*W:(3*col+3)*W] = np.squeeze(fake_dst_norm[i, :, :])
    imsave(img_name, img)





