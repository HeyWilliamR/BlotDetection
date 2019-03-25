import tensorflow as tf
import numpy as np
import os
from ConstValue.global_variable import *


def get_file(file_dir):
    images = []
    temp = []
    for root, sub_folders, files in os.walk(file_dir):
        for name in files:
            images.append(os.path.join(root, name))
        for name in sub_folders:
            temp.append(os.path.join(root, name))
    labels = []
    for one_folder in temp:
        n_img = len(os.listdir(one_folder))
        letter = one_folder.split("\\")[-1]
        if letter == "blot":
            labels = np.append(labels, n_img * [1])
        elif letter == "nut":
            labels = np.append(labels, n_img * [2])
        else:
            labels = np.append(labels, n_img * [0])
    # shuffle
    temp = np.array([images, labels])
    temp = temp.transpose()
    np.random.shuffle(temp)
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(float(i)) for i in label_list]
    return image_list, label_list


def get_batch(image_list, label_list, img_width, img_height, batch_size, epoch_num, capcity):
    image = tf.cast(image_list, tf.string)
    label = tf.cast(label_list, tf.int32)
    input_queue = tf.train.slice_input_producer([image, label],shuffle=False, num_epochs=epoch_num)
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image, img_width, img_height)
    image = tf.image.per_image_standardization(image)
    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=4, capacity=capcity)
    label_batch = tf.reshape(label_batch, [batch_size])
    return image_batch, label_batch

#K折交叉验证数据集合划分
def split_dataset(imagelist,labellist,flod):
    test_size = int(len(imagelist) / FOLD_VALUE)
    test_begin_index = int(len(imagelist) * (flod / FOLD_VALUE))
    test_end_index = test_begin_index + test_size
    test_img_set = imagelist[test_begin_index:test_end_index]
    test_label_set = labellist[test_begin_index:test_end_index]
    train_img_set = imagelist[0:test_begin_index] + imagelist[test_end_index:]
    train_label_set = imagelist[0:test_begin_index] + imagelist[test_end_index:]
    return train_img_set, train_label_set, test_img_set, test_label_set


def one_hot(labels):
    n_sample = len(labels)
    n_class = max(labels) + 1
    onehot_labels = np.zeros((n_sample, n_class))
    onehot_labels[np.arange(n_sample), labels] = 1
    return onehot_labels

