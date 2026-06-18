#!/usr/bin/env python
# coding: utf-8

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from sklearn.model_selection import train_test_split

import numpy as np
import random
import os
import cv2


def delete_index(data, pixel):
    index = list(map(lambda x: x[0], filter(lambda x: x[1].sum() == 0, enumerate(pixel))))
    data  = data.drop(index, axis=0).reset_index(drop=True)
    pixel = np.delete(pixel, index, axis=0)
    return data, pixel


def get_target_label_idx(labels, target):
    return np.argwhere(np.isin(labels, target)).flatten().tolist()


def global_contrast_normalization(x: torch.tensor, scale='l2'):
    assert scale in ('l1', 'l2')
    mean = torch.mean(x)
    x -= mean
    if scale == 'l1':
        x_scale = torch.mean(torch.abs(x))
    elif scale == 'l2':
        x_scale = torch.sqrt(torch.sum(x ** 2)) / int(np.prod(x.shape))
    x /= x_scale
    return x


def min_max_cal(data, scale='l2'):
    list_ = []
    for x in data:
        scaling = global_contrast_normalization(torch.Tensor(x), scale=scale)
        list_.append(scaling)
    list_r = np.stack(list_, axis=0).reshape(-1)
    return min(list_r), max(list_r)


def data_augmentation(root, d, p, list_1, list_2):
    transform = transforms.Compose([
        transforms.RandomRotation((-d, d)),
        transforms.RandomHorizontalFlip(p),
    ])
    dataset = ImageFolder(root=root, transform=transform)
    X = []; Y = []
    for x, y in dataset:
        if y != 6:
            x = np.array(x)
            X.append(list(x))
            label_str = list_1[y]
            Y.append(list_2[label_str])
    return X, Y


def make_imbalance(label, outlier_fraction, seed):
    count = 1200
    normal_count   = int(count * (1 - outlier_fraction))
    abnormal_count = int(count * outlier_fraction)
    print("Count of normal dataset: ",   normal_count)
    print("Count of abnormal dataset: ", abnormal_count)

    np.random.seed(seed)
    normal_indices   = list(np.random.choice(np.where(label == 0)[0], size=normal_count,   replace=False))
    abnormal_indices = list(np.random.choice(np.where(label == 1)[0], size=abnormal_count, replace=False))
    return normal_indices + abnormal_indices


def make_dataset(root, outlier_fraction, augmentation, iteration, seed):
    # BUG FIX: original code had nested loop bug where inner loop
    # only ran with the last 'f' from the outer loop.
    classes_list_1 = {'anger': 1, 'disgust': 1, 'fear': 1, 'happy': 0, 'sadness': 1}
    classes_list_2 = {0: 'happy'}

    X = []; Y = []

    for f in os.listdir(root):
        if f not in classes_list_1:
            continue
        folder = os.path.join(root, f)
        files  = [os.path.join(folder, fn) for fn in os.listdir(folder)
                  if fn.lower().endswith('.png')]
        for file in files:
            image = cv2.imread(file)
            X.append(image)
            Y.append(classes_list_1[f])

    if augmentation:
        d_list = [20, 20, 30, 30, 40, 40]
        p_list = [0.5, 0.7, 0.5, 0.7, 0.5, 0.7]
        for i in range(iteration):
            print(str(i), "th iteration")
            d = d_list[i]; p = p_list[i]
            x, y = data_augmentation(root + 'normal/', d, p, classes_list_2, classes_list_1)
            X += x; Y += y

    X = np.array(X)
    Y = np.array(Y)

    indices = make_imbalance(Y, outlier_fraction, seed)
    X = X[indices]; Y = Y[indices]

    X_tr, X_te, y_tr, y_te = train_test_split(X, Y, stratify=Y, test_size=0.2,
                                               random_state=seed, shuffle=True)
    X_tr  = np.array(X_tr);  y_tr  = np.array(y_tr)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, stratify=y_tr, test_size=0.2,
                                                 random_state=seed, shuffle=True)
    return (X_tr, y_tr), (X_val, y_val), (X_te, y_te)
