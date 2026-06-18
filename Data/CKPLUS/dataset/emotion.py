#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

from .preprocessing import get_target_label_idx, global_contrast_normalization
from base.torchvision_dataset import TorchvisionDataset


class Emotion_DataLoader(TorchvisionDataset):
    def __init__(self, train_data, train_labels, valid_data, valid_labels, test_data, test_labels,
                 normal_class=0, outlier_class=[1], scale='l1'):
        super().__init__()
        self.n_classes       = 2
        self.normal_classes  = normal_class
        self.outlier_classes = outlier_class

        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: global_contrast_normalization(x, scale=scale)),
        ])
        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        full_set = EmotionDataset(
            name='Train',
            train_data=train_data, train_labels=train_labels,
            valid_data=valid_data, valid_labels=valid_labels,
            test_data=test_data,   test_labels=test_labels,
            transform=transform, target_transform=target_transform,
        )
        train_idx_normal = get_target_label_idx(full_set.train_labels, 0)
        self.train_set = Subset(full_set, train_idx_normal)

        self.valid_set = EmotionDataset(
            name='Valid',
            train_data=train_data, train_labels=train_labels,
            valid_data=valid_data, valid_labels=valid_labels,
            test_data=test_data,   test_labels=test_labels,
            transform=transform, target_transform=target_transform,
        )
        self.test_set = EmotionDataset(
            name='Test',
            train_data=train_data, train_labels=train_labels,
            valid_data=valid_data, valid_labels=valid_labels,
            test_data=test_data,   test_labels=test_labels,
            transform=transform, target_transform=target_transform,
        )

    def loaders(self, train_batch_size: int, test_batch_size: int,
                shuffle_train=True, shuffle_test=False, num_workers: int = 0):
        train_loader = DataLoader(self.train_set, batch_size=train_batch_size,
                                  shuffle=shuffle_train, num_workers=num_workers, drop_last=False)
        valid_loader = DataLoader(self.valid_set, batch_size=test_batch_size,
                                  shuffle=shuffle_test, num_workers=num_workers, drop_last=False)
        test_loader  = DataLoader(self.test_set,  batch_size=test_batch_size,
                                  shuffle=shuffle_test, num_workers=num_workers, drop_last=False)
        return train_loader, valid_loader, test_loader


class EmotionDataset(Dataset):
    def __init__(self, name, train_data, train_labels, valid_data, valid_labels,
                 test_data, test_labels, transform, target_transform):
        self.name = name
        self.train_data   = train_data;   self.train_labels = train_labels
        self.valid_data   = valid_data;   self.valid_labels = valid_labels
        self.test_data    = test_data;    self.test_labels  = test_labels
        self.transform        = transform
        self.target_transform = target_transform

    def __len__(self):
        if self.name == 'Train': return len(self.train_data)
        if self.name == 'Valid': return len(self.valid_data)
        if self.name == 'Test':  return len(self.test_data)

    def __getitem__(self, index):
        if self.name == 'Train':
            img, target = self.train_data[index], self.train_labels[index]
        elif self.name == 'Valid':
            img, target = self.valid_data[index], self.valid_labels[index]
        elif self.name == 'Test':
            img, target = self.test_data[index],  self.test_labels[index]

        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index
