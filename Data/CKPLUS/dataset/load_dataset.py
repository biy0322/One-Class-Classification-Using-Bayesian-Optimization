#!/usr/bin/env python
# coding: utf-8

from .emotion import Emotion_DataLoader

def load_dataset(train_data, train_labels, valid_data, valid_labels, test_data, test_labels,
                 normal_class=0, outlier_class=[1], scale='l1'):
    dataset = Emotion_DataLoader(
        train_data, train_labels, valid_data, valid_labels, test_data, test_labels,
        normal_class=normal_class, outlier_class=outlier_class, scale=scale,
    )
    return dataset
