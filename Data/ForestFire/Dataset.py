#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
np.random.seed(1500)

from sklearn.model_selection import StratifiedShuffleSplit

class Dataset:
    def __init__(self, load_path):
        dataset = pd.read_csv(load_path)
        self.X = dataset[['PREC', 'RH', 'WIND_SPEED', 'TEMP']]
        self.y = np.array([1 if i == 'pos' else 0 for i in dataset.y])

    def get_target_label_idx(self, labels, target):
        return np.argwhere(np.isin(labels, target)).flatten().tolist()

    def train_valid_set(self, cv):
        skf = StratifiedShuffleSplit(n_splits=cv, test_size=0.2, random_state=1500)

        x_train_folds = []; x_val_folds = []; y_train_folds = []; y_val_folds = []
        for train_index, valid_index in skf.split(self.X, self.y):
            X_train, X_valid = self.X.iloc[train_index], self.X.iloc[valid_index]
            y_train, y_valid = self.y[train_index], self.y[valid_index]

            normal_index = self.get_target_label_idx(y_train, target=0)
            X_train, y_train = X_train.iloc[normal_index], y_train[normal_index]

            x_train_folds.append(X_train)
            x_val_folds.append(X_valid)
            y_train_folds.append(y_train)
            y_val_folds.append(y_valid)

        for i in range(cv):
            print(f"Fold {i+1} -")
            print("Train shapes:", x_train_folds[i].shape, y_train_folds[i].shape)
            print("Validation shapes:", x_val_folds[i].shape, y_val_folds[i].shape)
            print()
        return x_train_folds, x_val_folds, y_train_folds, y_val_folds
