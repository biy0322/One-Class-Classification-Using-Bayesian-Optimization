#!/usr/bin/env python
# coding: utf-8
import numpy as np
from sklearn.model_selection import train_test_split, KFold

class Dataset:
    def generate_data(self, n_samples, outlier_fraction):
        n_inliers = int((1. - outlier_fraction)*n_samples)
        n_outliers = int(outlier_fraction*n_samples)
    
        ## normal dataset => sampling from normal distribution
        X1 = 0.5 * np.random.randn(n_inliers//2,2) + 0.01
        X2 = 0.5 * np.random.randn(n_inliers//2,2) - 0.01
    
        X = np.r_[X1,X2]
    
        ## abnormal dataset => sampling from abnormal dataset
        offset = np.random.randint(low=1, high=10)
        X = np.r_[X,np.random.uniform(low=-offset, high=offset, size=(n_outliers, 2))]
    
        y = np.zeros(n_samples, dtype=int)
        y[-n_outliers:] = 1
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=123)
        return X_train, X_test, y_train, y_test
    
    def dataset(self, n_samples, outlier_fraction):
        X_train_dataset = [[[0 for _ in range(2)] for _ in range(800)] for _ in range(100)]
        y_train_dataset = [[[0 for _ in range(1)] for _ in range(800)] for _ in range(100)]

        X_test_dataset = [[[0 for _ in range(2)] for _ in range(200)] for _ in range(100)]
        y_test_dataset = [[[0 for _ in range(1)] for _ in range(200)] for _ in range(100)]

    ## make total 100 datasets
        for i in range(100):
            X_train, X_test, y_train, y_test = self.generate_data(n_samples, outlier_fraction)
    
            X_train_dataset[i] = X_train
            y_train_dataset[i] = y_train
    
            X_test_dataset[i] = X_test
            y_test_dataset[i] = y_test
    
        return X_train_dataset, y_train_dataset, X_test_dataset, y_test_dataset 

    def get_target_label_idx(self, labels, target):
        return np.argwhere(np.isin(labels, target)).flatten().tolist()
    
    def train_valid_set(self, X, y, cv):
        kf = KFold(n_splits=cv, shuffle=True, random_state=123)
        
        x_train_folds = []; x_val_folds = []; y_train_folds = []; y_val_folds = []

        for train_index, valid_index in kf.split(X):
            X_train, X_valid = X[train_index], X[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
            
            normal_index = self.get_target_label_idx(y_train, target=0)
            X_train, y_train = X_train[normal_index], y_train[normal_index]
            X_valid, y_valid = X_valid, y_valid
            
            x_train_folds.append(X_train); x_val_folds.append(X_valid); y_train_folds.append(y_train); y_val_folds.append(y_valid)
        
        return x_train_folds, x_val_folds, y_train_folds, y_val_folds