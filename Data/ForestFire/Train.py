#!/usr/bin/env python
# coding: utf-8

from Hopt import GridSearch, RandomSearch, Bayesian_hopt, Hyperband_hopt, BOHB_hopt
from Dataset import Dataset

import numpy as np
import pickle

from pyod.models.ocsvm import OCSVM
from Summary import summary

import warnings
warnings.filterwarnings('ignore')


class Train:
    def __init__(self, save_path, scoring=None, utility='ucb', xi=0.01, kappa=2.576, beta=None):
        self.dataset = Dataset(load_path='./Data/data_manipulated.csv')
        self.save_path = save_path
        self.scoring = scoring
        self.beta = beta
        self.utility = utility
        self.xi = xi
        self.kappa = kappa
        self.summary = summary()

    def train(self, n_splits=9):
        X_train, X_valid, y_train, y_valid = self.dataset.train_valid_set(cv=n_splits)

        for j in range(n_splits):
            x_tr = X_train[j]; x_val = X_valid[j]; y_tr = y_train[j]; y_val = y_valid[j]

            # before hopt (default OCSVM, no tuning)
            print("########## Before hopt ##########")
            be_hopt = OCSVM(kernel='rbf')
            be_hopt.fit(x_tr)
            with open(f'{self.save_path}_model_cv_{j}', 'wb') as f:
                pickle.dump(be_hopt, f)

            # grid search
            print("########## Grid Search ##########")
            self.grid = GridSearch(X_train=x_tr, X_valid=x_val, y_train=y_tr, y_valid=y_val,
                                   save_path=self.save_path, scoring=self.scoring, beta=self.beta)
            self.grid.fit(fold=j)

            # random search
            print("########## Random Search ##########")
            self.random = RandomSearch(X_train=x_tr, X_valid=x_val, y_train=y_tr, y_valid=y_val,
                                       save_path=self.save_path, scoring=self.scoring, beta=self.beta)
            self.random.fit(fold=j)

            # bayesian optimization
            print("########## Bayesian Optimization ##########")
            self.bayes = Bayesian_hopt(X_train=x_tr, X_valid=x_val, y_train=y_tr, y_valid=y_val,
                                       save_path=self.save_path, scoring=self.scoring,
                                       utility=self.utility, xi=self.xi, kappa=self.kappa, beta=self.beta)
            self.bayes.fit(fold=j)

            # hyperband
            print("########## Hyperband ##########")
            self.hyperband = Hyperband_hopt(X_train=x_tr, X_valid=x_val, y_train=y_tr, y_valid=y_val,
                                            save_path=self.save_path, scoring=self.scoring, beta=self.beta)
            self.hyperband.fit(fold=j)

            # BOHB
            print("########## BOHB ##########")
            self.bohb = BOHB_hopt(X_train=x_tr, X_valid=x_val, y_train=y_tr, y_valid=y_val,
                                  save_path=self.save_path, scoring=self.scoring, beta=self.beta)
            self.bohb.fit(fold=j)
