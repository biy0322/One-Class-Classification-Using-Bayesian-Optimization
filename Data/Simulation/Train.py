#!/usr/bin/env python
# coding: utf-8

# In[1]:

from synthetic_dataset import Dataset
from Visualization import Plotly_Visualization, Seaborn_Visualization
from Hopt import GridSearch, RandomSearch, Bayesian_hopt, Hyperband_hopt, BOHB_hopt
from Summary import summary

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score, roc_auc_score

from pyod.models.ocsvm import OCSVM

import warnings
warnings.filterwarnings('ignore')

import pickle
import time

class Train:
    def __init__(self, x_train_dataset, x_test_dataset, y_train_dataset, y_test_dataset, save_path, scoring=None, utility='ucb', xi=0.01, kappa=2.576, beta=None):

        self.x_train_dataset = x_train_dataset
        self.x_test_dataset = x_test_dataset
        self.y_train_dataset = y_train_dataset
        self.y_test_dataset = y_test_dataset

        self.save_path = save_path

        self.scoring = scoring
        self.utility = utility
        self.xi = xi
        self.kappa = kappa
        self.beta = beta

        self.grid_df = pd.DataFrame(columns = ['grid_recall_score','grid_fl_score', 'grid_roc_auc_score'])
        self.random_df = pd.DataFrame(columns = ['random_recall_score','random_fl_score', 'random_roc_auc_score'])
        self.bayesian_df = pd.DataFrame(columns = ['bayes_recall_score','bayes_f1_score', 'bayes_roc_auc_score'])
        self.hyperband_df = pd.DataFrame(columns = ['hyperband_recall_score','hyperband_f1_score', 'hyperband_roc_auc_score'])
        self.bohb_df = pd.DataFrame(columns = ['bohb_recall_score','bohb_f1_score', 'bohb_roc_auc_score'])

        self.dataset = Dataset()

        self.summary = summary()

    def train(self, n_splits=5, n_repeats=None, timing_path=None, history_path=None):
        n_runs = len(self.x_train_dataset) if n_repeats is None else n_repeats
        self.timing_records = []
        self.history_records = []

        def fit_with_timing(method_name, estimator, number, fold, budget_evaluations):
            start = time.perf_counter()
            estimator.fit(number=number, fold=fold)
            elapsed = time.perf_counter() - start
            self.timing_records.append({
                'repeat': number,
                'fold': fold,
                'method': method_name,
                'wall_time_sec': elapsed,
                'budget_evaluations': budget_evaluations,
            })
            for record in getattr(estimator, 'history', []):
                enriched = dict(record)
                enriched.update({
                    'repeat': number,
                    'fold': fold,
                    'method': method_name,
                })
                self.history_records.append(enriched)

        for i in range(n_runs):
            X_train = self.x_train_dataset[i]; y_train = self.y_train_dataset[i]
            X_train, X_valid, y_train, y_valid = self.dataset.train_valid_set(X=X_train, y=y_train, cv=n_splits)

            print(f"{i}-th iteration is started.")
            for j in range(n_splits):
                x_tr = X_train[j]; x_val = X_valid[j]; y_tr = y_train[j]; y_val = y_valid[j]

                self.grid = GridSearch(X_train=x_tr, X_valid=x_val, y_train=y_tr, y_valid=y_val,
                                       save_path=self.save_path, scoring=self.scoring, beta=self.beta)
                fit_with_timing('grid', self.grid, i, j, budget_evaluations=100)

                self.random = RandomSearch(X_train=x_tr, X_valid=x_val, y_train=y_tr, y_valid=y_val,
                                           save_path=self.save_path, scoring=self.scoring, beta=self.beta)
                fit_with_timing('random', self.random, i, j, budget_evaluations=100)

                self.bayes = Bayesian_hopt(X_train=x_tr, X_valid=x_val, y_train=y_tr, y_valid=y_val,
                                           save_path=self.save_path, scoring=self.scoring, utility=self.utility,
                                           xi=self.xi, kappa=self.kappa, beta=self.beta)
                fit_with_timing('bayesian', self.bayes, i, j, budget_evaluations=102)

                self.hyperband = Hyperband_hopt(X_train=x_tr, X_valid=x_val, y_train=y_tr, y_valid=y_val,
                                                save_path=self.save_path, scoring=self.scoring, beta=self.beta)
                fit_with_timing('hyperband', self.hyperband, i, j, budget_evaluations=100)

                self.bohb = BOHB_hopt(X_train=x_tr, X_valid=x_val, y_train=y_tr, y_valid=y_val,
                                      save_path=self.save_path, scoring=self.scoring, beta=self.beta)
                fit_with_timing('bohb', self.bohb, i, j, budget_evaluations=100)

        if timing_path is not None:
            pd.DataFrame(self.timing_records).to_csv(timing_path, index=False)
        if history_path is not None:
            pd.DataFrame(self.history_records).to_csv(history_path, index=False)
