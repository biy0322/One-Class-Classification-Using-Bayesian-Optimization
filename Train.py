#!/usr/bin/env python
# coding: utf-8

# In[1]:

from Virtual_Dataset import Dataset
from Visualization import Plotly_Visualization, Seaborn_Visualization
from Hopt import GridSearch, RandomSearch, Bayesian_hopt
from Summary import summary

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score, roc_auc_score

from pyod.models.ocsvm import OCSVM

import warnings
warnings.filterwarnings('ignore')

import pickle

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

        self.dataset = Dataset()
        
        self.summary = summary()
        
    def train(self, n_splits=5):
        for i in range(100):
           # print("############################ start "+ str(i) + "-th dataset ############################")
           # print()
           # print("############################ Start grid search ############################")
            X_train = self.x_train_dataset[i]; y_train = self.y_train_dataset[i]
            X_train, X_valid, y_train, y_valid = self.dataset.train_valid_set(X=X_train, y=y_train, cv=n_splits)
            
            for j in range(n_splits):
                x_tr = X_train[j]; x_val = X_valid[j]; y_tr = y_train[j]; y_val = y_valid[j]
                ## grid search
              #  print("#####Grid Search####")
                self.grid = GridSearch(X_train = x_tr, X_valid = x_val, y_train = y_tr, y_valid = y_val,
                                       save_path = self.save_path, scoring = self.scoring)
                self.grid.fit(number=i, fold=j)
                
                ## random search
              #  print("#####Random Search#####")
                self.random = RandomSearch(X_train = x_tr, X_valid = x_val, y_train = y_tr, y_valid = y_val,
                                           save_path = self.save_path, scoring = self.scoring)
                self.random.fit(number=i, fold=j)
                
                ## bayesian_optimization
              #  print("#####Bayesian Optimization#####")
                self.bayes = Bayesian_hopt(X_train = x_tr, X_valid = x_val, y_train = y_tr, y_valid = y_val, 
                                           save_path = self.save_path, scoring=self.scoring, utility=self.utility, xi=self.xi, kappa=self.kappa, beta=self.beta)
                self.bayes.fit(number=i, fold=j)