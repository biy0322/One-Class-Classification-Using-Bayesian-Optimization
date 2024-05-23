#!/usr/bin/env python
# coding: utf-8
from pyod.models.ocsvm import OCSVM

from sklearn.metrics import recall_score, fbeta_score, make_scorer
from Summary import summary

from itertools import product
import random
import numpy as np

from pyod.models.ocsvm import OCSVM

from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction

import pickle

class GridSearch:
    def __init__(self, X_train, X_valid, y_train, y_valid, save_path, scoring=None, beta=3, verbose=0):
        self.X_train, self.X_valid, self.y_train, self.y_valid = X_train, X_valid, y_train, y_valid
        
        self.save_path = save_path
        
        self.estimator = OCSVM(kernel='rbf')
        self.param_grid = param_grid = {   
                                        'nu' : list(np.linspace(0.001,0.999999, 10)),
                                        'gamma' :  list(np.linspace(0.001,30,10))}
        self.scoring = scoring
        self.beta = beta
        self.verbose = verbose
        
        self.summary = summary()
    def fit(self, number, fold):
        param_combinations = list(product(*self.param_grid.values()))
        best_score = None
        best_params = None

        for params in param_combinations:
            if self.verbose:
                print("Fitting with params:", params)

            self.estimator.set_params(**dict(zip(self.param_grid.keys(), params)))
            
            self.estimator.fit(self.X_train)
            pred = self.estimator.predict(self.X_valid)
                
            if self.scoring == "recall":
                score = recall_score(pred, self.y_valid)
            elif self.scoring == 'f_beta':
                score = fbeta_score(pred, self.y_valid, beta=self.beta)

            if best_score is None or score > best_score:
                best_score = score
                best_params = params
            
        if self.verbose:
            print("Best parameters:", {k: v for k, v in zip(self.param_grid.keys(), best_params)})
            print("Best score:", best_score)
            
        self.best_params_ = {k: v for k, v in zip(self.param_grid.keys(), best_params)}
        self.best_score_ = best_score
        
        cv_best_model = OCSVM(kernel='rbf', gamma=self.best_params_['gamma'], nu=self.best_params_['nu'])
        
        cv_best_model.fit(self.X_train)
        pred = cv_best_model.predict(self.X_valid)
       
        self.summary.get_clf_eval(self.y_valid, pred, fold)
        
        with open(f'{self.save_path}_grid_cv_{number}_{fold}','wb') as f:
            pickle.dump(cv_best_model, f)


class RandomSearch:
    def __init__(self, X_train, X_valid, y_train, y_valid, save_path, scoring=None, beta=3, verbose=0):
        self.X_train, self.X_valid, self.y_train, self.y_valid = X_train, X_valid, y_train, y_valid
        
        self.save_path = save_path
        
        
        self.estimator = OCSVM(kernel='rbf')
        random.seed(1500)
        self.param_random = {
                            'nu': [random.uniform(0.001,0.9999) for i in range(10)],
                            'gamma': [random.uniform(0.001,30) for i in range(10)]}
        self.scoring = scoring
        self.beta = beta
        self.verbose = verbose
  
        self.summary = summary()
        
    def fit(self, number, fold):
        param_combinations = list(product(*self.param_random.values()))
        best_score = None
        best_params = None

        for params in param_combinations:
            if self.verbose:
                print("Fitting with params:", params)

            self.estimator.set_params(**dict(zip(self.param_random.keys(), params)))
            
            self.estimator.fit(self.X_train)
            pred = self.estimator.predict(self.X_valid)
                
            if self.scoring == "recall":
                score = recall_score(pred, self.y_valid)
            elif self.scoring == 'f_beta':
                score = fbeta_score(pred, self.y_valid, beta=self.beta)

            if best_score is None or score > best_score:
                best_score = score
                best_params = params
            
        if self.verbose:
            print("Best parameters:", {k: v for k, v in zip(self.param_random.keys(), best_params)})
            print("Best score:", best_score)
            
        self.best_params_ = {k: v for k, v in zip(self.param_random.keys(), best_params)}
        self.best_score_ = best_score
        
        cv_best_model = OCSVM(kernel='rbf', gamma=self.best_params_['gamma'], nu=self.best_params_['nu'])
        
        cv_best_model.fit(self.X_train)
        pred = cv_best_model.predict(self.X_valid)
        self.summary.get_clf_eval(self.y_valid, pred, fold)
        
        with open(f'{self.save_path}_random_cv_{number}_{fold}','wb') as f:
            pickle.dump(cv_best_model, f)
        
class Bayesian_hopt:
    def __init__(self, X_train, X_valid, y_train, y_valid, save_path, scoring=None, utility = 'ucb', kappa=2.576, xi=0.01, beta=None, verbose=0):
        
        self.X_train = X_train; self.X_valid = X_valid; self.y_train = y_train; self.y_valid = y_valid
        
        self.save_path = save_path
        self.scoring = scoring
        self.kappa = kappa
        self.utility = utility
        self.xi = xi
        self.beta = beta
  
        self.verbose= verbose
        
        self.summary = summary()
        
    def bayesian_optimization(self, init_nu, init_gamma):
        self.ocsvm = OCSVM(kernel='rbf', gamma = init_gamma, nu=init_nu)
        self.ocsvm.fit(self.X_train)
        self.y_pred = self.ocsvm.predict(self.X_valid)
        if self.scoring=="recall":
            score = recall_score(self.y_pred, self.y_valid)
        elif self.scoring=='f_beta':
             score = fbeta_score(self.y_pred, self.y_valid, beta=self.beta)
        self.score = score
        return score
    
    def bayesian_optimzation_function(self):
        self.pbounds = {'init_gamma': (0.001,30), 'init_nu':(0.001,0.99999)}
        if self.utility == "ucb":
            self.acquisition_function = UtilityFunction(kind="ucb", kappa=self.kappa)
        elif self.utility == "ei":
            self.acquisition_function = UtilityFunction(kind="ei", xi=self.xi)
        bo = BayesianOptimization(f=self.bayesian_optimization, pbounds=self.pbounds, verbose=self.verbose,random_state=1, allow_duplicate_points = True)
        bo.maximize(init_points=2, n_iter=100, acquisition_function=self.acquisition_function)
        bayesian_best_parameter = bo.max
        return bayesian_best_parameter
    
    def fit(self, number, fold):   
        bayes_hopt = self.bayesian_optimzation_function()
        score = bayes_hopt['target']
        self.optim_gamma = bayes_hopt['params']['init_gamma']
        self.optim_nu = bayes_hopt['params']['init_nu']
                 
        cv_best_model = OCSVM(kernel='rbf', gamma = self.optim_gamma, nu=self.optim_nu)
        
        cv_best_model.fit(self.X_train)
        pred = cv_best_model.predict(self.X_valid)
        self.summary.get_clf_eval(self.y_valid, pred, fold)
                    
        with open(f'{self.save_path}_bayes_cv_{number}_{fold}','wb') as f:
               pickle.dump(cv_best_model, f)