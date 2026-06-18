#!/usr/bin/env python
# coding: utf-8
from pyod.models.ocsvm import OCSVM

from sklearn.metrics import recall_score, fbeta_score, make_scorer
from Summary import summary

from itertools import product
import random
import numpy as np
import time

from pyod.models.ocsvm import OCSVM

from bayes_opt import BayesianOptimization
from bayes_opt.acquisition import UpperConfidenceBound, ExpectedImprovement

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

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
        self.history = []
        self._fit_start_time = time.perf_counter()
        param_combinations = list(product(*self.param_grid.values()))
        best_score = None
        best_params = None

        for eval_index, params in enumerate(param_combinations, start=1):
            if self.verbose:
                print("Fitting with params:", params)

            self.estimator.set_params(**dict(zip(self.param_grid.keys(), params)))

            self.estimator.fit(self.X_train)
            pred = self.estimator.predict(self.X_valid)

            if self.scoring == "recall":
                score = recall_score(self.y_valid, pred)
            elif self.scoring == 'f_beta':
                score = fbeta_score(self.y_valid, pred, beta=self.beta)

            if best_score is None or score > best_score:
                best_score = score
                best_params = params
            self.history.append({
                'eval_index': eval_index,
                'resource_step': 1,
                'nu': params[0],
                'gamma': params[1],
                'score': score,
                'best_so_far': best_score,
                'elapsed_sec': time.perf_counter() - self._fit_start_time,
                'status': 'complete',
            })

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
        self.history = []
        self._fit_start_time = time.perf_counter()
        param_combinations = list(product(*self.param_random.values()))
        best_score = None
        best_params = None

        for eval_index, params in enumerate(param_combinations, start=1):
            if self.verbose:
                print("Fitting with params:", params)

            self.estimator.set_params(**dict(zip(self.param_random.keys(), params)))

            self.estimator.fit(self.X_train)
            pred = self.estimator.predict(self.X_valid)

            if self.scoring == "recall":
                score = recall_score(self.y_valid, pred)
            elif self.scoring == 'f_beta':
                score = fbeta_score(self.y_valid, pred, beta=self.beta)

            if best_score is None or score > best_score:
                best_score = score
                best_params = params
            self.history.append({
                'eval_index': eval_index,
                'resource_step': 1,
                'nu': params[0],
                'gamma': params[1],
                'score': score,
                'best_so_far': best_score,
                'elapsed_sec': time.perf_counter() - self._fit_start_time,
                'status': 'complete',
            })

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
        self.history = []

    def bayesian_optimization(self, init_nu, init_gamma):
        self.ocsvm = OCSVM(kernel='rbf', gamma = init_gamma, nu=init_nu)
        self.ocsvm.fit(self.X_train)
        self.y_pred = self.ocsvm.predict(self.X_valid)
        if self.scoring=="recall":
            score = recall_score(self.y_valid, self.y_pred)
        elif self.scoring=='f_beta':
             score = fbeta_score(self.y_valid, self.y_pred, beta=self.beta)
        self.score = score
        best_so_far = max([h['best_so_far'] for h in self.history], default=float('-inf'))
        best_so_far = max(best_so_far, score)
        self.history.append({
            'eval_index': len(self.history) + 1,
            'resource_step': 1,
            'nu': init_nu,
            'gamma': init_gamma,
            'score': score,
            'best_so_far': best_so_far,
            'elapsed_sec': time.perf_counter() - self._fit_start_time,
            'status': 'complete',
        })
        return score

    def bayesian_optimzation_function(self):
        self.pbounds = {'init_gamma': (0.001,30), 'init_nu':(0.001,0.99999)}
        if self.utility == "ucb":
            self.acquisition_function = UpperConfidenceBound(kappa=self.kappa)
        elif self.utility == "ei":
            self.acquisition_function = ExpectedImprovement(xi=self.xi)
        bo = BayesianOptimization(f=self.bayesian_optimization, pbounds=self.pbounds, verbose=self.verbose, random_state=1, allow_duplicate_points=True, acquisition_function=self.acquisition_function)
        bo.maximize(init_points=2, n_iter=100)
        bayesian_best_parameter = bo.max
        return bayesian_best_parameter

    def fit(self, number, fold):
        self.history = []
        self._fit_start_time = time.perf_counter()
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


class Hyperband_hopt:
    """Hyperband: Random sampling + successive halving via HyperbandPruner.

    Resource = training subset fraction (step/max_steps of X_train).
    Prunes unpromising configs early, promoting top performers to full data.
    Total budget is comparable to Grid/Random (100 full-data evaluations).
    """
    def __init__(self, X_train, X_valid, y_train, y_valid, save_path, scoring=None, beta=3,
                 n_trials=100, max_resource=9, min_resource=1, reduction_factor=3, verbose=0):
        self.X_train = X_train
        self.X_valid = X_valid
        self.y_train = y_train
        self.y_valid = y_valid
        self.save_path = save_path
        self.scoring = scoring
        self.beta = beta
        self.n_trials = n_trials
        self.max_resource = max_resource
        self.min_resource = min_resource
        self.reduction_factor = reduction_factor
        self.verbose = verbose
        self.summary = summary()
        self.history = []

    def _objective(self, trial):
        nu = trial.suggest_float('nu', 0.001, 0.99999)
        gamma = trial.suggest_float('gamma', 0.001, 30.0)

        n_samples = len(self.X_train)
        score = 0.0
        for step in range(self.min_resource, self.max_resource + 1):
            subset_size = max(5, int(n_samples * step / self.max_resource))
            X_sub = self.X_train[:subset_size]

            model = OCSVM(kernel='rbf', gamma=gamma, nu=nu)
            model.fit(X_sub)
            pred = model.predict(self.X_valid)

            if self.scoring == 'recall':
                score = recall_score(self.y_valid, pred)
            elif self.scoring == 'f_beta':
                score = fbeta_score(self.y_valid, pred, beta=self.beta)

            trial.report(score, step)
            best_so_far = max([h['best_so_far'] for h in self.history], default=float('-inf'))
            best_so_far = max(best_so_far, score)
            self.history.append({
                'eval_index': len(self.history) + 1,
                'trial_number': trial.number,
                'resource_step': step,
                'nu': nu,
                'gamma': gamma,
                'score': score,
                'best_so_far': best_so_far,
                'elapsed_sec': time.perf_counter() - self._fit_start_time,
                'status': 'observed',
            })
            if trial.should_prune():
                self.history[-1]['status'] = 'pruned'
                raise optuna.exceptions.TrialPruned()

        return score

    def fit(self, number, fold):
        self.history = []
        self._fit_start_time = time.perf_counter()
        sampler = optuna.samplers.RandomSampler(seed=1500)
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=self.min_resource,
            max_resource=self.max_resource,
            reduction_factor=self.reduction_factor
        )
        study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
        study.optimize(self._objective, n_trials=self.n_trials, show_progress_bar=False)

        best_params = study.best_params
        cv_best_model = OCSVM(kernel='rbf', gamma=best_params['gamma'], nu=best_params['nu'])
        cv_best_model.fit(self.X_train)
        pred = cv_best_model.predict(self.X_valid)
        self.summary.get_clf_eval(self.y_valid, pred, fold)

        with open(f'{self.save_path}_hyperband_cv_{number}_{fold}', 'wb') as f:
            pickle.dump(cv_best_model, f)


class BOHB_hopt:
    """BOHB: Bayesian Optimization (TPE) + HyperBand successive halving.

    Same resource schedule as Hyperband_hopt, but uses TPESampler
    (Tree-structured Parzen Estimator) instead of random sampling to
    exploit previously observed results when proposing new candidates.
    """
    def __init__(self, X_train, X_valid, y_train, y_valid, save_path, scoring=None, beta=3,
                 n_trials=100, max_resource=9, min_resource=1, reduction_factor=3, verbose=0):
        self.X_train = X_train
        self.X_valid = X_valid
        self.y_train = y_train
        self.y_valid = y_valid
        self.save_path = save_path
        self.scoring = scoring
        self.beta = beta
        self.n_trials = n_trials
        self.max_resource = max_resource
        self.min_resource = min_resource
        self.reduction_factor = reduction_factor
        self.verbose = verbose
        self.summary = summary()
        self.history = []

    def _objective(self, trial):
        nu = trial.suggest_float('nu', 0.001, 0.99999)
        gamma = trial.suggest_float('gamma', 0.001, 30.0)

        n_samples = len(self.X_train)
        score = 0.0
        for step in range(self.min_resource, self.max_resource + 1):
            subset_size = max(5, int(n_samples * step / self.max_resource))
            X_sub = self.X_train[:subset_size]

            model = OCSVM(kernel='rbf', gamma=gamma, nu=nu)
            model.fit(X_sub)
            pred = model.predict(self.X_valid)

            if self.scoring == 'recall':
                score = recall_score(self.y_valid, pred)
            elif self.scoring == 'f_beta':
                score = fbeta_score(self.y_valid, pred, beta=self.beta)

            trial.report(score, step)
            best_so_far = max([h['best_so_far'] for h in self.history], default=float('-inf'))
            best_so_far = max(best_so_far, score)
            self.history.append({
                'eval_index': len(self.history) + 1,
                'trial_number': trial.number,
                'resource_step': step,
                'nu': nu,
                'gamma': gamma,
                'score': score,
                'best_so_far': best_so_far,
                'elapsed_sec': time.perf_counter() - self._fit_start_time,
                'status': 'observed',
            })
            if trial.should_prune():
                self.history[-1]['status'] = 'pruned'
                raise optuna.exceptions.TrialPruned()

        return score

    def fit(self, number, fold):
        self.history = []
        self._fit_start_time = time.perf_counter()
        # TPESampler = Bayesian (Tree-structured Parzen Estimator)
        sampler = optuna.samplers.TPESampler(seed=1500)
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=self.min_resource,
            max_resource=self.max_resource,
            reduction_factor=self.reduction_factor
        )
        study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
        study.optimize(self._objective, n_trials=self.n_trials, show_progress_bar=False)

        best_params = study.best_params
        cv_best_model = OCSVM(kernel='rbf', gamma=best_params['gamma'], nu=best_params['nu'])
        cv_best_model.fit(self.X_train)
        pred = cv_best_model.predict(self.X_valid)
        self.summary.get_clf_eval(self.y_valid, pred, fold)

        with open(f'{self.save_path}_bohb_cv_{number}_{fold}', 'wb') as f:
            pickle.dump(cv_best_model, f)
