#!/usr/bin/env python
# coding: utf-8

from pyod.models.ocsvm import OCSVM

from sklearn.metrics import recall_score, fbeta_score

from itertools import product
import random
import numpy as np

from bayes_opt import BayesianOptimization
from bayes_opt.acquisition import UpperConfidenceBound, ExpectedImprovement

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import pickle

from Summary import summary


class GridSearch:
    def __init__(self, X_train, X_valid, y_train, y_valid, save_path, scoring=None, beta=3, verbose=0):
        self.X_train, self.X_valid, self.y_train, self.y_valid = X_train, X_valid, y_train, y_valid
        self.save_path = save_path
        self.estimator = OCSVM(kernel='rbf')
        self.param_grid = {
            'nu':    list(np.linspace(0.001, 0.9999, 10)),
            'gamma': list(np.linspace(0.001, 30, 10)),
        }
        self.scoring = scoring
        self.beta = beta
        self.verbose = verbose
        self.summary = summary()

    def fit(self, fold):
        param_combinations = list(product(*self.param_grid.values()))
        best_score = None
        best_params = None

        for params in param_combinations:
            self.estimator.set_params(**dict(zip(self.param_grid.keys(), params)))
            self.estimator.fit(self.X_train)
            pred = self.estimator.predict(self.X_valid)

            if self.scoring == 'recall':
                score = recall_score(self.y_valid, pred)
            elif self.scoring == 'f_beta':
                score = fbeta_score(self.y_valid, pred, beta=self.beta)

            if best_score is None or score > best_score:
                best_score = score
                best_params = params

        self.best_params_ = {k: v for k, v in zip(self.param_grid.keys(), best_params)}
        self.best_score_ = best_score

        cv_best_model = OCSVM(kernel='rbf', gamma=self.best_params_['gamma'], nu=self.best_params_['nu'])
        cv_best_model.fit(self.X_train)

        with open(f'{self.save_path}_grid_cv_{fold}', 'wb') as f:
            pickle.dump(cv_best_model, f)


class RandomSearch:
    def __init__(self, X_train, X_valid, y_train, y_valid, save_path, scoring=None, beta=3, verbose=0):
        self.X_train, self.X_valid, self.y_train, self.y_valid = X_train, X_valid, y_train, y_valid
        self.save_path = save_path
        self.estimator = OCSVM(kernel='rbf')
        random.seed(1500)
        self.param_random = {
            'nu':    [random.uniform(0.001, 0.9999) for _ in range(10)],
            'gamma': [random.uniform(0.001, 30)     for _ in range(10)],
        }
        self.scoring = scoring
        self.beta = beta
        self.verbose = verbose
        self.summary = summary()

    def fit(self, fold):
        param_combinations = list(product(*self.param_random.values()))
        best_score = None
        best_params = None

        for params in param_combinations:
            self.estimator.set_params(**dict(zip(self.param_random.keys(), params)))
            self.estimator.fit(self.X_train)
            pred = self.estimator.predict(self.X_valid)

            if self.scoring == 'recall':
                score = recall_score(self.y_valid, pred)
            elif self.scoring == 'f_beta':
                score = fbeta_score(self.y_valid, pred, beta=self.beta)

            if best_score is None or score > best_score:
                best_score = score
                best_params = params

        self.best_params_ = {k: v for k, v in zip(self.param_random.keys(), best_params)}
        self.best_score_ = best_score

        cv_best_model = OCSVM(kernel='rbf', gamma=self.best_params_['gamma'], nu=self.best_params_['nu'])
        cv_best_model.fit(self.X_train)

        with open(f'{self.save_path}_random_cv_{fold}', 'wb') as f:
            pickle.dump(cv_best_model, f)


class Bayesian_hopt:
    def __init__(self, X_train, X_valid, y_train, y_valid, save_path, scoring=None,
                 utility='ucb', kappa=2.576, xi=0.01, beta=None, verbose=0):
        self.X_train = X_train; self.X_valid = X_valid
        self.y_train = y_train; self.y_valid = y_valid
        self.save_path = save_path
        self.scoring = scoring
        self.kappa = kappa
        self.utility = utility
        self.xi = xi
        self.beta = beta
        self.verbose = verbose
        self.summary = summary()

    def bayesian_optimization(self, init_nu, init_gamma):
        ocsvm = OCSVM(kernel='rbf', gamma=init_gamma, nu=init_nu)
        ocsvm.fit(self.X_train)
        y_pred = ocsvm.predict(self.X_valid)
        if self.scoring == 'recall':
            score = recall_score(self.y_valid, y_pred)
        elif self.scoring == 'f_beta':
            score = fbeta_score(self.y_valid, y_pred, beta=self.beta)
        return score

    def bayesian_optimzation_function(self):
        self.pbounds = {'init_gamma': (0.001, 30), 'init_nu': (0.001, 0.99999)}
        if self.utility == 'ucb':
            acq = UpperConfidenceBound(kappa=self.kappa)
        elif self.utility == 'ei':
            acq = ExpectedImprovement(xi=self.xi)
        bo = BayesianOptimization(
            f=self.bayesian_optimization, pbounds=self.pbounds,
            verbose=self.verbose, random_state=123,
            allow_duplicate_points=True, acquisition_function=acq,
        )
        bo.maximize(init_points=2, n_iter=100)
        return bo.max

    def fit(self, fold):
        best = self.bayesian_optimzation_function()
        cv_best_model = OCSVM(kernel='rbf', gamma=best['params']['init_gamma'], nu=best['params']['init_nu'])
        cv_best_model.fit(self.X_train)

        with open(f'{self.save_path}_bayes_cv_{fold}', 'wb') as f:
            pickle.dump(cv_best_model, f)


class Hyperband_hopt:
    """Hyperband: Random sampling + successive halving via HyperbandPruner.

    Resource = training subset fraction (step/max_steps of X_train).
    """
    def __init__(self, X_train, X_valid, y_train, y_valid, save_path, scoring=None, beta=3,
                 n_trials=100, max_resource=9, min_resource=1, reduction_factor=3, verbose=0):
        self.X_train = X_train; self.X_valid = X_valid
        self.y_train = y_train; self.y_valid = y_valid
        self.save_path = save_path
        self.scoring = scoring
        self.beta = beta
        self.n_trials = n_trials
        self.max_resource = max_resource
        self.min_resource = min_resource
        self.reduction_factor = reduction_factor
        self.verbose = verbose
        self.summary = summary()

    def _objective(self, trial):
        nu    = trial.suggest_float('nu',    0.001, 0.99999)
        gamma = trial.suggest_float('gamma', 0.001, 30.0)

        n_samples = len(self.X_train)
        score = 0.0
        for step in range(self.min_resource, self.max_resource + 1):
            subset_size = max(5, int(n_samples * step / self.max_resource))
            X_sub = self.X_train.iloc[:subset_size] if hasattr(self.X_train, 'iloc') else self.X_train[:subset_size]

            model = OCSVM(kernel='rbf', gamma=gamma, nu=nu)
            model.fit(X_sub)
            pred = model.predict(self.X_valid)

            if self.scoring == 'recall':
                score = recall_score(self.y_valid, pred)
            elif self.scoring == 'f_beta':
                score = fbeta_score(self.y_valid, pred, beta=self.beta)

            trial.report(score, step)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return score

    def fit(self, fold):
        sampler = optuna.samplers.RandomSampler(seed=1500)
        pruner  = optuna.pruners.HyperbandPruner(
            min_resource=self.min_resource,
            max_resource=self.max_resource,
            reduction_factor=self.reduction_factor,
        )
        study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
        study.optimize(self._objective, n_trials=self.n_trials, show_progress_bar=False)

        best_params = study.best_params
        cv_best_model = OCSVM(kernel='rbf', gamma=best_params['gamma'], nu=best_params['nu'])
        cv_best_model.fit(self.X_train)

        with open(f'{self.save_path}_hyperband_cv_{fold}', 'wb') as f:
            pickle.dump(cv_best_model, f)


class BOHB_hopt:
    """BOHB: Bayesian Optimization (TPE) + HyperBand successive halving."""
    def __init__(self, X_train, X_valid, y_train, y_valid, save_path, scoring=None, beta=3,
                 n_trials=100, max_resource=9, min_resource=1, reduction_factor=3, verbose=0):
        self.X_train = X_train; self.X_valid = X_valid
        self.y_train = y_train; self.y_valid = y_valid
        self.save_path = save_path
        self.scoring = scoring
        self.beta = beta
        self.n_trials = n_trials
        self.max_resource = max_resource
        self.min_resource = min_resource
        self.reduction_factor = reduction_factor
        self.verbose = verbose
        self.summary = summary()

    def _objective(self, trial):
        nu    = trial.suggest_float('nu',    0.001, 0.99999)
        gamma = trial.suggest_float('gamma', 0.001, 30.0)

        n_samples = len(self.X_train)
        score = 0.0
        for step in range(self.min_resource, self.max_resource + 1):
            subset_size = max(5, int(n_samples * step / self.max_resource))
            X_sub = self.X_train.iloc[:subset_size] if hasattr(self.X_train, 'iloc') else self.X_train[:subset_size]

            model = OCSVM(kernel='rbf', gamma=gamma, nu=nu)
            model.fit(X_sub)
            pred = model.predict(self.X_valid)

            if self.scoring == 'recall':
                score = recall_score(self.y_valid, pred)
            elif self.scoring == 'f_beta':
                score = fbeta_score(self.y_valid, pred, beta=self.beta)

            trial.report(score, step)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return score

    def fit(self, fold):
        sampler = optuna.samplers.TPESampler(seed=1500)
        pruner  = optuna.pruners.HyperbandPruner(
            min_resource=self.min_resource,
            max_resource=self.max_resource,
            reduction_factor=self.reduction_factor,
        )
        study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
        study.optimize(self._objective, n_trials=self.n_trials, show_progress_bar=False)

        best_params = study.best_params
        cv_best_model = OCSVM(kernel='rbf', gamma=best_params['gamma'], nu=best_params['nu'])
        cv_best_model.fit(self.X_train)

        with open(f'{self.save_path}_bohb_cv_{fold}', 'wb') as f:
            pickle.dump(cv_best_model, f)
