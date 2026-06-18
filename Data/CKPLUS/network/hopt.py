#!/usr/bin/env python
# coding: utf-8

from itertools import product
import random
import numpy as np
import copy

import torch

from sklearn.metrics import recall_score, fbeta_score, roc_auc_score

from bayes_opt import BayesianOptimization
from bayes_opt.acquisition import UpperConfidenceBound

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from .network import LeNet5
from .Train import TrainerDeepSVDD
from .evaluation import eval
from .epsilon_thres import epsilon_threshold


# ---------------------------------------------------------------------------
# Grid Search
# ---------------------------------------------------------------------------
class GridSearch:
    def __init__(self, args, train_loader, valid_loader, path, device,
                 verbose=0, objective='soft-boundary', f_beta_param=1):
        self.args         = args
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.path         = path
        self.device       = device
        self.verbose      = verbose
        self.objective    = objective
        self.f_beta_param = f_beta_param

        if objective == 'soft-boundary':
            self.param_grid = {
                'lr': list(np.linspace(0.0001, 0.01, 5)),
                'nu': list(np.linspace(0.001, 0.9999, 5)),
            }

    def fit(self, early_stopping_epochs, simulation):
        param_combinations = list(product(*self.param_grid.values()))
        best_score  = None
        best_lr = best_nu = None
        best_model_state = best_c = best_R = best_t1 = best_t2 = None

        for param in param_combinations:
            lr, nu = param
            trainer = TrainerDeepSVDD(self.args, self.train_loader, self.device,
                                      self.objective, lr=lr, R=0, nu=nu, warm_up_n_epochs=5)
            net, c, R = trainer.train(early_stopping_epochs)
            true_v, score_v = eval(net, c, R, self.objective, self.valid_loader, self.device)
            pred_v   = [0 if s <= 0 else 1 for s in score_v]
            f_beta   = fbeta_score(true_v, pred_v, beta=self.f_beta_param)

            if best_score is None or f_beta > best_score:
                best_score       = f_beta
                best_lr, best_nu = lr, nu
                best_model_state = copy.deepcopy(net.state_dict())
                best_c = c.clone(); best_R = R.clone()
                best_t1 = trainer.threshold_1; best_t2 = trainer.threshold_2

        torch.save({
            'net': best_model_state, 'c': best_c, 'R': best_R,
            'lr': best_lr, 'nu': best_nu,
            'threshold_1': best_t1, 'threshold_2': best_t2,
        }, f'{self.path}best_model/{simulation}/{simulation}_result.pt')

        self.best_model_state = best_model_state
        self.best_c = best_c; self.best_R = best_R


# ---------------------------------------------------------------------------
# Random Search
# ---------------------------------------------------------------------------
class RandomSearch:
    def __init__(self, args, train_loader, valid_loader, path, device,
                 verbose=0, objective='soft-boundary', f_beta_param=1):
        self.args         = args
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.path         = path
        self.device       = device
        self.verbose      = verbose
        self.objective    = objective
        self.f_beta_param = f_beta_param

        random.seed(1500)
        if objective == 'soft-boundary':
            self.param_random = {
                'lr': [random.uniform(0.0001, 0.01)    for _ in range(5)],
                'nu': [random.uniform(0.001,  0.9999)  for _ in range(5)],
            }

    def fit(self, early_stopping_epochs, simulation):
        param_combinations = list(product(*self.param_random.values()))
        best_score  = None
        best_lr = best_nu = None
        best_model_state = best_c = best_R = best_t1 = best_t2 = None

        for param in param_combinations:
            lr, nu = param
            trainer = TrainerDeepSVDD(self.args, self.train_loader, self.device,
                                      self.objective, lr=lr, R=0, nu=nu, warm_up_n_epochs=5)
            net, c, R = trainer.train(early_stopping_epochs)
            true_v, score_v = eval(net, c, R, self.objective, self.valid_loader, self.device)
            pred_v   = [0 if s <= 0 else 1 for s in score_v]
            f_beta   = fbeta_score(true_v, pred_v, beta=self.f_beta_param)

            if best_score is None or f_beta > best_score:
                best_score       = f_beta
                best_lr, best_nu = lr, nu
                best_model_state = copy.deepcopy(net.state_dict())
                best_c = c.clone(); best_R = R.clone()
                best_t1 = trainer.threshold_1; best_t2 = trainer.threshold_2

        torch.save({
            'net': best_model_state, 'c': best_c, 'R': best_R,
            'lr': best_lr, 'nu': best_nu,
            'threshold_1': best_t1, 'threshold_2': best_t2,
        }, f'{self.path}best_model/{simulation}/{simulation}_result.pt')

        self.best_model_state = best_model_state
        self.best_c = best_c; self.best_R = best_R


# ---------------------------------------------------------------------------
# Bayesian Optimization
# ---------------------------------------------------------------------------
class Bayesian:
    def __init__(self, args, train_loader, valid_loader, path, device, name,
                 early_stopping_epochs, objective='soft-boundary',
                 n_iter=25, f_beta_param=1, kappa=15):
        self.args             = args
        self.train_loader     = train_loader
        self.valid_loader     = valid_loader
        self.path             = path
        self.device           = device
        self.name             = name
        self.early_stopping_epochs = early_stopping_epochs
        self.objective        = objective
        self.n_iter           = n_iter
        self.f_beta_param     = f_beta_param
        self.kappa            = kappa

    def _bayesian_opt(self, init_lr, init_nu):
        trainer = TrainerDeepSVDD(self.args, self.train_loader, self.device,
                                  self.objective, lr=init_lr, R=0, nu=init_nu, warm_up_n_epochs=5)
        net, c, R = trainer.train(self.early_stopping_epochs)
        true_v, score_v = eval(net, c, R, self.objective, self.valid_loader, self.device)
        pred_v  = [0 if s <= 0 else 1 for s in score_v]
        f_beta  = fbeta_score(true_v, pred_v, beta=self.f_beta_param)

        # save candidate model for later retrieval via best params
        model_path = f'{self.path}bayes/{self.name}/hopt_lr_{init_lr:.6f}_nu_{init_nu:.6f}.pt'
        torch.save({
            'net': copy.deepcopy(net.state_dict()), 'c': c.clone(), 'R': R.clone(),
            'lr': init_lr, 'nu': init_nu,
            'threshold_1': trainer.threshold_1, 'threshold_2': trainer.threshold_2,
        }, model_path)
        return f_beta

    def fit(self, simulation):
        import os
        os.makedirs(f'{self.path}bayes/{self.name}', exist_ok=True)
        pbounds = {'init_lr': (0.0001, 0.01), 'init_nu': (0.001, 0.9999)}
        acq = UpperConfidenceBound(kappa=self.kappa)
        bo = BayesianOptimization(
            f=self._bayesian_opt, pbounds=pbounds,
            verbose=0, random_state=123, allow_duplicate_points=True,
            acquisition_function=acq,
        )
        bo.maximize(init_points=3, n_iter=self.n_iter)

        best = bo.max['params']
        model_path = f'{self.path}bayes/{self.name}/hopt_lr_{best["init_lr"]:.6f}_nu_{best["init_nu"]:.6f}.pt'
        model_dict = torch.load(model_path, weights_only=False)
        torch.save(model_dict, f'{self.path}best_model/{simulation}/{simulation}_result.pt')
        return model_dict


# ---------------------------------------------------------------------------
# Hyperband  (resource = epoch, RandomSampler + HyperbandPruner)
# ---------------------------------------------------------------------------
class Hyperband_hopt:
    """Hyperband: random sampling + successive halving via HyperbandPruner.
    Resource = training epochs. Trains epoch-by-epoch and prunes poor configs early.
    """
    def __init__(self, args, train_loader, valid_loader, path, device,
                 objective='soft-boundary', f_beta_param=1,
                 n_trials=25, max_resource=30, min_resource=1, reduction_factor=3):
        self.args         = args
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.path         = path
        self.device       = device
        self.objective    = objective
        self.f_beta_param = f_beta_param
        self.n_trials     = n_trials
        self.max_resource = max_resource
        self.min_resource = min_resource
        self.reduction_factor = reduction_factor

    def _objective(self, trial):
        lr = trial.suggest_float('lr', 0.0001, 0.01, log=True)
        nu = trial.suggest_float('nu', 0.001, 0.9999)

        net = LeNet5(self.args.latent_dim).to(self.device)
        net.apply(self._weights_init)
        c = torch.randn(self.args.latent_dim).to(self.device)
        R = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr,
                                     weight_decay=self.args.weight_decay)
        warm_up = 5
        best_score = 0.0

        for epoch in range(1, self.max_resource + 1):
            net.train()
            for data in self.train_loader:
                x, _, _ = data
                x = x.float().to(self.device)
                optimizer.zero_grad()
                z    = net(x)
                dist = torch.sum((z - c) ** 2, dim=1)
                score_t = dist - R ** 2
                loss    = R ** 2 + (1 / nu) * torch.mean(
                    torch.max(torch.zeros_like(score_t), score_t))
                if epoch >= warm_up:
                    R.data = torch.tensor(
                        np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu),
                        device=self.device, dtype=torch.float32)
                loss.backward()
                optimizer.step()

            true_v, score_v = eval(net, c, R, self.objective, self.valid_loader, self.device)
            pred_v  = [0 if s <= 0 else 1 for s in score_v]
            f_beta  = fbeta_score(true_v, pred_v, beta=self.f_beta_param)
            best_score = max(best_score, f_beta)

            trial.report(f_beta, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return best_score

    def _weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 and classname != 'Conv':
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('Linear') != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

    def fit(self, simulation):
        sampler = optuna.samplers.RandomSampler(seed=1500)
        pruner  = optuna.pruners.HyperbandPruner(
            min_resource=self.min_resource,
            max_resource=self.max_resource,
            reduction_factor=self.reduction_factor,
        )
        study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
        study.optimize(self._objective, n_trials=self.n_trials, show_progress_bar=False)

        best = study.best_params
        trainer = TrainerDeepSVDD(self.args, self.train_loader, self.device, self.objective,
                                  lr=best['lr'], R=0, nu=best['nu'], warm_up_n_epochs=5)
        net, c, R = trainer.train(early_stopping_epochs=5)

        torch.save({
            'net': net.state_dict(), 'c': c, 'R': R,
            'lr': best['lr'], 'nu': best['nu'],
            'threshold_1': trainer.threshold_1, 'threshold_2': trainer.threshold_2,
        }, f'{self.path}best_model/{simulation}/{simulation}_result.pt')


# ---------------------------------------------------------------------------
# BOHB  (resource = epoch, TPESampler + HyperbandPruner)
# ---------------------------------------------------------------------------
class BOHB_hopt:
    """BOHB: Bayesian optimization (TPE) + successive halving via HyperbandPruner.
    Same epoch-based resource schedule as Hyperband_hopt, with TPE instead of random sampling.
    """
    def __init__(self, args, train_loader, valid_loader, path, device,
                 objective='soft-boundary', f_beta_param=1,
                 n_trials=25, max_resource=30, min_resource=1, reduction_factor=3):
        self.args         = args
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.path         = path
        self.device       = device
        self.objective    = objective
        self.f_beta_param = f_beta_param
        self.n_trials     = n_trials
        self.max_resource = max_resource
        self.min_resource = min_resource
        self.reduction_factor = reduction_factor

    def _objective(self, trial):
        lr = trial.suggest_float('lr', 0.0001, 0.01, log=True)
        nu = trial.suggest_float('nu', 0.001, 0.9999)

        net = LeNet5(self.args.latent_dim).to(self.device)
        net.apply(self._weights_init)
        c = torch.randn(self.args.latent_dim).to(self.device)
        R = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr,
                                     weight_decay=self.args.weight_decay)
        warm_up = 5
        best_score = 0.0

        for epoch in range(1, self.max_resource + 1):
            net.train()
            for data in self.train_loader:
                x, _, _ = data
                x = x.float().to(self.device)
                optimizer.zero_grad()
                z    = net(x)
                dist = torch.sum((z - c) ** 2, dim=1)
                score_t = dist - R ** 2
                loss    = R ** 2 + (1 / nu) * torch.mean(
                    torch.max(torch.zeros_like(score_t), score_t))
                if epoch >= warm_up:
                    R.data = torch.tensor(
                        np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu),
                        device=self.device, dtype=torch.float32)
                loss.backward()
                optimizer.step()

            true_v, score_v = eval(net, c, R, self.objective, self.valid_loader, self.device)
            pred_v  = [0 if s <= 0 else 1 for s in score_v]
            f_beta  = fbeta_score(true_v, pred_v, beta=self.f_beta_param)
            best_score = max(best_score, f_beta)

            trial.report(f_beta, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return best_score

    def _weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 and classname != 'Conv':
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('Linear') != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

    def fit(self, simulation):
        sampler = optuna.samplers.TPESampler(seed=1500)
        pruner  = optuna.pruners.HyperbandPruner(
            min_resource=self.min_resource,
            max_resource=self.max_resource,
            reduction_factor=self.reduction_factor,
        )
        study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
        study.optimize(self._objective, n_trials=self.n_trials, show_progress_bar=False)

        best = study.best_params
        trainer = TrainerDeepSVDD(self.args, self.train_loader, self.device, self.objective,
                                  lr=best['lr'], R=0, nu=best['nu'], warm_up_n_epochs=5)
        net, c, R = trainer.train(early_stopping_epochs=5)

        torch.save({
            'net': net.state_dict(), 'c': c, 'R': R,
            'lr': best['lr'], 'nu': best['nu'],
            'threshold_1': trainer.threshold_1, 'threshold_2': trainer.threshold_2,
        }, f'{self.path}best_model/{simulation}/{simulation}_result.pt')
