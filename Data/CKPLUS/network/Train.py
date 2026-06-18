#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import random

from .network import LeNet5
from .epsilon_thres import epsilon_threshold


class TrainerDeepSVDD:
    def __init__(self, args, data_loader, device, objective,
                 lr=None, beta_1=None, beta_2=None,
                 R=None, nu=None, warm_up_n_epochs=None):
        self.args         = args
        self.train_loader = data_loader
        self.device       = device
        self.objective    = objective
        self.lr           = lr
        self.beta_1       = beta_1
        self.beta_2       = beta_2
        self.R            = torch.tensor(R if R is not None else 0.0, device=self.device, dtype=torch.float32)
        self.nu           = nu
        self.warm_up_n_epochs = warm_up_n_epochs if warm_up_n_epochs is not None else 5

        # reproducibility
        seed = 1500
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark     = True

    def train(self, early_stopping_epochs):
        net = LeNet5(self.args.latent_dim).to(self.device)
        net.apply(self.weights_init_normal)
        c = torch.randn(self.args.latent_dim).to(self.device)

        if self.objective == 'one-class':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.lr,
                                         betas=(self.beta_1, self.beta_2),
                                         weight_decay=self.args.weight_decay)
        elif self.objective == 'soft-boundary':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.lr,
                                         weight_decay=self.args.weight_decay)

        early_stop_counter = 0
        min_loss   = float('inf')
        best_model = None

        net.train()
        for epoch in range(self.args.num_epochs):
            total_loss  = 0.0
            total_score = []

            for data in self.train_loader:
                x, _, _ = data
                x = x.float().to(self.device)
                optimizer.zero_grad()
                z    = net(x)
                dist = torch.sum((z - c) ** 2, dim=1)

                if self.objective == 'one-class':
                    loss = torch.mean(dist)

                elif self.objective == 'soft-boundary':
                    score = dist - self.R ** 2
                    loss  = self.R ** 2 + (1 / self.nu) * torch.mean(
                        torch.max(torch.zeros_like(score), score))
                    if epoch >= self.warm_up_n_epochs:
                        self.R.data = torch.tensor(
                            self.get_radius(dist, self.nu), device=self.device, dtype=torch.float32)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_score += list(dist.detach().cpu().numpy())

            if total_loss < min_loss:
                best_model   = net
                threshold_1  = epsilon_threshold(np.array(total_score))
                q3 = np.quantile(total_score, 0.75)
                q1 = np.quantile(total_score, 0.25)
                threshold_2  = q3 + (q3 - q1) * 1.5
                min_loss     = total_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if epoch >= 10 and early_stop_counter >= early_stopping_epochs:
                break

        self.best_model  = best_model
        self.threshold_1 = threshold_1
        self.threshold_2 = threshold_2
        self.net = net
        self.c   = c
        return self.best_model, c, self.R

    def weights_init_normal(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 and classname != 'Conv':
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('Linear') != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

    def get_radius(self, dist: torch.Tensor, nu: float):
        return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
