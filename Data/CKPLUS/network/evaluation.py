#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import random

from .network import LeNet5


def eval(net, c, R, objective, dataloader, device):
    """Test the Deep SVDD model and return (labels, scores)."""
    scores = []; labels = []
    net.eval()
    with torch.no_grad():
        for data in dataloader:
            x, y, _ = data
            x = x.float().to(device)
            z = net(x)
            if objective == 'one-class':
                score = torch.sum((z - c) ** 2, dim=1)
            elif objective == 'soft-boundary':
                score = torch.sum((z - c) ** 2, dim=1) - R ** 2
            scores.append(score.detach().cpu())
            labels.append(y.cpu())
    labels = torch.cat(labels).numpy()
    scores = torch.cat(scores).numpy()
    return labels, scores
