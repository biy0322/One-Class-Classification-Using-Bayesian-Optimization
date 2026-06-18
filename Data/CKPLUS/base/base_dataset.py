#!/usr/bin/env python
# coding: utf-8

from abc import ABC, abstractmethod
from torch.utils.data import DataLoader

class BaseADDataset(ABC):
    def __init__(self):
        super().__init__()
        self.n_classes      = 2
        self.normal_classes  = None
        self.outlier_classes = None
        self.train_set = None
        self.test_set  = None

    @abstractmethod
    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False,
                num_workers: int = 0) -> (DataLoader, DataLoader):
        pass

    def __repr__(self):
        return self.__class__.__name__
