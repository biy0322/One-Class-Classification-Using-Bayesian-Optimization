#!/usr/bin/env python
# coding: utf-8

from .base_dataset import BaseADDataset
from torch.utils.data import DataLoader

class TorchvisionDataset(BaseADDataset):
    def __init__(self):
        super().__init__()

    def __getitem__(self, index):
        raise NotImplementedError("Subclasses must implement __getitem__ method!")

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False,
                num_workers: int = 0) -> (DataLoader, DataLoader):
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size,
                                  shuffle=shuffle_train, num_workers=num_workers, drop_last=False)
        test_loader  = DataLoader(dataset=self.test_set,  batch_size=batch_size,
                                  shuffle=shuffle_test,  num_workers=num_workers, drop_last=False)
        return train_loader, test_loader
