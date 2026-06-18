#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)

class summary:
    def get_clf_eval(self, y_test, y_pred, fold=None):
        confusion = confusion_matrix(y_test, y_pred)
        accuracy  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        rc        = recall_score(y_test, y_pred)
        f1        = f1_score(y_test, y_pred, average='macro')
        roc_auc   = roc_auc_score(y_test, y_pred)

        self.rc      = rc
        self.f1      = f1
        self.roc_auc = roc_auc

        return rc, f1, roc_auc
