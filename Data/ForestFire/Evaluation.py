#!/usr/bin/env python
# coding: utf-8

from Summary import summary
from Dataset import Dataset

import pickle
import pandas as pd
import numpy as np

summary = summary()


def predict(n_splits=9, save_path=None):
    dataset = Dataset(load_path='./Data/data_manipulated.csv')
    x_test = dataset.X
    y_test = dataset.y

    be_hopt = []; grid = []; random = []; bayes = []; hyperband = []; bohb = []

    for j in range(n_splits):
        with open(f'{save_path}_model_cv_{j}', 'rb') as f:
            be_hopt.append(pickle.load(f))
        with open(f'{save_path}_grid_cv_{j}', 'rb') as f:
            grid.append(pickle.load(f))
        with open(f'{save_path}_random_cv_{j}', 'rb') as f:
            random.append(pickle.load(f))
        with open(f'{save_path}_bayes_cv_{j}', 'rb') as f:
            bayes.append(pickle.load(f))
        with open(f'{save_path}_hyperband_cv_{j}', 'rb') as f:
            hyperband.append(pickle.load(f))
        with open(f'{save_path}_bohb_cv_{j}', 'rb') as f:
            bohb.append(pickle.load(f))

    def majority_vote(models):
        preds = [m.predict(x_test) for m in models]
        return (np.mean([p.astype(int) for p in preds], axis=0) > 0.5).astype(int)

    be_result       = majority_vote(be_hopt)
    grid_result     = majority_vote(grid)
    random_result   = majority_vote(random)
    bayes_result    = majority_vote(bayes)
    hband_result    = majority_vote(hyperband)
    bohb_result     = majority_vote(bohb)

    print("########### before hopt ###########")
    rc_0, f1_0, roc_0 = summary.get_clf_eval(y_test, be_result)
    print("########### grid search ###########")
    rc_1, f1_1, roc_1 = summary.get_clf_eval(y_test, grid_result)
    print("########### random search ###########")
    rc_2, f1_2, roc_2 = summary.get_clf_eval(y_test, random_result)
    print("########### bayesian optimization ###########")
    rc_3, f1_3, roc_3 = summary.get_clf_eval(y_test, bayes_result)
    print("########### hyperband ###########")
    rc_4, f1_4, roc_4 = summary.get_clf_eval(y_test, hband_result)
    print("########### BOHB ###########")
    rc_5, f1_5, roc_5 = summary.get_clf_eval(y_test, bohb_result)

    total_result = pd.DataFrame(
        {
            'recall_score': [rc_0, rc_1, rc_2, rc_3, rc_4, rc_5],
            'f1_score':     [f1_0, f1_1, f1_2, f1_3, f1_4, f1_5],
            'roc_auc_score':[roc_0, roc_1, roc_2, roc_3, roc_4, roc_5],
        },
        index=['before_hopt', 'grid', 'random', 'bayesian', 'hyperband', 'bohb'],
    )

    return y_test, [be_result, grid_result, random_result, bayes_result, hband_result, bohb_result], total_result
