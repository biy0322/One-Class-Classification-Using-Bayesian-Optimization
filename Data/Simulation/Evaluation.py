#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
import numpy as np
import pandas as pd

from Summary import summary
summary = summary()

def predict(n_splits, x_test_dataset, y_test_dataset, save_path, n_repeats=None):
    n_runs = len(x_test_dataset) if n_repeats is None else n_repeats
    grid_df      = pd.DataFrame(columns=['grid_recall_score',      'grid_f1_score',      'grid_roc_auc_score'])
    random_df    = pd.DataFrame(columns=['random_recall_score',    'random_f1_score',    'random_roc_auc_score'])
    bayesian_df  = pd.DataFrame(columns=['bayes_recall_score',     'bayes_f1_score',     'bayes_roc_auc_score'])
    hyperband_df = pd.DataFrame(columns=['hyperband_recall_score', 'hyperband_f1_score', 'hyperband_roc_auc_score'])
    bohb_df      = pd.DataFrame(columns=['bohb_recall_score',      'bohb_f1_score',      'bohb_roc_auc_score'])

    for i in range(n_runs):
        print(f"{i}-th fold evaluation...")
        grid = []; random = []; bayes = []; hyperband = []; bohb = []
        result1 = []; result2 = []; result3 = []; result4 = []; result5 = []

        x_test = x_test_dataset[i]; y_test = y_test_dataset[i]

        for j in range(n_splits):
            with open(f'{save_path}_grid_cv_{i}_{j}', 'rb') as f:
                grid.append(pickle.load(f))

            with open(f'{save_path}_random_cv_{i}_{j}', 'rb') as f:
                random.append(pickle.load(f))

            with open(f'{save_path}_bayes_cv_{i}_{j}', 'rb') as f:
                bayes.append(pickle.load(f))

            with open(f'{save_path}_hyperband_cv_{i}_{j}', 'rb') as f:
                hyperband.append(pickle.load(f))

            with open(f'{save_path}_bohb_cv_{i}_{j}', 'rb') as f:
                bohb.append(pickle.load(f))

        for model in grid:
            result1.append(model.predict(x_test))

        for model in random:
            result2.append(model.predict(x_test))

        for model in bayes:
            result3.append(model.predict(x_test))

        for model in hyperband:
            result4.append(model.predict(x_test))

        for model in bohb:
            result5.append(model.predict(x_test))

        # majority vote across CV folds
        grid_result      = (np.mean([arr.astype(int) for arr in result1], axis=0) > 0.5).astype(int)
        random_result    = (np.mean([arr.astype(int) for arr in result2], axis=0) > 0.5).astype(int)
        bayes_result     = (np.mean([arr.astype(int) for arr in result3], axis=0) > 0.5).astype(int)
        hyperband_result = (np.mean([arr.astype(int) for arr in result4], axis=0) > 0.5).astype(int)
        bohb_result      = (np.mean([arr.astype(int) for arr in result5], axis=0) > 0.5).astype(int)

        rc_1, f1_1, roc_auc_1 = summary.get_clf_eval(y_test, grid_result)
        rc_2, f1_2, roc_auc_2 = summary.get_clf_eval(y_test, random_result)
        rc_3, f1_3, roc_auc_3 = summary.get_clf_eval(y_test, bayes_result)
        rc_4, f1_4, roc_auc_4 = summary.get_clf_eval(y_test, hyperband_result)
        rc_5, f1_5, roc_auc_5 = summary.get_clf_eval(y_test, bohb_result)

        grid_df.loc[i]      = [rc_1, f1_1, roc_auc_1]
        random_df.loc[i]    = [rc_2, f1_2, roc_auc_2]
        bayesian_df.loc[i]  = [rc_3, f1_3, roc_auc_3]
        hyperband_df.loc[i] = [rc_4, f1_4, roc_auc_4]
        bohb_df.loc[i]      = [rc_5, f1_5, roc_auc_5]

    test_sensitivity_score = pd.DataFrame({
        'grid':      grid_df['grid_recall_score'],
        'random':    random_df['random_recall_score'],
        'bayesian':  bayesian_df['bayes_recall_score'],
        'hyperband': hyperband_df['hyperband_recall_score'],
        'bohb':      bohb_df['bohb_recall_score'],
    })
    test_f1_score = pd.DataFrame({
        'grid':      grid_df['grid_f1_score'],
        'random':    random_df['random_f1_score'],
        'bayesian':  bayesian_df['bayes_f1_score'],
        'hyperband': hyperband_df['hyperband_f1_score'],
        'bohb':      bohb_df['bohb_f1_score'],
    })
    test_roc_auc_score = pd.DataFrame({
        'grid':      grid_df['grid_roc_auc_score'],
        'random':    random_df['random_roc_auc_score'],
        'bayesian':  bayesian_df['bayes_roc_auc_score'],
        'hyperband': hyperband_df['hyperband_roc_auc_score'],
        'bohb':      bohb_df['bohb_roc_auc_score'],
    })

    return test_sensitivity_score, test_f1_score, test_roc_auc_score
