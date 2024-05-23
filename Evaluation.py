#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
import numpy as np
import pandas as pd

from Summary import summary
summary = summary()
def predict(n_splits, x_test_dataset, y_test_dataset, save_path):
    grid_df = pd.DataFrame(columns = ['grid_recall_score','grid_fl_score', 'grid_roc_auc_score'])
    random_df = pd.DataFrame(columns = ['random_recall_score','random_fl_score', 'random_roc_auc_score'])
    bayesian_df = pd.DataFrame(columns = ['bayes_recall_score','bayes_f1_score', 'bayes_roc_auc_score'])
        
    for i in range(100):
        grid = []; random = []; bayes = []
        result1 = []; result2 = []; result3 = []
        
        x_test = x_test_dataset[i]; y_test = y_test_dataset[i]
        
        for j in range(n_splits):
            with open(f'{save_path}_grid_cv_{i}_{j}','rb') as f:
                grid.append(pickle.load(f))
            
            with open(f'{save_path}_random_cv_{i}_{j}','rb') as f:
                random.append(pickle.load(f))
            
            with open(f'{save_path}_bayes_cv_{i}_{j}','rb') as f:
                bayes.append(pickle.load(f))
            
        ## gird search
        for model in grid:
            # model.fit(X_normal_train)
            pred = model.predict(x_test)
            result1.append(pred)
                
        ## random search
        for model in random:
            # model.fit(X_normal_train)
            pred = model.predict(x_test)
            result2.append(pred)
            
       ## bayesian optimization
        for model in bayes:
            # model.fit(X_normal_train)
            pred = model.predict(x_test)
            result3.append(pred)
            
        grid_result = (np.mean([arr.astype(int) for arr in result1], axis=0) > 0.5).astype(int)
        random_result = (np.mean([arr.astype(int) for arr in result2], axis=0) > 0.5).astype(int)
        bayes_result = (np.mean([arr.astype(int) for arr in result3], axis=0) > 0.5).astype(int)
        
        print("############ grid search ############")
        rc_1, f1_1, roc_auc_1 = summary.get_clf_eval(y_test, grid_result)
        print("############ random search ############")
        rc_2, f1_2, roc_auc_2 = summary.get_clf_eval(y_test, random_result)
        print("############ bayesian optimization ############")
        rc_3, f1_3, roc_auc_3 = summary.get_clf_eval(y_test, bayes_result)
            
        grid_df.loc[i] = [rc_1, f1_1, roc_auc_1]
        random_df.loc[i] = [rc_2, f1_2, roc_auc_2]
        bayesian_df.loc[i] = [rc_3, f1_3, roc_auc_3]
            
    test_sensitivity_score = pd.DataFrame({'grid': grid_df['grid_recall_score'], 'random': random_df['random_recall_score'], 'bayesian': bayesian_df['bayes_recall_score']})
    test_f1_score = pd.DataFrame({'grid': grid_df['grid_fl_score'], 'random': random_df['random_fl_score'], 'bayesian': bayesian_df['bayes_f1_score']})
    test_roc_auc_score = pd.DataFrame({'grid': grid_df['grid_roc_auc_score'], 'random': random_df['random_roc_auc_score'], 'bayesian': bayesian_df['bayes_roc_auc_score']})

    return test_sensitivity_score, test_f1_score, test_roc_auc_score
