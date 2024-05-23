#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
    def result(self, data, scoring):
        test_summary = data.describe().iloc[1:,:]
        
        index = test_summary.index
        if scoring=='recall':
            columns = pd.MultiIndex.from_product([['test_sensitivity_score'], test_summary.columns])
            summary_df = pd.DataFrame(index=index, columns=columns)
            summary_df.loc[:,'test_sensitivity_score'] = test_summary.values
            return summary_df
        elif scoring=='f1':
            columns = pd.MultiIndex.from_product([['test_F1_score'], test_summary.columns])
            summary_df = pd.DataFrame(index=index, columns=columns)
            summary_df.loc[:,'test_F1_score'] = test_summary.values
            return summary_df
        elif scoring=='roc-auc':
            columns = pd.MultiIndex.from_product([['test_roc_auc_score'], test_summary.columns])
            summary_df = pd.DataFrame(index=index, columns=columns)
            summary_df.loc[:,'test_roc_auc_score'] = test_summary.values
            return summary_df
            
    def get_clf_eval(self, y_test, y_pred, fold=None):
        confusion = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro")
        rc = recall_score(y_test, y_pred)
        F1 = f1_score(y_test, y_pred, average = "macro")
        roc_auc = roc_auc_score(y_test, y_pred)
        
          
   #     print(f"오차행렬:\n", confusion)
   #     print("\n정확도: {:.4f}".format(accuracy))
   #     print("정밀도: {:.4f}".format(precision))
   #     print("재현율: {:.4f}".format(rc))
   #     print("F1: {:.4f}".format(F1))
        
        return rc, F1, roc_auc