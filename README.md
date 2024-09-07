# One-Class-Classification-Using-Bayesian-Optimization

This paper explores one-class classification (OCC), a pivotal technique for identifying anomalies in datasets where normal and abnormal data are imbalanced. In particular, we delve into two significant OCC methodologies: the one class support vector machines (OC-SVM) and deep support vector data description (Deep SVDD). Both approaches rely heavily on the precise adjustment of hyperparameters. To optimize these settings, most studies, including ours, evaluate three common methodologies: grid search, random search, and Bayesian optimization. In this work, we compare the performance of these techniques to determine which method best enhances model accuracy. We demonstrate that Bayesian optimization, which utilizes Gaussian processes, consistently outperforms the other two methods in both synthetic and real-data applications, including forest fire risk data for artillery military training and the emotional data from the Extended Cohn-Kanade (CK+) dataset.

### Performance Summary of Synthetic Dataset
We investigated this by varying the ratio of normal to abnormal data to 95:5, 90:10, 80:20, 70:30, and 60:40. The training dataset to test dataset ratio was maintained at 8:2. Below table presents the comparison of hyperparameters $\nu$ and $\gamma$ obtained through grid search, random search, and Bayesian optimization using three performance evaluation metrics. 

| Ratio<sup>1</sup> | Recall (GC)<sup>2</sup> | Recall (RC) | Recall (BO) | F-1 (GC) | F-1 (RC) | F-1 (BO) | AUC (GC) | AUC (RC) | AUC (BO) |
|-------------------|-------------------------|-------------|-------------|----------|----------|----------|----------|----------|----------|
| 95:5              | 0.7790 (0.0311)         | 0.8070 (0.0303) | 0.7880 (0.0312) | 0.7076 (0.0102) | 0.6814 (0.0087) | 0.7510 (0.0130) | 0.8567 (0.0160) | 0.8514 (0.0154) | 0.8620 (0.0165) |
| 90:10             | 0.8035 (0.0300)         | 0.8040 (0.0304) | 0.7945 (0.0300) | 0.7656 (0.0121) | 0.7488 (0.0112) | 0.7923 (0.0139) | 0.8536 (0.0158) | 0.8482 (0.0155) | 0.8588 (0.0161) |
| 80:20             | 0.8527 (0.0249)         | 0.8558 (0.0253) | 0.8355 (0.0266) | 0.8425 (0.0121) | 0.8302 (0.0115) | 0.8562 (0.0133) | 0.8795 (0.0133) | 0.8746 (0.0131) | 0.8821 (0.0141) |
| 70:30             | 0.8438 (0.0240)         | 0.8505 (0.0236) | 0.8407 (0.0228) | 0.8580 (0.0123) | 0.8499 (0.0116) | 0.8713 (0.0123) | 0.8735 (0.0127) | 0.8694 (0.0122) | 0.8807 (0.0126) |
| 60:40             | 0.8515 (0.0228)         | 0.8508 (0.0236) | 0.8476 (0.0224) | 0.8689 (0.0130) | 0.8614 (0.0129) | 0.8793 (0.0134) | 0.8754 (0.0127) | 0.8698 (0.0126) | 0.8832 (0.0131) |

<sup>1</sup> The ratio of normal to abnormal  
<sup>2</sup> GC, RC, and BO refer to grid search, random search, and Bayesian optimization, respectively.  
Numbers in parentheses are standard errors.



Bayesian optimization consistently demonstrated superior performance across all cases based on the F-1 score and AUC. Moreover, as the ratio of normal to abnormal data became more imbalanced, the performance of Bayesian optimization improved, indicating its efficacy in handling unbalanced datasets.
