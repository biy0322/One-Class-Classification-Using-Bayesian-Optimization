#!/usr/bin/env python
# coding: utf-8

import numpy as np

def epsilon_threshold(train_scores, reg_level=1):
    e_s = train_scores
    best_threshold = None
    max_score = -10000000
    mean_e_s = np.mean(e_s)
    sd_e_s   = np.std(e_s)

    for z in np.arange(2.5, 12, 0.5):
        epsilon    = mean_e_s + sd_e_s * z
        pruned_e_s = e_s[e_s < epsilon]
        i_anom     = np.argwhere(e_s >= epsilon).reshape(-1,)
        buffer     = np.arange(1, 50)
        i_anom = np.sort(np.unique(np.concatenate((
            i_anom,
            np.array([i + buffer for i in i_anom]).flatten(),
            np.array([i - buffer for i in i_anom]).flatten(),
        ))))
        i_anom = i_anom[(i_anom < len(e_s)) & (i_anom >= 0)]

        if len(i_anom) > 0:
            mean_perc_decrease = (mean_e_s - np.mean(pruned_e_s)) / mean_e_s
            sd_perc_decrease   = (sd_e_s   - np.std(pruned_e_s))  / sd_e_s
            denom = {0: 1, 1: len(i_anom), 2: len(i_anom) ** 2}.get(reg_level, len(i_anom))
            score = (mean_perc_decrease + sd_perc_decrease) / denom

            if score >= max_score and len(i_anom) < len(e_s) * 0.5:
                max_score      = score
                best_threshold = epsilon

    if best_threshold is None:
        best_threshold = np.max(e_s)
    return best_threshold
