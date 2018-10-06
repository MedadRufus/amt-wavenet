from __future__ import division

import sys
import numpy as np

from .piano_roll import thresh


def calc_stats(predictions, labels, thr):
    p = thresh(predictions, thr, True).astype(bool)
    l = thresh(labels, thr, True).astype(bool)

    P = np.sum(l)
    N = np.size(l) - P
    TP = np.sum(p * l)
    TN = np.sum((~p) * (~l))
    FP = np.sum((p ^ l) * p)
    FN = np.sum((p ^ l) * l)

    return P, N, TP, TN, FP, FN


def calc_metrics(predictions, labels, thr=0.5, stats=None):
    '''Calculates and returns dicitonary of most relevant
    MIR metrics.
    '''

    if stats:
        P, N, TP, TN, FP, FN = stats
    else:
        P, N, TP, TN, FP, FN = calc_stats(predictions,
                                          labels,
                                          thr)

    c = sys.float_info.min # prevent zero division

    metrics = {}
    # Positive Predictive Value - Precision
    metrics['precision_rate'] = (TP) / (TP + FP + c)
    # True Positive Rate - Recall
    metrics['recall_rate'] = (TP) / (P + c)
    # False Discovery Rate
    metrics['false_discovery_rate'] = (FP) / (FP + TP + c)
    # False Positive Rate - Fall-out
    metrics['false_positive_rate'] = (FP) / (N + c)

    # F1-Score
    metrics['f1_measure'] = ((metrics['precision_rate'] *
                              metrics['recall_rate']) /
                             (metrics['precision_rate'] +
                              metrics['recall_rate'] + c)
                            ) * 2
    # Frame-level Accuracy as proposed by Dixon [2000]
    metrics['acc_measure'] = (TP) / (FP + FN + TP + c)

    return metrics


def metrics_empty_dict():
    return {'acc_measure' : None,
            'f1_measure' : None,
            'precision_rate' : None,
            'recall_rate' : None,
            'false_discovery_rate' : None,
            'false_positive_rate' : None
           }
