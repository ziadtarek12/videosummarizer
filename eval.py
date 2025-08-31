"""
eval.py
- fscore_vs_gt(pred_shots, gt_shots)
- diversity metrics (avg pairwise cosine distance)
- kendall_tau between predicted importance ranking and ground-truth importance scores
"""

import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics.pairwise import cosine_similarity

def overlap(a,b):
    return max(0.0, min(a[1],b[1]) - max(a[0],b[0]))

def fscore_vs_gt(pred_shots, gt_shots):
    pred_total = sum(e-s for s,e in pred_shots)
    gt_total = sum(e-s for s,e in gt_shots)
    inter = 0.0
    for p in pred_shots:
        for g in gt_shots:
            inter += overlap(p,g)
    prec = inter / (pred_total + 1e-9)
    rec  = inter / (gt_total + 1e-9)
    f = 2*prec*rec/(prec+rec+1e-9)
    return dict(precision=prec, recall=rec, fscore=f)

def avg_pairwise_dist(embs):
    if len(embs) < 2: return 0.0
    sims = cosine_similarity(embs)
    n = sims.shape[0]
    np.fill_diagonal(sims, 0.0)
    pairs = n*(n-1)
    return float(sims.sum()/pairs)

def kendall_rank(pred_scores, gt_scores):
    # both should be 1D arrays of same length; higher is more important
    tau, p = kendalltau(pred_scores, gt_scores)
    return dict(tau=tau, pvalue=p)
