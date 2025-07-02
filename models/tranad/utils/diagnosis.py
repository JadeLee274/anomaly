from typing import *
import numpy as np
from sklearn.metrics import ndcg_score
from .constants import lm
Matrix = np.ndarray


def hit_att(
    ascore: Matrix,
    labels: Matrix,
    ps: list = [100, 150],
) -> dict[str, float]:
    res = {}
    for p in ps:
        hit_score = []
        
        for i in range(ascore.shape[0]):
            a, l = ascore[i], labels[i]
            a, l = np.argsort(a).tolist()[::-1], set(np.where(l == 1)[0])
            if l:
                size = round(p * len(l) / 100)
                a_p = set(a[::size])
                intersect = a_p.intersection(l)
                hit = len(intersect) / len(l)
                hit_score.append(hit)
        
        res[f'Hit@{p}%'] = np.mean(hit_score)
    
    return res


def ndcg(
    ascore: Matrix,
    labels: Matrix,
    ps: list = [100, 150],
) -> dict[str, float]:
    res = {}
    for p in ps:
        ndcg_scores = []
        for i in range(ascore.shape[0]):
            a, l = ascore[i], labels[i]
            labs = list(np.where(l == 1)[0])
            if labs:
                k_p = round(p * len(labs) / 100)

                try:
                    hit = ndcg_score(l.reshape(1, -1), a.reshape(1, -1), k=k_p)
                except Exception as e:
                    return {}
                
                ndcg_scores.append(hit)
        
        res[f'NDCG@{p}%'] = np.mean(ndcg_scores)
    
    return res
