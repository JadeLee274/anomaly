from typing import *
import numpy as np
from sklearn.metrics import *
from .spot import SPOT
from .constants import *
Vector = np.ndarray
Matrix = np.ndarray


def calc_point2point(
    predict,
    actual
) -> Tuple[float, float, float, float, float, float, float, float]:
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall  / (precision + recall + 0.00001)
    
    try:
        roc_auc = roc_auc_score(actual, predict)
    except:
        roc_auc = 0
    
    return f1, precision, recall, TP, TN, FP, FN, roc_auc


def adjust_predicts(
    score,
    label,
    threshold=None,
    pred=None,
    calc_latency: bool = False,
):
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0

    if pred is None:
        predict = score > threshold
    else:
        predict = pred

    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0

    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    if not predict[j]:
                        predict[j] = True
                        latency += 1
        elif not actual[i]:
            anomaly_state = False
        
        if anomaly_state:
            predict[i] = True
    
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict
    

def calc_seq(
    score,
    label,
    threshold,
    calc_latency: bool = False,
):
    if calc_latency:
        predict, latency = adjust_predicts(
            score,
            label,
            calc_latency=calc_latency
        )
        t = list(calc_point2point(predict, label))
        t.append(latency)
        return t
    else:
        predict = adjust_predicts(
            score,
            label,
            threshold,
            calc_latency=calc_latency,
        )
        return calc_point2point(predict, label)
    

def bf_search(
    score,
    label,
    start,
    end=None,
    step_num: int = 1,
    display_freq: int = 1,
    verbose: bool = True,
):
    if step_num is None or end is None:
        end = start
        step_num = 1
    
    search_step, search_range, search_lower_bound = step_num, end - start, start

    if verbose:
        print(
            'search range: ',
            search_lower_bound,
            search_lower_bound + search_range
        )
    
    threshold = search_lower_bound

    m = (-1., -1., -1.)
    m_t = 0.0

    for i in range(search_step):
        threshold += search_range / float(search_step)
        target = calc_seq(score, label, threshold, calc_latency=True)
        if target[0] > m[0]:
            m_t = threshold
            m = target
        if verbose and i % display_freq == 0:
            print('cur thr: ', threshold, target, m, m_t)
    
    print(m, m_t)
    return m, m_t


def pot_eval(
    init_score: Matrix,
    score: Matrix,
    label: Matrix,
    q: float = 1e-5,
    level: float = 0.02,
):
    lms = lm[0]

    while True:
        try:
            s = SPOT(q)
            s.fit(init_score, score)
            s.initialize(level=lms, min_extrema=False, verbose=False)
        except:
            lms = lms * 0.999
        else:
            break
    
    ret = s.run(dynamic=False)
    pot_th = np.mean(ret['thresholds']) * lm[1]
    pred, p_latency = adjust_predicts(
        score,
        label,
        pot_th,
        calc_latency=True,
    )
    p_t = calc_point2point(pred, label)

    return {
        'f1': p_t[0],
        'precision': p_t[1],
        'recall': p_t[2],
        'TP': p_t[3],
        'TN': p_t[4],
        'FP': p_t[5],
        'FN': p_t[6],
        'ROC/AUC': p_t[7],
        'threshold': pot_th,
    }, np.array(pred)
