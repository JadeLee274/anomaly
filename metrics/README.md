# Metrics

## Limitations of F1-score
F1 metric is widely-used performance score in time series anomaly detection. But, the vanilla F1 is said to underestimate the performance of models, so many researchers (especially the research on state-of-the-art models) use the point-adjustment (PA) strategy. PA is based on the assumption that since the time series is the set of data collected continuously, anomalies will not only given in point, but also as a range.

Anomaly detection model is trained with the normal data, which are verified by the professionals of the field where each dataset is given. When it comes to unsupervised learning, the label is not given. But, the test set is given with the label. When the label is 0, then the data is normal, and when the label is 1, then the data is abnormal. The output of the model is the sequence of anomaly score (e.g. reconstruction loss, association discrepancy) of each data points. If a score is larger than the threshold, then the score becomes 1; otherwise 0.

So, the ground truth label (that is, test set label) and the model output is given as the binary sequence. By the assumption on time series, there is a sequence of 1's in the ground truth label. Assume that there is a range of anomalies in ground truth labels, where the index is from $M$ to $N$. Now, suppose that there exists $p \in [M, N]$ such that the $p$-th point is 1. Then PA adjusts $M$~$N$-th points of the output to 1.

By doing so, PA 'complements' the underestimated performance of the model. But, this leads to the overestimation of the performance of the model, since if the model can find only one point in the anomaly range, then PA considers that the model finds all point in the range. For example, when it comes to Anomaly Transformer trained and tested on PSM dataset, precision, recall, and F1 extremely varies. (For more examples and details, go to ../models/anomaly_transformer and run scripts.)

| | Precision | Recall | F1 |
| :-------: | :-------: | :-----: | :-----: |
| with PA | 0.9723 | 0.9810 | 0.9760 |
| w/o PA | 0.2834 | 0.0109 | 0.0210 |

So, the need for more reliable metric is needed for reliable research. This repository is on the suggested metrics.

## Suggested metrics
To be added soon