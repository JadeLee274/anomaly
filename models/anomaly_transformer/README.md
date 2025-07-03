# Anomaly Transformer (ICLR 2022, Spotlight)
- Paper Link: https://arxiv.org/pdf/2110.02642

- Github link: https://github.com/thuml/Anomaly-Transformer/tree/main

## Assumption
Due to the continuity of time series, not only the point anomalies are given, but also anomaly segments are given. Authors assume that anomalies are strongly related to each other, but weakly related to noraml data. They suggest the additional attention layer based on the Gaussian kernel, which is named Anomaly Attention. This is the layer that calculates the attention score between data within the window, which is called Prior-Association. By the assumption on anomalies, anomalies have the high prior association, whereas the normal data don't. The Series-Association, is just the attention score of the vanilla transformer.

Also, authors suggests the novel anomaly score, which is called Association Discrepancy (AssDis). This is calculated as the symmetrized KL divergnce between Prior- and Series- Associations. As the PA/SA of anomaly is high/low and the PA/SA of normal data is medium/medium, the AssDis of anomalies will be lower that that of the normal ones. The training chooses minimax strategy.

Using the point-adjustment strategy, the model achieved the state-of-the-art performance compared to its baselines:

<p align='center'>
<img src='./pics/f1_scores.png' height='250' alt='' align=center />
</p>