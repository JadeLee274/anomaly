# anomaly

Repo for anomaly detection study. 
Mainly for studying papers, codes, and running experiments.

# Topics and github links

## Anomaly Transformer (ICLR 2022 Spotlight)
- Github link: https://github.com/thuml/Anomaly-Transformer/tree/main

- This paper assumes that, due to the continuity of time series, anomalies are strongly related to each other, but weakly related to noraml data.

- Authors suggest the additional attention layer based on the Gaussian kernel, which is named Anomaly Attention. This is the layer that calculates the attention score between data within the window, which is called Prior-Association. By the assumption on anomalies, anomalies have the high prior association, whereas the normal data don't.

- The Series-Association, is just the attention score of the vanilla transformer.

- Also, authors suggests the novel anomaly score, which is called Association Discrepancy. This is calculated as the symmetrized KL divergnce between Prior- and Series- Associations. As the PA/SA of anomaly is high/low and the PA/SA of normal data is medium/medium, the Association Discrepancy of anomalies will be lower that that of the normal ones.


## TranAD (VLDB 2022)
- Github link: https://github.com/imperial-qore/TranAD.git

## TimesNet (ICLR 2023)
- Github link: https://github.com/thuml/TimesNet
