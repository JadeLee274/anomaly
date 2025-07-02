# Models
This repository is about the various models for the time sereis anomlay detection tasks. Based on the F1-score, each models gained the state-of-the-art title title when the papers on them were released. But, note that some of them used the point-adjustment strategy for their F1-score. Such strategy is for the complement of vanilla F1, which can underestimate the performance of the model; but is critisized since it can overestimate the performance. Check the [metric] repository is for more detail. 

## Anomaly Transformer [ICLR 2022, Spotlight]
- Detect anomalies with transformer architecture with new layer called anomlay attention.

- Based on the assumption that since time series is continuous, the anomalies will have less association with other data points except themselves, leading to the novel loss called association discrepancy.

## TranAD [VLDB 2022]
- Train the transformer-based model with adversarial training. (Similar to USAD)

- Data- and time-efficient, and also able to perform the root cause analysis for each dataset.

## TimesNet [ICLR 2023]
- Foundation model for general time seires task (anomaly detection, forecasting, imputation, classification)

- Used FFT to find the preiod and frequency of each features, making 1D data to 2D data. And then applied CNN-based netrowks (e.g. ResNet, Inception) to catch temporal informations.