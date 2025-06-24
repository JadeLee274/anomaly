# About this repository:

This repository is for anomaly detection study; mainly about studying papers, codes, and running experiments by myself. Summaries for each directories are liseted below.

## Anomaly Transformer [ICLR 2022, Spotlight]
- Detect anomalies with transformer architecture with new layer called anomlay attention.

- Based on the assumption that since time series is continuous, the anomalies will have less association with other data points except themselves, leading to the novel loss called association discrepancy.


## TranAD [VLDB 2022]
- Train the transformer-based model with adversarial training. (Similar to USAD)

- Data- and time-efficient, and also able to perform the root cause analysis for each dataset.


## TimesNet [ICLR 2023]
- Foundation model for general time seires task (anomaly detection, forecasting, imputation, classification)

- Used FFT to find the preiod and frequency of each features, making 1D data to 2D data. And then applied CNN-based netrowks (e.g. ResNet, Inception) to catch temporal informations.


## AERCA [ICLR 2025, Oral]
- Autoencoder-based Granger causal discovery, the effective statistical hypothesis test for determining whether one time series is usiful in forecasting another. 

- Decompose the t-th data point to the linear combination of previous data (i.e., window) and exogenous variables, and aims to learn the weights of them. 