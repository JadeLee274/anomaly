# About this repository:

This repository is for anomaly detection study; mainly about studying papers, codes, and running experiments by myself. Summaries for each directories are liseted below.

# Transformer-based Models
- Since the length of time series is long, transformer backbones are the good architecture in dealing with time series, modeling the temporal patterns of time sereis and catching the relationships between data points better than the classical recurrent neural netrowks such as vanilla RNN, LSTM, or GRU.

- The following directories are about the models that has transformer backbones, which are showing the state-of-the-art performance compared to the baselines such as classical machine learning models, and deep-learning models such as LSTM-VAE, OmniAnomaly, MAD-GAN.

## Anomaly Transformer [ICLR 2022, Spotlight]
- Detect anomalies with transformer architecture with new layer called anomlay attention.

- Based on the assumption that since time series is continuous, the anomalies will have less association with other data points except themselves, leading to the novel loss called association discrepancy.

## TranAD [VLDB 2022]
- Train the transformer-based model with adversarial training. (Similar to USAD)

- Data- and time-efficient, and also able to perform the root cause analysis for each dataset.


# Foundataion Models
- Foundation models can handle not only anomaly detection, but also generalizes to various time series tasks such as short/long term forecasting, imputation, and classification, with the same backbone.

- I added such model since I think that the foundataion model can handle such tasks since it can catch the rich information about the dataset, so that I can think about how to catch such information.

## TimesNet [ICLR 2023]
- Foundation model for general time seires task (anomaly detection, forecasting, imputation, classification)

- Used FFT to find the preiod and frequency of each features, making 1D data to 2D data. And then applied CNN-based netrowks (e.g. ResNet, Inception) to catch temporal informations.


# Root Cause Analysis
- Root cause analysis is useful in explaining how and where the anomaly happened, leading to the explainability of the anomaly detection models.

- It is also related to the real-life applications of the time series anomaliy detection task. For example, when it comes to the SWaT dataset, if someone poisined the water in some area, then it can lead to the entire pollution. So by RCA, one can cope with such problem before or after such problems.

## AERCA [ICLR 2025, Oral]
- Autoencoder-based Granger causal discovery, the effective statistical hypothesis test for determining whether one time series is usiful in forecasting another. 

- Decompose the t-th data point to the linear combination of previous data (i.e., window) and exogenous variables, and aims to learn the weights of them. 