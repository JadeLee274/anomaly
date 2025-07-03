# Models
This repository is about the various models for the time sereis anomlay detection tasks. Based on the F1-score, each models gained the state-of-the-art title title when the papers on them were released. But, note that some of them used the point-adjustment strategy for their F1-score. Such strategy is for the complement of vanilla F1, which can underestimate the performance of the model; but is critisized since it can overestimate the performance. Check ../metric for more detail. 

## Anomaly Transformer [ICLR 2022, Spotlight]
Since time series is a set of sequential data, the anomalies will also be given as segments. Using the newly suggested attention layer called Anomaly Attention and loss called Association Discrepancy, Anomaly Transformer achieved the state-of-the-art performance compared to its baselines.

## TranAD [VLDB 2022]
Following the USAD, TranAD trains the transformer-based model with adversarial training, making the robust weights. Also, it is data- and time-efficient, and also able to do root cause analysis for each dataset.

## TimesNet [ICLR 2023]
TimesNet is a foundation model for general time seires task (anomaly detection, forecasting, imputation, classification). Using Fast Fourier Transform (FFT) to find the preiod and frequency of each features, the model makes 1D data to 2D data, applying CNN-based netrowks (e.g. ResNet, Inception) to catch the intra- inter-period variations of data.