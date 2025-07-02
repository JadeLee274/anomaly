# Benchmarks
In real life, the anomalies are extremely rare, and has various patterns and densities (for example, the continuity assumption of the Anomaly Transformer may not work well in some datasets), meaning that there is no generalizable anomaly detection model. This directory deals with generating the time series benchmark datasets, which helps analyzing the performance of anomaly detection models in various situations, such as different anomaly ratio, different anomaly pattern, etc. The generation of the dataset is based on the specific formulation of various kinds of anomalies. 

## Revisiting Time Series Outlier Detection [NeurIPS 2021]
- Defines and formulates the various kinds of anomalies, wich are coded so that the various kinds of datasets can be generated to analyze the performance of various models in various situations.

- In many situations, the classical machine learning models may perform better than the deep learning models, especially Autoregression models.