# Root Cause Analysis
Root cause analysis is useful in explaining how and where the anomaly happened, leading to the explainability of the anomaly detection models. It is also related to the real-life applications of the time series anomaliy detection task. For example, when it comes to the SWaT dataset, if someone poisined the water in some area, then it can lead to the entire pollution. So by RCA, one can cope with such problem before or after such problems.

## AERCA [ICLR 2025, Oral]
- Autoencoder-based Granger causal discovery, the effective statistical hypothesis test for determining whether one time series is usiful in forecasting another.

- Decompose the t-th data point to the linear combination of previous data (i.e., window) and exogenous variables, and aims to learn the weights of them. 