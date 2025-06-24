## TimesNet (ICLR 2023)
- Paper link: https://arxiv.org/pdf/2210.02186

- Github link: https://github.com/thuml/TimesNet

- TimesNet is a foundation model for time series, that is, it is not only for the anomaly detection task, but also for forecasting, classification, and imputation. Before this paper, few attempts were made to catch more intrinsic variations of the data; for example, seasonality, trend, etc. TimesNet is a model for catching such variations, so that it can perform multiple tasks on time series. Although it lacks F1-score on the widely-used datasets in anomaly detection task (such as SMD, PSM, SMAP, MSL, SWaT) compared to Anomaly Transformre, it is notable that this model tries to analyze time series more intrinsically. (This is the reason why I put it in my repository.)

- This architecture aims to catch such variations. Specifically, by applying the Fast Fourier Transform to data, one can decompose data into linear combination of periodic functions, with each period and frequency. By doing so, the 1D-tensor data is reshaped into 2D-tensor data, where the shape is period x frequency.

- With the process above, authors try to find out two types on variations: the intraperiod variation, and the interperiod variation. The former is the variation within each period, and the latter is the variation between each period.

- By using backbones for the computer vision tasks (in this paper, Conv2D-based architectures), TimesNet can simultaneously catch such variations, generalizing to various multivariate time series analysis task; anomaly detection, long/short time series forecasting, classification, imputation.

- But in the anomaly detection task, where the Anomaly Transformer is one of the baselines, authors changed the anomaly score of Anomaly Transformer from Association Discrepancy to Reconstruction error, for the fair comparison. This seems to be the tradeoff between focusing on the specific task and generalizing into various task.

- Following the github link of TimesNet, the model is in the time series library made by authors (https://github.com/thuml/Time-Series-Library/tree/main), and we can run an experiment not only on TimesNet, but also on various models such as Autoformer, FEDformer, iTransformer, etc. But in this repository, I simplified the structure of the files related to TimesNet, so that we can only focus on the time series anomaly detection task. But remember, TimesNet is a foundation model, so it will be more useful to go to the time series library link and look at the original code.