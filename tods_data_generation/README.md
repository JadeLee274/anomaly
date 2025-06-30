# Revisiting Time Series Outlier Detection [NeurIPS 2021]
- Paper link: https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/ec5decca5ed3d6b8079e2e7e7bacc9f2-Paper-round1.pdf

- Github link: https://github.com/datamllab/tods/tree/benchmark/benchmark/synthetic (This link is a part of tods, the anomaly detection package of the authors of this paper. To see the full project, go to https://github.com/datamllab/tods/tree/master.)

- Due to the ambiguity of anomalies, various types of time series anomalies are ill-defined in many previous studies. This paper classifies and formulates the types of anomalies and synthesizes the univariate and multivariate time series based on the formulation. Time series anomlies are classified into two categories: point anomalies and pattern anomalies.

## Point anomalies
- Point anomalies are global anomlies (extreme values) and contextual anomalies (relatively small/large in context, e.g. windows; but not globally). Point anomalies are formulated into $|x_{t} - \hat{x}_{t}| > \delta$, where $\delta$ is a threshold. 

- The threshold of global anomaly can be formulated as $\delta = \lambda \cdot \sigma(X)$, where $\lambda$ is the scale about how far the anomlay is from the normal data, and $\sigma$ is the standard deviation of the entire dataset X

- The threshold of contextual anomaly can be formulated as $\delta = \lambda \cdot \sigma(X_{(t-k: t+k)})$, where $\lambda$ is similar to that of global anomaly, and $\sigma$ is the standard deviation of the window $X_{(t-k: t+k)}$.


## Pattern anomalies
- Pattern anomalies are anomalous subsequences. They can be classified into shapelet anomalies (a.k.a. collective anomalies, the subsequences with dissimilar shapelet compared to the normal shapelet), seasonal anomalies (unusual seasonality), and trend anomalies (unusual trend).

- Pattern anomalies can be formulated as $X_{(i: j)} = \rho(2\pi\omega T_{(i: j)}) + \tau(T_{(i: j)})$, where $T_{(i: j)}$ is the sub-timestamps; $\rho$ models the shape of the subsequence (the most familiar example is fourier transform); and $\tau$ models the trend.

- Here, the threshold and measure of dissimilarity are given as $\delta$ and $s$, respectively.

- The shapelet anomaly is formulated as $s(\rho(.), \hat{\rho}(.)) > \delta$, where $\rho$ is the shapelet of the original data, and $\hat{\rho}$ is that of the output of some ML/DL model - that is, a prediction.

- The seasonal anomaly is formulated as $s(\omega, \hat{\omega}) > \delta$, where $\omega$ is the seasonality of the original data, and $\hat{\omega}$ is the seasonality of the output of a prediction.

- The trend anomaly is formulated as $s(\tau, \hat{\tau}) > \delta$, where $\tau$ is the trend of the original data, and $\hat{\tau}$ is the trend of the output of a prediction.

## Data Generation and F1-based Scores 
- First of all, note that the deep-learning-based anomaly detection models are based on Autoencoder, RNN+LSTM, and GAN. Transformer or Diffusion backbones are not included in the paper. (Anomaly Transformer gained better/similar F1 score compared to the classical models that will be discussed in this section.)

- Based on the formulations, authors coded the data generation process in both univariate, multivariate version. So that we can generate the synthesize time series anomaly detection dataset. I modified the multivariate dataset generation code so that we can flexibly generate data. 

- Surprisingly, classical machine learning algorithms beat the Autoencoder-, LSTM+RNN- and GAN- based deep learning algorithms on the datasets.

- Especially, Autoregression has shown the best performance in detecting contextual/shapelet. For the latter case, this is probably because AR uses contextuality for self-regression in windows.

- Also, for the real world datasets, classical algorithms tend to detect anomalies better than the deep-learning-based algorithms. Especially GAN performed very poorly, possibly because of the various patterns of anomalies.