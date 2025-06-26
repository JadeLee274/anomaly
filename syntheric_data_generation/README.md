# Revisiting Time Series Outlier Detection [NeurIPS 2021]
- Paper link: https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/ec5decca5ed3d6b8079e2e7e7bacc9f2-Paper-round1.pdf

- Github link: https://github.com/datamllab/tods/tree/benchmark/benchmark/synthetic (This link is a part of tods, the anomaly detection package of the authors of this paper. To see the full project, go to https://github.com/datamllab/tods/tree/master.)

- Due to the ambiguity of anomalies, various types of time series anomalies are ill-defined in many previous studies. This paper classifies and formulates the types of anomalies and synthesizes the univariate and multivariate time series based on the formulation. Time series anomlies are classified into two categories: point anomalies and pattern anomalies.

## Point anomalies
- Point anomalies are global anomlies (extreme values) and contextual anomalies (relatively small/large in context, e.g. windows; but not globally). Point anomalies are formulated into $|x_{t} - \hat{x}_{t}| > \delta$, where $\delta$ is a threshold. 

- The threshold of global anomaly can be formulated as $\delta = \lambda \cdot \sigma(X)$, where $\lambda$ is the range and $\sigma$ is the standard deviation of the entire dataset X

- The threshold of contextual anomaly can be formulated as $\delta = \lambda \cdot \sigma(X_(t-k: t+k))$, where $\lambda$ is the coefficient of threshold, and $\sigma$ is the standard deviation of the window $X_{(t-k: t+k)}$.


## Pattern anomalies
- Pattern anomalies are anomalous subsequences. They can be classified into shapelet anomalies (a.k.a. collective anomalies, the subsequences with dissimilar shapelet compared to the normal shapelet), seasonal anomalies (unusual seasonality), and trend anomalies (unusual trend).

- Pattern anomalies can be formulated as $X_{(i: j)} = \rho(2\pi\omega T_{(i: j)}) + \tau(T_{(i: j)})$, where $T_{(i: j)}$ is the sub-timestamps; $\rho$ models the shape of the subsequence (the most familiar example is fourier transform); and $\tau$ models the trend.

- Here, the threshold and measure of dissimilarity are given as $\delta$ and $s$, respectively.

- The shapelet anomaly is formulated as $s\left(\rho(.), \hat{\rho}(.)\right)$, where $\rho$ is the basic shapelet of the output of some ML/DL model.