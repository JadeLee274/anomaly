# anomaly

Repo for anomaly detection study. 
Mainly for studying papers, codes, and running experiments.
Each architecture-based topics are listed below.

# Transformer-based

## Anomaly Transformer (ICLR 2022, Spotlight)
- Github link: https://github.com/thuml/Anomaly-Transformer/tree/main

- This paper assumes that, due to the continuity of time series, anomalies are strongly related to each other, but weakly related to noraml data.

- Authors suggest the additional attention layer based on the Gaussian kernel, which is named Anomaly Attention. This is the layer that calculates the attention score between data within the window, which is called Prior-Association. By the assumption on anomalies, anomalies have the high prior association, whereas the normal data don't.

- The Series-Association, is just the attention score of the vanilla transformer.

- Also, authors suggests the novel anomaly score, which is called Association Discrepancy (AssDis). This is calculated as the symmetrized KL divergnce between Prior- and Series- Associations. As the PA/SA of anomaly is high/low and the PA/SA of normal data is medium/medium, the AssDis of anomalies will be lower that that of the normal ones. The training chooses minimax strategy.


## TranAD (VLDB 2022)
- Github link: https://github.com/imperial-qore/TranAD.git

- The model is composed of one encoder and two decoders. At the first phase of training, the input passes the encoder, and then passes the first decoder. At the second phase, the output of first decoder passes the encoder, and then passes the second decoder.

- Letting L1, L2 be the reconstruction error of the first decoder and the second decoder, respectively. Then the first decoder tries to minimize (epsilon^n) * L1 + (epsilon^n) * L2, and the second decoder tries to minimize (epsilon^n) * L1 - (epsilon^n) * L2, where epslion is some training parameter, and n is in {1, ..., #epochs}. The anomaly score is given as the mean of reconstuction loss of two decoders.

- Such approach follows the previous training process of USAD. (See https://github.com/manigalati/usad.)

- The vanilla version of TranAD has a little number of parameters, so that it can be run on the CPU, with few amount of training time (reduced up to 99% compared to the baselines of TranAD), and decent score (increased up to 17% compared to the baselines).



# CNN-based

## TimesNet (ICLR 2023)
- Github link: https://github.com/thuml/TimesNet

- There are many attemps to catch the temporal variaions of the data for the time series analysis (such as Anomaly Transformer). But before this paper, few attempts were made to catch more intrinsic variations of the data; for example, seasonality, trend, etc. This can be the matter of generalization to the real world data.

- This architecture aims to catch such variations. Specifically, by applying the Fast Fourier Transform to data, one can decompose data into linear combination of periodic functions, with each period and frequency. By doing so, the 1D-tensor data is reshaped into 2D-tensor data, where the shape is period x frequency.

- With the process above, authors try to find out two types on variations: the intraperiod variation, and the interperiod variation. The former is the variation within each period, and the latter is the variation between each period.

- By using backbones for the computer vision tasks (in this paper, Conv2D-based architectures), TimesNet can simultaneously catch such variations, generalizing to various multivariate time series analysis task; anomaly detection, long/short time series forecasting, classification, imputation.

- But in the anomaly detection task, where the Anomaly Transformer is one of the baselines, authors changed the anomaly score of Anomaly Transformer from Association Discrepancy to Reconstruction error, for the fair comparison. This seems to be the tradeoff between focusing on the specific task and generalizing into various task.

- Following the github link of TimesNet, the model is in the time series library made by authors (https://github.com/thuml/Time-Series-Library/tree/main), and we can run an experiment not only on TimesNet, but also on various models such as Autoformer, FEDformer, iTransformer, etc. I simplified the structure of the files related to TimesNet, so that we can only focus on the TSAD task.
