# About this repository:

This repository is for anomaly detection study; mainly about studying papers, codes, and running experiments by myself. Each architecture-based topics are listed below.

# Attention & Transformer-based

## Anomaly Transformer (ICLR 2022, Spotlight)
- Paper Link: https://arxiv.org/pdf/2110.02642

- Github link: https://github.com/thuml/Anomaly-Transformer/tree/main

- This paper assumes that, due to the continuity of time series, anomalies are strongly related to each other, but weakly related to noraml data.

- Authors suggest the additional attention layer based on the Gaussian kernel, which is named Anomaly Attention. This is the layer that calculates the attention score between data within the window, which is called Prior-Association. By the assumption on anomalies, anomalies have the high prior association, whereas the normal data don't.

- The Series-Association, is just the attention score of the vanilla transformer.

- Also, authors suggests the novel anomaly score, which is called Association Discrepancy (AssDis). This is calculated as the symmetrized KL divergnce between Prior- and Series- Associations. As the PA/SA of anomaly is high/low and the PA/SA of normal data is medium/medium, the AssDis of anomalies will be lower that that of the normal ones. The training chooses minimax strategy.


## TranAD (VLDB 2022)
- Paper link: https://arxiv.org/pdf/2201.07284

- Github link: https://github.com/imperial-qore/TranAD.git

- The model is composed of one encoder and two decoders. At the first phase of training, the input passes the encoder, and then passes the first decoder. At the second phase, the output of first decoder passes the encoder, and then passes the second decoder.

- Letting L1, L2 be the reconstruction error of the first decoder and the second decoder, respectively. Then the first decoder tries to minimize (epsilon^n) * L1 + (epsilon^n) * L2, and the second decoder tries to minimize (epsilon^n) * L1 - (epsilon^n) * L2, where epslion is some training parameter, and n is in {1, ..., #epochs}. The anomaly score is given as the mean of reconstuction loss of two decoders.

- Such approach is motivated by the training process of USAD. (See https://dl.acm.org/doi/pdf/10.1145/3394486.3403392 and https://github.com/manigalati/usad for details about USAD.)

- The vanilla version of TranAD has a little number of parameters, so that it can be run on the CPU, with few amount of training time (reduced up to 99% compared to the baselines of TranAD), and decent score (increased up to 17% compared to the baselines).


## DCDetector (KDD 2023)
- Paper link: https://arxiv.org/pdf/2306.10347

- Github link: https://github.com/DAMO-DI-ML/KDD2023-DCdetector

- 



# CNN-based

## TimesNet (ICLR 2023)
- Paper link: https://arxiv.org/pdf/2210.02186

- Github link: https://github.com/thuml/TimesNet

- TimesNet is a foundation model for time series, that is, it is not only for the anomaly detection task, but also for forecasting, classification, and imputation. Before this paper, few attempts were made to catch more intrinsic variations of the data; for example, seasonality, trend, etc. TimesNet is a model for catching such variations, so that it can perform multiple tasks on time series. Although it lacks F1-score on the widely-used datasets in anomaly detection task (such as SMD, PSM, SMAP, MSL, SWaT) compared to Anomaly Transformre, it is notable that this model tries to analyze time series more intrinsically. (This is the reason why I put it in my repository.)

- This architecture aims to catch such variations. Specifically, by applying the Fast Fourier Transform to data, one can decompose data into linear combination of periodic functions, with each period and frequency. By doing so, the 1D-tensor data is reshaped into 2D-tensor data, where the shape is period x frequency.

- With the process above, authors try to find out two types on variations: the intraperiod variation, and the interperiod variation. The former is the variation within each period, and the latter is the variation between each period.

- By using backbones for the computer vision tasks (in this paper, Conv2D-based architectures), TimesNet can simultaneously catch such variations, generalizing to various multivariate time series analysis task; anomaly detection, long/short time series forecasting, classification, imputation.

- But in the anomaly detection task, where the Anomaly Transformer is one of the baselines, authors changed the anomaly score of Anomaly Transformer from Association Discrepancy to Reconstruction error, for the fair comparison. This seems to be the tradeoff between focusing on the specific task and generalizing into various task.

- Following the github link of TimesNet, the model is in the time series library made by authors (https://github.com/thuml/Time-Series-Library/tree/main), and we can run an experiment not only on TimesNet, but also on various models such as Autoformer, FEDformer, iTransformer, etc. But in this repository, I simplified the structure of the files related to TimesNet, so that we can only focus on the time series anomaly detection task. But remember, TimesNet is a foundation model, so it will be more useful to go to the time series library link and look at the original code.


# Diffusion-based

## DiffAD (KDD 2023)
- Paper link: https://dl.acm.org/doi/pdf/10.1145/3580305.3599391

- Github link: https://github.com/ChunjingXiao/DiffAD

- The time series data imputation task is to fill the gap between observed data.

- DiffAD suggests that comparing the predicted data made by imputating the subset of training set using Diffusion model and the original data, it can detect anomaly on the test set. And it gained the better F1-score compared to the state-of-the-art-models like Anomaly Transformer.

- Notably, the anomaly detection using diffusion-based imputation model like CSDI (See https://arxiv.org/pdf/2107.03502 and https://github.com/ermongroup/CSDI for more details on CSDI) gained the better F1-score compared to the baselines, except Anomaly Transformer. This indicates the possibility of the application of the methods for time series imputation to the time series anomaly detection.
