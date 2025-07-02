# TranAD (VLDB 2022)
- Paper link: https://arxiv.org/pdf/2201.07284

- Github link: https://github.com/imperial-qore/TranAD.git

- The model is composed of one encoder and two decoders. At the first phase of training, the input passes the encoder, and then passes the first decoder. At the second phase, the output of first decoder passes the encoder, and then passes the second decoder.

- Letting L1, L2 be the reconstruction error of the first decoder and the second decoder, respectively. Then the first decoder tries to minimize (epsilon^n) * L1 + (epsilon^n) * L2, and the second decoder tries to minimize (epsilon^n) * L1 - (epsilon^n) * L2, where epslion is some training parameter, and n is in {1, ..., #epochs}. The anomaly score is given as the mean of reconstuction loss of two decoders.

- Such approach is motivated by the training process of USAD. (See https://dl.acm.org/doi/pdf/10.1145/3394486.3403392 and https://github.com/manigalati/usad for details about USAD.)

- The vanilla version of TranAD has a little number of parameters, so that it can be run on the CPU, with few amount of training time (reduced up to 99% compared to the baselines of TranAD), and decent score (increased up to 17% compared to the baselines).