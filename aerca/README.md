## AERCA [ICLR 2025, Oral]
- Paper link: https://openreview.net/pdf?id=k38Th3x4d9

- Github link: https://github.com/hanxiao0607/AERCA

- $$\mathbb{x}_{t} = \sum_{m=1}^{K} \alpha_{K-m}\mathbb{x}_{t-(K-m)} + \alpha_{K}\mathbb{x}_{t-K} + \sum_{m=2}^{K+1} \alpha_{K+1-m} \sum_{k=m}^{K} \omega_{k}\mathbb{x}_{t-k-(K+1-m)}$$, where $\omega_{k}$ is the parameter of Granger causality, and $\alpha_{n} = \sum_{i=1}^{n} \omega_{k}\alpha_{n-i}, 1 \leq n \leq K$ is a recursive equation with $\alpha_{0} = 1$.

Currently working on some files, issues, and summary. To be updated.