# stock-transaction-abnormal-detection-based-on-DTW-
My undergrad thesis.

Insider trading undermines the integrity and orderliness of financial markets. In order to better monitor investors' trading behavior and provide a references for authorities, this study proposes a non-parametric approach to identifying insider trading based on multiple Dynamic time warping (DTW), combined with extreme value theory for threshold delineation. In terms of insider trading, combining empirical studies at home and abroad, this study refers to the definition of insider trading behavior in the Shenzhen Stock Exchange and uses indicators such as pulling and suppressing share prices and false declarations as the basis for determining the characteristics of insider trading. In the model evaluation and empirical analysis, the study analyses two cases of insider trading disclosed by the Shenzhen Stock Exchange and is able to effectively discern insider trading sequences within the disclosed existence date. Meanwhile, this paper combines previous research and improves the DTW method by extending it to multivariate DTW, transforming unequal sequences, and introducing multiple lower bound distances in calculating the DTW distance. The experimental results show that the improved NN DTW algorithm can handle the unequal-length multivariate DTW problem that cannot be handled by the traditional DTW algorithm, and has better performance in identifying insider trading.

## Environment

please install correct version of [torch](https://pytorch.org/) in your environment.

Then, run the

```bash
pip install -r requirements.txt
```


