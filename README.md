# netreg

Evaluating various approaches to network/pathway regularization for gene expression data

This is an adaptation of the pipeline used in [BioBombe](https://github.com/greenelab/BioBombe)
for predicting gene alteration status using compressed TCGA gene expression data.

Current goals of this project:
  * Benchmark [PLIER](https://github.com/wgmao/PLIER) against the
    current models (e.g. PCA, NMF) for gene alteration prediction
  * Explore network regularized methods for gene alteration prediction
    (e.g. graph LASSO as implemented
    [here](https://github.com/suinleelab/attributionpriors/blob/master/graph/Graph_Attribution_Prior.ipynb),
    or network-regularized NMF as implemented [here](https://github.com/raphael-group/netNMF-sc))
    using a variety of biological networks.
