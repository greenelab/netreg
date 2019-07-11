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

## Setup

We recommend using the conda environment specified in the `environment.yml` file to run these analyses. To build and activate this environment, run:

```
# conda version 4.5.0
conda env create --file environment.yml

conda activate netreg
```
