"""
Adapted from:
https://github.com/greenelab/BioBombe/blob/master/9.tcga-classify/classify-with-raw-expression.py

"""

import os
import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import config as cfg
from tcga_util import (
    load_pancancer_data,
    load_top_50,
    subset_genes_by_mad,
    get_threshold_metrics,
    summarize_results,
    extract_coefficients,
    align_matrices,
    process_y_matrix,
    train_model,
    process_y_matrix_cancertype,
    check_status,
)

p = argparse.ArgumentParser()
p.add_argument('--gene_list', nargs='*', default=None,
               help='<Optional> Provide a list of genes to run\
                     mutation classification for; default is all genes')
p.add_argument('--seed', type=int, default=cfg.default_seed)
p.add_argument('--verbose', action='store_true')
args = p.parse_args()

if args.verbose:
    logging.basicConfig(level=logging.DEBUG, format='%(message)s')

np.random.seed(args.seed)
algorithm = "raw"

# load data
logging.debug('Loading gene label data...')
genes_df = load_top_50()
if args.gene_list is not None:
    genes_df = genes_df[genes_df['gene'].isin(args.gene_list)]
    genes_df.reset_index(drop=True, inplace=True)

logging.debug('Loading pan-cancer data...')
# TODO: this data can probably be cached somewhere to speed things up,
#       loading this data is currently v. slow
(sample_freeze_df,
 mutation_df,
 copy_loss_df,
 copy_gain_df,
 mut_burden_df) = load_pancancer_data()

# Load and process X matrix
logging.debug('Loading raw gene expression data...')
rnaseq_train = (
    os.path.join(cfg.data_dir,
                 'train_tcga_expression_matrix_processed.tsv.gz')
    )
rnaseq_test = (
    os.path.join(cfg.data_dir,
                 'test_tcga_expression_matrix_processed.tsv.gz')
    )

rnaseq_train_df = pd.read_csv(rnaseq_train, index_col=0, sep='\t')
rnaseq_test_df = pd.read_csv(rnaseq_test, index_col=0, sep='\t')

# TODO: is option for subset by mad genes necessary?
mad_file = os.path.join(cfg.data_dir, 'tcga_mad_genes.tsv')
rnaseq_train_df, rnaseq_test_df = subset_genes_by_mad(
    rnaseq_train_df, rnaseq_test_df, mad_file, cfg.num_features_raw)

# Scale RNAseq matrix the same way RNAseq was scaled for
# compression algorithms
train_fitted_scaler = MinMaxScaler().fit(rnaseq_train_df)
rnaseq_train_df = pd.DataFrame(
    train_fitted_scaler.transform(rnaseq_train_df),
    columns=rnaseq_train_df.columns,
    index=rnaseq_train_df.index,
)

test_fitted_scaler = MinMaxScaler().fit(rnaseq_test_df)
rnaseq_test_df = pd.DataFrame(
    test_fitted_scaler.transform(rnaseq_test_df),
    columns=rnaseq_test_df.columns,
    index=rnaseq_test_df.index,
)

# Track total metrics for each gene in one file
metric_cols = [
    "auroc",
    "aupr",
    "gene_or_cancertype",
    "signal",
    "z_dim",
    "seed",
    "algorithm",
    "data_type",
]

num_genes = len(genes_df)

for gene_idx, gene_series in genes_df.iterrows():

    gene_name = gene_series.gene
    classification = gene_series.classification

    # Create list to store gene specific results
    gene_auc_list = []
    gene_aupr_list = []
    gene_coef_list = []
    gene_metrics_list = []

    # Create directory for the gene
    gene_dir = os.path.join(cfg.results_dir, "mutation", gene_name)
    os.makedirs(gene_dir, exist_ok=True)

    # Check if gene has been processed already
    check_file = os.path.join(gene_dir,
                              "{}_raw_coefficients.tsv.gz".format(gene_name))
    if check_status(check_file):
        continue

    # Process the y matrix for the given gene or pathway
    y_mutation_df = mutation_df.loc[:, gene_name]

    # Include copy number gains for oncogenes
    # and copy number loss for tumor suppressor genes (TSG)
    include_copy = True
    if classification == "Oncogene":
        y_copy_number_df = copy_gain_df.loc[:, gene_name]
    elif classification == "TSG":
        y_copy_number_df = copy_loss_df.loc[:, gene_name]
    else:
        y_copy_number_df = pd.DataFrame()
        include_copy = False

    y_df = process_y_matrix(
        y_mutation=y_mutation_df,
        y_copy=y_copy_number_df,
        include_copy=include_copy,
        gene=gene_name,
        sample_freeze=sample_freeze_df,
        mutation_burden=mut_burden_df,
        filter_count=cfg.filter_count,
        filter_prop=cfg.filter_prop,
        output_directory=gene_dir,
        hyper_filter=5,
    )

    model_no = 1

    for signal in ["signal", "shuffled"]:
        if signal == "shuffled":
            # Shuffle training data
            x_train_raw_df = rnaseq_train_df.apply(
                lambda x: np.random.permutation(x.tolist()),
                axis=1,
                result_type="expand",
            )

            x_train_raw_df.columns = rnaseq_train_df.columns
            x_train_raw_df.index = rnaseq_train_df.index

            # Shuffle testing data
            x_test_raw_df = rnaseq_test_df.apply(
                lambda x: np.random.permutation(x.tolist()),
                axis=1,
                result_type="expand",
            )

            x_test_raw_df.columns = rnaseq_test_df.columns
            x_test_raw_df.index = rnaseq_test_df.index

        else:
            x_train_raw_df = rnaseq_train_df
            x_test_raw_df = rnaseq_test_df

        # Now, perform all the analyses for each X matrix
        train_samples, x_train_df, y_train_df = align_matrices(
            x_file_or_df=x_train_raw_df, y=y_df
        )

        test_samples, x_test_df, y_test_df = align_matrices(
            x_file_or_df=x_test_raw_df, y=y_df
        )

        # Train the model
        logging.debug(
            "Training model {} of 2 for gene {} of {}".format(
                model_no, gene_idx+1, num_genes)
        )

        model_no += 1

        logging.debug(
            "-- gene: {}, for raw {} features".format(gene_name, signal)
        )

        # Fit the model
        cv_pipeline, y_pred_train_df, y_pred_test_df, y_cv_df = train_model(
            x_train=x_train_df,
            x_test=x_test_df,
            y_train=y_train_df,
            alphas=cfg.alphas,
            l1_ratios=cfg.l1_ratios,
            n_folds=cfg.folds,
            max_iter=cfg.max_iter,
        )

        # Get metric  predictions
        y_train_results = get_threshold_metrics(
            y_train_df.status, y_pred_train_df, drop=False
        )
        y_test_results = get_threshold_metrics(
            y_test_df.status, y_pred_test_df, drop=False
        )
        y_cv_results = get_threshold_metrics(y_train_df.status, y_cv_df, drop=False)

        # Get coefficients
        coef_df = extract_coefficients(
            cv_pipeline=cv_pipeline,
            feature_names=x_train_df.columns,
            signal=signal,
            z_dim=cfg.num_features_raw,
            seed=args.seed,
            algorithm=algorithm,
        )

        coef_df = coef_df.assign(gene=gene_name)

        # Store all results
        train_metrics_, train_roc_df, train_pr_df = summarize_results(
            y_train_results, gene_name, signal, cfg.num_features_raw,
            args.seed, algorithm, "train"
        )
        test_metrics_, test_roc_df, test_pr_df = summarize_results(
            y_test_results, gene_name, signal, cfg.num_features_raw,
            args.seed, algorithm, "test"
        )
        cv_metrics_, cv_roc_df, cv_pr_df = summarize_results(
            y_cv_results, gene_name, signal, cfg.num_features_raw,
            args.seed, algorithm, "cv"
        )

        # Compile summary metrics
        metrics_ = [train_metrics_, test_metrics_, cv_metrics_]
        metric_df_ = pd.DataFrame(metrics_, columns=metric_cols)
        gene_metrics_list.append(metric_df_)

        gene_auc_df = pd.concat([train_roc_df, test_roc_df, cv_roc_df])
        gene_auc_list.append(gene_auc_df)

        gene_aupr_df = pd.concat([train_pr_df, test_pr_df, cv_pr_df])
        gene_aupr_list.append(gene_aupr_df)

        gene_coef_list.append(coef_df)

    gene_auc_df = pd.concat(gene_auc_list)
    gene_aupr_df = pd.concat(gene_aupr_list)
    gene_coef_df = pd.concat(gene_coef_list)
    gene_metrics_df = pd.concat(gene_metrics_list)

    file = os.path.join(
        gene_dir, "{}_raw_auc_threshold_metrics.tsv.gz".format(gene_name)
    )
    gene_auc_df.to_csv(
        file, sep="\t", index=False, compression="gzip", float_format="%.5g"
    )

    file = os.path.join(
        gene_dir, "{}_raw_aupr_threshold_metrics.tsv.gz".format(gene_name)
    )
    gene_aupr_df.to_csv(
        file, sep="\t", index=False, compression="gzip", float_format="%.5g"
    )

    gene_coef_df.to_csv(
        check_file, sep="\t", index=False, compression="gzip", float_format="%.5g"
    )

    file = os.path.join(gene_dir, "{}_raw_classify_metrics.tsv.gz".format(gene_name))
    gene_metrics_df.to_csv(
        file, sep="\t", index=False, compression="gzip", float_format="%.5g"
    )

