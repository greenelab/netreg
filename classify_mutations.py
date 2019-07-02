"""
Adapted from:
https://github.com/greenelab/BioBombe/blob/master/9.tcga-classify/classify-top-mutations.py

"""

import os
import argparse
import numpy as np
import pandas as pd

import config as cfg
from tcga_util import (
    get_threshold_metrics,
    summarize_results,
    extract_coefficients,
    align_matrices,
    process_y_matrix,
    train_model,
    build_feature_dictionary,
    check_status,
)

def load_top_50():
    # load top 50 mutated genes from BioBombe repo, these are
    # precomputed there
    base_url = "https://github.com/greenelab/BioBombe/raw"
    commit = "aedc9dfd0503edfc5f25611f5eb112675b99edc9"

    file = "{}/{}/9.tcga-classify/data/top50_mutated_genes.tsv".format(
            base_url, commit)
    genes_df = pd.read_csv(file, sep='\t')

    return genes_df

def load_pancancer_data():
    # Load data to build y matrices
    base_url = "https://github.com/greenelab/pancancer/raw"
    commit = "2a0683b68017fb226f4053e63415e4356191734f"

    # Load data
    file = "{}/{}/data/sample_freeze.tsv".format(base_url, commit)
    sample_freeze_df = pd.read_csv(file, index_col=0, sep='\t')

    file = "{}/{}/data/pancan_mutation_freeze.tsv.gz".format(base_url, commit)
    mutation_df = pd.read_csv(file, index_col=0, sep='\t')

    file = "{}/{}/data/copy_number_loss_status.tsv.gz".format(base_url, commit)
    copy_loss_df = pd.read_csv(file, index_col=0, sep='\t')

    file = "{}/{}/data/copy_number_gain_status.tsv.gz".format(base_url, commit)
    copy_gain_df = pd.read_csv(file, index_col=0, sep='\t')

    file = "{}/{}/data/mutation_burden_freeze.tsv".format(base_url, commit)
    mut_burden_df = pd.read_csv(file, index_col=0, sep='\t')

    return (
        sample_freeze_df,
        mutation_df,
        copy_loss_df,
        copy_gain_df,
        mut_burden_df
    )


if __name__ == '__main__':

    p = argparse.ArgumentParser()
    p.add_argument('--gene_list', nargs='*', default=None,
                   help='<Optional> Provide a list of genes to run\
                         mutation classification for; default is all genes')
    args = p.parse_args()

    # load data
    genes_df = load_top_50()
    if args.gene_list is not None:
        genes_df = genes_df[genes_df['gene'].isin(args.gene_list)]
        genes_df.reset_index(drop=True, inplace=True)

    (sample_freeze_df,
     mutation_df,
     copy_loss_df,
     copy_gain_df,
     mut_burden_df) = load_pancancer_data()

    # Setup column names for output files
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

    # Obtain a dictionary of file directories for loading each feature matrix (X)
    z_matrix_dict, num_models = build_feature_dictionary()

    num_genes = len(genes_df)

    for gene_idx, gene_series in genes_df.iterrows():
        if gene_idx > 5: exit()

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
                                  "{}_coefficients.tsv.gz".format(gene_name))

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

        # Now, perform all the analyses for each X matrix

        model_no = 1

        for signal in z_matrix_dict.keys():
            z_dim_dict = z_matrix_dict[signal]
            for z_dim in z_dim_dict.keys():
                seed_z_dim_dict = z_dim_dict[z_dim]
                for seed in seed_z_dim_dict.keys():
                    z_train_file = z_matrix_dict[signal][z_dim][seed]["train"]
                    z_test_file = z_matrix_dict[signal][z_dim][seed]["test"]

                    for alg in cfg.algorithms:
                        # Load and process data
                        train_samples, x_train_df, y_train_df = align_matrices(
                            x_file_or_df=z_train_file, y=y_df, algorithm=alg
                        )

                        test_samples, x_test_df, y_test_df = align_matrices(
                            x_file_or_df=z_test_file, y=y_df, algorithm=alg
                        )

                        # Train the model
                        print(
                            "Training model {} of {} for gene {} of {}".format(
                                model_no, num_models, gene_idx+1, num_genes)
                        )

                        model_no += 1

                        print(
                            "-- gene: {}, algorithm: {}, signal: {}, z_dim: {}, "
                            "seed: {}".format(gene_name, alg, signal, z_dim, seed)
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
                        y_cv_results = get_threshold_metrics(
                            y_train_df.status, y_cv_df, drop=False
                        )

                        # Get coefficients
                        coef_df = extract_coefficients(
                            cv_pipeline=cv_pipeline,
                            feature_names=x_train_df.columns,
                            signal=signal,
                            z_dim=z_dim,
                            seed=seed,
                            algorithm=alg,
                        )

                        coef_df = coef_df.assign(gene=gene_name)

                        # Store all results
                        train_metrics_, train_roc_df, train_pr_df = summarize_results(
                            y_train_results, gene_name, signal, z_dim, seed, alg, "train"
                        )
                        test_metrics_, test_roc_df, test_pr_df = summarize_results(
                            y_test_results, gene_name, signal, z_dim, seed, alg, "test"
                        )
                        cv_metrics_, cv_roc_df, cv_pr_df = summarize_results(
                            y_cv_results, gene_name, signal, z_dim, seed, alg, "cv"
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

        file = os.path.join(gene_dir, "{}_auc_threshold_metrics.tsv.gz".format(gene_name))
        gene_auc_df.to_csv(
            file, sep="\t", index=False, compression="gzip", float_format="%.5g"
        )

        file = os.path.join(gene_dir, "{}_aupr_threshold_metrics.tsv.gz".format(gene_name))
        gene_aupr_df.to_csv(
            file, sep="\t", index=False, compression="gzip", float_format="%.5g"
        )

        gene_coef_df.to_csv(
            check_file, sep="\t", index=False, compression="gzip", float_format="%.5g"
        )

        file = os.path.join(gene_dir, "{}_classify_metrics.tsv.gz".format(gene_name))
        gene_metrics_df.to_csv(
            file, sep="\t", index=False, compression="gzip", float_format="%.5g"
        )

