"""
Adapted from
https://github.com/greenelab/BioBombe/blob/master/2.sequential-compression/scripts/train_models_given_z.py

"""
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

import config as cfg
from data_models import DataModel

def subset_genes_by_mad(train_df, test_df, mad_file, subset_mad_genes):
    # subset genes by mean absolute deviation
    mad_genes_df = pd.read_csv(mad_file, sep='\t')
    mad_genes = mad_genes_df.iloc[0:subset_mad_genes, ].gene_id.astype(str)

    train_df = train_df.reindex(mad_genes, axis='columns')
    test_df = test_df.reindex(mad_genes, axis='columns')
    return (train_df, test_df)

def shuffle_train_genes(train_df):
    # randomly permute genes of each sample in the rnaseq matrix
    shuf_df = train_df.apply(lambda x:
                             np.random.permutation(x.tolist()),
                             axis=1)

    # Setup new pandas dataframe
    shuf_df = pd.DataFrame(shuf_df, columns=['gene_list'])
    shuf_df = pd.DataFrame(shuf_df.gene_list.values.tolist(),
                           columns=rnaseq_train_df.columns,
                           index=rnaseq_train_df.index)
    return shuf_df

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--data_dir', default=cfg.data_dir,
                   help='location of processed expression data')
    p.add_argument('-k', '--num_components', type=int,
                   help='dimensionality of z')
    p.add_argument('-n', '--num_seeds', type=int, default=5,
                   help='number of different seeds to run on current data')
    p.add_argument('-m', '--subset_mad_genes', type=int, default=8000,
                   help='subset num genes based on mean absolute deviation')
    p.add_argument('-o', '--output_dir',
                   help='where to save the output files')
    p.add_argument('-s', '--shuffle', action='store_true',
                   help='randomize gene expression data for negative control')
    args = p.parse_args()

    if args.shuffle:
        file_prefix = '{}_components_shuffled_'.format(args.num_components)
    else:
        file_prefix = '{}_components_'.format(args.num_components)

    recon_file = os.path.join(args.output_dir,
                              '{}reconstruction.tsv'.format(file_prefix))

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    rnaseq_train = (
        os.path.join(args.data_dir,
                     'train_tcga_expression_matrix_processed.tsv.gz')
        )
    rnaseq_test = (
        os.path.join(args.data_dir,
                     'test_tcga_expression_matrix_processed.tsv.gz')
        )

    rnaseq_train_df = pd.read_csv(rnaseq_train, index_col=0, sep='\t')
    rnaseq_test_df = pd.read_csv(rnaseq_test, index_col=0, sep='\t')

    # Determine most variably expressed genes and subset
    if args.subset_mad_genes is not None:
        mad_file = os.path.join(cfg.data_dir, 'tcga_mad_genes.tsv')
        rnaseq_train_df, rnaseq_test_df = subset_genes_by_mad(
            rnaseq_train_df, rnaseq_test_df, mad_file, args.subset_mad_genes)

    np.random.seed(cfg.default_seed)
    random_seeds = np.random.randint(0, high=1000000, size=args.num_seeds)

    comp_out_dir = os.path.join(os.path.abspath(args.output_dir),
                                'ensemble_z_matrices',
                                'components_{}'.format(args.num_components))

    if not os.path.exists(comp_out_dir):
        os.makedirs(comp_out_dir)

    dm = DataModel(df=rnaseq_train_df, test_df=rnaseq_test_df)
    dm.transform(how='zeroone')

    reconstruction_results = []
    test_reconstruction_results = []

    for seed in random_seeds:
        np.random.seed(seed)
        seed_file = os.path.join(comp_out_dir, 'model_{}'.format(seed))
        if args.shuffle:
            seed_file = '{}_shuffled'.format(seed_file)
            shuffled_train_df = shuffle_train_genes(rnaseq_train_df)
            dm = DataModel(df=shuffled_train_df,
                           test_df=rnaseq_test_df)
            dm.transform(how='zeroone')

        # add other models here
        dm.nmf(n_components=args.num_components, transform_test_df=True)

        # Obtain z matrix (sample scores per latent space feature) for all models
        full_z_file = os.path.join(cfg.models_dir,
                        '{}_z_matrix.tsv.gz'.format(seed_file))
        dm.combine_models().to_csv(full_z_file, sep='\t', compression='gzip')

        full_test_z_file = os.path.join(cfg.models_dir,
                        '{}_z_test_matrix.tsv.gz'.format(seed_file))
        dm.combine_models(test_set=True).to_csv(full_test_z_file, sep='\t',
                                                compression='gzip')

        # Obtain weight matrices (gene by latent space feature) for all models
        full_weight_file = os.path.join(cfg.models_dir,
                        '{}_weight_matrix.tsv.gz'.format(seed_file))
        dm.combine_weight_matrix().to_csv(full_weight_file, sep='\t',
                                          compression='gzip')

        # Store reconstruction costs and reconstructed input at training end
        full_reconstruction, reconstructed_matrices = dm.compile_reconstruction()

        # Store reconstruction evaluation and data for test set
        full_test_recon, test_recon_mat = dm.compile_reconstruction(test_set=True)

        reconstruction_results.append(
            full_reconstruction.assign(seed=seed, shuffled=args.shuffle)
            )

        test_reconstruction_results.append(
            full_test_recon.assign(seed=seed, shuffled=args.shuffle)
            )

    # Save reconstruction results
    pd.concat([
        pd.concat(reconstruction_results).assign(data_type='training'),
        pd.concat(test_reconstruction_results).assign(data_type='testing')
    ]).reset_index(drop=True).to_csv(recon_file, sep='\t', index=False)

