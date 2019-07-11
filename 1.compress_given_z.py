"""
Adapted from:
https://github.com/greenelab/BioBombe/blob/master/2.sequential-compression/scripts/train_models_given_z.py

"""
import os
import argparse
import logging
import numpy as np
import pandas as pd

import config as cfg
from data_models import DataModel
from tcga_util import subset_genes_by_mad

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

p = argparse.ArgumentParser()
p.add_argument('-a', '--algorithm', default=None,
               help='which transform to run, default runs all\
                     of the transforms that are implemented',
               choices=DataModel.list_algorithms())
p.add_argument('-k', '--num_components', type=int,
               help='dimensionality of z')
p.add_argument('-n', '--num_seeds', type=int, default=5,
               help='number of different seeds to run on current data')
p.add_argument('-m', '--subset_mad_genes', type=int, default=8000,
               help='subset num genes based on mean absolute deviation')
p.add_argument('-o', '--output_dir', default=cfg.models_dir,
               help='where to save the output files')
p.add_argument('-s', '--shuffle', action='store_true',
               help='randomize gene expression data for negative control')
p.add_argument('-v', '--verbose', action='store_true')
args = p.parse_args()

algs_to_run = ([args.algorithm] if args.algorithm
                                else DataModel.list_algorithms())

if args.verbose:
    logging.basicConfig(level=logging.DEBUG, format='%(message)s')

# load input expression data
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

# determine most variably expressed genes and subset, if necessary
if args.subset_mad_genes is not None:
    mad_file = os.path.join(cfg.data_dir, 'tcga_mad_genes.tsv')
    rnaseq_train_df, rnaseq_test_df = subset_genes_by_mad(
        rnaseq_train_df, rnaseq_test_df, mad_file, args.subset_mad_genes)

dm = DataModel(df=rnaseq_train_df, test_df=rnaseq_test_df)
dm.transform(how='zeroone')

if args.shuffle:
    file_prefix = '{}_components_shuffled_'.format(args.num_components)
else:
    file_prefix = '{}_components_'.format(args.num_components)

# specify location of output files

comp_out_dir = os.path.join(os.path.abspath(args.output_dir),
                            'ensemble_z_matrices',
                            'components_{}'.format(args.num_components))

if not os.path.exists(comp_out_dir):
    os.makedirs(comp_out_dir)

np.random.seed(cfg.default_seed)
random_seeds = np.random.randint(0, high=1000000, size=args.num_seeds)

reconstruction_results = []
test_reconstruction_results = []

logging.debug('Fitting compressed models...')
recon_file = os.path.join(args.output_dir,
                          '{}reconstruction.tsv'.format(
                          file_prefix))

for ix, seed in enumerate(random_seeds, 1):
    np.random.seed(seed)
    seed_file = os.path.join(comp_out_dir, 'model_{}'.format(seed))
    if args.shuffle:
        seed_file = '{}_shuffled'.format(seed_file)
        shuffled_train_df = shuffle_train_genes(rnaseq_train_df)
        dm = DataModel(df=shuffled_train_df,
                       test_df=rnaseq_test_df)
        dm.transform(how='zeroone')

    if 'pca' in algs_to_run:
        logging.debug('-- Fitting pca model for random seed {} of {}'.format(
                      ix, len(random_seeds)))

        dm.pca(n_components=args.num_components,
               transform_test_df=True)
    if 'nmf' in algs_to_run:
        logging.debug('-- Fitting nmf model for random seed {} of {}'.format(
                      ix, len(random_seeds)))
        dm.nmf(n_components=args.num_components,
               transform_test_df=True,
               seed=seed)

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

