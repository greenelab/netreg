import os

pj = lambda *paths: os.path.abspath(os.path.join(*paths))

repo_root = os.getcwd()

data_dir = pj(repo_root, 'data')
models_dir = pj(repo_root, 'models')
results_dir = pj(repo_root, 'results')

default_seed = 42

# hyperparameters for classification experiments
filter_prop = 0.05
filter_count = 15
folds = 5
max_iter = 100
alphas = [0.1, 0.13, 0.15, 0.2, 0.25, 0.3]
l1_ratios = [0.15, 0.16, 0.2, 0.25, 0.3, 0.4]

# parameters for classification using raw gene expression
num_features_raw = 8000
