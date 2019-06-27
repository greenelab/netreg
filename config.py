import os

pj = lambda *paths: os.path.abspath(os.path.join(*paths))

data_dir = pj(os.getcwd(), 'data')
models_dir = pj(os.getcwd(), 'models')

default_seed = 42
