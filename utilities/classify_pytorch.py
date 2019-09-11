import time
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold,
    cross_val_predict
)
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.utils.data as data_utils


class LogisticRegression(nn.Module):
    """Model for PyTorch logistic regression."""

    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        # one output for binary classification
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)

class TorchLR:
    """Class to run hyperparameter search/cross-validation.

    Maintains state for training/cross-validation results.
    """
    def __init__(self,
                 params_map,
                 seed=1,
                 num_iters=10,
                 num_inner_folds=4,
                 use_gpu=False,
                 verbose=False):

        self.seed = seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        max_params_length = max(len(vs) for k, vs in params_map.items())
        # if there's only one choice provided for each hyperparameter,
        # we'll skip the parameter search later
        #
        # if there are multiple choices, select num_iters parameter
        # combinations to be tested during the parameter search
        if max_params_length > 1:
            params_map = self.get_params_map(params_map,
                                             num_iters=num_iters)
        self.params_map = params_map
        self.num_inner_folds = num_inner_folds
        self.use_gpu = use_gpu
        self.verbose = verbose


    def get_params_map(self, param_choices, num_iters=10):
        """Get random combinations of hyperparameters to search over.

        Currently combinations are selected with replacement, i.e. duplicates can
        happen.

        TODO: could make this sample from continuous distributions too,
        might be useful for some params

        Parameters
        ----------
        param_choices: dict, (str: list)
            Maps hyperparameter names to choices (currently only works with
            discrete values). Example:
            param_choices = {
                'learning_rate': [0.005, 0.001, 0.0001, 0.00005],
                'batch_size': [10, 20, 50, 100],
                'num_epochs': [200, 500, 1000],
                'l1_penalty': [0, 0.01, 0.1, 1, 10]
            }

        num_iters : int
            The number of combinations to search over.

        Returns
        -------
        dict, (str: list)
            Maps hyperparameter names to lists of values to try.

        """
        import random; random.seed(self.seed)
        # sorting here ensures that results for models that share the same
        # parameters will have the same choices, and thus will be easily
        # comparable
        param_options = sorted(param_choices.items())
        params_map = {p: [random.choice(vals) for _ in range(num_iters)]
                         for p, vals in param_options}
        return params_map


    def train_torch_model(self, X_train, X_test, y_train, y_test):
        """Wrapper function for PyTorch model training.

        If multiple hyperparameter choices are provided, get the best
        set of hyperparameters from a random search. Otherwise, just use
        the hyperparameters provided to train/evaluate the model.
        """
        max_params_length = max(len(vs) for k, vs in self.params_map.items())
        if max_params_length > 1:
            results_df, best_params = self.torch_param_selection(X_train, y_train)
            self.results_df = results_df
        else:
            best_params = {k: vs[0] for k, vs in self.params_map.items()}

        self.best_params = best_params

        losses, preds, preds_bn = self.torch_model(X_train, X_test, y_train, y_test,
                                                   best_params)

        return losses, preds, preds_bn


    def torch_model(self, X_train, X_test, y_train, y_test, params,
                    save_weights=False):

        """Main function for training PyTorch model.

        Parameters
        ----------
        X_train : array_like, [n_samples, n_features]
            Training data

        X_test : array_like, [n_samples, n_features]
            Data to evaluate model on

        y_train : array_like, [n_samples]
            Training labels

        y_test : array_like, [n_samples]
            Labels to evaluate model on

        params : dict, (str: mixed)
            Maps hyperparameter names to a single value, used to train the model

        save_weights: bool
            Whether or not to save weights (coefficients) from trained model

        Returns
        -------
        tuple : ((list, list), (list, list), (list, list))
            ((loss on training data, loss on test data),
             (predictions on training data, predictions on testing data),
             (binarized predictions on training/test data))
        """
        if self.verbose:
            t = time.time()

        learning_rate = params['learning_rate']
        batch_size = params['batch_size']
        num_epochs = params['num_epochs']
        l1_penalty = params['l1_penalty']

        # Weight loss function based on training data label imbalance
        # see, e.g. https://discuss.pytorch.org/t/about-bcewithlogitslosss-pos-weights/22567/2
        #
        # TODO: could add a function argument to turn this on/off (but in
        # general it seems to give slightly better results)
        train_count = np.bincount(y_train)
        pos_weight = train_count[0] / train_count[1]
        if self.verbose:
            print('\n[0, 1]: {} (pos_weight={:.4f})'.format(train_count, pos_weight))

        if self.use_gpu:
            X_tr = torch.stack([torch.Tensor(x) for x in X_train]).cuda()
            X_ts = torch.stack([torch.Tensor(x) for x in X_test]).cuda()
            y_tr = torch.Tensor(y_train).view(-1, 1).cuda()
            y_ts = torch.Tensor(y_test).view(-1, 1).cuda()
            pos_weight = torch.Tensor([pos_weight]).cuda()
        else:
            X_tr = torch.stack([torch.Tensor(x) for x in X_train])
            X_ts = torch.stack([torch.Tensor(x) for x in X_test])
            y_tr = torch.Tensor(y_train).view(-1, 1)
            y_ts = torch.Tensor(y_test).view(-1, 1)
            pos_weight = torch.Tensor([pos_weight])

        train_loader = data_utils.DataLoader(
                data_utils.TensorDataset(X_tr, y_tr),
                batch_size=batch_size, shuffle=True)

        model = LogisticRegression(X_train.shape[1])
        if self.use_gpu:
            model = model.cuda()

        # pos_weight is a scalar, the weight for the 1 class
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, patience=5)

        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, (X_batch, y_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                l1_loss = sum(torch.norm(param, 1) for param in model.parameters())
                loss += l1_penalty * l1_loss
                running_loss += loss
                loss.backward()
                optimizer.step()
            scheduler.step(running_loss)

        if save_weights:
            if self.use_gpu:
                self.last_weights = model.linear.weight.data.cpu().numpy()
            else:
                self.last_weights = model.linear.weight.data.numpy()

        y_pred_train = model(X_tr)
        y_pred_test = model(X_ts)

        train_loss = float((
                criterion(y_pred_train, y_tr) +
                l1_penalty * sum(torch.norm(param, 1) for param in model.parameters())
        ).detach())

        test_loss = float((
                criterion(y_pred_test, y_ts) +
                l1_penalty * sum(torch.norm(param, 1) for param in model.parameters())
        ).detach())

        if self.verbose:
            print('(time: {:.3f} sec)'.format(time.time() - t))

        if self.use_gpu:
            y_pred_train = y_pred_train.cpu().detach().numpy()
            y_pred_test = y_pred_test.cpu().detach().numpy()
        else:
            y_pred_train = y_pred_train.detach().numpy()
            y_pred_test = y_pred_test.detach().numpy()


        y_pred_bn_train = (y_pred_train > 0).astype('int')
        y_pred_bn_test = (y_pred_test > 0).astype('int')

        return ((train_loss, test_loss),
                (y_pred_train, y_pred_test),
                (y_pred_bn_train, y_pred_bn_test))


    def torch_param_selection(self, X_train, y_train):
        """Cross-validate to select best parameters from a set of possibilities.

        Dataset terminology: (subtrain | tune) = train | test
        (avoiding the term "validation" since it's overloaded in biology)
        """
        # k-fold cross-validation over the training data
        kf = KFold(n_splits=self.num_inner_folds, shuffle=True,
                   random_state=self.seed)
        results_df = None
        for fold, (subtrain_ixs, tune_ixs) in enumerate(kf.split(X_train), 1):
            X_subtrain, X_tune = X_train[subtrain_ixs], X_train[tune_ixs]
            y_subtrain, y_tune = y_train[subtrain_ixs], y_train[tune_ixs]
            if self.verbose:
                print('Running inner CV fold {} of {}'.format(
                        fold, self.num_inner_folds))
            result_df = self.torch_tuning(X_subtrain, X_tune, y_subtrain, y_tune)
            result_df['fold'] = fold
            if results_df is None:
                results_df = result_df
            else:
                results_df = pd.concat((results_df, result_df), ignore_index=True)

        # get the index of the parameter set that performed the best on
        # average across folds
        sorted_df = (
            results_df.loc[results_df['train/tune'] == 'tune']
                      .groupby('param_set')
                      .mean()
                      .sort_values(by='loss')
                      .reset_index()
        )
        best_ix = sorted_df.loc[0, 'param_set']
        best_params = {k: v[best_ix] for k, v in self.params_map.items()}

        # save CV results for best parameter set, for analysis later
        self.cv_results_df = results_df[results_df['param_set'] == best_ix]

        return results_df, best_params


    def torch_tuning(self, X_subtrain, X_tune, y_subtrain, y_tune):
        """Run parameter search on a single subtrain/tune split."""
        result = {
            'param_set': [],
            'train/tune': [],
            'loss': [],
            'auroc': []
        }
        for param in self.params_map.keys():
            result[param] = []
        num_iters = len(self.params_map[list(self.params_map.keys())[0]])
        for ix in range(num_iters):
            if self.verbose:
                print('-- Running parameter set {} of {}...'.format(ix+1, num_iters),
                      end='')
            params = {k: v[ix] for k, v in self.params_map.items()}
            print(params)
            losses, y_preds, __ = self.torch_model(X_subtrain,
                                                   X_tune,
                                                   y_subtrain,
                                                   y_tune,
                                                   params)
            y_pred_subtrain, y_pred_tune = y_preds
            subtrain_loss, tune_loss = losses
            subtrain_auroc = roc_auc_score(y_subtrain, y_pred_subtrain, average="weighted")
            tune_auroc = roc_auc_score(y_tune, y_pred_tune, average="weighted")
            if self.verbose:
                print('subtrain_loss: {:.4f}, tune_loss: {:.4f}'.format(
                        subtrain_loss, tune_loss))
            result['param_set'].append(ix)
            result['train/tune'].append('train')
            result['loss'].append(subtrain_loss)
            result['auroc'].append(subtrain_auroc)
            for param in self.params_map.keys():
                result[param].append(self.params_map[param][ix])
            result['param_set'].append(ix)
            result['train/tune'].append('tune')
            result['loss'].append(tune_loss)
            result['auroc'].append(tune_auroc)
            for param in self.params_map.keys():
                result[param].append(self.params_map[param][ix])
        return pd.DataFrame(result)


if __name__ == '__main__':
    # code to test the implementation quickly against sklearn
    # using breast cancer dataset from sklearn.datasets
    # original: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

    import argparse

    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    import sys; sys.path.append('.')
    import config as cfg
    from tcga_util import get_threshold_metrics
    from classify_sklearn import train_sklearn_model

    p = argparse.ArgumentParser()
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--seed', type=int, default=cfg.default_seed)
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    # hyperparameter choices to do a random search over
    sklearn_param_choices = {
        'alpha': [0.1, 0.13, 0.15, 0.2, 0.25, 0.3],
        'l1_ratio': [0.15, 0.16, 0.2, 0.25, 0.3, 0.4]
    }

    torch_param_choices = {
        'learning_rate': [0.001, 0.0001, 0.00001],
        'batch_size': [10, 20, 50, 100],
        'num_epochs': [100, 200, 500, 1000],
        'l1_penalty': [0, 0.01, 0.1, 1, 10]
    }

    num_iters = 8
    num_inner_folds = 3
    model = TorchLR(torch_param_choices,
                    seed=args.seed,
                    num_iters=num_iters,
                    num_inner_folds=num_inner_folds,
                    use_gpu=args.gpu,
                    verbose=args.verbose)

    # load data and split into train/test sets
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=args.seed)

    # classify using sklearn SGDClassifier
    y_pred = train_sklearn_model(X_train, X_test,
                                 y_train,
                                 sklearn_param_choices['alpha'],
                                 sklearn_param_choices['l1_ratio'],
                                 seed=args.seed)

    y_pred_train, y_pred_test, y_pred_bn_train, y_pred_bn_test = y_pred
    print(y_pred_train[:20].flatten())
    print(y_pred_bn_train[:20].flatten())
    print(y_train[:20])

    sk_train_acc = sum(
            [1 for i in range(len(y_pred_train))
               if y_pred_bn_train[i] == y_train[i]]
    ) / len(y_pred_train)
    sk_test_acc = sum(
            [1 for i in range(len(y_pred_test))
               if y_pred_bn_test[i] == y_test[i]]
    ) / len(y_pred_test)

    sk_train_results = get_threshold_metrics(y_train, y_pred_train)
    sk_test_results = get_threshold_metrics(y_test, y_pred_test)

    losses, preds, preds_bn = model.train_torch_model(X_train, X_test,
                                                      y_train, y_test)

    y_pred_train, y_pred_test = preds
    y_pred_bn_train, y_pred_bn_test = preds_bn
    print(y_pred_train[:20].flatten())
    print(y_pred_bn_train[:20].flatten())
    print(y_train[:20])

    def calculate_accuracy(y, y_pred):
        return sum(1 for i in range(len(y)) if y[i] == y_pred[i]) / len(y)

    torch_train_acc = calculate_accuracy(y_train, y_pred_bn_train)
    torch_test_acc = calculate_accuracy(y_test, y_pred_bn_test)

    torch_train_results = get_threshold_metrics(y_train, y_pred_train)
    torch_test_results = get_threshold_metrics(y_test, y_pred_test)

    print('Sklearn train accuracy: {:.3f}, test accuracy: {:.3f}'.format(
        sk_train_acc, sk_test_acc))
    print('Sklearn train AUROC: {:.3f}, test AUROC: {:.3f}'.format(
        sk_train_results['auroc'], sk_test_results['auroc']))
    print('Sklearn train AUPRC: {:.3f}, test AUPRC: {:.3f}'.format(
        sk_train_results['aupr'], sk_test_results['aupr']))

    print('Torch train accuracy: {:.3f}, test accuracy: {:.3f}'.format(
        torch_train_acc, torch_test_acc))
    print('Torch train AUROC: {:.3f}, test AUROC: {:.3f}'.format(
        torch_train_results['auroc'], torch_test_results['auroc']))
    print('Torch train AUPRC: {:.3f}, test AUPRC: {:.3f}'.format(
        torch_train_results['aupr'], torch_test_results['aupr']))

    y_pred_train = np.random.uniform(size=(len(y_train),))
    y_pred_test = np.random.uniform(size=(len(y_test),))
    y_pred_bn_train = (y_pred_train > 0.5).astype('int')
    y_pred_bn_test = (y_pred_test > 0.5).astype('int')

    random_train_acc = calculate_accuracy(y_train, y_pred_bn_train)
    random_test_acc = calculate_accuracy(y_test, y_pred_bn_test)

    random_train_results = get_threshold_metrics(y_train, y_pred_train)
    random_test_results = get_threshold_metrics(y_test, y_pred_test)

    print('Random guessing train accuracy: {:.3f}, test accuracy: {:.3f}'.format(
        random_train_acc, random_test_acc))
    print('Random guessing train AUROC: {:.3f}, test AUROC: {:.3f}'.format(
        random_train_results['auroc'], random_test_results['auroc']))
    print('Random guessing train AUPRC: {:.3f}, test AUPRC: {:.3f}'.format(
        random_train_results['aupr'], random_test_results['aupr']))