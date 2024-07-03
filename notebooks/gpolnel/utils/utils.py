import math
from joblib import cpu_count
from scipy.stats import spearmanr, pearsonr

import torch


# +++++++++++++++++++++++++++ Train/Test split
def train_test_split(X, y, p_test=0.3, shuffle=True, indices_only=False, seed=0):
    """ Splits X and y tensors into train and test subsets

    This method replicates the behaviour of Sklearn's 'train_test_split'.

    Parameters
    ----------
    X : torch.Tensor
        Input data instances,
    y : torch.Tensor
        Target vector.
    p_test : float (default=0.3)
        The proportion of the dataset to include in the test split.
    shuffle : bool (default=True)
        Whether or not to shuffle the data before splitting.
    indices_only : bool (default=False)
        Whether or not to return only the indices representing training and test partition.
    seed : int (default=0)
        The seed for random numbers generators.

    Returns
    -------
    X_train : torch.Tensor
        Training data instances.
    y_train : torch.Tensor
        Training target vector.
    X_test : torch.Tensor
        Test data instances.
    y_test : torch.Tensor
        Test target vector.
    train_indices : torch.Tensor
        Indices representing the training partition.
    test_indices : torch.Tensor
    Indices representing the test partition.
    """
    # Sets the seed before generating partition's indexes
    torch.manual_seed(seed)
    # Generates random indices
    if shuffle:
        indices = torch.randperm(X.shape[0])
    else:
        indices = torch.arange(0, X.shape[0], 1)
    # Splits indices
    split = int(math.floor(p_test * X.shape[0]))
    train_indices, test_indices = indices[split:], indices[:split]

    if indices_only:
        return train_indices, test_indices
    else:
        # Generates train/test partitions
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        return X_train, X_test, y_train, y_test


def _get_tasks_per_job(total, n_jobs):
    """

    Parameters
    ----------


    Returns
    -------
    tasks_per_job : torch.Tensor
    """
    # Verifies parameter's validity
    if n_jobs == 0:
        raise ValueError('Parameter n_jobs == 0 has no meaning.')
    # Estimates the effective number of jobs
    n_jobs_ = min(cpu_count(), n_jobs)
    tasks_per_job = (total // n_jobs_) * torch.ones(n_jobs, dtype=torch.int)
    return tasks_per_job[:tasks_per_job % n_jobs] + 1


# +++++++++++++++++++++++++++ Fitness (a.k.a. Cost) Functions
def phi(l=None, no=None, nao=None, naoc=None, sol=None):
    '''
    Calculates the PHI (Proxy for Human Interpretability) of mathematical expressions according to
    Virgolin, M., De Lorenzo, A., Medvet, E., Randone, F. (2020) Learning a Formula of Interpretability to Learn
    Interpretable Formulas. In: Parallel Problem Solving from Nature (PPSN), XVI, 79–93, Springer. Cham, Switzerland.
    Parameters
    ----------
        l : length (=size or number of elements) in the mathematical expression
        no : number of operators in the mathematical expression
        nao : number of non-arithmetic operators in the mathematical expression
        naoc : number of consecutive non-arithmetic operators in the mathematical expression
        sol : solution to be evaluated

    Return
    ----------
        float : PHI linear model
    '''
    if sol is None:
        return 79.1 - .2 * l - 0.5 * no - 3.4 * nao - 4.5 * naoc
    return 79.1 - .2 * sol.get_size() - 0.5 * sol.get_no() - 3.4 * sol.get_nao() - 4.5 * sol.get_naoc()