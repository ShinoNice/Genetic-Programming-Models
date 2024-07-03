import torch
from gpolnel.utils.inductive_programming import _get_tree_depth


class Ffunctions():
    """ Class of fitnesses functions.

    This class allows the dynamic call for fitness functions and sets its following attributes:
    - name : it is used for defining all other attributes
    - mode_min : if the fitness should be minimized it is True; if the fitness should be maximized, it id False.
    - best_value : sets the ideal value to which solutions should convert.
    - symbols : the symbol that should be used to represent the fitness function
    - structural : if the fitness is structural (regarding the tree structure)
    """

    # ------------------------
    # -- Constructor ---------
    # ------------------------
    def __init__(self, name, symbol=None, best_value=None):
        self.evaluate = {
            'complexity': self.complexity,
            'depth': self.depth,
            'mae': self.mae,
            'mse': self.mse,
            'n_features': self.n_features,
            'no': self.no,
            'nao': self.nao,
            'naoc': self.naoc,
            'phi': self.phi,
            'rmse': self.rmse,
            'size': self.size,
            'visitation_length': self.visitation_length
        }

        self.mode_min = {
            'complexity': True,
            'depth': True,
            'mae': True,
            'mse': True,
            'n_features': True,
            'no': True,
            'nao': True,
            'naoc': True,
            'phi': False,
            'rmse': True,
            'size': True,
            'visitation_length': True
        }

        self.best_value = {
            'complexity': 0,
            'depth': 0,
            'mae': 0.,
            'mse': 0.,
            'n_features': 1,
            'no': 0,
            'nao': 0,
            'naoc': 0,
            'phi': 79.1,
            'rmse': 0.,
            'size': 0,
            'visitation_length': 0
        }

        self.symbols = {
            'complexity': 'COMPLEXITY',
            'depth': 'DEPTH',
            'mae': 'MAE',
            'mse': 'MSE',
            'n_features': 'N_FEATURES',
            'no': 'NO',
            'nao': 'NAO',
            'naoc': 'NAOC',
            'phi': 'PHI',
            'rmse': 'RMSE',
            'size': 'SIZE',
            'visitation_length': 'VISITATION_LENGTH'
        }

        self.structural = {
            'complexity': True,
            'depth': True,
            'mae': False,
            'mse': False,
            'n_features': True,
            'no': True,
            'nao': True,
            'naoc': True,
            'phi': True,
            'rmse': False,
            'size': True,
            'visitation_length': True
        }

        self.name = name
        self.min_ = self.mode_min[self.name]
        self.symbol = self.symbols[name] if symbol is None else symbol
        self.best_value = self.best_value[name] if best_value is None else best_value
        self.is_structural = self.structural[name]

    def __call__(self, **kwargs):
        return self.evaluate[self.name](**kwargs)

    #
    # Structural Ffunctions
    #
    def complexity(self, **kwargs):
        return kwargs['sol'].get_complexity()

    def depth(self, **kwargs):
        return kwargs['sol'].get_depth()

    def no(self, **kwargs):
        return kwargs['sol'].get_no()

    def nao(self, **kwargs):
        return kwargs['sol'].get_nao()

    def naoc(self, **kwargs):
        return kwargs['sol'].get_naoc()
    
    def phi( self, **kwargs ):
        return kwargs['sol'].get_phi()

    def size( self, **kwargs ):
        return kwargs['sol'].get_size()

    def n_features(self, **kwargs):
        return kwargs['sol'].get_n_features()

    def visitation_length( self, **kwargs ):
        return kwargs['sol'].get_visitation_length()


    #
    # Error-based Ffunctions
    #
    # MAE
    def mae(self, **kwargs):
        if kwargs['call'] == 'dl':
            return self.mae_dl(**kwargs)
        if kwargs['call'] == 'join':
            return self.mae_join(**kwargs)
    def mae_dl(self, **kwargs):
        # Returns the sum of squared errors
        return torch.sum(torch.abs(torch.sub(kwargs['y_true'], kwargs['y_pred'])))
    def mae_join(self, **kwargs):
        return kwargs['fit_dl'] / kwargs['n']

    # MSE
    def mse(self, **kwargs):
        if kwargs['call'] == 'dl':
            return self.mse_dl(**kwargs)
        if kwargs['call'] == 'join':
            return self.mse_join(**kwargs)

    def mse_dl(self, **kwargs):
        # Returns the sum of squared errors
        return torch.sum(torch.pow(torch.sub(kwargs['y_true'], kwargs['y_pred']), 2))

    def mse_join(self, **kwargs):
        return kwargs['fit_dl']/kwargs['n']

    # RMSE
    def rmse(self, **kwargs):
        if kwargs['call'] == 'dl':
            return self.rmse_dl(**kwargs)
        if kwargs['call'] == 'join':
            return self.rmse_join(**kwargs)
        if kwargs['call'] == 'semantic':
            return torch.sqrt(torch.mean(torch.pow(torch.sub(kwargs['y_true'], kwargs['y_pred']), 2), len(kwargs['y_pred'].shape)-1))

    def rmse_dl(self, **kwargs):
        # Returns the sum of squared errors
        return torch.sum(torch.pow(torch.sub(kwargs['y_true'], kwargs['y_pred']), 2), len(kwargs['y_pred'].shape)-1)

    def rmse_join(self, **kwargs):
        return torch.sqrt(kwargs['fit_dl']/kwargs['n'])


