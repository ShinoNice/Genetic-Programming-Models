import os
import errno
import pickle
import random

import torch
import torch.nn as nn
from gpolnel.utils.solution import Solution

# +++++++++++++++++++++++++++ GP's constants set
def get_constant_set(start, end, size, device="cpu"):
    step, i = (end - start) / size, 0
    constant_set = []
    for _ in range(size):
        constant_set.append(torch.tensor([start + i], device=device))
        i += step
    return constant_set


# +++++++++++++++++++++++++++ GP's functions set
class _Function(object):
    """ Implements a function program element

    This object is able to be called with NumPy vectorized arguments
    and return a resulting vector based on a mathematical relationship.
    This class was strongly inspired on gplearn's implementation (for
    more details, visit https://github.com/trevorstephens/gplearn).

    Attributes
    ----------
    function_ : callable
        A function with signature function(x1, *args) that returns a Numpy
        array of the same shape as its arguments.
    name : str
        The name for the function as it should be represented in the program
        and its visualizations.
    arity : int
        The number of arguments that the ``function`` takes.
    """
    def __init__(self, function_, name, arity, arithmetic):
        self.function_ = function_
        self.name = name
        self.arity = arity
        self.arithmetic = arithmetic

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    def __call__(self, *args):
        return self.function_(*args)


def make_function(function_, name, arity, arithmetic):
    """ Creates an instance of type Function

    This factory function creates a function node, one of the core nodes in any
    program. The resulting object is able to be called with NumPy vectorized
    arguments and return a resulting vector based on a mathematical
    relationship.

    Parameters
    ----------
    function_ : callable
        A function with signature `function(x1, *args)` that returns a
        torch.Tensor of the same shape as its arguments.
    name : str
        Function's name as it should be represented in the solution's
        representation.
    arity : int
        The number of arguments the function takes.
    arithmetic : Boolean
        If the operator is arithmetic

    Returns
    -------
    _Function
        An instance of type _Function.
    """
    return _Function(function_, name, arity, arithmetic)


def protected_div(x1, x2):
    """ Implements the division protected against zero denominator

    Performs division between x1 and x2. If x2 is (or has) zero(s), the
    function returns the numerator's value(s).

    Parameters
    ----------
    x1 : torch.Tensor
        The numerator.
    x2 : torch.Tensor
        The denominator.

    Returns
    -------
    torch.Tensor
        Result of protected division between x1 and x2.
    """
    return torch.where(torch.abs(x2) > 0.001, torch.div(x1, x2), torch.tensor(1.0, dtype=x2.dtype, device=x2.device))


def protected_log(x):
    """ Implements the logarithm protected against non-positives

    Applies the natural logarithm function of on the elements of the
    input tensor. When the value(s) are smaller than 1e-4, returns the
    natural logarithm of 1e-4. When the value(s) are greather than 1e4,
    returns the natural logarithm of 1e4.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor.

    Returns
    -------
    torch.Tensor
        Returns a new tensor with the natural logarithm of the
        elements of the input.
    """
    x[x > 1e4] = 1e4
    x[x < 1e-4] = 1e-4
    return torch.log(x)


def protected_exp(x):
    """ Implements the expoenential protected against overflow

    Applies the exponential function of on the elements of the
    input tensor. When the value(s) are greather than 10, returns 6e4,
    which is close to 5.9874e+04 (exp(10)).

    Parameters
    ----------
    x : torch.Tensor
        The input tensor.

    Returns
    -------
    torch.Tensor
        Returns a new tensor with the exponential of the elements of the input.
    """
    return torch.where(x < 10, torch.exp(x), torch.tensor(6e4, dtype=x.dtype, device=x.device))


def _protected_stack(x1, x2):
    """ Performs a protected stacking of two tensors

     The original torch.stack function cannot stack different shaped
     tensors, which is frequently necessary when stacking some tensor
     with a constant (also a tensor) during min/max/mean operations
     involving two operands. This function performs an appropriate
     re-shaping.

    Parameters
    ----------
    x1 : torch.Tensor
        First operand.
    x2 : torch.Tensor
        Second operand.

    Returns
    -------
    torch.Tensor
        Stacked tensors.
    """
    if x1.shape == x2.shape:
        return torch.stack([x1, x2])
    else:
        t_b = x1 if x1.shape > x2.shape else x2
        t_s = x1 if x1.shape < x2.shape else x2
        return torch.stack([t_b, t_s.repeat(*t_b.shape)])


def protected_min(t1, t2):
    """ Returns the minimum between two tensors at each index

    To perform the min operation between the values of the two tensors
    at the same index.

    Returns
    -------
    torch.Tensor
        Tensor of the same shape as t1 or t2, containing the smallest
        value between the two.
    """
    return torch.min(_protected_stack(t1, t2), dim=0)[0]


def protected_max(t1, t2):
    """ Returns the maximum between two tensors at each index

    To perform the max operation between the values of the two tensors
    at the same index.

    Returns
    -------
    torch.Tensor
        Tensor of the same shape as t1 or t2, containing the largest
        value between the two.
    """
    return torch.max(_protected_stack(t1, t2), dim=0)[0]


def protected_mean(t1, t2):
    """ Returns the average between two tensors at each index

    To perform the max operation between the values of the two tensors
    at the same index.

    Returns
    -------
    torch.Tensor
        Tensor of the same shape as t1 or t2, containing the average
        value between the two.
    """
    return torch.mean(_protected_stack(t1, t2), dim=0)


# +++++++++++++++++++++++++++ Creates the functions' set
# Traditional operators
add2 = _Function(function_=torch.add, name='add', arity=2, arithmetic=True)
sub2 = _Function(function_=torch.sub, name='sub', arity=2, arithmetic=True)
mul2 = _Function(function_=torch.mul, name='mul', arity=2, arithmetic=True)
div2 = _Function(function_=protected_div, name='div', arity=2, arithmetic=True)
# Higher arity operators
add3 = _Function(function_=torch.add, name='add', arity=3, arithmetic=True)
# Vectorial operators
mean2 = _Function(function_=protected_mean, name='mean', arity=2, arithmetic=False)
max2 = _Function(function_=protected_max, name='max', arity=2, arithmetic=False)
min2 = _Function(function_=protected_min, name='min', arity=2, arithmetic=False)
# Non-linearities
log1 = _Function(function_=protected_log, name='log', arity=1, arithmetic=False)
exp1 = _Function(function_=protected_exp, name='exp', arity=1, arithmetic=False)
sin1 = _Function(function_=torch.sin, name='sin', arity=1, arithmetic=False)
cos1 = _Function(function_=torch.cos, name='cos', arity=1, arithmetic=False)
# Activation functions
lf1 = _Function(function_=nn.Sigmoid(), name='lf', arity=1, arithmetic=False)
tanh1 = _Function(function_=nn.Tanh(), name='tanh', arity=1, arithmetic=False)

# Puts everything into a dictionary
function_map = {'add': add2, 'sub': sub2, 'mul': mul2, 'div': div2,
                'log': log1, 'exp': exp1, 'sin': sin1, 'cos': cos1,
                'lf': lf1, 'tanh': tanh1,
                'mean': mean2, 'max': max2, 'min': min2}


# +++++++++++++++++++++++++++ Trees execution
def get_subtree(tree):
    # Check tree's length: if too small, return the full tree
    if len(tree) <= 3:
        start, end = 0, len(tree)
    else:
        probs = torch.tensor([0.9 if isinstance(node, _Function) else 0.1 for node in tree])
        probs = torch.cumsum(torch.div(probs, probs.sum()), dim=0)
        rnd = random.uniform(0.00001, 0.99999)
        start = (probs >= rnd).nonzero()[0][0].item()
        stack = 1
        end = start
        while stack > end - start:
            node = tree[end]
            if isinstance(node, _Function):
                stack += node.arity
            end += 1

    return start, end


def _set_tree_depth(sol):
    sol.depth = _get_tree_depth(sol.repr_)
    return sol


def _get_tree_depth(repr_):
    terminals = [0]
    depth = 1
    for node in repr_:
        if isinstance(node, _Function):
            terminals.append(node.arity)
            depth = max(len(terminals), depth)
        else:
            terminals[-1] -= 1
            while terminals[-1] == 0:
                terminals.pop()
                terminals[-1] -= 1
    return depth - 1


def _execute_tree(repr_, X):
    node = repr_[0]
    # Secure against constants' tree
    if isinstance(node, torch.Tensor):
        return node.repeat_interleave(len(X))  # return node
    if isinstance(node, int):
        return X[:, node]
    apply_stack = []
    for node in repr_:
        if isinstance(node, _Function):
            apply_stack.append([node])
        else:
            apply_stack[-1].append(node)

        while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
            function_ = apply_stack[-1][0]
            terminals = [X[:, t] if isinstance(t, int) else t for t in apply_stack[-1][1:]]
            intermediate_result = function_(*terminals)
            if len(apply_stack) != 1:
                apply_stack.pop()
                apply_stack[-1].append(intermediate_result)
            else:
                # Secure against constants' tree
                if len(intermediate_result.shape) == 0:
                    return torch.cat(X.shape[0]*[intermediate_result[None]])
                else:
                    return intermediate_result
    return None


# +++++++++++++++++++++++++++ GSGP reconstruction algorithm
def prm_reconstruct_tree(history, path_init_pop, path_rts, device='cpu'):
    """ Implements GSGP trees' reconstruction

    This function is used to provide the reconstruct_tree (the inner
    function) with the necessary environment (the outer scope) - the
    table which stores the evolution's history, the paths towards the
    initial and the random trees, and the processing device that was
    used in the underlying experiment.

    Parameters
    ----------
    history : pandas.DataFrame
        Stores the evolution's history in the following columns:
            - "Iter": iteration's number;
            - "Operator": the variation operator that was applied on
             a given offspring;
            - "T1": the ID of the first parent;
            - "T2": the ID of the second parent (if GSC was applied);
            - "Tr": the ID of a random tree generated (assumes only
             one random tree is necessary to apply an operator);
            - "ms": mutation's step (if GSM was applied);
            - "Fitness": offspring's training fitness;
    path_init_pop : str
        Paths towards the initial trees.
    path_rts : str
        Paths towards the random trees.
    device : str (default="cpu")
        Specification of the processing device that was used in the
        underling experiment.

    Returns
    -------
    reconstruct_tree : function
        A function that reconstructs the user-specified tree.
    """
    # Verifies initial trees' directory
    if not os.path.isdir(path_init_pop):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path_init_pop)
    # Verifies random trees' directory
    if not os.path.isdir(path_rts):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path_rts)

    def reconstruct_tree(id):
        """ Implements GSGP trees' reconstruction

        Reconstructs the user-specified tree following its evolutionary
        history.

        Parameters
        ----------
        id : str
            Row's index in history pandas.DataFrame, which serves as an
            identification of the tree to be reconstructed.

        Returns
        -------
        list
            GSGP tree stored as a LISP tree.
        """
        if history.loc[id, "Iter"] == 1:
            if history.loc[id, "Operator"] == "crossover":
                # Loads trees
                with open(os.path.join(path_init_pop, history.loc[id, "T1"] + '.pickle'), 'rb') as f:
                    t1 = pickle.load(f)
                with open(os.path.join(path_init_pop, history.loc[id, "T2"] + '.pickle'), 'rb') as f:
                    t2 = pickle.load(f)
                with open(os.path.join(path_rts, history.loc[id, "Tr"] + '.pickle'), 'rb') as f:
                    tr = pickle.load(f)
                # Returns as geometric semantic crossover
                return [add2, mul2] + tr + t1 + [mul2, sub2, torch.tensor([1.0], device=device)] + tr + t2
            else:
                # Loads trees
                with open(os.path.join(path_init_pop, history.loc[id, "T1"] + '.pickle'), 'rb') as f:
                    t1 = pickle.load(f)
                with open(os.path.join(path_rts, history.loc[id, "Tr"] + '.pickle'), 'rb') as f:
                    tr = pickle.load(f)
                ms = torch.tensor([history.loc[id, "ms"]], device=device)

                # Returns as geometric semantic mutation
                return [add2] + t1 + [mul2, ms] + tr
        else:
            if history.loc[id, "Operator"] == "crossover":
                # Loads the random tree
                with open(os.path.join(path_rts, history.loc[id, "Tr"] + '.pickle'), 'rb') as f:
                    tr = pickle.load(f)
                # Returns as geometric semantic crossover recursively
                return [add2, mul2]+tr+reconstruct_tree(history.loc[id, "T1"]) + \
                       [mul2, sub2, torch.tensor([1.0], device=device)]+tr+reconstruct_tree(history.loc[id, "T2"])
            else:
                # Loads the random tree
                with open(os.path.join(path_rts, history.loc[id, "Tr"] + '.pickle'), 'rb') as f:
                    tr = pickle.load(f)
                ms = torch.tensor([history.loc[id, "ms"]], device=device)

                # Returns as geometric semantic mutation
                return [add2]+reconstruct_tree(history.loc[id, "T1"])+[mul2, ms]+tr

    return reconstruct_tree
