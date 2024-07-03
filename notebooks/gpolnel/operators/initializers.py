""" Initialization operators create initial candidate solutions
The module `gpol.initializers` contains some relevant initialization
operators (initializers) used to create one (for single-point solve)
or several (for population-based solve) random initial candidate
solutions in the solve space. Given the fact this library supports
different types of optimization problems (OPs), the module contains a
collection of initializers suitable for every kind of OP implemented
in this library.
"""

import copy
import math
import random
from joblib import Parallel, delayed

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from gpolnel.problems.inductive_programming import SML
from gpolnel.algorithms.genetic_algorithm import GeneticAlgorithm
from gpolnel.operators.selectors import prm_tournament
from gpolnel.operators.variators import prm_subtree_mtn, swap_xo

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# INDUCTIVE PROGRAMMING PROBLEMS
#
class Terminal:
    """Tree terminal class

    This class represents the terminal nodes of Genetic Programming trees.

    """
    def __init__(self, constant_set, p_constants, n_dims, device):
        self.constant_set = constant_set
        self.p_constants = p_constants
        self.n_dims = n_dims
        self.device = device
        self.generate = {
            'erc': self.erc,
            'cte': self.cte,
            'dataset_feature': self.dataset_feature
        }

    def initialize(self):
        """Initializes the Terminal.

        Terminal nodes can be constants or dataset features.
        According to the probability of using constants (p_constants attribute), the Terminal node will be
        initialized by a Constant of by a DatasetFeature

        Parameters
        ----------
        Returns
        -------
        Terminal
            The generated terminal node.
        """
        if random.uniform(0, 1) < self.p_constants:
            return self.generate[self.constant_set.name]()
        else:
            return self.dataset_feature()

    def erc(self):
        """Initializes the Ephemeral Random Constant (Koza, 1992).

        Reference: Koza, J. R. (1992). Genetic Programming: On the Programming of
        Computers by Means of Natural Selection. MIT Press.

        Parameters
        ----------
        Returns
        -------
        Tensor
            The generated constant.
        """
        return torch.tensor(random.uniform(self.constant_set.min, self.constant_set.max), device=self.device)

    def cte(self):
        """Initializes the Constant

        Parameters
        ----------
        Returns
        -------
        Tensor
            The generated constant.
        """
        return torch.tensor(self.constant_set.values[random.randint(0, len(self.constant_set.values) - 1)], device=self.device)

    def dataset_feature(self):
        """Initializes the Dataset Feature

        Parameters
        ----------
        Returns
        -------
        int
            The index of the dataset feature.
        """
        return random.randint(0, self.n_dims - 1)


class ERC:
    """"
    Ephemeral Random Constant Class (Koza, 1992).

    Reference: Koza, J. R. (1992). Genetic Programming: On the Programming of
        Computers by Means of Natural Selection. MIT Press.
    """
    name = 'erc'

    def __init__(self, min, max):
        self.min = min
        self.max = max
    
    def __str__(self):
        return 'ERC: min {:.4f}, max {:.4f}'.format(self.min, self.max)


class Constant:
    """"
    Constant Class.
    """
    name = 'cte'

    def __init__(self, values):
        self.values = values

    def __str__(self):
        return 'Constant set: values {}'.format(self.values)

def grow(sspace, n_sols):
    return [grow_individual(sspace) for _ in range(n_sols)]


def grow_individual(sspace):
    """ Implements Grow initialization algorithm for GP

    The implementation assumes the probability of sampling a program
    element from the set of functions is the same as from the set of
    terminals until achieving the maximum depth (i.e., 50%). The
    probability of selecting a constant from the set of terminals
    is controlled by the value in "p_constants" key, defined in sspace
    dictionary, and equals 0.5*sspace["p_constants"].

    Parameters
    ----------
    sspace : dict
        Problem instance's solve-space.

    Returns
    -------
    program : list
        A list of program elements which represents an initial computer
        program (candidate solution). The program follows LISP-based
        formulation and Polish pre-fix notation.
    """
    # Starts the tree with a function
    function_ = random.choice(sspace['function_set'])
    program = [function_]
    terminal_stack = [function_.arity]
    max_depth = random.randint(1, sspace['max_init_depth'])

    # While there are open branches
    while terminal_stack:
        depth = len(terminal_stack)
        choice = random.randint(0, 1)  # 0: function_, 1: terminal (50/50)

        # If max init depth allows and the random choice was for a function,
        # Adds a function node to the tree structure
        if (depth < max_depth) and choice == 0:
            function_ = random.choice(sspace['function_set'])
            program.append(function_)
            terminal_stack.append(function_.arity)
        else:
            # Otherwise, adds a terminal node to the tree structure
            terminal = Terminal(
                constant_set=sspace['constant_set'],
                p_constants=sspace['p_constants'],
                n_dims=sspace['n_dims'],
                device=sspace['device']
            ).initialize()
            program.append(terminal)
            terminal_stack[-1] -= 1
            # Updates the terminal stack controller
            while terminal_stack[-1] == 0:
                terminal_stack.pop()
                if not terminal_stack:  # If terminal stack is empty, returns the tree structure
                    return program
                terminal_stack[-1] -= 1
    return None


def prm_grow(sspace):
    """ Implements Grow initialization algorithm

    The library's interface restricts variation operators' parameters
    to solutions' representations only. However, the functioning of
    some of the GP's variation operators requires random trees'
    generation - this is the case of the sub-tree mutation, the
    geometric semantic operators, ... In this sense, the variation
    functions' enclosing scope does not contain enough information to
    generate the initial trees. To remedy this situation, closures are
    used as they provide the variation functions the necessary outer
    scope for trees' initialization: the solve space. Moreover, this
    solution, allows one to have a deeper control over the operators'
    functioning - an important feature for the research purposes.

    Parameters
    ----------
    sspace : dict
        Problem instance's solve-space.

    Returns
    -------
    grow_ : function
        A function which implements Grow initialization algorithm,
        which uses the user-specified solve space for trees'
        initialization.
    """
    def grow_():
        """ Implements Grow initialization algorithm

        Implements Grow initialization algorithm, which uses the user-
        specified solve space for trees' initialization.

        Returns
        -------
        program : list
            A list of program elements which represents an initial computer
            program (candidate solution). The program follows LISP-based
            formulation and Polish pre-fix notation.
        """
        function_ = random.choice(sspace['function_set'])
        program = [function_]
        terminal_stack = [function_.arity]
        max_depth = random.randint(1, sspace['max_init_depth'])

        while terminal_stack:
            depth = len(terminal_stack)
            choice = random.randint(0, 1)  # 0: function_, 1: terminal

            if (depth < max_depth) and choice == 0:
                function_ = random.choice(sspace['function_set'])
                program.append(function_)
                terminal_stack.append(function_.arity)
            else:
                terminal = Terminal(
                    constant_set=sspace['constant_set'],
                    p_constants=sspace['p_constants'],
                    n_dims=sspace['n_dims'],
                    device=sspace['device']
                ).initialize()

                program.append(terminal)
                terminal_stack[-1] -= 1
                while terminal_stack[-1] == 0:
                    terminal_stack.pop()
                    if not terminal_stack:
                        return program
                    terminal_stack[-1] -= 1
        return None

    return grow_


def full(sspace, n_sols):
    return [full_individual(sspace) for _ in range(n_sols)]


def full_individual(sspace):
    """ Implements Full initialization algorithm

    The probability of selecting a constant from the set of terminals
    is controlled by the value in "p_constants" key, defined in sspace
    dictionary, and equals 0.5*sspace["p_constants"].

    Parameters
    ----------
    sspace : dict
        Problem instance's solve-space.

    Returns
    -------
    program : list
        A list of program elements which represents an initial computer
        program (candidate solution). The program follows LISP-based
        formulation and Polish pre-fix notation.
    """
    function_ = random.choice(sspace["function_set"])
    program = [function_]
    terminal_stack = [function_.arity]

    while terminal_stack:
        depth = len(terminal_stack)

        if depth < sspace["max_init_depth"]:
            function_ = random.choice(sspace["function_set"])
            program.append(function_)
            terminal_stack.append(function_.arity)
        else:
            terminal = Terminal(
                constant_set=sspace["constant_set"],
                p_constants=sspace["p_constants"],
                n_dims=sspace["n_dims"],
                device=sspace["device"]
            ).initialize()
            program.append(terminal)
            terminal_stack[-1] -= 1
            while terminal_stack[-1] == 0:
                terminal_stack.pop()
                if not terminal_stack:
                    return program
                terminal_stack[-1] -= 1
    return None


def prm_full(sspace):
    """ Implements Full initialization algorithm

    The library's interface restricts variation operators' parameters
    to solutions' representations only. However, the functioning of
    some of the GP's variation operators requires random trees'
    generation - this is the case of the sub-tree mutation, the
    geometric semantic operators, ... In this sense, the variation
    functions' enclosing scope does not contain enough information to
    generate the initial trees. To remedy this situation, closures are
    used as they provide the variation functions the necessary outer
    scope for trees' initialization: the solve space. Moreover, this
    solution, allows one to have a deeper control over the operators'
    functioning - an important feature for the research purposes.

    Parameters
    ----------
    sspace : dict
        Problem instance's solve-space.

    Returns
    -------
    full_ : function
        A function which implements Full initialization algorithm,
        which uses the user-specified solve space for trees'
        initialization.
    """
    def full_():
        """ Implements Full initialization algorithm

        Implements Full initialization algorithm, which uses the user-
        specified solve space for trees' initialization.

        Returns
        -------
        program : list
            A list of program elements which represents an initial computer
            program (candidate solution). The program follows LISP-based
            formulation and Polish pre-fix notation.
        """
        function_ = random.choice(sspace["function_set"])
        program = [function_]
        terminal_stack = [function_.arity]

        while terminal_stack:
            depth = len(terminal_stack)

            if depth < sspace["max_init_depth"]:
                function_ = random.choice(sspace["function_set"])
                program.append(function_)
                terminal_stack.append(function_.arity)
            else:
                terminal = Terminal(
                    constant_set=sspace["constant_set"],
                    p_constants=sspace["p_constants"],
                    n_dims=sspace["n_dims"],
                    device=sspace["device"]
                ).initialize()
                program.append(terminal)
                terminal_stack[-1] -= 1
                while terminal_stack[-1] == 0:
                    terminal_stack.pop()
                    if not terminal_stack:
                        return program
                    terminal_stack[-1] -= 1
        return None

    return full_


def rhh(sspace, n_sols):
    """ Implements Ramped Half and Half initialization algorithm

    Implements the Ramped Half and Half, which, by itself, uses
    Full and Grow.

    Parameters
    ----------
    sspace : dict
        Problem-specific solve space (𝑆).
    n_sols : int
        The number of solutions in the population

    Returns
    -------
    pop : list
        A list of program elements which represents the population
        initial of computer programs (candidate solutions). Each
        program is a list of program's elements that follows a
        LISP-based formulation and Polish pre-fix notation.
    """
    pop = []
    n_groups = sspace["max_init_depth"]
    group_size = math.floor((n_sols / 2.) / n_groups)
    for group in range(n_groups):
        max_depth_group = group + 1
        for i in range(group_size):
            sspace_group = sspace
            sspace_group["max_init_depth"] = max_depth_group
            pop.append(full_individual(sspace_group))
            pop.append(grow_individual(sspace_group))
    while len(pop) < n_sols:
        pop.append(grow_individual(sspace_group) if random.randint(0, 1) else full_individual(sspace_group))
    return pop

def nn_init_individual(sspace):
    n_neurons = [sspace['input_shape']] + sspace['n_hidden_neurons'] + [sspace['n_output']]
    # Weights
    sol = []
    for i_nn in range(len(n_neurons) - 1):
        sol += [torch.randn(n_neurons[i_nn], n_neurons[i_nn + 1], device=sspace['device'])*sspace['init_factor']]
    # Biases
    sol = [sol, [torch.randn(n, device=sspace['device']) for n in n_neurons[1:]]]
    return sol


def nn_init(sspace, n_sols):
    return [nn_init_individual(sspace) for _ in range(n_sols)]

