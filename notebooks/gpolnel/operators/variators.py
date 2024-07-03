""" Variation operators "move" the candidate solutions across S
The module `gpol.variators` contains some relevant variation operators
(variators) used to "move" the candidate solutions across the solve
space in between the iterations. Given the fact this library supports
different types of iterative solve algorithms (ISAs), the module
contains a collection of variators suitable for every kind of ISA
implemented in this library.
"""

import copy
import random

import torch
import numpy as np

from gpolnel.utils.inductive_programming import tanh1, lf1, add2, sub2, mul2, _execute_tree, get_subtree, _Function


# +++++++++++++++++++++++++++ Inductive Programming
def swap_xo(p1, p2):
    """ Implements the swap crossover

    The swap crossover (a.k.a. standard GP's crossover) consists of
    exchanging (swapping) parents' two randomly selected subtrees.

    Parameters
    ----------
    p1 : list
        Representation of the first parent.
    p2 : list
        Representation of the second parent.

    Returns
    -------
    list, list
        Tuple of two lists, each representing an offspring obtained
        from swapping two randomly selected sub-trees in the parents.
    """
    # Selects start and end indexes of the first parent's subtree
    p1_start, p1_end = get_subtree(p1)
    # Selects start and end indexes of the second parent's subtree
    p2_start, p2_end = get_subtree(p2)

    return p1[:p1_start] + p2[p2_start:p2_end] + p1[p1_end:], p2[:p2_start] + p1[p1_start:p1_end] + p2[p2_end:]


def prm_hoist_mtn():
    def hoist_mtn(repr_):
        """ Implements the hoist mutation

        The hoist mutation selects a random subtree R from solution's
        representation and replaces it with a random subtree R' taken
        from itself, i.e., a random subtree R' is selected from R and
        replaces it in the representation (it is 'hoisted').

        Parameters
        ----------
        repr_ : list
            Parent's representation.

        Returns
        -------
        list
            The offspring obtained from replacing a randomly selected
            subtree in the parent by a random tree.
        """
        # Get a subtree (R)
        start, end = get_subtree(repr_)
        subtree = repr_[start:end]
        # Get a subtree of the subtree to hoist (R')
        sub_start, sub_end = get_subtree(subtree)
        hoist = subtree[sub_start:sub_end]
        # Returns the result as lists' concatenation
        return repr_[:start] + hoist + repr_[end:]
    return hoist_mtn


def prm_point_mtn(sspace, prob):
    """ Implements the point mutation

    This function is used to provide the point_mtn (inner function)
    with the necessary environment (the outer scope) - the solve
    space (𝑆), necessary to for the mutation function to access the
    set of terminals and functions.

    Parameters
    ----------
    sspace : dict
        The formal definition of the 𝑆. For GP-based algorithms, it
        contains the set of constants, functions and terminals used
        to perform point mutation.
    prob : float
        Probability of mutating one node in the representation.

    Returns
    -------
    point_mtn : function
        The function which implements the point mutation for GP.
    """
    def point_mtn(repr_):
        """ Implements the point mutation

        The point mutation randomly replaces some randomly selected
        nodes from the individual's representation. The terminals are
        replaced by other terminals and functions are replaced by other
        functions with the same arity.

        Parameters
        ----------
        repr_ : list
            Parent's representation.

        Returns
        -------
        list
            The offspring obtained from replacing a randomly selected
            subtree in the parent by a random tree.
        """
        # Creates a copy of parent's representation
        repr_copy = copy.deepcopy(repr_)
        # Performs point replacement
        for i, node in enumerate(repr_copy):
            if random.random() < prob:
                if isinstance(node, _Function):
                    # Finds a valid replacement with same arity
                    node_ = sspace["function_set"][random.randint(0, len(sspace["function_set"])-1)]
                    while node.arity != node_.arity:
                        node_ = sspace["function_set"][random.randint(0, len(sspace["function_set"]) - 1)]
                    # Performs the replacement, once a valid function was found
                    repr_copy[i] = node_
                else:
                    if random.random() < sspace["p_constants"]:
                        # repr_copy[i] = sspace["constant_set"][random.randint(0, len(sspace["constant_set"]) - 1)] not working
                        # torch.tensor(random.uniform(self.constant_set.min, self.constant_set.max), device=self.device) code from erc 
                        repr_copy[i] = torch.tensor(random.uniform(sspace['constant_set'].min, sspace['constant_set'].max), device=sspace['device']) # directly generate constant
                    else:
                        repr_copy[i] = random.randint(0, sspace["n_dims"] - 1)

        return repr_copy

    return point_mtn


def prm_subtree_mtn(initializer):
    """ Implements the the subtree mutation

    This function is used to provide the gs_xo (inner function) with
    the necessary environment (the outer scope) - the random trees'
    generator, required to perform the mutation itself.

    Parameters
    ----------
    initializer : function
        Parametrized initialization function to generate random trees.

    Returns
    -------
    subtree_mtn : function
        The function which implements the sub-tree mutation for GP.
    """
    def subtree_mtn(repr_):
        """ Implements the the subtree mutation

        The subtree mutation (a.k.a. standard GP's mutation) replaces a
        randomly selected subtree of the parent individual by a completely
        random tree.

        Parameters
        ----------
        repr_ : list
            Parent's representation.

        Returns
        -------
        list
            The offspring obtained from replacing a randomly selected
            subtree in the parent by a random tree.
        """
        # Generates a random tree
        random_tree = initializer()
        # Calls swap crossover to swap repr_ with random_tree
        return swap_xo(repr_, random_tree)[0]

    return subtree_mtn


def prm_gs_xo(initializer, device):
    """ Implements the geometric semantic crossover (GSC)

    This function is used to provide the gs_xo (inner function) with
    the necessary environment (the outer scope) - the random trees'
    generator and the processing device. The former is necessary to
    generate a random tree that is required to perform the crossover
    itself, whereas the latter is used to create a tensor that holds
    a single value (-1) and store it in the outer scope of gs_xo (this
    is done to avoid allocating it on the GPU at every function's call).

    Parameters
    ----------
    initializer : function
        Parametrized initialization function to generate random trees.
    device : str
        Processing device.

    Returns
    -------
    gs_xo : function
        A function which returns two offsprings after applying the GSC
        on the parents' representation.
    """
    c1 = torch.Tensor([1.0]).to(device)
    def gs_xo(p1, p2):
        """ Implements the geometric semantic crossover

        The GSO corresponds to the geometric crossover in the semantic
        space. This function stores individuals' representations (the
        trees) in memory.

        Parameters
        ----------
        p1 : list
            Representation of the first parent.
        p2 : list
            Representation of the second parent.

        Returns
        -------
        list, list
            Tuple of two lists, each representing for an offspring obtained
            from applying the GSC on parents' representation.
        """
        rt = [lf1] + initializer()
        # Performs GSC on trees and returns the result
        return [add2, mul2] + rt + p1 + [mul2, sub2, c1] + rt + p2, \
               [add2, mul2] + rt + p2 + [mul2, sub2, c1] + rt + p1

    return gs_xo


def prm_gs_mtn(initializer, ms):
    """ Implements the geometric semantic mutation (GSM)

    This function is used to provide the gs_mtn (inner function) with
    the necessary environment (the outer scope) - the random trees'
    generator and the mutation's step(s). The former is necessary to
    generate a random tree that is required to perform the mutation
    itself, whereas the latter is used to moderate random tree's effect
    on the parent tree.

    Parameters
    ----------
    initializer : float
        Parametrized initialization function to generate random trees.
    ms : torch.Tensor
        A 1D tensor of length m. If it is a single-valued tensor, then
        the mutation step equals ms; if it is a 1D tensor of length m,
        then the mutation step is selected from it at random, at each
        call of gs_mtn.

    Returns
    -------
    gs_mtn : function
        A function which implements the GSM.
    """
    def gs_mtn(repr_):
        """ Implements the geometric semantic mutation (GSM)

        The GSM corresponds to the ball mutation in the semantic space.
        This function stores individuals' representations (the trees)
        in memory.

        Parameters
        ----------
        repr_ : list
            Parent's representation.

        Returns
        -------
        list
            The offspring obtained from adding a random tree, which
            output is bounded in [-ms, ms].
        """
        ms_ = ms if len(ms) == 1 else ms[random.randint(0, len(ms) - 1)]
        return [add2] + repr_ + [mul2, ms_, tanh1] + initializer()

    return gs_mtn


def prm_efficient_gs_xo(X, initializer):
    """ Implements the an efficient variant of GSC

    This function is used to provide the efficient_gs_xo (inner
    function) the necessary environment (the outer scope) - the
    input data and the random trees' generator. The former is necessary
    to generate a random tree that is required to latter the crossover
    itself, whereas the former is used the latter is necessary to
    execute the aforementioned random tree and store its semantics
    (along with some other features).

    Moreover, this function creates a tensor that holds a single value
    (-1) and store it in the outer scope of gs_xo (this is done to
    avoid allocating it on the GPU at every function's call).

    Parameters
    ----------
    X : torch.tensor
        The input data.
    initializer : function
        Initialization function. Used to generate random trees.

    Returns
    -------
    efficient_gs_xo : function
        A function which returns offsprings' semantics (and some other
        important features), after applying the GSC on the parents'
        representation.
    """
    c1 = torch.tensor([1.0], device=X.device)

    def efficient_gs_xo(p1, p2):
        """ Implements the an efficient variant of GSC

        Implements an efficient variant of GSC that acts on solutions'
        semantics instead of trees' structure. That is, the trees are
        never stored in computers' memory, only one random tree is
        temporarily generated at each function call to allow the
        calculations to happen (its output and other features are
        stored).
         For more details, consult "A new implementation of geometric
        semantic GP and its application to problems in pharmacokinetics"
        by L. Vanneschi et at. (2013).

        Parameters
        ----------
        p1 : list
            Representation of the first parent.
        p2 : list
            Representation of the second parent.

        Returns
        -------
        torch.Tensor, torch.Tensor
            Resulting offsprings' semantics.
        list
            Random tree generated to perform the GSC.
        """
        # Creates a random tree (bounded in [0, 1])
        rt = [lf1] + initializer()
        # Executes the tree to obtain random tree's semantics on X
        rt_s = _execute_tree(rt, X)
        # Performs GSC on semantics and returns parent's semantics and the random tree
        return rt_s * p1 + (c1 - rt_s) * p2, rt_s * p2 + (c1 - rt_s) * p1, rt

    return efficient_gs_xo


def prm_efficient_gs_mtn(X, initializer, ms):
    """ Implements the an efficient variant of GSM

    This function is used to provide the efficient_gs_mtn (inner
    function) the necessary environment (the outer scope) - the
    input data, the random trees' generator and the mutation's step(s).

    Parameters
    ----------
    X : torch.tensor
        The input data.
    initializer : function
        Initialization function. Used to generate random trees.
    ms : torch.Tensor
        A 1D tensor of length m. If it is a single-valued tensor, then
        the mutation step equals ms; if it is a 1D tensor of length m,
        then the mutation step is selected from it at random, at each
        call of gs_mnt.

    Returns
    -------
    efficient_gs_mtn : function
        A function which implements the efficient GSM.
    """
    def efficient_gs_mtn(repr_):
        """ Implements the an efficient variant of GSM

        Implements an efficient variant of GSM that acts on solutions'
        semantics instead of trees' structure. That is, the trees are
        never stored in computers' memory, only one random tree is
        temporarily generated at each function call to allow the
        calculations to happen (its output and other features are
        stored).
         For more details, consult "A new implementation of geometric
        semantic GP and its application to problems in pharmacokinetics"
        by L. Vanneschi et at. (2013).

        Parameters
        ----------
        repr_ : list
            Parent's representation.

        Returns
        -------
        torch.Tensor
            The offspring's representation stored as semantics vector,
            obtained from adding a random tree bounded in [-ms, ms].
        list
            Random tree generated to perform the GSM.
        torch.Tensor
            The GSM's mutation step used to create the offspring.
        """
        # Chooses the mutation step
        ms_ = ms if len(ms) == 1 else ms[random.randint(0, len(ms) - 1)]
        # Creates a random tree bounded in [-1, 1]
        rt = [tanh1] + initializer()
        # Performs GSM and returns the semantics, the random tree and the mutation's step
        return repr_ + ms_ * _execute_tree(rt, X), rt, ms_

    return efficient_gs_mtn


# +++++++++++++++++++++++++++ Neuroevolution
def nn_xo(p1, p2):

    weights1, biases1 = p1
    weights2, biases2 = p2
    
    new_weights1, new_weights2 = [], []
    new_biases1, new_biases2 = [], []

    # crossover 
    for i in range(len(weights1)):

        # weights
        w1 = weights1[i].flatten()
        w2 = weights2[i].flatten()

        new_w1 = w1.clone()
        new_w2 = w2.clone()

        crossover_point_weight = random.randint(0, weights1[i].numel())
        new_w1[:crossover_point_weight] = w2[:crossover_point_weight]
        new_w2[:crossover_point_weight] = w1[:crossover_point_weight]

        new_weights1.append(new_w1.reshape(weights1[i].shape))
        new_weights2.append(new_w2.reshape(weights1[i].shape))

        # biases
        b1 = biases1[i].flatten()
        b2 = biases2[i].flatten()

        new_b1 = b1.clone()
        new_b2 = b2.clone()

        crossover_point_bias = random.randint(0, biases1[i].numel())
        new_b1[:crossover_point_bias] = b2[:crossover_point_bias]
        new_b2[:crossover_point_bias] = b1[:crossover_point_bias]

        new_biases1.append(new_b1.reshape(biases1[i].shape))
        new_biases2.append(new_b2.reshape(biases1[i].shape))

    p1 = [new_weights1, new_biases1]
    p2 = [new_weights2, new_biases2]
    
    return p1, p2


def prm_nn_mtn(ms, sspace):
    
    def nn_mtn(repr_):
        weights, biases = repr_
        
        # Mutate weight
        mutated_weights = [layer + ms * (2 * torch.rand(layer.shape, device=sspace['device']) - 1) for layer in weights]
        
        # Mutate biases
        mutated_biases = [bias + ms * (2 * torch.rand(bias.shape, device=sspace['device']) - 1) for bias in biases]

        return [mutated_weights, mutated_biases]
    
    return nn_mtn

