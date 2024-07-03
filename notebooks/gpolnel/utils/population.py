import copy
import sys
import random

import torch
from copy import deepcopy
from gpolnel.utils.tree import Tree
from gpolnel.utils.solution import Solution

class Population:
    pop_id = 0

    def __init__(self, repr_=None):
        """ Object's constructor.

        Parameters
        ----------
        repr_ : Object
            The solutions' representation in the population.
        """
        self._pop_id = Population.pop_id
        Population.pop_id += 1
        self.individuals = None
        self.repr_ = None
        if repr_ is not None:
            self._set_individuals(repr_)
            self._set_repr(repr_)
        self.valid = None
        self.fit = None


    def __len__(self):
        return len(self.repr_)

    def __getitem__(self, index):
        return self.repr_[index]

    def __str__(self):
        s = ''
        for i in range(len(self.individuals)):
            s += str(i)
            s += ')\t'
            s += str(self.individuals[i].repr_)
            s += ' (fit: '
            s += str(self.individuals[i].fit)
            s += ')\n'
        return s

    def _set_individuals(self, repr_):
        """Sets the population individuals whose repr_ are given by repr_ arg.

        Parameters
        ----------
        repr_ : list
        A list of repr_ attributes of population individuals.
        Returns
        -------
        """
        self.individuals = [Solution(r) for r in repr_]

    def _set_repr(self, repr_):
        """Sets the population representations.

        Parameters
        ----------
        repr_ : list
        A list of repr_ attributes of its individuals.
        Returns
        -------
        """
        self.repr_ = repr_

    def _get_copy(self):
        """ Makes a copy of the calling Population object.

        Returns
        -------
        pop : Population
            An object of type Population, copy of self.
        """
        pop_copy = Population()
        if type(self.repr_) is torch.Tensor:
            repr_ = self.repr_.clone()
        else:
            repr_ = self.repr_.copy()
        pop_copy._populate([Solution(r) for r in repr_])
        return pop_copy

    def _get_best_pop_index(self, min_, fit):
        """Returns the index of the best solution in the population.
            If there is a tie, returns a random amongst the best.

        Parameters
        ----------
        min_ : Boolean
            True if the fitnesses of individuals should be minimized,
            False if the fitnesses of individuals should be maximized.
        fit : tensor
            A tensor with the population fitnesses.

        Returns
        -------
        int
            The index of best candidate solution in the population.
        """
        # Finds the index of the best candidate-solution(s)
        best_fit = fit.min() if min_ else fit.max()
        best_sols_indexes = [int(x) for x in torch.nonzero(fit == best_fit)]
        # If there is not a tie, returns the index
        if len(best_sols_indexes) == 1:
            return best_sols_indexes[0]
        # If there is a tie, returns a random solution
        return torch.tensor(random.choice(best_sols_indexes))

    def populate(self, individuals):
        """Set the individuals of the population and all of its attributes related to the individuals.

        Parameters
        ----------
        individuals : list
            List of solutions.

        Returns
        -------
        """
        # Set Population solutions
        self.individuals = individuals.copy()
        # Set Population structural attributes
        self._set_repr([ind.repr_ for ind in individuals])
        # Set Population fitnesses attributes
        self.fit = torch.stack([ind.fit for ind in self.individuals], dim=0)
        self.valid = [ind.valid for ind in self.individuals]

    def replace_individual(self, index, individual):
        """Replaces the individual at position index with in the population.

        Parameters
        ----------
        index :
            The index of the individual to be replaced
        individual : Solution
            The new individuals to be added in the population

        Returns
        -------
        : void

        """
        new_individuals = deepcopy(self.individuals)
        new_individuals[index] = individual
        self.populate(new_individuals)

    def get_best_pop_index(self, min_, fit=None):
        """Encapsulates the method for getting the index of best solution in the population.

        Parameters
        ----------
        min_ : Boolean
            True if the fitnesses of individuals should be minimized,
            False if the fitnesses of individuals should be maximized.
        fit : tensor
            A tensor with the population fitnesses.
        Returns
        -------
        int
            The index of best candidate solution in the population.
        """
        if fit is None: fit = self.fit
        return self._get_best_pop_index(min_, fit)

    def get_best_pop(self, min_, fit=None):
        """Returns the best solution in the population.

        Parameters
        ----------
        min_ : Boolean
            True if the fitnesses of individuals should be minimized,
            False if the fitnesses of individuals should be maximized.
        fit : tensor
            A tensor with the population fitnesses.

        Returns
        -------
        Solution
            The best candidate solution in the population.
        """
        return self.individuals[self.get_best_pop_index(min_=min_, fit=fit)]._get_copy()

    def get_worst_pop_index(self, min_, fit=None):
        """Returns the index of the worst solution in the population, by calling the method to find the index of
        the best solution with the min_ inverted.

        Parameters
        ----------
        min_ : Boolean
            True if the fitnesses of individuals should be minimized,
            False if the fitnesses of individuals should be maximized.
        fit : tensor
            A tensor with the population fitnesses.

        Returns
        -------
        int
            The index of worst candidate solution in the population.
        """
        return self.get_best_pop_index(min_=(not min_), fit=fit)

    def get_worst_pop(self, min_, fit=None):
        """Returns the worst solution in the population, by calling the method to find the best solution with
        the min_ inverted.

        Parameters
        ----------
        min_ : Boolean
            True if the fitnesses of individuals should be minimized,
            False if the fitnesses of individuals should be maximized.
        fit : tensor
            A tensor with the population fitnesses.

        Returns
        -------
        Solution
            The worst candidate-solution in the population.
        """
        return self.get_best_pop(min_=(not min_), fit=fit)


class PopulationTree(Population):
    """ Implementation of a Tree Population class for GP.

    Attributes
    ----------
    _pop_id : int
        A unique identification of a population object.
    repr_ : Object
        The solutions' representation in the population.
    valid : torch.Tensor
        The solutions' validity state under the light of 𝑆.
    fit : torch.Tensor
        A tensor representing solutions' quality in 𝑆. It is assigned
        by a given problem instance (PI), using fitness function (𝑓).
    size : torch.Tensor
        A tensor representing solutions' sizes.
    depth : torch.Tensor
        A tensor representing solutions' depth.
    """

    def _set_individuals(self, repr_):
        """Sets the population individuals whose repr_ are given by repr_ arg.

        Parameters
        ----------
        repr_ : list
        A list of repr_ attributes of population individuals.

        Return
        ----------
        """
        self.individuals = [Tree(r) for r in repr_]

    def _set_repr(self, repr_):
        """Sets the population representations and all of its structural attributes.

        Parameters
        ----------
        repr_ : list
        A list of repr_ attributes of its individuals.
        Returns
        -------
        """
        self.repr_ = repr_
        self._set_size()
        self._set_complexity()
        self._set_depth()
        self._set_no()
        self._set_nao()
        self._set_naoc()
        self._set_phi()
        self._set_n_features()
        self._set_visitation_length()

    def _get_copy(self):
        """ Makes a copy of the calling PopulationTree object.

        Returns
        -------
        pop : PopulationTree
            An object of type PopulationTree, copy of self.
        """
        pop_copy = PopulationTree()
        pop_copy.populate(deepcopy(self.individuals))
        return pop_copy

    # Size
    def _get_size(self):
        return torch.tensor([ind.get_size() for ind in self.individuals])

    def get_size(self):
        return self.size

    def _set_size(self):
        self.size = self._get_size()

    # Complexity
    def _get_complexity(self):
        return torch.tensor([ind.get_complexity() for ind in self.individuals], dtype=torch.long)

    def get_complexity(self):
        return self.complexity

    def _set_complexity(self):
        self.complexity = self._get_complexity()

    # Depth
    def _get_depth(self):
        return torch.tensor([ind.get_depth() for ind in self.individuals])

    def get_depth(self):
        return self.depth

    def _set_depth(self):
        self.depth = self._get_depth()

    # Number of Operators
    def _get_no(self):
        return torch.tensor([ind.get_no() for ind in self.individuals])

    def get_no(self):
        return self.no

    def _set_no(self):
        self.no = self._get_no()

    # Number of Non-arithmetic Operators
    def _get_nao(self):
        return torch.tensor([ind.get_nao() for ind in self.individuals])

    def get_nao(self):
        return self.nao

    def _set_nao(self):
        self.nao = self._get_nao()

    # Number of Consecutive Non-arithmetic Operators
    def _get_naoc(self):
        return torch.tensor([ind.get_naoc() for ind in self.individuals])

    def get_naoc(self):
        return self.naoc

    def _set_naoc(self):
        self.naoc = self._get_naoc()

    # PHI
    def _get_phi(self):
        return torch.tensor([ind.get_phi() for ind in self.individuals])

    def get_phi(self):
        return self.phi

    def _set_phi(self):
        self.phi = self._get_phi()

    # Number of (unique) Features
    def _get_n_features(self):
        return torch.tensor([ind.get_n_features() for ind in self.individuals])

    def get_n_features(self):
        return self.n_features

    def _set_n_features(self):
        self.n_features = self._get_n_features()

    # Visitation Length
    def _get_visitation_length(self):
        return torch.tensor([ind.get_visitation_length() for ind in self.individuals])

    def get_visitation_length(self):
        return self.visitation_length

    def _set_visitation_length(self):
        self.visitation_length = self._get_visitation_length()

