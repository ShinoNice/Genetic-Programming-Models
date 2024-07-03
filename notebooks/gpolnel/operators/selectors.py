""" Selection operators select candidate solutions for breeding
The module `gpol.selectors` contains some popular selection
operators (selectors) used to select one or several (for DE) candidate
solutions based on their fitness (except when it comes to random
selection, also implemented here). Given the fact solutions' selection
is traditionally fitness-based, as such invariant to solutions'
representation, almost all the selectors can be applied with any kind
of OPs.
"""

import math
import random
import torch



def prm_tournament(pressure):
    """ Implements tournament selection algorithm

    This function is used to provide the tournament (inner function)
    with the necessary environment (the outer scope) - the selection's
    pressure.

    Note that tournament selection returns the index of the selected
    solution, not the representation. This function can be used for the
    GeneticAlgorithm or the GSGP instances.

    Parameters
    ----------
    pressure : float
        Selection pressure.

    Returns
    -------
    tournament : function
        A function which implements tournament selection algorithm
        with a pool calculated as 'int(len(population) * pressure)'.
    """
    def tournament(pop, min_):
        """ Implements tournament selection algorithm

        The tournament selection algorithm returns the most-fit
        solution from a pool of randomly selected solutions. The pool
        is calculated as 'int(len(population) * pressure)'. This
        function can be used for the GeneticAlgorithm or the GSGP
        instances.

        Parameters
        ----------
        pop : Population
            The pointer to the population to select individuals from.
        min_ : bool
            The purpose of optimization.

        Returns
        -------
        int
            The index of the most-fit solution from the random pool.
        """
        # Computes tournament pool size with respect to the population
        pool_size = math.ceil(len(pop) * pressure)
        # Gets random indices of the individuals
        indices = torch.randint(low=0, high=len(pop), size=(pool_size, ))
        # Returns the best individual in the pool
        return indices[torch.argmin(pop.fit[indices])] if min_ else indices[torch.argmax(pop.fit[indices])]

    return tournament


def prm_roulette_wheel():
    def roulette_wheel(pop, min_):
        """ Implements roulette wheel selection algorithm

        Generates and returns the index in [0, len(pop)[ range
        after fitness proportionate (a.k.a. roulette wheel)
        selection algorithm.

        Parameters
        ----------
        pop : Population
            The pointer to the population to select individuals from.
        min_ : bool
            The purpose of optimization. In this procedure, as selection
            is performed randomly, it exists only for to obey library's
            standards.

        Returns
        -------
        int
            The index of the solution after fitness proportionate selection.
        """
        prop_fit = pop.fit/pop.fit.sum()
        _, indices = torch.sort(prop_fit, descending=min_)
        cum_fit = torch.cumsum(prop_fit, dim=0)
        return indices[cum_fit > random.uniform(0, 1)][0]
    return roulette_wheel

def prm_rank_selection():
    def rank_selection(pop, min_):
        """ Implements rank selection algorithm

        Generates and returns the index in [0, len(pop)[ range. Parents'
        selection depends on the relative rank of the fitness and not the
        fitness itself. The higher the rank of a given parent, the higher
        its probability of being selected. It is recommended to use when
        the individuals in the population have very close fitness values.

        Parameters
        ----------
        pop : Population
            The pointer to the population to select individuals from.
        min_ : bool
            The purpose of optimization. In this procedure, as selection
            is performed randomly, it exists only for to obey library's
            standards.

        Returns
        -------
        int
            The index of the solution after fitness rank selection.
        """
        _, indices = torch.sort(pop.fit, descending=min_)
        indices_ = indices + 1
        indices_prop = indices_/indices_.sum()
        cum_indices = torch.cumsum(indices_prop, dim=0)
        sel = random.uniform(0, 1)
        return torch.flip(indices, (0, ))[cum_indices > sel][0] if min_ else indices[cum_indices > sel][0]
    return rank_selection

def prm_rnd_selection():
    def rnd_selection(pop, min_):
        """ Implements random selection algorithm

        Generates and returns random index in [0, len(pop)[ range.

        Parameters
        ----------
        pop : Population
            The pointer to the population to select individuals from.
        min_ : bool
            The purpose of optimization. In this procedure, as selection
            is performed randomly, it exists only for to obey library's
            standards.

        Returns
        -------
        int
            A random index in [0, len(pop)[ range.
        """
        return random.randint(0, len(pop)-1)
    return rnd_selection



def prm_double_tournament(pressure1, pressure2):

    def tournament1(pop, min_):
        """ Implements tournament selection algorithm

        The tournament selection algorithm returns the most-fit
        solution from a pool of randomly selected solutions. The pool
        is calculated as 'int(len(population) * pressure)'. This
        function can be used for the GeneticAlgorithm or the GSGP
        instances.

        Parameters
        ----------
        pop : Population
            The pointer to the population to select individuals from.
        min_ : bool
            The purpose of optimization.

        Returns
        -------
        int
            The index of the most-fit solution from the random pool.
        """
        # Computes tournament pool size with respect to the population
        pool_size = math.ceil(len(pop) * pressure1)
        # Gets random indices of the individuals
        indices = torch.randint(low=0, high=len(pop), size=(pool_size, ))
        # Returns the best individual in the pool
        return indices[torch.argmin(pop.fit[indices])] if min_ else indices[torch.argmax(pop.fit[indices])]
    
    
    def double_tournament(pop, min_criteria):
        """ Implements double tournament selection algorithm

        The double tournament selection algorithm returns the most-fit
        solution based on two sequential tournaments, one for each objective.
        The pool is calculated as 'int(len(population) * pressure)'. This
        function can be used for the GeneticAlgorithm or the GSGP
        instances.

        Parameters
        ----------
        pop : Population
            The pointer to the population to select individuals from.
        min_criteria : tuple of bool
            The purpose of optimization for each objective.

        Returns
        -------
        int
            The index of the most-fit solution from the sequential tournaments.
        """
        # First tournament based on the first objective (e.g., RMSE)
        pop2_indices = []
        for _ in range(math.ceil(len(pop) * pressure2)):
            selected_index = tournament1(pop, min_criteria)
            pop2_indices.append(selected_index)
        
        # Convert list to tensor
        pop2_indices = torch.tensor(pop2_indices, dtype=torch.int32)
        
        # Second tournament based on the second objective (e.g., tree size)
        pool_size = math.ceil(len(pop2_indices) * pressure2)
        indices = torch.randint(low=0, high=len(pop2_indices), size=(pool_size,))
        selected_indices = pop2_indices[indices]
        
        # Returns the best individual in the pool based on tree size
        final_winner_index = selected_indices[torch.argmin(pop.size[selected_indices])] if min_criteria else selected_indices[torch.argmax(pop.size[selected_indices])]

        return final_winner_index

    return double_tournament
