import sys
import torch

from gpolnel.utils.solution import Solution


class Problem:  # STATUS: read
    """ Defines an Optimization Problem (OP)

    "Formally, an Optimization Problem (OP) is a pair of objects (𝑆, 𝑓)
    where 𝑆 is the set of all possible solutions, also known as the
    solve-space, and 𝑓 : 𝑆 → 𝐼𝑅  is a mapping between 𝑆 and the set of
    real numbers 𝐼𝑅, commonly known as the fitness function."
        - CIFO's lecture notes, by prof. Leonardo Vanneschi (2015).

    "OP is either minimization or maximization problem and is specified
    by a set of problem instances."
        - Simulated Annealing and Boltzmann Machines, E. Aarts and
         J. Kost (1989).

    The code contained in this class constitutes a general definition
    of an OP, following the previous two statements. As such, an OP's 
    instance is characterized by the following attributes:
        1) the solve space 𝑆;
        3) the fitness function 𝑓;
        4) the flag representing the purpose of optimization - whether
        it is minimization (True) or maximization (False).

    Provided 𝑓 and 𝑆, it makes sense that the fitness evaluation of
    candidate solutions should be executed by an OP's instance.

    Note that 𝑆 is highly dependent on the OP's type, reason why,
    within this framework, it is defined as a dictionary of varied
    (problem-specific) content.

    Since this library is implemented for both single-point and
    population-based iterative solve algorithms (ISAs), every kind of
    OP is required to evaluate_pop and validate both individual solutions
    and their collective representations ('evaluate_pop').

    Note that, by definition, the fitness is assigned as torch.Tensor
    instead of float. This happens to avoid unnecessary burden when
    detaching objects from GPU (if processing on GPU).

    Attributes
    ----------
    sspace : dict
        𝑆.
    ffunction : function
        𝑓 : 𝑆 → 𝐼𝑅.
    min_ : bool
        A flag which defines the purpose of optimization. If it's value
        is True, then the OP is a minimization problem; otherwise it is
        a maximization problem.
    """

    def __init__(self, sspace, ffunction, min_=True):
        """ Objects' constructor

        The constructor takes a dictionary that explicitly defines
        the solve-space (𝑆) of an OP, a fitness function that maps
        the elements from 𝑆 to the space of real numbers and a flag
        that states the purpose of the optimization.

        Parameters
        ----------
        sspace : dict
            𝑆.
        ffunction  : function
            𝑓 : 𝑆 → 𝐼𝑅.
        min_: bool
            A flag which defines the purpose of optimization.
        """
        self.sspace = sspace
        self.ffunction = ffunction
        self.min_ = min_

    def _is_feasible_sol(self, repr_):
        """ Assesses solution's feasibility under 𝑆's constraints.

        Assesses solution's feasibility after constraints specified
        in 𝑆 (if any).

        Parameters
        ----------
        repr_ : object
            Representation of a candidate solution.

        Returns
        -------
        bool
            Representations's feasibility state.
        """
        pass

    def _is_feasible_pop(self, repr_):
        """ Assesses population's feasibility under 𝑆's constraints.

        Assesses population's feasibility after constraints specified
        in 𝑆 (if any). This method was particularly designed to include
        more efficient assessment procedure for a set of solutions.

        Parameters
        ----------
        repr_ : object
            Candidate solutions's collective representation.

        Returns
        -------
        torch.Tensor
            Representations' feasibility state.
        """
        pass

    def evaluate_sol(self, sol):
        """ Evaluates a candidate solution

        Given the logic embedded in this library and the fact that,
        formally, an instance of an OP is a pair of objects (𝑆, 𝑓),
        one of the main operations a problem instance (PI) performs
        is to evaluate_pop candidate solutions. The purpose of this method
        is to evaluate_pop one single solutions at a call. At the method's
        call, the individual solutions should be provided as instances
        of type Solution.

        Parameters
        ----------
        sol : Solution
            A candidate solution to be evaluated.
        """
        pass

    def evaluate_pop(self, pop):
        """  Evaluates a population of candidate solutions

        This method allows to carry a more efficient evaluation of a
        set of candidate solutions, at a single call. At the method's
        call, the set of candidate solutions should be encapsulated
        into a special object of type Population.

        Parameters
        ----------
        pop : Population
            The object which holds population's representation and
            other important attributes (e.g. fitness cases, validity
            states, etc.).
        """
        pass

    def _set_bad_fit_sol(self, sol, test=False, device="cpu"):
        """ Sets a 'very bad' fitness for a given solution

        This method assigns a 'very bad' fitness for a given
        candidate solution. It is usually used called by the
        methods which perform solutions' validation.

        Parameters
        ----------
        sol : Solution
            A candidate solution whose fitness has to be defined as a
            'very bad'.
        test : bool (default=False)
            A flag which defines which data partition to use when
            evaluating the solution.
        device : str (default="cpu")
            The processing device to allocate the fitness value on.
        """
        if self.min_:
            sol.fit = torch.tensor(sys.maxsize, device=device)
            if test:
                sol.test_fit = torch.tensor(sys.maxsize, device=device)
        else:
            sol.fit = torch.tensor(-sys.maxsize, device=device)
            if test:
                sol.test_fit = torch.tensor(-sys.maxsize, device=device)

    def _set_bad_fit_pop(self, pop, device="cpu"):
        """ Sets a 'very bad' fitness for a given population

        This method assigns a 'very bad' fitness for a given
        population. It is usually used called by the methods
        which perform solutions' validation.

        Parameters
        ----------
        pop : Population
            A population whose fitness has to be defined as a
            'very bad'.
        device : str (default="cpu")
            The processing device to allocate the fitness values on.
        """
        if self.min_:
            pop.fit = torch.ones(len(pop), device=device) * sys.maxsize
        else:
            pop.fit = torch.zeros(len(pop), device=device)
