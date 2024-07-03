import torch
import random

from gpolnel.utils.solution import Solution
from gpolnel.utils.population import Population
from gpolnel.algorithms.random_search import RandomSearch


class PopulationBased(RandomSearch):
    """Population-based ISA (PB-ISAs).

    Based on the number of candidate solutions they handle at each
    step, the optimization algorithms can be categorized into
    Single-Point (SP) and Population-Based (PB) approaches. The solve
    procedure in the SP algorithms is generally guided by the
    information provided by a single candidate solution from 𝑆,
    usually the best-so-far solution, that is gradually evolved in a
    well defined manner in hope to find the global optimum. The HC is
    an example of a SP algorithm as the solve is performed by
    exploring the neighborhood of the current best solution.
    Contrarily, the solve procedure in PB algorithms is generally
    guided by the information shared by a set of candidate solutions
    and the exploitation of its collective behavior of different ways.
    In abstract terms, one can say that PB algorithms share, at least,
    the following two features: an object representing the set of
    simultaneously exploited candidate solutions (i.e., the
    population), and a procedure to "move" them across 𝑆.

    An instance of a PB-ISA is characterized by the following features:
        1) a PI (i.e., what to solve/optimize);
        2) a function to initialize the solve at a given point of the
         solve space (𝑆);
        3) the best solution found by the PB-ISA;
        4) the number of simultaneously exploited solution (the
         population's size);
        6) a collection of candidate solutions - the population;
        7) a random state for random numbers generation;
        8) the processing device (CPU or GPU).

    Attributes
    ----------
    pi : Problem (inherited from RandomSearch)
        An instance of OP.
    pop_size : int
        The population's size.
    best_sol : Solution (inherited from RandomSearch)
        The best solution found.
    pop_size : int
        Population's size.
    pop : Population
        Object of type Population which holds population's collective
        representation, feasibility states and fitness values.
    initializer : function (inherited from RandomSearch)
        The initialization procedure.
    mutator : function
        A function to move solutions across 𝑆.
    seed : int (inherited from RandomSearch)
        The seed for random numbers generators.
    device : str (inherited from RandomSearch)
        Specification of the processing device.
    """

    def __init__(self, pi, initializer, mutator, pop_size=100, seed=0, device="cpu"):
        """Objects' constructor.

        Parameters
        ----------
        pi : Problem
            Instance of an optimization problem (PI).
        initializer : function
            The initialization procedure.
        mutator : function
            A function to move solutions across the solve space.
        pop_size : int (default=100)
            Population's size.
        seed : int str (default=0)
            The seed for random numbers generators.
        device : str (default="cpu")
            Specification of the processing device.
        """
        RandomSearch.__init__(self, pi, initializer, seed, device)
        self.mutator = mutator
        self.pop_size = pop_size
        # Initializes the population's object (None by default)
        self.pop = None

    def _initialize(self, start_at=None, tree=False):
        """Initializes the solve at a given point in 𝑆.

        Note that the user-specified start_at is assumed to be feasible
        under 𝑆's constraints.

        Parameters
        ----------
        start_at : object (default=None)
            A user-specified initial starting point in 𝑆.
        """
        # Creates as empty list for the population's representation
        pop_size, pop_repr = self.pop_size, []
        # Recomputes populations' size and extends the list with user-specified initial seed, is such exists
        if start_at is not None:
            pop_size -= len(start_at)
            pop_repr.extend(start_at)
        # Initializes pop_size individuals by means of 'initializer' function
        pop_repr.extend(self.initializer(sspace=self.pi.sspace, n_sols=pop_size))
        # Stacks population's representation, if candidate solutions are objects of type torch.tensor
        if isinstance(pop_repr[0], torch.Tensor):
            pop_repr = torch.stack(pop_repr)
        # Set pop and best solution
        self._set_pop(pop_repr=pop_repr, tree=tree)

    def _set_pop(self, pop_repr):
        """Encapsulates the set method of the population attribute of PopulationBased algorithm.

        Parameters
        ----------
        pop_repr : list
            A list of solutions' representation.

        Returns
        -------
        None
        """
        # Creates an object of type 'Population', given the initial representation
        self.pop = Population(pop_repr)
        # Evaluates population on the problem instance
        self.pi.evaluate_pop(self.pop)
        # Sets the best solution
        self._set_best_sol()

    def _set_best_sol(self):
        """Encapsulates the set method of the best_sol attribute of PopulationBased algorithm.

        Parameters
        ----------
            self
        Returns
        -------
            None
        """
        self.best_sol = self.pop.get_best_pop(min_=self.pi.min_)

    def _create_log_event(self, it, timing, pop, log, log_xp='GPOLNEL'):
        """Implements a standardized log-event

        Creates a log-event for the underlying best-so-far solution.

        Parameters
        ----------
        it : int
            Iteration's number.
        timing : float
            Iterations's running time in seconds.
        pop : Population
            An object of type Population which represents the current
            population (at the end of iteration i).
        log : int, optional
            An integer that controls the verbosity of the log file. The
            following nomenclature is applied in this class:
                - log = 0: do not write any log data.
                - log = 1: writes the current iteration, its timing, and
                    elite's length and fitness.
                - log = 2: also, writes population's average and
                    standard deviation (in terms of fitness).
                - log = 3: also, writes elite's representation
        log_xp : string
            A reference for the experiment being performed. Deafult value is set generically as 'GPOL'.
        """
        # Appends the current iteration, its timing, and elite's length
        log_event = [log_xp, self.seed, it, timing, len(self.best_sol), self.best_sol.fit]
        if hasattr(self.best_sol, 'test_fit'):
            log_event.append(self.best_sol.test_fit)
        if log >= 2:
            log_event.extend([pop.fit_avg, pop.fit_std])
        # Also, writes elite's representation
        if log >= 3:
            log_event.append(self.best_sol.repr_)
        # Return log event
        return log_event

    def _verbose_reporter(self, it, timing, pop, verbose=0):
        """Reports the progress of the solve on the console.

        Parameters
        ----------
        it : int
            Integer that represents the current iteration.
        timing : float
            Floating-point that represents the processing time of the
            current iteration.
        pop : Population
            An object of type Population that represents the current
            population/swarm.
        verbose : int, optional (default=0)
            An integer that controls the verbosity of the solve. The
            following nomenclature is applied in this class:
                - verbose = 0: do not print anything (controlled from
                 the solve method).
                - verbose = 1: prints current iteration, its timing,
                    and elite's length and fitness (default).
                - verbose = 2: also prints population's average
                    and standard deviation (in terms of fitness).
        """
        if it == -1:
            if hasattr(self.best_sol, "test_fit"):
                print('-' * 103)
                print(' ' * 11 + '|{:^53}  |{:^34}|'.format("Best solution", "Population"))
                print('-' * 103)
                if len(self.best_sol.fit.shape) == 0:
                    line_format = '{:<10} {:<1} {:<8} {:<16} {:<16} {:>10} {:<1} {:<16} {:>16}'
                else:
                    line_format = '{:<10} {:<1} {:<8} {:<16} {:<16} {:>10} {:<1} {:<16} {:>16}'
                print(line_format.format('Generation', "|", 'Length', 'Fitness', 'Test Fitness',
                                         "Timing", "|", "AVG Fitness", "STD Fitness"))
                print('-' * 103)
            else:
                print('-' * 86)
                print(' ' * 11 + '|{:^36}  |{:^34}|'.format("Best solution", "Population"))
                print('-' * 86)
                line_format = '{:<10} {:<1} {:<8} {:<16} {:>10} {:<1} {:<16} {:>16}'
                print(line_format.format('Generation', "|", 'Length', 'Fitness', "Timing",
                                         "|", "AVG Fitness", "STD Fitness"))
        else:
            if hasattr(self.best_sol, "test_fit"):
                if len(self.best_sol.fit.shape) == 0:
                    line_format = '{:<10d} {:<1} {:<8d} {:<16g} {:<16g} {:>10.3f} {:<1} {:<16g} {:>16g}'
                else:
                    line_format = '{:<10d} {:<1} {:<8d} {} {} {:>10.3f} {:<1} {:<16g} {:>16g}'
                length = len(self.best_sol)
                if verbose >= 2:
                    avgfit, stdfit = pop.fit_avg, pop.fit_std
                else:
                    avgfit, stdfit = -1.0, -1.0
                print(line_format.format(it, "|", length, self.best_sol.fit.tolist(), self.best_sol.test_fit.tolist(), timing, "|",
                                         avgfit, stdfit))
            else:
                if len(self.best_sol.fit.shape) == 0:
                    line_format = '{:<10d} {:<1} {:<8d} {:<16g} {:>10.3f} {:<1} {:<16g} {:>16g}'
                else:
                    line_format = '{:<10d} {:<1} {:<8d} {} {:>10.3f} {:<1} {:<16g} {:>16g}'
                # If the the type of OP is of knapsack's family, then sum the vector, otherwise the length
                length = int(self.best_sol.repr_.sum().item()) if isinstance(self.pi, Knapsack01) else len(self.best_sol)
                if verbose >= 2:
                    avgfit, stdfit = pop.fit_avg, pop.fit_std
                else:
                    avgfit, stdfit = -1.0, -1.0

                print(line_format.format(it, "|", length, self.best_sol.fit.tolist(), timing, "|", avgfit, stdfit))

    def solve(self, n_iter=20, tol=None, n_iter_tol=5, start_at=None, test_elite=False, verbose=0, log=0):
        """Defines the solve procedure of a PB-ISA.

        This method implements the pseudo-code of a given PB-ISA.

        Parameters
        ----------
        n_iter : int (default=20)
            The number of iterations.
        tol : float (default=None)
            Minimum required fitness improvement for n_iter_tol
            consecutive iterations to continue the solve. When best
            solution's fitness is not improving by at least tol for
            n_iter_tol consecutive iterations, the solve will be
            automatically interrupted.
        n_iter_tol : int (default=5)
            Maximum number of iterations to continue the solve while
            not meeting the tol improvement.
        start_at : object (default=None)
            The initial starting point in 𝑆 (it is is assumed to be
            feasible under 𝑆's constraints, if any).
        test_elite : bool (default=False)
            Indicates whether assess the best-so-far solution on the
            test partition (this regards SML-based OPs).
        verbose : int, optional (default=0)
            An integer that controls the verbosity of the solve. The
            following nomenclature is applied in this class:
                - verbose = 0: do not print anything.
                - verbose = 1: prints current iteration, its timing,
                    and elite's length and fitness.
                - verbose = 2: also prints population's average
                    and standard deviation (in terms of fitness).
        log : int, optional (default=0)
            An integer that controls the verbosity of the log file. The
            following nomenclature is applied in this class:
                - log = 0: do not write any log data;
                - log = 1: writes the current iteration, its timing, and
                    elite's length and fitness;
                - log = 2: also, writes population's average and
                    standard deviation (in terms of fitness);
                - log = 3: also, writes elite's representation.
        """
        pass

    def elite_replacement(self, offs_pop, min_=None, best_parent=None):
        """Single-individual elite replacement.

        If the best_parent is better than the best individual in offs_pop, replaces the worst individual
        in offs_pop by the best_parent.

        Parameters
        ----------
        off_pop : Population
            Offspring population.
        min_ : Boolean
            If True, the problem is a minimization problem; if False, it is a maximization problem.
        best_parent : Solution
            Best individual of current (parents) population.

        Returns
        -------
        offs_pop : Population
            The offspring population after application of elitism.
        """
        if min_ is None: min_ = self.pi.min_
        if best_parent is None: best_parent = self.pop.get_best_pop(min_=min_)
        best_offs = offs_pop.get_best_pop(min_=min_)
        if best_parent.is_better(best_offs, min_):
            index = offs_pop.get_worst_pop_index(min_=min_)
            offs_pop.replace_individual(index=index, individual=best_parent)
        return offs_pop

    @staticmethod
    def _get_phen_div(pop):
        """Returns the phenotypic diversity of a population.

        Parameters
        ----------
        pop : Population
            An object of type population, after evaluation.

        Returns
        -------
        torch.tensor
            The standard deviation of population's fitness values.
        """
        # dim=0 is set to handle with single and multi-objective problems
        return torch.std(pop.fit, dim=0)
