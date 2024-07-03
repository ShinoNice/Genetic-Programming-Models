import os
import time
import random
import pickle
import logging
import torch
import pandas as pd
from copy import deepcopy

from gpolnel.algorithms.population_based import PopulationBased
from gpolnel.utils.population import Population, PopulationTree
from gpolnel.utils.tree import Tree
from gpolnel.utils.inductive_programming import _execute_tree, _get_tree_depth


class GeneticAlgorithm(PopulationBased):
    """Implements Genetic Algorithm (GA).

    Genetic Algorithm (GA) is a meta-heuristic introduced by John
    Holland, strongly inspired by Darwin's Theory of Evolution.
    Conceptually, the algorithm starts with a random-like population of
    candidate-solutions (called chromosomes). Then, by resembling the
    principles of natural selection and the genetically inspired
    variation operators, such as crossover and mutation, the algorithm
    breeds a population of next-generation candidate-solutions (called
    the offspring population), which replaces the previous population
    (a.k.a. the population of parents). The procedure is iterated until
    reaching some stopping criteria, like a predefined number of
    iterations (also called generations).

    An instance of GA can be characterized by the following features:
        1) an instance of an OP, i.e., what to solve/optimize;
        2) a function to initialize the solve at a given point in 𝑆;
        3) a function to select candidate solutions for variation phase;
        4) a function to mutate candidate solutions;
        5) the probability of applying mutation;
        6) a function to crossover two solutions (the parents);
        7) the probability of applying crossover;
        8) the population's size;
        9) the best solution found by the PB-ISA;
        10) a collection of candidate solutions - the population;
        11) a random state for random numbers generation;
        12) the processing device (CPU or GPU).

    Attributes
    ----------
    pi : Problem (inherited from PopulationBased)
        An instance of OP.
    best_sol : Solution (inherited from PopulationBased))
        The best solution found.
    pop_size : int (inherited from PopulationBased)
        The population's size.
    pop : Population (inherited from PopulationBased)
        Object of type Population which holds population's collective
        representation, validity state and fitness values.
    initializer : function (inherited from PopulationBased))
        The initialization procedure.
    selector : function
        The selection procedure.
    mutator : function (inherited from PopulationBased)
        The mutation procedure.
    p_m : float
        The probability of applying mutation.
    crossover : function
        The crossover procedure.
    p_c : float
        The probability of applying crossover.
    elitism : bool
        A flag which activates elitism during the evolutionary process.
    reproduction : bool
        A flag which states if reproduction should happen (reproduction
        is True), when the crossover is not applied. If reproduction is
        False, then either crossover or mutation will be applied.
    seed : int (inherited from PopulationBased)
        The seed for random numbers generators.
    device : str (inherited from PopulationBased)
        Specification of the processing device.
    """
    __name__ = "GeneticAlgorithm"

    def __init__(self, pi, initializer, selector, mutator, crossover, p_m=0.2, p_c=0.8, pop_size=100, elitism=True,
                 reproduction=False, seed=0, device="cpu"):
        """ Objects' constructor

        Following the main purpose of a PB-ISA, the constructor takes a
        problem instance (PI) to solve, the population's size and an
        initialization procedure. Moreover it receives the mutation and
        the crossover functions along with the respective probabilities.
        The constructor also takes two boolean values indicating whether
        elitism and reproduction should be applied. Finally, it takes
        some technical parameters like the random seed and the processing
        device.

        Parameters
        ----------
        pi : Problem
            Instance of an optimization problem (PI).
        initializer : function
            The initialization procedure.
        selector : function
            The selection procedure.
        mutator : function
            A function to move solutions across the solve space.
        crossover : function
            The crossover function.
        p_m : float (default=0.2)
            Probability of applying mutation.
        p_c : float (default=0.8)
            Probability of applying crossover.
        pop_size : int (default=100)
            Population's size.
        elitism : bool (default=True)
            A flag which activates elitism during the evolutionary process.
        reproduction : bool (default=False)
            A flag which states if reproduction should happen (reproduction
            is True), when the crossover is not applied.
        seed : int (default=0)
            The seed for random numbers generators.
        device : str (default="cpu")
            Specification of the processing device.
        """
        PopulationBased.__init__(self, pi, initializer, mutator, pop_size, seed, device)  # at this point, it has the pop attribute, but it is None
        self.selector = selector
        self.p_m = p_m
        self.crossover = crossover
        self.p_c = p_c
        self.elitism = elitism
        self.reproduction = reproduction

    def _set_pop(self, pop_repr, tree=True):
        """Encapsulates the set method of the population attribute of GeneticAlgorithm algorithm.

        Parameters
        ----------
        pop_repr : list
            A list of solutions' representation.

        Returns
        -------
        None
        """
        # Creates an object of type 'PopulationTree', given the initial representation
        if tree:
            self.pop = PopulationTree(pop_repr)
        else:
            self.pop = Population(pop_repr)
        # Evaluates population on the problem instance
        self.pi.evaluate_pop(self.pop)
        # Gets the best in the initial population
        self._set_best_sol()

    def solve(self, n_iter=20, tol=None, n_iter_tol=5, start_at=None, test_elite=False, verbose=0, log=0, log_path='./log/', log_xp='GPOL'):
        """Defines the solve procedure of a GA.

        This method implements the following pseudo-code:
            1) Create a random initial population of size n (𝑃);
            2) Repeat until satisfying some termination condition,
             typically the number of generations:
                1) Calculate fitness ∀ individual in 𝑃;
                2) Create an empty population 𝑃’, the population of
                 offsprings;
                3) Repeat until 𝑃’ contains 𝑛 individuals:
                    1) Chose the main genetic operator – crossover,
                     with probability p_c or reproduction with
                     probability (1 − p_c);
                    2) Select two individuals, the parents, by means
                     of a selection algorithm;
                    3) Apply operator selected in 2) 3) 1) to the
                     individuals selected in 2) 3) 2);
                    4) Apply mutation on the resulting offspring with
                     probability p_m;
                    5) Insert individuals from 2) 3) 4) into 𝑃’;
                4) Replace 𝑃 with 𝑃’;
            3) Return the best individual in 𝑃 (the elite).

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
        # Initialization time
        start = time.time()
        # Set log path
        if log > 0:
            logging.basicConfig(filename=log_path, filemode='w', level=logging.INFO)

        # 1)
        tree_based = 'function_set' in self.pi.sspace
        self._initialize(start_at=start_at, tree=tree_based)

        # Optionally, evaluates the elite on the test partition
        if test_elite:
            # Workaround proposed by L. Rosenfeld to maintain the dataloader's seed when test_elite changes
            state = torch.get_rng_state()
            self.pi.evaluate_sol(self.best_sol, train=False, test=True)
            torch.set_rng_state(state)

        # Optionally, computes population's AVG and STD (in terms of fitness)
        if log >= 2 or verbose >= 2:
            self.pop.fit_avg = self.pop.fit.mean().item()
            self.pop.fit_std = self.pop.fit.std().item()

        # Optionally, reports initializations' summary results on the console
        if verbose > 0:
            # Creates reporter's header reports the result of initialization
            self._verbose_reporter(-1, 0, None, 1)
            self._verbose_reporter(0, time.time() - start, self.pop, verbose)

        # Optionally, writes the log-data
        if log > 0:
            log_event = [self.pi.__name__, self.__name__, self.seed]
            logger = logging.getLogger(','.join(list(map(str, log_event))))
            log_event = self._create_log_event(0, 0, self.pop, log, log_xp)
            logger.info(','.join(list(map(str, log_event))))

        # Optionally, creates local variables to account for the tolerance-based stopping criteria
        if tol:
            n_iter_bare, last_fit = 0, self.best_sol.fit.clone()

        # 2)
        for it in range(1, n_iter + 1, 1):
            # 2) 2)
            offs_pop, start = [], time.time()

            # 2) 3)
            pop_size = self.pop_size - self.pop_size % 2
            while len(offs_pop) < pop_size:
                # 2) 3) 2)
                p1_idx = p2_idx = self.selector(self.pop, self.pi.min_)
                # Avoids selecting the same parent twice
                while p1_idx == p2_idx:
                    p2_idx = self.selector(self.pop, self.pi.min_)

                if not self.reproduction:  # performs GP-like variation
                    if random.uniform(0, 1) < self.p_c:
                        # 2) 3) 3)
                        offs1, offs2 = self.crossover(self.pop[p1_idx], self.pop[p2_idx])
                    else:
                        # 2) 3) 4)
                        offs1 = self.mutator(self.pop[p1_idx])
                        offs2 = self.mutator(self.pop[p2_idx])
                else:  # performs GA-like variation
                    offs1, offs2 = self.pop[p1_idx], self.pop[p2_idx]
                    if random.uniform(0, 1) < self.p_c:
                        # 2) 3) 3)
                        offs1, offs2 = self.crossover(self.pop[p1_idx], self.pop[p2_idx])
                    if random.uniform(0, 1) < self.p_m:
                        # 2) 3) 4)
                        offs1 = self.mutator(self.pop[p1_idx])
                        offs2 = self.mutator(self.pop[p2_idx])

                # 2) 3) 5)
                offs_pop.extend([offs1, offs2])

            # Adds one more individual, if the population size is odd
            if pop_size < self.pop_size:
                offs_pop.append(self.mutator(self.pop[self.selector(self.pop, self.pi.min_)]))

            # If batch training, appends the elite to evaluate_pop it on the same batch(es) as the offspring population
            if self._batch_training:
                offs_pop.append(self.best_sol.repr_)

            # If solutions are objects of type torch.tensor, stacks their representations in the same tensor
            if isinstance(offs_pop[0], torch.Tensor):
                offs_pop = torch.stack(offs_pop)

            # 2) 1)
            offs_pop = globals()[self.pop.__class__.__name__](offs_pop)
            self.pi.evaluate_pop(offs_pop)


            # Overrides elites's information, if it was re-evaluated, and removes it from 'offsprings'
            if self._batch_training:
                self.best_sol.valid = offs_pop.valid[-1]
                self.best_sol.fit = offs_pop.fit[-1]
                # Removes the elite from the object 'offsprings'
                offs_pop.repr_ = offs_pop.repr_[0:-1]
                offs_pop.valid = offs_pop.valid[0: -1]
                offs_pop.fit = offs_pop.fit[0: -1]

            # Performs population's replacement
            if self.elitism:
                offs_pop = self.elite_replacement(offs_pop)
            self.pop = offs_pop
            self._set_best_sol()

            # Optionally, evaluates the elite on the test partition
            if test_elite:
                # Workaround proposed by L. Rosenfeld to maintain the dataloader's seed when test_elite changes
                state = torch.get_rng_state()
                self.pi.evaluate_sol(self.best_sol, train=False, test=True)
                torch.set_rng_state(state)

            # Optionally, computes iteration's timing
            if (log + verbose) > 0:
                timing = time.time() - start

            # Optionally, computes population's AVG and STD (in terms of fitness)
            if log >= 2 or verbose >= 2:
                self.pop.fit_avg = self.pop.fit.mean().item()
                self.pop.fit_std = self.pop.fit.std().item()

            # Optionally, writes the log-data on the file
            if log > 0:
                log_event = self._create_log_event(it, timing, self.pop, log, log_xp)
                logger.info(','.join(list(map(str, log_event))))

            # Optionally, reports the progress on the console
            if verbose > 0:
                self._verbose_reporter(it, timing, self.pop, verbose)

            # Optionally, verifies the tolerance-based stopping criteria
            if tol:
                n_iter_bare, last_fit = self._check_tol(last_fit, tol, n_iter_bare)
                if n_iter_bare == n_iter_tol:
                    break


class GSGP(GeneticAlgorithm):
    """Re-implements Genetic Algorithm (GA) for GSGP.

    Given the growing importance of the Geometric Semantic Operators
    (GSOs), proposed by Moraglio et al., we decided to include them in
    our library, following the efficient implementation proposed by
    Castelli et al. More specifically, we implemented Geometric
    Semantic Genetic Programming (GSGP) through a specialized class
    called GSGP, subclass of the GeneticAlgorithm, that encapsulates
    the efficient implementation of GSOs.

    An instance of GSGP can be characterized by the following features:
        1) an instance of an OP, i.e., what to solve/optimize;
        2) the population's size;
        3) a function to initialize the solve at a given point in 𝑆;
        4) a function to select candidate solutions for variation phase;
        5) a function to mutate candidate solutions;
        6) the probability of applying mutation;
        7) a function to crossover two solutions (the parents);
        8) the probability of applying crossover;
        9) whether to apply elite replacement or not;
        10) whether to apply reproduction or not;
        11) the best solution found by the PB-ISA;
        12) a collection of candidate solutions - the population;
        11) a random state for random numbers generation;
        12) the processing device (CPU or GPU).

    Attributes
    ----------
    pi : Problem (inherited from GeneticAlgorithm)
        An instance of OP.
    best_sol : Solution (inherited from GeneticAlgorithm))
        The best solution found.
    pop_size : int (inherited from GeneticAlgorithm)
        The population's size.
    pop : Population (inherited from GeneticAlgorithm)
        Object of type Population which holds population's collective
        representation, validity state and fitness values.
    initializer : function (inherited from GeneticAlgorithm)
        The initialization procedure.
    selector : function (inherited from GeneticAlgorithm)
        The selection procedure.
    mutator : function (inherited)
        The mutation procedure.
    crossover : function (inherited from GeneticAlgorithm)
        The crossover procedure.
    p_m : float (inherited from GeneticAlgorithm)
        The probability of applying mutation.
    p_c : float (inherited from GeneticAlgorithm)
        The probability of applying crossover.
    elitism : bool (inherited from GeneticAlgorithm)
        A flag which activates elitism during the evolutionary process.
    reproduction : bool (inherited from GeneticAlgorithm)
        A flag which states if reproduction should happen (reproduction
        is True), when the crossover is not applied. If reproduction is
        False, then either crossover or mutation will be applied.
        path_init_pop : str
        Connection string towards initial population's repository.
    path_init_pop : str
        Connection string towards initial trees' repository.
    path_rts : str
        Connection string towards random trees' repository.
    history : dict
        Dictionary which stores the history of operations applied on
        each offspring. In abstract terms, it stores 1 level family
        tree of a given offspring. More specifically, history stores
        as a key the offspring's ID, as a value a dictionary with the
        following structure:
            - "Iter": iteration's number;
            - "Operator": the variation operator that was applied on
             a given offspring;
            - "T1": the ID of the first parent;
            - "T2": the ID of the second parent (if GSC was applied);
            - "Tr": the ID of a random tree generated (assumes only
             one random tree is necessary to apply an operator);
            - "ms": mutation's step (if GSM was applied);
            - "Fitness": offspring's training fitness;
    pop_ids : lst
        IDs of the current population (the population of parents).
    seed : int (inherited from GeneticAlgorithm)
        The seed for random numbers generators.
    device : str (inherited from GeneticAlgorithm)
        Specification of the processing device.

    References
    ----------
    Alberto Moraglio, Krzysztof Krawiec and Colin G. Johnson.
        "Geometric Semantic Genetic Programming". Parallel Problem
        Solving from Nature - PPSN XII. 2012
    Mauro Castelli, Sara Silva and Leonardo Vanneschi
        "A C++ framework for geometric semantic genetic programming".
        Genetic Programming and Evolvable Machines. 2015.
    """
    __name__ = "GSGP"

    def __init__(self, pi, initializer, selector, mutator, crossover, p_m=0.95, p_c=0.05, pop_size=100, elitism=True,
                 reproduction=False, path_init_pop=None, path_rts=None, seed=0, device='cpu'):

        """ Objects' constructor

        Following the main purpose of a PB-ISA, the constructor takes a
        problem instance (PI) to solve, the population's size and an
        initialization procedure. Moreover it receives the mutation and
        the crossover functions along with the respective probabilities.
        The constructor also takes two boolean values indicating whether
        elitism and reproduction should be applied. Finally, it takes
        some technical parameters like the random seed and the processing
        device.

        Parameters
        ----------
        pi : Problem
            Optimization problem's instance (PI).
        path_init_pop : str
            Connection string towards initial trees' repository.
        path_rts : str
            Connection string towards random trees' repository.
        initializer : function
            The initialization procedure.
        selector : function
            Selection procedure.
        mutator : function
            A function to move solutions across the solve space.
        crossover : function
            Crossover.
        p_m : float (default=0.05)
            Probability of applying mutation.
        p_c : float (default=0.95)
            Probability of applying crossover.
        pop_size : int (default=100)
            Population's size.
        elitism : bool (default=True)
            A flag which activates elitism during the evolutionary process.
        reproduction : bool (default=False)
            A flag which states if reproduction should happen (reproduction
            is True), when the crossover is not applied.
        seed : int (default=0)
            The seed for random numbers generators.
        device : str (default="cpu")
            Specification of the processing device.
        """
        GeneticAlgorithm.__init__(self, pi, initializer, selector, mutator, crossover, p_m, p_c, pop_size, elitism,
                                  reproduction, seed, device)
        if path_init_pop and path_rts:
            self.reconstruct = True
            self.path_init_pop = path_init_pop
            self.path_rts = path_rts
            self.history = {}
            self.pop_ids = []
        else:
            self.reconstruct = False

    def _set_best_sol(self):
        """Encapsulates the set method of the best_sol attribute of PopulationBased algorithm.

        Parameters
        ----------
            self
        Returns
        -------
            None
        """
        best_idx = self.pop._get_best_pop_index(min_=self.pi.min_, fit=self.pop.fit)
        self.best_sol = self.pop.individuals[best_idx]
        self.best_sol.size = self.pop.size[best_idx]
        self.best_sol.depth = self.pop.depth[best_idx]

    def _initialize(self, start_at=None):
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
        # Recomputes populations' size and extends the list with user-specified initial seed (if any)
        if start_at is not None:
            pop_size -= len(start_at)
            pop_repr.extend(start_at)
        # Initializes pop_size individuals by means of 'initializer' function
        pop_repr.extend(self.initializer(sspace=self.pi.sspace, n_sols=pop_size))
        #
        if self.reconstruct:
            # Stores populations' representation as individual trees (each tree is stored as a .pickle)
            for i, tree in enumerate(pop_repr):
                # Appends representations' ID to the list
                self.pop_ids.append(str(self.seed) + "_" + str(0) + "_" + str(i))  # iteration 0
                # Writes the pickles
                with open(os.path.join(self.path_init_pop, self.pop_ids[-1] + ".pickle"), "wb") as handle:
                    pickle.dump(tree, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # Extracts trees' structural metrics
        pop = Population(pop_repr)
        pop.size = torch.tensor([Tree(r).get_size() for r in pop.repr_])
        pop.depth = torch.tensor([Tree(r).get_depth() for r in pop.repr_])
        # Sets semantics for the entire dataset (From this point on, pop_repr stores trees' semantics only)
        pop_repr = [_execute_tree(repr_, self.pi.X) for repr_ in pop_repr]
        # Expands the semantics vectors when individuals are only constants
        pop_repr = [torch.cat(len(self.pi.X)*[repr_[None]]) if len(repr_.shape) == 0 else repr_ for repr_ in pop_repr]
        # Sets semantic representation as torch tensor for population
        pop.repr_ = torch.stack(pop_repr)
        [pop.individuals[i].__setattr__('repr_', f) for i, f in enumerate(pop.repr_)]
        # Sets GSGP population
        self.pop = deepcopy(pop)
        # Evaluates the population on a given problem instance for train and test partitions
        self.pi.evaluate_pop(self.pop)
        # self.pi.evaluate_pop(self.pop, test=True)
        # Sets the elite
        self._set_best_sol()

    def solve(self, n_iter=20, tol=None, n_iter_tol=5, start_at=None, test_elite=False, verbose=0,
              log=0, log_path='./log/gsgp.log', log_xp='GSGP-GPOLNEL'):
        """Defines the solve procedure of a GSGP.

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
        log_path : string
            Path to the log file.
        log_xp : string
            Experiment name for log event.
        """
        # Optionally, tracks initialization's timing for console's output
        start = time.time()
        # Set log path
        if log > 0:
            logging.basicConfig(filename=log_path, filemode='w', level=logging.INFO)

        # 1)
        self._initialize(start_at=start_at)

        # Optionally, evaluates the elite on the test partition
        if test_elite:
            self.pi.evaluate_sol(self.best_sol, test=True)

        # Optionally, computes population's AVG and STD (in terms of fitness)
        if log >= 2 or verbose >= 2:
            self.pop.fit_avg = self.pop.fit.mean().item()
            self.pop.fit_std = self.pop.fit.std().item()

        # Optionally, reports initializations' summary results on the console
        if verbose > 0:
            # Creates reporter's header reports the result of initialization
            self._verbose_reporter(-1, 0, None, 1)
            self._verbose_reporter(0, time.time() - start, self.pop, verbose)

        # Optionally, writes the log-data
        if log > 0:
            log_event = [self.pi.__name__, self.__name__, self.seed]
            logger = logging.getLogger(','.join(list(map(str, log_event))))
            log_event = self._create_log_event(0, 0, self.pop, log, log_xp)
            logger.info(','.join(list(map(str, log_event))))

        # Optionally, creates local variables to account for the tolerance-based stopping criteria
        if tol:
            n_iter_bare, last_fit = 0, self.best_sol.fit.clone()

        # 2)
        id_count = 0
        for it in range(1, n_iter + 1, 1):
            id_it = str(self.seed) + "_" + str(it) + "_"
            # 2) 2)
            offs_pop_ids, offs_pop_repr, offs_pop_size, offs_pop_depth, start = [], [], [], [], time.time()

            # 2) 3)
            pop_size = self.pop_size - self.pop_size % 2
            while len(offs_pop_repr) < pop_size:
                # 2) 3) 2)
                p1_idx = self.selector(self.pop, self.pi.min_)
                p2_idx = self.selector(self.pop, self.pi.min_)
                # Avoids selecting the same parent twice
                while p1_idx == p2_idx:
                    p2_idx = self.selector(self.pop, self.pi.min_)
                # Performs GP-like variation (no reproduction)
                if random.uniform(0, 1) < self.p_c:
                    # 2) 3) 3)
                    offs1_repr, offs2_repr, rt = self.crossover(self.pop[p1_idx], self.pop[p2_idx])
                    offs1_rt_size = offs2_rt_size = len(rt)
                    offs1_rt_depth = offs2_rt_depth = _get_tree_depth(rt)
                    if self.reconstruct:
                        # Stores the random tree as a .pickle
                        rt_id = id_it + "rt_xo_" + str(id_count)
                        id_count += 1
                        with open(os.path.join(self.path_rts, rt_id + ".pickle"), "wb") as handle:
                            pickle.dump(rt, handle, protocol=pickle.HIGHEST_PROTOCOL)

                        # Writes the history: when crossover, assumes ms = -1.0
                        offs_pop_ids.append(id_it + "o1_xo_" + str(id_count))
                        id_count += 1
                        self.history[offs_pop_ids[-1]] = {"Iter": it, "Operator": "crossover",
                                                          "T1": self.pop_ids[p1_idx],
                                                          "T2": self.pop_ids[p2_idx], "Tr": rt_id, "ms": -1.0}
                        offs_pop_ids.append(id_it + "o2_xo_" + str(id_count))
                        id_count += 1
                        self.history[offs_pop_ids[-1]] = {"Iter": it, "Operator": "crossover",
                                                          "T1": self.pop_ids[p2_idx],
                                                          "T2": self.pop_ids[p1_idx], "Tr": rt_id, "ms": -1.0}
                else:
                    # 2) 3) 4)
                    offs1_repr, rt1, ms1 = self.mutator(self.pop[p1_idx])
                    offs2_repr, rt2, ms2 = self.mutator(self.pop[p2_idx])
                    offs1_rt_size, offs2_rt_size = len(rt1), len(rt2)
                    offs1_rt_depth, offs2_rt_depth = _get_tree_depth(rt1), _get_tree_depth(rt2)
                    if self.reconstruct:
                        # Stores random trees as .pickle and writes the history
                        for rt, offs, p_idx, ms in zip([rt1, rt2], [offs1_repr, offs2_repr], [p1_idx, p2_idx], [ms1, ms2]):
                            rt_id = id_it + "rt_mtn_" + str(id_count)
                            id_count += 1
                            with open(os.path.join(self.path_rts, rt_id + ".pickle"), "wb") as handle:
                                pickle.dump(rt, handle, protocol=pickle.HIGHEST_PROTOCOL)

                            # When mutation, assumes T1 as the parent and T2 as -1.0
                            offs_pop_ids.append(id_it + "o1_mtn_" + str(id_count))
                            id_count += 1
                            self.history[offs_pop_ids[-1]] = {"Iter": it, "Operator": "mutation",
                                                              "T1": self.pop_ids[p_idx], "T2": -1.0,
                                                              "Tr": rt_id, "ms": ms.item()}

                # 2) 3) 5)
                offs_pop_repr.extend([offs1_repr, offs2_repr])
                offs_pop_size.extend([self.pop.size[p1_idx]+offs1_rt_size, self.pop.size[p2_idx]+offs2_rt_size])
                offs_pop_depth.extend([self.pop.depth[p1_idx]+offs1_rt_depth, self.pop.depth[p2_idx]+offs2_rt_depth])

            # Stacks the population's representation, size and depth
            offs_pop_repr = torch.stack(offs_pop_repr)
            offs_pop_size = torch.stack(offs_pop_size)
            offs_pop_depth = torch.stack(offs_pop_depth)

            # 2) 1)
            offs_pop = Population(offs_pop_repr)
            offs_pop.size = offs_pop_size
            offs_pop.depth = offs_pop_depth
            self.pi.evaluate_pop(offs_pop)

            # Updates offspring's history with fitness
            for off, fit in zip(offs_pop_ids, offs_pop.fit):
                self.history[off]["Fitness"] = fit.item()

            # Updates population's IDs
            self.pop_ids = offs_pop_ids

            # Performs population's replacement
            if self.elitism:
                offs_pop = self.elite_replacement(offs_pop)
            self.pop = offs_pop
            self._set_best_sol()

            # Optionally, evaluates the elite on the test partition
            if test_elite:
                self.pi.evaluate_sol(self.best_sol, test=True)

            # Optionally, computes iteration's timing
            if (log + verbose) > 0:
                timing = time.time() - start

            # Optionally, computes population's AVG and STD (in terms of fitness)
            if log >= 2 or verbose >= 2:
                self.pop.fit_avg = self.pop.fit.mean().item()
                self.pop.fit_std = self.pop.fit.std().item()

            # Optionally, writes the log-data on the file
            if log > 0:
                log_event = self._create_log_event(it, timing, self.pop, log, log_xp)
                logger.info(','.join(list(map(str, log_event))))

            # Optionally, reports the progress on the console
            if verbose > 0:
                self._verbose_reporter(it, timing, self.pop, verbose)

            # Optionally, verifies the tolerance-based stopping criteria
            if tol:
                n_iter_bare, last_fit = self._check_tol(last_fit, tol, n_iter_bare)

                if n_iter_bare == n_iter_tol:
                    break

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
            if self.best_sol.test_fit is not None:
                print('-' * 103)
                print(' ' * 11 + '|{:^53}  |{:^34}|'.format("Best solution", "Population"))
                print('-' * 103)
                if len(self.best_sol.fit.shape) == 0:
                    line_format = '{:<10} {:<1} {:<8} {:<16} {:<16} {:>10} {:<1} {:<16} {:>16}'
                else:
                    line_format = '{:<10} {:<1} {:<8} {:<16} {:<16} {:>10} {:<1} {:<16} {:>16}'
                print(line_format.format('Generation', "|", 'Depth', 'Fitness', 'Test Fitness',
                                         "Timing", "|", "AVG Fitness", "STD Fitness"))
                print('-' * 103)
            else:
                print('-' * 86)
                print(' ' * 11 + '|{:^36}  |{:^34}|'.format("Best solution", "Population"))
                print('-' * 86)
                line_format = '{:<10} {:<1} {:<8} {:<16} {:>10} {:<1} {:<16} {:>16}'
                print(line_format.format('Generation', "|", 'Depth', 'Fitness', "Timing",
                                         "|", "AVG Fitness", "STD Fitness"))
        else:
            if self.best_sol.test_fit is not None:
                if len(self.best_sol.fit.shape) == 0:
                    line_format = '{:<10d} {:<1} {:<8d} {:<16g} {:<16g} {:>10.3f} {:<1} {:<16g} {:>16g}'
                else:
                    line_format = '{:<10d} {:<1} {:<8d} {} {} {:>10.3f} {:<1} {:<16g} {:>16g}'
                depth = self.best_sol.depth
                if verbose >= 2:
                    avgfit, stdfit = pop.fit_avg, pop.fit_std
                else:
                    avgfit, stdfit = -1.0, -1.0
                print(line_format.format(it, "|", depth, self.best_sol.fit.tolist(), self.best_sol.test_fit.tolist(), timing, "|",
                                         avgfit, stdfit))
            else:
                if len(self.best_sol.fit.shape) == 0:
                    line_format = '{:<10d} {:<1} {:<8d} {:<16g} {:>10.3f} {:<1} {:<16g} {:>16g}'
                else:
                    line_format = '{:<10d} {:<1} {:<8d} {} {:>10.3f} {:<1} {:<16g} {:>16g}'
                depth = self.best_sol.depth
                if verbose >= 2:
                    avgfit, stdfit = pop.fit_avg, pop.fit_std
                else:
                    avgfit, stdfit = -1.0, -1.0
                print(line_format.format(it, "|", depth, self.best_sol.fit.tolist(), timing, "|", avgfit, stdfit))

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
        log_event = [log_xp, self.seed, it, timing, self.best_sol.size, self.best_sol.fit]
        if hasattr(self.best_sol, 'test_fit'):
            log_event.append(self.best_sol.test_fit)
        if log >= 2:
            log_event.extend([pop.fit_avg, pop.fit_std])
        # Also, writes elite's representation
        if log >= 3:
            log_event.append(self.best_sol.repr_)
        # Return log event
        return log_event

    def write_history(self, path):
        """Writes evolution's history on disk.

        The file that is written will be then used by the
        reconstruction algorithm.

        Parameters
        ----------
        path : str
            File path.
        """
        if self.reconstruct:
            pd.DataFrame.from_dict(self.history, orient="index").to_csv(path)
        else:
            print("Cannot write population's genealogical history since the reconstruction was not activated!")
