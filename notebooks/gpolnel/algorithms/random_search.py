import time
import random
import logging

import torch

from gpolnel.utils.solution import Solution
from gpolnel.algorithms.search_algorithm import SearchAlgorithm


class RandomSearch(SearchAlgorithm):
    """Random Search (RS) Algorithm.

    Random Search (RS) can be seen as the very first and rudimentary
    stochastic iterative solve algorithm (SISA) for problem-solving.
    Its strategy, far away from being intelligent, consists of
    randomly sampling S for a given number of iterations. RS is
    frequently used in benchmarks as the baseline for assessing
    algorithms' performance. Following this rationale, one can
    conceptualize RS at the root of the hierarchy of intelligent
    SISAs; under this perspective, it is meaningful to assume that the
    SISAs donated with intelligence, like Hill Climbing and Genetic
    Algorithms, might be seen as improvements upon RS, thus branching
    from it.

    An instance of a RS can be characterized by the following features:
        1) a PI (i.e., what to solve/optimize);
        2) a function to initialize the solve at a given point of the
         solve space (𝑆);
        3) the best solution found by the ISA;
        4) a random state for random numbers generation;
        5) the processing device (CPU or GPU).

    To solve a PI, the RS:
        1) initializes the solve at a given point in 𝑆 (normally, by
         sampling candidate solution(s) at random);
        2) searches throughout 𝑆, in iterative manner, for the best
         possible solution by randomly sampling candidate solutions
         from it. Traditionally, the termination condition for an ISA
         is the number of iterations, the default stopping criteria in
         this library.

    Attributes
    ----------
    pi : Problem (inherited from SearchAlgorithm)
        An instance of an OP.
    best_sol : Solution (inherited from SearchAlgorithm)
        The best solution found.
    initializer : function (inherited)
        The initialization procedure.
    seed : int
        The seed for random numbers generators.
    device : str (inherited from SearchAlgorithm)
        Specification of the processing device.
    """
    __name__ = "RandomSearch"

    def __init__(self, pi, initializer, seed=0, device="cpu"):
        """Objects' constructor.

        Parameters
        ----------
        pi : Problem
            An instance of an OP.
        initializer : function
            The initialization procedure.
        seed : int (default=0)
            The seed for random numbers generators.
        device : str (default="cpu")
            Specification of the processing device.
        """
        SearchAlgorithm.__init__(self, pi, initializer, device)
        # Sets the random seed for torch and random
        self.seed = seed
        torch.manual_seed(self.seed)
        random.seed(self.seed)

    def _initialize(self, start_at=None):
        """Initializes the solve at a given point in 𝑆.

        Note that the user-specified start_at is assumed to be feasible
        under 𝑆's constraints.

        Parameters
        ----------
        start_at : object (default=None)
            A user-specified initial starting point in 𝑆.
        """
        if start_at:
            # Given the initial_seed, crates an instance of type Solution
            self.best_sol = Solution(start_at)
            # Evaluates the candidate solution
            self.pi.evaluate_sol(self.best_sol)
        else:
            # Generates an valid random initial solution (already evaluated)
            self.best_sol = self._get_random_sol()
            while not self.best_sol.valid:
                self.best_sol = self._get_random_sol()

    def _get_random_sol(self):
        """Generates one random initial solution.

        This method (1) generates a random representation of a
        candidate solution by means of the initializer function, (2)
        creates an instance of type Solution, (3) evaluates  instance's
        representation and (4) returns the evaluated object.
        Notice that the solution can be feasible under 𝑆's constraints
        or not.

        Returns
        -------
        Solution
            A random initial solution.
        """
        # 1)
        repr_ = self.initializer(sspace=self.pi.sspace, device=self.device)
        # 2)
        sol = Solution(repr_)
        # 3)
        self.pi.evaluate_sol(sol)
        # 4)
        return sol

    def _create_log_event(self, it, timing, log):
        """Implements a standardized log-event.

        Creates a log-event for the underlying best-so-far solution.

        Parameters
        ----------
        it : int
            Iteration's number.
        timing : float
            Iterations's running time in seconds.
        log : int, optional (default=0)
            An integer that controls the verbosity of the log file. The
            following nomenclature is applied in this class:
                - log = 0: do not write any log data;
                - log = 1: writes the current iteration, its timing, and
                    elite's length and fitness;
                - log = 3: also, writes elite's representation.
        """
        log_event = [it, timing, len(self.best_sol), self.best_sol.fit.item()]
        if hasattr(self.best_sol, 'test_fit'):
            log_event.append(self.best_sol.test_fit.item())
        if log >= 2:
            log_event.append(self.best_sol.repr_)
        return log_event

    def _verbose_reporter(self, it, timing):
        """Reports the solve progress on the console.

        Prints the status of the solve at a given iteration. Uses the
        best-so-far solution.

        Parameters
        ----------
        it : int
            Iteration's number.
        timing : float
            Time the underlying iteration took to run.
        """
        if it == -1:
            if hasattr(self.best_sol, "test_fit"):
                print('-' * 66)
                print(' ' * 11 + '|{:^53}|'.format("Best solution"))
                print('-' * 66)
                line_format = '{:<12} {:<8} {:<16} {:<16} {:>10}'
                print(line_format.format('Generation', 'Length', 'Fitness', 'Test Fitness', "Timing"))
            else:
                print('-' * 49)
                print(' ' * 11 + '|{:^36}|'.format("Best solution"))
                print('-' * 49)
                line_format = '{:<12} {:<8} {:<16} {:>10}'
                print(line_format.format('Generation', 'Length', 'Fitness', "Timing"))
        else:
            if hasattr(self.best_sol, "test_fit"):
                line_format = '{:<10d} {:<1} {:<8d} {:<16g} {:<16g} {:>10.3f}'
                # If the the type of OP is of knapsack's family, then sum the vector, otherwise the length
                length = self.best_sol.repr_.sum() if isinstance(self.pi, Knapsack01) else len(self.best_sol)
                print(line_format.format(it, " ", length, self.best_sol.fit, self.best_sol.test_fit, timing))
            else:
                line_format = '{:<10d} {:<1} {:<8d} {:<16g} {:>10.3f}'
                # If the the type of OP is of knapsack's family, then sum the vector, otherwise the length
                length = int(self.best_sol.repr_.sum().item()) if isinstance(self.pi, Knapsack01) else len(self.best_sol)
                print(line_format.format(it, " ", length, self.best_sol.fit, timing))

    def solve(self, n_iter=20, tol=None, n_iter_tol=5, start_at=None, test_elite=False, verbose=0, log=0):
        """Implements the solve procedure of a RS algorithm.

        This method implements the following pseudo-code:
            1) _initialize: generate a valid random initial solution 𝑖;
            2) repeat: until satisfying some stopping criteria (usually
             number of iterations):
                1) generate one random solution 𝑗;
                2) if the fitness of solution 𝑗 is better or equal
                 than the fitness of solution 𝑖, 𝑖=𝑗.

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
        log : int, optional (default=0)
            An integer that controls the verbosity of the log file. The
            following nomenclature is applied in this class:
                - log = 0: do not write any log data;
                - log = 1: writes the current iteration, its timing, and
                    elite's length and fitness;
                - log = 2: also, writes elite's representation.
        """
        # Optionally, tracks initialization's timing for console's output
        if verbose > 0:
            start = time.time()

        # 1)
        self._initialize()

        # Optionally, evaluates the elite on the test partition
        if test_elite:
            # Workaround proposed by L. Rosenfeld to maintain the dataloader's seed when test_elite changes
            state = torch.get_rng_state()
            self.pi.evaluate_sol(self.best_sol, test=test_elite)
            torch.set_rng_state(state)

        # Optionally, reports initializations' summary results on the console
        if verbose > 0:
            # Creates reporter's header reports the result of initialization
            self._verbose_reporter(-1, 0)
            self._verbose_reporter(0, time.time() - start)

        # Optionally, writes the log-data
        if log > 0:
            # Optionally, evaluates the elite on the test partition
            if test_elite:
                # Workaround proposed by L. Rosenfeld to maintain the dataloader's seed when test_elite changes
                state = torch.get_rng_state()
                self.pi.evaluate_sol(self.best_sol, test=test_elite)
                torch.set_rng_state(state)

            log_event = [self.pi.__name__, self.__name__, self.seed]
            logger = logging.getLogger(','.join(list(map(str, log_event))))
            log_event = self._create_log_event(it=0, timing=0, log=log)
            logger.info(','.join(list(map(str, log_event))))

        # Optionally, creates local variables to account for the tolerance-based stopping criteria
        if tol:
            n_iter_bare, last_fit = 0, self.best_sol.fit.clone()

        # 2)
        for it in range(1, n_iter + 1):
            # 2) 1)
            rand_cand_sol, start = self._get_random_sol(), time.time()
            # 2) 2)
            if rand_cand_sol.valid:
                self.best_sol = self._get_best(self.best_sol, rand_cand_sol)

                # Optionally, evaluates the elite on the test partition
                if test_elite:
                    # Workaround proposed by L. Rosenfeld to maintain the dataloader's seed when test_elite changes
                    state = torch.get_rng_state()
                    self.pi.evaluate_sol(self.best_sol, test=test_elite)
                    torch.set_rng_state(state)

            # Optionally, computes iteration's timing
            if (log + verbose) > 0:
                timing = time.time() - start

            # Optionally, writes in the log file
            if log > 0:
                log_event = self._create_log_event(it=it, timing=timing, log=log)
                logger.info(','.join(list(map(str, log_event))))

            # Optionally, reports the progress on the console
            if verbose > 0:
                self._verbose_reporter(it, timing)

            # Optionally, verifies the tolerance-based stopping criteria
            if tol:
                n_iter_bare, last_fit = self._check_tol(last_fit, tol, n_iter_bare)

                if n_iter_bare == n_iter_tol:
                    break
