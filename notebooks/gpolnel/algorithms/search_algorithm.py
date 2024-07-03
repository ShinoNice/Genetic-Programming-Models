from gpolnel.utils.solution import Solution


class SearchAlgorithm:
    """Iterative Search Algorithm (ISA).

    To solve an an optimization problem's instance (PI), one needs to
    define a procedure (an algorithm). This library focuses on
    Iterative Search Algorithms (ISAs) - algorithms that,
    iteration after iteration, take decisions (probabilistic or not)
    to improve upon the current solution.

    An instance of an ISA is characterized by the following features:
        1) a PI (i.e., what to solve/optimize);
        2) a function to initialize the solve at a given point of the
         solve space (𝑆);
        3) the best solution found by the ISA.

    To solve a PI, an ISA:
        1) initializes the solve at a given point in 𝑆 (normally, by
         sampling candidate solution(s) at random);
        2) searches throughout 𝑆, in iterative manner, for the best
         possible solution, according to the criteria specified in PI.
         Traditionally, the termination condition for an ISA is the
         number of iterations, the default stopping criteria in this
         library.
    These two steps are implemented as abstract methods "_initialize"
    and "solve", meaning that every subclass of type SearchAlgorithm
    has to implement them.

    Attributes
    ----------
    pi: Problem
        An instance of an OP.
    best_sol : Solution
        The best solution found.
    initializer : function
        The initialization procedure.
    device : str
        Specification of the processing device.
    """
    def __init__(self, pi, initializer, device):
        """ Objects' constructor.

        Parameters
        ----------
        pi : Problem instance
            An instance of an Optimization Problem.
        initializer : function
            The initialization procedure.
        device : str
            Specification of the processing device.
        """
        self.pi = pi
        self.initializer = initializer
        self.device = device
        self.best_sol = None  # initializes the best solution: None, by default
        # Utility flag regarding batch training
        self._batch_training = True if ("batch_training" in self.pi.sspace and self.pi.sspace["batch_training"]) else False

    def _initialize(self, start_at=None):
        """Initializes the solve at a given point in 𝑆.

        Note that the user-specified start_at is assumed to be feasible
        under 𝑆's constraints.

        Parameters
        ----------
        start_at : object (default=None)
            A user-specified initial starting point in 𝑆.
        """
        pass

    def _create_log_event(self):
        """Implements a standardized log-event.

        Creates a log-event for the underlying best-so-far solution.
        """
        pass

    def _verbose_reporter(self):
        """Reports the solve progress on the console.

        Prints the status of the solve at a given iteration. Uses the
        best-so-far solution.
        """
        pass

    def solve(self, n_iter=20, tol=None, n_iter_tol=5, start_at=None, test_elite=False, verbose=0, log=0):
        """Defines the general steps of an ISA.

        An ISA can be used to solve both pure OPs and Supervised
        Machine Learning (SML) types of optimization problem (OPs).
        Recall that the latter is concerned with solutions'
        generalization ability; in this sense, the parameter test_elite
        allows to assess solution(s) generalization ability when solving
        SML-OPs.

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
            An integer that controls the verbosity of the solve.
        log : int, optional (default=0)
            An integer that controls the verbosity of the log file.
        """
        pass

    def _check_tol(self, last_fit, tol, n_iter_bare):
        """

        Parameters
        ----------
        last_fit : torch.tensor
        tol : torch.tensor
        n_iter_bare : int

        Returns
        -------
        int, torch.tensor

        """
        if self.pi.min_:  # counts bare iterations for minimization
            if last_fit <= self.best_sol.fit + tol:
                n_iter_bare += 1
                if last_fit > self.best_sol.fit:
                    last_fit = self.best_sol.fit.clone()
                return n_iter_bare, last_fit
            else:
                return 0, self.best_sol.fit.clone()  # better then last, restart counting bare iterations
        else:  # counts bare iterations for maximization
            if last_fit >= self.best_sol.fit - tol:
                n_iter_bare += 1
                if last_fit < self.best_sol.fit:
                    last_fit = self.best_sol.fit.clone()
                return n_iter_bare, last_fit
            else:
                return 0, self.best_sol.fit.clone()

    def _get_best(self, cand_a, cand_b):
        """Compares two candidate solutions and returns the best.

        Provides a flexible and a simple mean of comparison between two
        candidate solutions based on their fitness. The method assumes
        both solutions are different.

        Parameters
        ----------
        cand_a : Solution
            An object of type Solution.
        cand_b : Solution
            Another object of type Solution.

        Returns
        -------
        Solution
            The best candidate solution.
        """
        if self.pi.min_:
            if cand_a.fit >= cand_b.fit:
                return cand_b
            else:
                return cand_a
        else:
            if cand_a.fit <= cand_b.fit:
                return cand_b
            else:
                return cand_a

    def _get_worst(self, cand_a, cand_b):
        """Compares two solutions and returns the worst.

        Provides a flexible and a simple mean of comparison between two
        candidate solutions based on their fitness. The method assumes
        both solutions are different.

        Parameters
        ----------
        cand_a : Solution
            An object of type Solution.
        cand_b : Solution
            Another object of type Solution.

        Returns
        -------
        Solution
            The worst candidate solution.
        """
        if self.pi.min_:
            if cand_a.fit >= cand_b.fit:
                return cand_a
            else:
                return cand_b
        else:
            if cand_a.fit <= cand_b.fit:
                return cand_a
            else:
                return cand_b


