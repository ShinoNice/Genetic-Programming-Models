import sys
from joblib import Parallel, delayed
import torch
import numpy as np

from gpolnel.problems.problem import Problem
from gpolnel.utils.solution import Solution
from gpolnel.utils.inductive_programming import _execute_tree, _get_tree_depth


class SML(Problem):
    """ Implements SML problem in the scope of IP-OPs

    "Inductive programming (IP) is a special area of automatic 
    programming, covering research from artificial intelligence and 
    programming, which addresses learning of typically declarative 
    (logic or functional) and often recursive programs from incomplete 
    specifications, such as input/output examples or constraints."
        - https://en.wikipedia.org/wiki/Inductive_programming
    
    In the context of Supervised Machine Learning (SML) problems, one
    can define the task of a Genetic Programming (GP) algorithm as the
    program/function induction that identifies the mapping 𝑓 : 𝑆 → 𝐼𝑅
    in the best possible way, generally measured through solutions'
    generalization ability. "The generalization ability (or simply
    generalization) of a model is defined by its performance in data
    other than the training data. In practice, this generalization
    ability is estimated by leaving out of the training data a part of
    the total available data. The data left out of the training data is
    usually referred to as unseen data, testing data, or test data. A
    model that is performing well in unseen data is said to be
    generalizing. However, performance in training and unseen data does
    not always agree."
        - An Exploration of Generalization and Overfitting in Genetic 
          Programming - Standard and Geometric Semantic Approaches, 
          I. Gonçalves (2016).

    In the context of this library, and in this release, GP is mainly
    used to solve SML problems, like regression or classification. As
    such, the solve space for an instance of inductive programming OP,
    is made of labeled input training and unseen data, and GP-specific
    parameters which characterize and bound the solve space (i.e., the
    set of functions and constants, the range of randomly  generated
    constants, the maximum boundaries for initial trees' depth, etc.).

    An instance of this class receives the training and unseen data as the
    instance features of type torch.utils.data.Dataset. Consequently the
    training of a GP system can be performed by batches or by using the whole
    dataset at a time.

    Attributes
    ----------
    sspace : dict
        The solve space of an instance of SML OP is composed by the
        following key-value pairs:
            <"n_dims"> int: the number of input features (a.k.a. input
             dimensions) in the underlying SML problem;
            <"function_set"> list: the set of primitive functions;
            <"constant_set"> torch.Tensor: the set of constants to draw
             terminals from;
            <"p_constants"> float: the probability of generating a
             constant when sampling a terminal;
            <"max_init_depth"> int: maximum trees' depth during the
             initialization;
            <"max_depth"> int: maximum trees' depth during the
             evolution;
            <"n_batches"> int: number of batches to use when evaluating
             solutions (more than one can be used).
    ffunction : function
        𝑓 : 𝑆 → 𝐼𝑅. Examples of possible fitness functions are:
         - mean absolute error;
         - mean squared error;
         - mean squared logarithmic error;
         - median absolute error;
         - etc.
    min_ : bool
        A flag which defines the purpose of optimization.
    dl_train : torch.utils.data.DataLoader
        Train data-loader.
    dl_test : torch.utils.data.DataLoader
        Test data-loader.
    n_jobs : int
        Number of jobs to run in parallel when executing trees.
    device : str
        Specification of the processing device.
    """
    __name__ = "IP-SML"

    def __init__(self, sspace, ffunction, dl_train, dl_test=None, min_=None, n_jobs=1):
        """ Object's constructor.

        Parameters
        ----------
        sspace : dict
            The solve space of an instance of SML OP is composed by the
            following key-value pairs:
                <"n_dims"> int: the number of input features (a.k.a.
                 input dimensions) in the underlying SML problem;
                <"function_set"> list: the set of primitive functions;
                <"constant_set"> torch.Tensor: the set of constants to
                 draw terminals from;
                <"p_constants"> float: the probability of generating a
                 constant when sampling a terminal;
                <"max_init_depth"> int: maximum trees' depth during the
                 initialization;
                <"max_depth"> int: maximum trees' depth during the
                 evolution;
                <"n_batches"> int: number of batches to use when
                 evaluating solutions (more than one can be used).
        ffunction : function
            𝑓 : 𝑆 → 𝐼𝑅.
        dl_train : torch.utils.data.DataLoader
            DataLoader for the training set.
        dl_test : torch.utils.data.DataLoader
            DataLoader for the testing set.
        n_jobs : int (default=1)
            Number of parallel processes used to execute the trees.
        min_ : bool
            A flag which defines the purpose of optimization.
        """
        if min_ is None: min_ = ffunction.min_
        Problem.__init__(self, sspace, ffunction, min_)
        self.dl_train = dl_train
        self.dl_test = dl_test
        self.n_jobs = n_jobs
        # Infers processing device from the constants' set
        self.device = self.sspace["device"]

    def _evaluate_sol(self, sol, test):
        """ Encapsulates the evaluation of a single solution with the fitness function
        and dataset of current SML instance.

        Parameters
        ----------
        sol : Solution
            A solution to be evaluated.
        test : Boolean
            If the train or test set should be used for evaluating the solution.
        """
        return self._evaluate_sol_ffunction(self.ffunction, sol, test)

    def _evaluate_sol_data_loader(self, sol, data_loader):
        """ Encapsulates the evaluation of a single solution with the fitness function
        of current SML instance using dataset given by data_loader arg.

        Parameters
        ----------
        sol : Solution
            A solution to be evaluated.
        data_loader : torch.DataLoader
            The data loader that should be used to evaluate the solution.
        """
        return self._evaluate_sol_ffunction_data_loader(self.ffunction, sol, data_loader)

    def _evaluate_sol_ffunction_data_loader(self, ffunction, sol, data_loader):
        """ Evaluates a single solution with the fitness function and dataset
        given by ffunction arg and the dataset of current SML instance.

        Parameters
        ----------
        sol : Solution
            A solution to be evaluated.
        ffunction : Ffunction
            The fitness function used to evaluate the solution.
        data_loader : torch.DataLoader
            The data loader that should be used to evaluate the solution.
        """
        if ffunction.is_structural:
            fit = ffunction(sol=sol)
        else:
            with torch.no_grad():
                # Creates a container to accumulate fitness across different batches
                fit_dl = 0.0
                n = 0
                # Iterates fitness calculation for n_batches or total number of batches
                # n_batches is user set so it can be larger than the total number of batches
                # and it is set for training data, but the number of batches in training and test sets
                # can be different.
                data_loader_iter = iter(data_loader)
                for i_batch in range(min(len(data_loader_iter), self.sspace["n_batches"])):
                    # Moves the data to the underlying processing device
                    batch = next(data_loader_iter)
                    X, y = batch[0].to(self.device), batch[1].to(self.device)
                    # Computes the semantics
                    y_pred = _execute_tree(sol.repr_, X)
                    # Computes the fitness for the current batch
                    fit_batch = ffunction(call='dl', y_true=y, y_pred=y_pred)
                    fit_dl += fit_batch
                    n += X.shape[0]
                    # Releases all unoccupied cached memory
                    torch.cuda.empty_cache()
            # Combines the batches into the final fit evaluation
            fit = ffunction(call='join', fit_dl=fit_dl, n=n)
        return fit

    def _evaluate_sol_ffunction(self, ffunction, sol, test):
        """ Evaluates a single solution with the fitness function given by ffunction arg and
        dataset of current SML instance..

        Parameters
        ----------
        sol : Solution
            A solution to be evaluated.
        ffunction : Ffunction
            The fitness function used to evaluate the solution.
        test : Boolean
            If the train or test set should be used for evaluating the solution.
        """
        if ffunction.is_structural:
            fit = ffunction(sol=sol)
        else:
            # Chooses which data partition to use
            data_loader = self.dl_test if test else self.dl_train
            with torch.no_grad():
                # Creates a container to accumulate fitness across different batches
                fit_dl = 0.0
                n = 0
                # Iterates fitness calculation for n_batches or total number of batches
                # n_batches is user set so it can be larger than the total number of batches
                # and it is set for training data, but the number of batches in training and test sets
                # can be different.
                data_loader_iter = iter(data_loader)
                for i_batch in range(min(len(data_loader_iter), self.sspace["n_batches"])):
                    # Moves the data to the underlying processing device
                    batch = next(data_loader_iter)
                    X, y = batch[0].to(self.device), batch[1].to(self.device)
                    # Computes the semantics
                    y_pred = _execute_tree(sol.repr_, X)
                    # Computes the fitness for the current batch
                    fit_batch = ffunction(call='dl', y_true=y, y_pred=y_pred)
                    fit_dl += fit_batch
                    n += X.shape[0]
                    # Releases all unoccupied cached memory
                    torch.cuda.empty_cache()
            # Combines the batches into the final fit evaluation
            fit = ffunction(call='join', fit_dl=fit_dl, n=n)
        return fit

    def _evaluate_pop_ffunction(self, ffunction, pop):
        """ Evaluates at a once the entire population pop with the fitness function
        given by ffunction arg and the dataset of current SML instance.

        Parameters
        ----------
        pop : Population
            The population whose solutions should be evaluated.
        ffunction : Ffunction
            The fitness function used to evaluate the solution.
        """
        # Set structural fitnesses
        if ffunction.is_structural:
            fit = ffunction(sol=pop).clone().detach().to(self.device)
        # Set error fitnesses
        else:
            # Assigns default fitness-cases
            fit_dl = torch.zeros(len(pop), device=self.device)
            n = 0
            # Temporarily sets all the requires_grad flag to False
            with torch.no_grad():
                # Iterates fitness calculation for n_batches or total number of batches
                data_loader_iter = iter(self.dl_train)
                for i_batch in range(min(len(data_loader_iter), self.sspace["n_batches"])):
                    # Moves the data to the underlying processing device
                    batch = next(data_loader_iter)
                    X, y = batch[0].to(self.device), batch[1].to(self.device)
                    if self.device == 'cpu':
                    # For the current batch, iterates over the population to evaluate_pop each individual
                        y_pred = Parallel(n_jobs=self.n_jobs)(delayed(_execute_tree)(repr_, X) for repr_ in pop.repr_)
                    else:
                        y_pred = []
                        for i, repr_ in enumerate(pop.repr_):
                            y_pred.append(_execute_tree(repr_, X))
                    fit_batch = ffunction(call='dl', y_true=y, y_pred=torch.stack(y_pred))
                    fit_dl += fit_batch
                    n += X.shape[0]
                    # Releases all unoccupied cached memory
                    torch.cuda.empty_cache()
            # Combines the batches into the final fit evaluation
            fit = ffunction(call='join', fit_dl=fit_dl, n=n)
        return fit

    def _is_feasible_sol(self, repr_):
        """ Assesses solution's feasibility under 𝑆's constraints.

        Assesses solution's feasibility after constraints specified
        in 𝑆 (if any). In the context of IP-OP, the feasibility relates
        with the maximum allowed depth of the tree representing a
        candidate solution.

        Parameters
        ----------
        repr_ : list
            LISP-based representation of a candidate solution.

        Returns
        -------
        bool
            Representation's feasibility state.
        """
        if 'max_depth' not in self.sspace or self.sspace['max_depth'] == -1:
            # No depth limit was specified, so the solution is valid.
            return True
        else:
            if _get_tree_depth(repr_) <= self.sspace["max_depth"]:
                return True
            else:
                return False

    def _is_feasible_pop(self, repr_):
        """ Assesses population's feasibility under 𝑆's constraints.

        Assesses population's feasibility after constraints specified
        in 𝑆 (if any). This method was particularly designed to include
        more efficient assessment procedure for a set of solutions.
        In the context of IP-OP, the feasibility relates with the
        maximum allowed depth of the tree representing a candidate
        solution.

        Parameters
        ----------
        repr_ : list
            A list of LISP-based representations of a set of candidate
            solutions.

        Returns
        -------
        list
            Representations' feasibility state.
        """
        if "max_depth" not in self.sspace or self.sspace["max_depth"] == -1:
            # No depth limit was specified, thus all the solutions are assumed to be valid
            return [True]*len(repr_)
        else:
            return [_get_tree_depth(t) <= self.sspace["max_depth"] for t in repr_]

    def evaluate_sol(self, sol, train=True, test=False):
        """ Evaluate a candidate solution.

        This method receives a candidate solution from 𝑆 and, after
        validating its representation by means of _is_feasible_sol,
        evaluates it by means of 𝑓. If the solution happens to be
        invalid, then it automatically receives a "very bad fitness":
        maximum possible integer in the case of minimization, zero
        otherwise.

        Parameters
        ----------
        sol : Solution
            A candidate solution from the solve space.
        train : bool
            A flag which defines if train data partition should be used when
            evaluating the solution.
        test : bool
            A flag which defines if test data partition should be used when
            evaluating the solution.
        """
        # Validates solution's representation
        sol.valid = self._is_feasible_sol(sol.repr_)
        # Evaluates solution, if it is valid
        if sol.valid:
            if train:
                sol.fit = self._evaluate_sol(sol=sol, test=False)
            if test:
                sol.test_fit = self._evaluate_sol(sol=sol, test=True)
        else:
            self._set_bad_fit_sol(sol, test, self.device)
        return sol

    def evaluate_sol_data_loader(self, sol, data_loader):
        """Encapsulates the evaluation of a single solution with the fitness function
        of current SML instance using dataset given by data_loader arg.

        Parameters
        ----------
        sol : Solution
            A solution to be evaluated.
        data_loader : torch.DataLoader
            The data loader that should be used to evaluate the solution.
        """
        return self._evaluate_sol_data_loader(sol=sol, data_loader=data_loader)

    def evaluate_pop(self, pop):
        """ Evaluates a population of candidate solutions.

        This method receives a population of candidate solutions from 𝑆 and, after
        validating its representation by means of _is_feasible_pop,
        evaluates it by means of 𝑓. If the solution happens to be
        invalid, then it automatically receives a "very bad fitness":
        maximum possible integer in the case of minimization, zero otherwise.

        Parameters
        ----------
        pop : Population
            population candidate solution to evaluate_pop.
        """
        # Validates population's representation
        pop.valid = self._is_feasible_pop(pop.repr_)
        # Assigns individuals fitness(es)
        pop.fit = self._evaluate_pop_ffunction(self.ffunction, pop)
        # Assigns the default fitness to invalid solutions in the population
        pop_invalid = ~torch.tensor(pop.valid)
        if any(pop_invalid):
            pop.fit[pop_invalid] = torch.ones(pop_invalid.sum(), device=self.device) * (
                sys.maxsize if self.min_ else -sys.maxsize)
        # Assigns the fitness values to individuals
        [pop.individuals[i].__setattr__('fit', f) for i, f in enumerate(pop.fit)]
        [pop.individuals[i].__setattr__('valid', f) for i, f in enumerate(pop.valid)]

    def predict_sol_data_loader(self, repr_, data_loader, device):
        """ Predicts the output of the solution with representation repr_
        using the dataset given by data_loader arg.

        Parameters
        ----------
        repr_ : list
            A list of the tree elements.
        data_loader : torch.DataLoader
            The data loader that should be used to evaluate the solution.
        device : string
            The device in which the solution should be evaluated.
        """
        with torch.no_grad():
            # Creates a container to accumulate fitness across different batches
            y_pred = []
            # Iterates fitness calculation for n_batches
            for b, batch in enumerate(data_loader):
                # Moves the data to the underlying processing device
                X, y = batch[0].to(device), batch[1].to(device)
                # Computes the semantics
                y_pred.append(_execute_tree(repr_, X))
                # Releases all unoccupied cached memory
                torch.cuda.empty_cache()
            # Combines the batches into the final predict object
            y_pred = torch.cat(y_pred, dim=0)
        return y_pred


class SMLGS(Problem):
    """ Implements inductive programming OP.

    "Inductive programming (IP) is a special area of automatic
    programming, covering research from artificial intelligence and
    programming, which addresses learning of typically declarative
    (logic or functional) and often recursive programs from incomplete
    specifications, such as input/output examples or constraints."
        - https://en.wikipedia.org/wiki/Inductive_programming

    In the context of Machine Learning (ML) OPs, one can define the
    task of Genetic Programming (GP) as of program/function induction
    that identifies the mapping 𝑓 : 𝑆 → 𝐼𝑅 in the best possible way,
    generally measured through tge generalization ability.
    "The generalization ability (or simply generalization) of a model
    is dened by its performance in data other than the training data.
    In practice, this generalization ability is estimated by leaving
    out of the training data a part of the total available data. The
    data left out of the training data is usually referred to as
    unseen data, testing data, or test data. A model that is
    performing well in unseen data is said to be generalizing. However,
    performance in training and unseen data does not always agree."
        - An Exploration of Generalization and Overfitting in Genetic
          Programming - Standard and Geometric Semantic Approaches,
          I. Gonçalves (2016).

    In the context of this library, GP is mainly used to solve
    Supervised Machine Learning (SML) tasks, like Regression or
    Classification. As such, the solve space for an instance of
    inductive programming OP, in the context of Machine Learning (ML)
    problem solving, is made of labeled input training and unseen data,
    and GP-specific parameters which characterize and delimit the
    solve space (i.e., the set of functions and constants, the range of
    randomly  generated constants, the maximum boundaries for initial
    trees' depth, etc.).

    This class implements another OP type and is subject to the
    following restrictions:
     - the training and unseen data are provided within the solve
       space as objects of type torch.utils.data.Dataset;
     - the training of GP system can be performed by batches or by
       using the whole dataset at a time.

    Attributes
    ----------
    sspace : dict
        The solve space composed by:
         - batch_size : int
         - function set: the set of all primitive functions used to
          compose trees;
         - % of constants: proportion of constants in regard to the number
          of input features, (the minor component of the terminal set);
         - range of constants: lower and upper bounds for random constants
          generation;
         - max_init_depth: maximum initialization depth;
         - max_growth_depth: maximum growth depth of trees.
    X : torch.Tensor
        A tensor representing the input samples. Internally, it will be
        converted to dtype=torch.float. It is assumed that the object
        was already allocated on the proper processing device.
    y : torch.Tensor
        A tensor representing the The target values. Internally, it
        will be converted to dtype=torch.float. It is assumed that the
        object was already allocated on the proper processing device.
    train_indices : torch.Tensor
        Indices representing the training partition of the data.
    test_indices : torch.Tensor
        Indices representing the testing partition of the data.
    ffunction : function
        𝑓 : 𝑆 → 𝐼𝑅. Examples of possible fitness functions are:
         - mean absolute error;
         - mean squared error;
         - mean squared logarithmic error;
         - median absolute error;
         - etc.
    device : str
        Specification of the processing device. Inferred from the
        solve-space's definition.
    """
    __name__ = "IPOP_GS"

    def __init__(self, sspace, ffunction, X, y, train_indices, test_indices, min_=None):
        """ Object's constructor.

        Parameters
        ----------
        sspace : dict
            The formal definition of 𝑆 (problem dependent).
        ffunction : function
            𝑓 : 𝑆 → 𝐼𝑅.
        X : torch.Tensor
            A tensor representing the input samples. Internally, it
            will be converted to dtype=torch.float. It is assumed that
            the object was already allocated on the proper processing
            device.
        y : torch.Tensor
            A tensor representing the The target values. Internally, it
            will be converted to dtype=torch.float. It is assumed that
            the object was already allocated on the proper processing
            device.
        train_indices : Tensor
            A tensor of int with the indices of data cases to be used for
            train partition
        test_indices : Tensor
            A tensor of int with the indices of data cases to be used for
            test partition
        min_ : bool (default=True)
            A flag which defines the purpose of optimization.
        """
        if min_ is None: min_ = ffunction.min_
        Problem.__init__(self, sspace, ffunction, min_)
        self.X = X
        self.y = y
        # Infers the processing device
        self.device = self.sspace['device']
        # Train and test data splits
        self.train_indices, self.test_indices = train_indices, test_indices

    def _evaluate_pop_ffunction(self, ffunction, pop, test=False):
        """ Evaluates at a once the entire population pop with the fitness function
        given by ffunction arg and the dataset of current SML instance.

        Parameters
        ----------
        ffunction : Ffunction
            The fitness function used to evaluate the solution.
        pop : Population
            The population whose solutions should be evaluated.
        """
        # Set structural fitnesses
        if ffunction.is_structural:
            return ffunction(sol=pop).clone().detach().to(self.device)
        # Set error fitnesses
        else:
            if test:
                if self.test_indices is None:
                    return None
                else:
                    return ffunction(call='semantic', y_true=self.y[self.test_indices], y_pred=pop.repr_[:, self.test_indices])
            else:
                return ffunction(call='semantic', y_true=self.y[self.train_indices], y_pred=pop.repr_[:, self.train_indices])

    def evaluate_pop(self, pop, test=False):
        """ Evaluates a population of candidate solutions.

        This method receives a population of candidate solutions from 𝑆 and, after
        validating its representation by means of _is_feasible_pop,
        evaluates it by means of 𝑓. If the solution happens to be
        invalid, then it automatically receives a "very bad fitness":
        maximum possible integer in the case of minimization, zero otherwise.

        Parameters
        ----------
        pop : Population
            population candidate solution to evaluate_pop.
        """
        # Temporarily sets all the requires_grad flag to False
        with torch.no_grad():
            # Assigns default validity state: with GSOs it is True, by default, for all the individuals
            pop.valid = torch.ones(len(pop), dtype=torch.bool, device=self.device)
            if test:
                pop.test_fit = self._evaluate_pop_ffunction(self.ffunction, pop, test=True)
            else:
                pop.fit = self._evaluate_pop_ffunction(self.ffunction, pop, test=False)
        # Releases all unoccupied cached memory
        torch.cuda.empty_cache()
        # Assigns the fitness values to individuals

        if test:
            if pop.test_fit is not None:
                [pop.individuals[i].__setattr__('test_fit', f) for i, f in enumerate(pop.test_fit)]
        else:
            [pop.individuals[i].__setattr__('fit', f) for i, f in enumerate(pop.fit)]
        [pop.individuals[i].__setattr__('valid', f) for i, f in enumerate(pop.valid)]

    def _evaluate_sol_ffunction(self, ffunction, sol, test):
        """ Evaluates a single solution with the fitness function given by ffunction arg and
        dataset of current SML instance..

        Parameters
        ----------
        sol : Solution
            A solution to be evaluated.
        ffunction : Ffunction
            The fitness function used to evaluate the solution.
        test : Boolean
            If the train or test set should be used for evaluating the solution.
        """
        if ffunction.is_structural:
            return ffunction(sol=sol).clone().detach().to(self.device)
        else:
            if test:
                if self.test_indices is None:
                    return None
                else:
                    return ffunction(call='semantic', y_true=self.y[self.test_indices], y_pred=sol.repr_[self.test_indices])
            else:
                return ffunction(call='semantic', y_true=self.y[self.train_indices], y_pred=sol.repr_[self.train_indices])

    def evaluate_sol(self, sol, test=False):
        """ Evaluate a candidate solution.

        This method receives a candidate solution from 𝑆 and, after
        validating its representation by means of _is_feasible_sol,
        evaluates it by means of 𝑓. If the solution happens to be
        invalid, then it automatically receives a "very bad fitness":
        maximum possible integer in the case of minimization, zero
        otherwise.

        Parameters
        ----------
        sol : Solution
            A candidate solution from the solve space.
        test : bool
            A flag which defines which data partition to use when
            evaluating the solution.
        """
        # Assigns default validity state: with GSOs it is true, by default
        sol.valid = True
        # Computes solution fitnesses
        if test:
            sol.test_fit = self._evaluate_sol_ffunction(ffunction=self.ffunction, sol=sol, test=True)
        else:
            sol.fit = self._evaluate_sol_ffunction(ffunction=self.ffunction, sol=sol, test=False)
        return sol

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
