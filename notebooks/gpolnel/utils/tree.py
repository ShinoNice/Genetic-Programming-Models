import sys

import torch
from math import prod
from copy import deepcopy
from gpolnel.utils.solution import Solution
from gpolnel.utils.inductive_programming import _Function
from gpolnel.utils.utils import phi


class Tree(Solution):
    """ Implementation of a Tree class for tree-based GP.

    Tree solutions need specific attributes and methods. Thus, this class
    centrilises these features.


    Attributes
    ----------
    _id : int
        A unique identification of a solution.
    valid : bool
        Solution's validity state under the light of 𝑆.
    repr_ : list
        The representation of a tree in 𝑆.
    fit: float
        A measure of quality in 𝑆. It is assigned by a given problem
        instance (PI), using fitness function (𝑓).
    depth: int
        The tree depth.
    """

    def __init__(self, repr_):
        """ Object's constructor.

        Parameters
        ----------
        repr_ : Object
            The representation of a tree in 𝑆.
        """
        Solution.__init__(self, repr_)
        self._set_size()
        self._set_complexity()
        self._set_depth()
        self._set_depths()
        self._set_no()
        self._set_nao()
        self._set_naoc()
        self._set_phi()
        self._set_n_features()
        self._set_visitation_length()

    def _get_copy(self):
        """ Makes a copy of the calling object.

        Notice that, to ease library's code-flexibility, the solutions'
        representation can take two forms: list or tensor. The former
        regards GP trees, the latter all the remaining representations
        (array-based).

        Returns
        -------
        solution : Tree
            An object of type Tree, copy of self.
        """
        if type(self.repr_) is torch.Tensor:
            sol_copy = Tree(self.repr_.clone())
        else:
            sol_copy = Tree(self.repr_.copy())
        sol_copy.valid = self.valid
        if self.fit is None:
            sol_copy.fit = None
        else:
            sol_copy.fit = self.fit.clone()
        if self.test_fit is None:
            sol_copy.test_fit = None
        else:
            sol_copy.test_fit = self.test_fit.clone()

        return sol_copy

    def printTree(self, feature_names=None, out=None):
        """ Encapsulated the method for converting the tree representation in a string,
        which can be printed or returned as a string, according to the out arg.

        Parameters
        ----------
        feature_names : list
            A list of strings with feature names.
        out : string
            If 'string' the string of the solution representation is returned as
            a string, otherwise it is printed in the default output device.
        """
        # One node trees
        if len(self.repr_) == 1:
            if isinstance(self.repr_[0], int):
                if feature_names is not None:
                    s = feature_names[self.repr_[0]]
                else:
                    s = 'x_' + str(self.repr_[0])
            else:
                s = '{:.4f}'.format(self.repr_[0].item())
            if out == 'string': return s
            print(s)
        else:
            # Larger trees
            _, s = self._printTree(s='', i=0, feature_names=feature_names)
            if out == 'string': return s[1:]
            print(s[1:])

    def _printTree(self, s, i, feature_names):
        """ Generates the tree representation as a string.

        Parameters
        ----------
        s : list
            The string of the tree representation part that has already been converted.
        i : int
            The index of the element to be read.
        feature_names : list
            A list of strings with feature names.

        Return
        ----------
        i : int
            The index of the next tree element to be read.
        s : string
            The string of the tree representation part that has already been converted.
        """
        branch = ''
        if isinstance(self.repr_[i], _Function):
            branch += ' ' + self.repr_[i].name + '('
            parent_arity = self.repr_[i].arity
            for i_arity in range(parent_arity):
                i += 1
                if isinstance(self.repr_[i], _Function):
                    i, subtree = self._printTree(s=s, i=i, feature_names=feature_names)
                    if subtree[-1] == ',': subtree = subtree[:-1]
                    branch += subtree
                else:
                    if isinstance(self.repr_[i], int):
                        if feature_names is not None:
                            branch += ' ' + feature_names[self.repr_[i]] + ','
                        else:
                            branch += ' x_' + str(self.repr_[i]) + ','
                    else:
                        branch += ' ' + '{:.4f}'.format(self.repr_[i].item()) + ','
            if branch[-1] == ',': branch = branch[:-1]
        else:
            i += 1
            # branch += ' x_' + self.repr_[i].name
            # branch += ' ' + self.repr_[i]
            if isinstance(self.repr_[i], int):
                if feature_names is not None:
                    branch += ' ' + feature_names[self.repr_[i]] + ','
                else:
                    branch += ' x_' + str(self.repr_[i]) + ','
            else:
                branch += ' ' + '{:.4f}'.format(self.repr_[i].item()) + ','
                # branch += ' ' + str(self.repr_[i].item()) + ','
        if branch[-1] == ',': branch = branch[:-1]
        s += branch + ' )'
        return i, s

    def _get_complexity(self):
        return self._evaluate_complexity()

    def get_complexity(self):
        return self.complexity

    def _set_complexity(self):
        self.complexity = self._get_complexity()

    # Depth
    def _get_depth(self):
        return self._evaluate_depth()

    def get_depth(self):
        return self.depth

    def _set_depth(self):
        self.depth = self._get_depth()

    # Depths
    def _get_depths(self):
        return self._evaluate_depths()

    def get_depths(self):
        return self.depths

    def _set_depths(self):
        self.depths = self._get_depths()

    # Number of (Unique) Features
    def _get_n_features(self):
        return sum([isinstance(el, int) for el in set(self.repr_)])

    def get_n_features(self):
        return self.n_features

    def _set_n_features(self):
        self.n_features = self._get_n_features()

    # Number of Non-arithmetic Operators
    def _get_nao(self):
        operators = [el for el in self.repr_ if isinstance(el, _Function)]
        return sum([not op.arithmetic for op in operators])

    def get_nao(self):
        return self.nao

    def _set_nao(self):
        self.nao = self._get_nao()

    # Number of Consecutive Non-arithmetic Operators
    def _get_naoc(self):
        nao_all = [not el.arithmetic if isinstance(el, _Function) else False for el in self.repr_]
        nao_indexes = [i for i in range(self.get_size()) if nao_all[i]]
        nao_consecutive = [(y - x) == 1 for x, y in zip(nao_indexes, nao_indexes[1:])]
        return sum(nao_consecutive)

    def get_naoc(self):
        return self.naoc

    def _set_naoc(self):
        self.naoc = self._get_naoc()

    # Number of Operators
    def _get_no(self):
        return sum([isinstance(el, _Function) for el in self.repr_])

    def get_no(self):
        return self.no

    def _set_no(self):
        self.no = self._get_no()

    # PHI
    def _get_phi(self):
        return phi(sol=self)

    def get_phi(self):
        return self.phi

    def _set_phi(self):
        self.phi = self._get_phi()

    # Size
    def _get_size(self):
        return len(self.repr_)

    def get_size(self):
        return self.size

    def _set_size(self):
        self.size = self._get_size()

    # Visitation length
    def _get_visitation_length(self):
        '''Evaluate and return the tree visitation length, defined by
            Reference:
                Maarten Keijzer and James Foster. 2007. Crossover Bias in Genetic Programming. In Genetic Programming
                , Marc Ebner, Michael O’Neill, Aniko Ekárt, Leonardo Vanneschi, and Anna Isabel Esparcia-Alcázar (Eds.).
                Springer Berlin Heidelberg, Berlin, Heidelberg, 33–44.
        Parameters
        ----------

        Return
        ----------
            visitation_length: int
        '''
        return self.get_size() + sum(self.get_depths())

    def get_visitation_length(self):
        return self.visitation_length

    def _set_visitation_length(self):
        self.visitation_length = self._get_visitation_length()

    # Utils
    def get_subtree_indexes(self, repr_=None):
        ''' Getter of the indexes of the elements of the tree

        Parameters
        ----------
        repr_ : list
            A list of the elements of the tree.

        Return
        ----------
            start: int
            end: int
        '''
        if repr_ is None: repr_ = self.repr_
        start = 1
        stack = 1
        end = start
        while stack > end - start:
            node = repr_[end]
            if isinstance(node, _Function):
                stack += node.arity
            end += 1
        return start, end - 1

    def get_all_subtrees(self, repr_=None):
        ''' Getter of the subtrees of the tree given by repr_ arg.

        Parameters
        ----------
        repr_ : list
            A list of the elements of the tree.

        Return
        ----------
            subtrees: list
        '''
        if repr_ is None: repr_ = self.repr_
        subtrees = []
        start = 0
        size = len(repr_)
        for i_arity in range(repr_[0].arity):
            r = repr_[start:size]
            start, end = self.get_subtree_indexes(r)
            subtrees.append(repr_[start:end])
            start = end
        return subtrees

    def _evaluate_depth(self, repr_=None):
        ''' Evaluates the depth of the tree given by repr_ arg.

        Parameters
        ----------
        repr_ : list
            A list of the elements of the tree.

        Return
        ----------
            depth: int
        '''
        if repr_ is None: repr_ = self.repr_
        terminals = [0]
        depth = 1
        for node in repr_:
            if isinstance(node, _Function):
                terminals.append(node.arity)
                depth = max(len(terminals), depth)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return depth - 1

    def _evaluate_depths(self, repr_=None):
        ''' Evaluates the depth of all elements of the tree given by repr_ arg.

        Parameters
        ----------
        repr_ : list
            A list of the elements of the tree.

        Return
        ----------
            depths: list
        '''
        if repr_ is None: repr_ = self.repr_
        depths = []
        for i_el in range(len(repr_)):
            if isinstance(repr_[i_el], _Function):
                _subtrees = self.get_all_subtrees(repr_[i_el:])
                subtree_depths = [self._evaluate_depth(s) for s in _subtrees]
                depths.append(max(subtree_depths) + 1)
            else:
                depths.append(0)
        return depths

    def _evaluate_complexity(self, repr_=None):
        '''Evaluate and return the tree complexity adapted from Komenda et al. 2016
            complexity(n) =
                sum(
                    1                           = constant
                    2                           = variable
                    sum( complexity(c) )        = +, -
                    2 * prod( complexity(c) )   = *, /
                    2.5 * complexity(n)^2       = square
                    2.5 * complexity(n)^3       = squareroot
                    3 * 2^complexity(n)         = sin, cos, tan, exp, log
                )
                n = node
                c = each child of the node
            Reference:
                Kommenda, M., Kronberger, G., Affenzeller, M., Winkler, S.M., Burlacu, B. (2016).
                Evolving Simple Symbolic Regression Models by Multi-Objective Genetic Programming.
                In: Riolo, R., Worzel, W., Kotanchek, M., Kordon, A. (eds) Genetic Programming
                Theory and Practice XIII. Genetic and Evolutionary Computation. Springer, Cham.
                https://doi.org/10.1007/978-3-319-34223-8_1
        Parameters
        ----------

        Return
        ----------
            complexity_k: float
        '''
        # Definitions
        def _nao_prod(x):
            return 5 * prod(x)
        def _gs_operators(x):
            return 10 * prod(x)
        complexity = {
            'cte': 2,
            'feature': 3,
            'add': sum,
            'sub': sum,
            'mul': prod,
            'div': prod,
            'sin': _nao_prod,
            'cos': _nao_prod,
            'log': _nao_prod,
            'exp': _nao_prod,
            'tanh': _gs_operators,
            'lf': _gs_operators,
        }
        # Assess complexity
        if repr_ is None: repr_ = self.repr_
        repr_ = deepcopy(repr_)
        # Sets features complexity values
        repr_ = ['feature' if isinstance(el, int) else el for el in repr_]
        # Sets constants complexity values
        repr_ = ['cte' if isinstance(el, torch.Tensor) else el for el in repr_]
        # Set nested complexity values
        node = repr_[0]
        # Safe assessment for trees size 1
        if len(repr_) == 1:
            return complexity[node]
        # Or proceed with nested assessment for bigger trees
        apply_stack = []
        for node in repr_:
            if isinstance(node, _Function):
                apply_stack.append([node])
            else:
                apply_stack[-1].append(node)
            while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
                complexity_ = complexity[apply_stack[-1][0].name]
                terminals = [complexity[t] if isinstance(t, str) else t for t in apply_stack[-1][1:]]
                c = complexity_(terminals)
                intermediate_result = c if c < sys.maxsize else sys.maxsize
                if len(apply_stack) != 1:
                    apply_stack.pop()
                    apply_stack[-1].append(intermediate_result)
                else:
                    return intermediate_result

