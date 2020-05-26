"""
The optimization module
"""
from typing import Dict, List, Union, Callable, Tuple
import math
import logging

import dlib

logger = logging.getLogger(__name__)


class EvaluationCandidate:
    def __init__(self, candidate, arg_names, categories, log_args, maximize, is_integer):
        self.candidate = candidate
        self.arg_names = arg_names
        self.maximize = maximize
        self.log_args = log_args
        self.categories = categories
        self.is_integer = is_integer

    @property
    def x(self):
        x = {}
        for name, val in zip(self.arg_names, self.candidate.x):
            if self.is_integer[name]:
                val = int(val)
            if name in self.categories:
                x[name] = self.categories[name][val]
            elif name in self.log_args:
                x[name] = math.exp(val)
            else:
                x[name] = val
        return x

    def set(self, y):
        if not self.maximize:
            y = -y
        self.candidate.set(y)


class GlobalOptimizer:
    """
    Global optimizer that uses an efficient derivative-free method to optimize.

    See
    `LIPO algorithm implementation <http://dlib.net/python/index.html#dlib.find_max_global>`_. A good explanation of
    how it works, can be found here: `Here <http://blog.dlib.net/2017/12/a-global-optimization-algorithm-worth.html>`_
    """

    def __init__(
        self,
        function: Union[Callable, None] = None,
        lower_bounds: Dict[str, Union[float, int]] = {},
        upper_bounds: Dict[str, Union[float, int]] = {},
        categories: Dict[str, List[str]] = {},
        log_args: Union[str, List[str]] = "auto",
        flexible_bounds: Dict[str, List[bool]] = {},
        flexible_bound_threshold: float = -1.0,
        evaluations: List[Tuple[Dict[str, Union[float, int, str]], float]] = [],
        maximize: bool = True,
        epsilon=0.0,
        random_state=None,
    ):
        """
        Init optimizer

        Args:
            function (callable): function to optimize
            lower_bounds (Dict[str]): lower bounds of optimization, integer arguments are automatically inferred
            upper_bounds (Dict[str]): upper bounds of optimization, integer arguments are automatically inferred
            log_args (Union[str, List[str]): list of arguments to treat as if in log space, if "auto", then
                a variable is optimized in log space if

                - The lower bound on the variable is > 0
                - The ratio of the upper bound to lower bound is > 1000
                - The variable is not an integer variable
            flexible_bounds (Dict[str, List[bool]]): dictionary of parameters and list of booleans indicating
                if parameters are deemed flexible or not. by default all parameters are deemed flexible
            flexible_bound_threshold (float): if to enlarge bounds if optimum is top or bottom
                ``flexible_bound_threshold`` quantile
            evaluations List[Tuple[Dict[str], float]]: list of tuples of x and y values
            maximize (bool): if to maximize or minimize (default ``True``)
            epsilon (float): accuracy below which exploration will be priorities vs exploitation; default = 0
            random_state (Union[None, int]): random state
        """
        self.function = function
        self.epsilon = epsilon
        self.random_state = random_state

        # check bounds
        assert len(lower_bounds) == len(upper_bounds), "Number of upper and lower bounds should be the same"
        for name in lower_bounds.keys():
            is_lower_integer = isinstance(lower_bounds[name], int)
            is_upper_integer = isinstance(upper_bounds[name], int)
            assert (is_lower_integer and is_upper_integer) or (
                not is_lower_integer and not is_upper_integer
            ), f"Argument {name} must be either integer or not integer"
            assert (
                lower_bounds[name] < upper_bounds[name]
            ), f"Lower bound should be smaller than upper bound for argument {name}"

        self.categories = categories
        self.arg_names = list(upper_bounds.keys()) + list(self.categories.keys())

        # set bounds
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        for name, cats in self.categories.items():
            self.lower_bounds[name] = 0
            self.upper_bounds[name] = len(cats) - 1

        # infer if variable is integer
        self.is_integer = {name: isinstance(self.lower_bounds[name], int) for name in self.arg_names}

        # set arguemnts in log space
        if isinstance(log_args, str) and log_args == "auto":
            self.log_args = []
            for name in self.arg_names:
                if (
                    not self.is_integer[name]
                    and self.lower_bounds[name] > 0
                    and self.upper_bounds[name] / self.lower_bounds[name] > 1e3
                ):
                    self.log_args.append(name)
        else:
            self.log_args = log_args
        # transform bounds
        for name in self.log_args:
            assert name not in self.categories, f"Log-space is not defined for categoricals such as {name}"
            assert not self.is_integer[name], f"Log-space is not defined for integer variables such as {name}"
            assert self.lower_bounds[name] > 0, f"Log-space is only defined for positive lower bounds"
            self.lower_bounds[name] = math.log(self.lower_bounds[name])
            self.upper_bounds[name] = math.log(self.upper_bounds[name])

        # check log args
        for name in self.log_args:
            assert not self.is_integer[name], f"Integer or categorical arguments such as {name} cannot be in log space"

        # convert initial evaluations
        self.init_evaluations = []
        for x, y in evaluations:
            e = {}
            for name, val in x.items():
                if name in self.categories:
                    e[name] = self.categories[name].index(val)
                elif name in self.log_args:
                    e[name] = math.log(val)
                else:
                    e[name] = val
            self.init_evaluations.append((e, y))

        # if to maximize
        self.maximize = maximize

        # check bound threshold
        assert flexible_bound_threshold < 0.5, "Quantile for bound flexibility has to be below 0.5"
        self.flexible_bound_threshold = flexible_bound_threshold
        self.flexible_bounds = {name: flexible_bounds.get(name, [True, True]) for name in self.arg_names}

        # initialize search object
        self._init_search()

    def _init_search(self):
        function_spec = dlib.function_spec(
            [self.lower_bounds[name] for name in self.arg_names],
            [self.upper_bounds[name] for name in self.arg_names],
            [self.is_integer[name] for name in self.arg_names],
        )
        self.search = dlib.global_function_search(
            functions=[function_spec],
            initial_function_evals=[
                [dlib.function_evaluation([x[0][name] for name in self.arg_names], x[1]) for x in self._raw_evaluations]
            ],
            relative_noise_magnitude=0.001,
        )
        self.search.set_solver_epsilon(self.epsilon)
        if self.random_state is not None:
            self.search.set_seed(self.random_state)

    def get_candidate(self):
        """
        get candidate for evaluation

        Returns:
            EvaluationCandidate: candidate has property `x` for candidate kwargs and method `set` to
                inform the optimizer of the value
        """
        if self.flexible_bound_threshold >= 0:  # if to flexibilize bounds

            if len(self.search.get_function_evaluations()[1][0]) > 1 / (
                max(self.flexible_bound_threshold, 0.05)
            ):  # ensure sufficient evaluations have happened -> not more than 20
                reinit = False
                # check for optima close to bounds
                optimum_args = self.optimum[0]
                for name in self.arg_names:
                    if name in self.categories:
                        continue

                    lower = self.lower_bounds[name]
                    upper = self.upper_bounds[name]
                    span = upper - lower

                    if name in self.log_args:
                        val = math.log(optimum_args[name])
                    else:
                        val = optimum_args[name]

                    # redefine lower bound
                    if (val - lower) / span <= self.flexible_bound_threshold and self.flexible_bounds[name][0]:
                        # center value
                        proposed_val = val - (upper - val)
                        # limit change in log space
                        if name in self.log_args:
                            proposed_val = max(self.upper_bounds[name] - 2, proposed_val, -15)

                        if proposed_val < self.lower_bounds[name]:
                            self.lower_bounds[name] = proposed_val
                            # restart search
                            reinit = True

                    # redefine upper bound
                    elif (upper - val) / span <= self.flexible_bound_threshold and self.flexible_bounds[name][1]:
                        # center value
                        proposed_val = val + (val - lower)
                        # limit log space redefinition
                        if name in self.log_args:
                            proposed_val = min(self.upper_bounds[name] + 2, proposed_val, 15)

                        if proposed_val > self.upper_bounds[name]:
                            self.upper_bounds[name] = proposed_val
                            # restart search
                            reinit = True

                    if self.is_integer[name]:
                        self.lower_bounds[name] = int(self.lower_bounds[name])
                        self.upper_bounds[name] = int(self.upper_bounds[name])

                if reinit:  # reinitialize optimization with new bounds
                    logger.debug(f"resetting bounds to {self.lower_bounds} to {self.upper_bounds}")
                    self._init_search()

        return EvaluationCandidate(
            candidate=self.search.get_next_x(),
            arg_names=self.arg_names,
            categories=self.categories,
            maximize=self.maximize,
            log_args=self.log_args,
            is_integer=self.is_integer,
        )

    @property
    def _raw_evaluations(self):
        if hasattr(self, "search"):
            evals = [
                ({name: val for name, val in zip(self.arg_names, e.x)}, e.y)
                for e in self.search.get_function_evaluations()[1][0]
            ]
        else:
            evals = self.init_evaluations
        return evals

    @property
    def evaluations(self):
        """
        evaluations (as initialized and carried out)

        Returns:
            List[Tuple[Dict[str], float]]]: list of x and y value pairs
        """
        # convert log space and categories
        converted_evals = []
        for x, y in self._raw_evaluations:
            e = {}
            for name, val in x.items():
                if self.is_integer[name]:
                    val = int(val)
                if name in self.categories:
                    e[name] = self.categories[name][val]
                elif name in self.log_args:
                    e[name] = math.exp(val)
                else:
                    e[name] = val
            converted_evals.append((e, y))
        return converted_evals

    @property
    def optimum(self):
        """
        Current optimum

        Returns:
            Tuple[Dict[str], float, float]: tuple of optimal x, corresponding y and idx or optimal evaluation
        """
        x, y, idx = self.search.get_best_function_eval()
        new_x = {}
        for name, val in zip(self.arg_names, x):
            if self.is_integer[name]:
                val = int(val)
            if name in self.categories:
                new_x[name] = self.categories[name][val]
            elif name in self.log_args:
                new_x[name] = math.exp(val)
            else:
                new_x[name] = val

        return new_x, y, idx

    def run(self, num_function_calls: int = 1):
        """
        run optimization

        Args:
            num_function_calls (int): number of function calls
        """
        for _ in range(num_function_calls):
            candidate = self.get_candidate()
            candidate.set(self.function(**candidate.x))

    @property
    def running_optimum(self):
        """
        maximum by evaluation step

        Returns:
            list: value of optimum for each evaluation
        """
        optima = []
        for e in self.evaluations:
            if len(optima) == 0:
                optima.append(e[1])
            else:
                if self.maximize:
                    if optima[-1] > e[1]:
                        optima.append(optima[-1])
                    else:
                        optima.append(e[1])
                else:
                    if optima[-1] < e[1]:
                        optima.append(optima[-1])
                    else:
                        optima.append(e[1])
        return optima
