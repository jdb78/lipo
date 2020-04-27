"""
The optimization module
"""
import math
from typing import Dict, List, Union, Callable, Tuple
import dlib


class EvaluationCandidate:
    def __init__(self, candidate, arg_names, categories, log_args, maximize):
        self.candidate = candidate
        self.arg_names = arg_names
        self.maximize = maximize
        self.log_args = log_args
        self.categories = categories

    @property
    def x(self):
        x = {}
        for name, val in zip(self.arg_names, self.candidate.x):
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
        function: Callable,
        lower_bounds: Dict[str, Union[float, int]],
        upper_bounds: Dict[str, Union[float, int]],
        categories: Dict[str, List[str]],
        log_args: Union[str, List[str]] = "auto",
        flexible_bound_threshold: float = 0.1,
        evaluations: List[Tuple[Dict[str], float]] = [],
        maximize: bool = True,
    ):
        """
        Init optimizer

        Args:
            function (callable):
            lower_bounds (Dict[str]): lower bounds of optimization, integer arguments are automatically inferred
            upper_bounds (Dict[str]): upper bounds of optimization, integer arguments are automatically inferred
            log_args (Union[str, List[str]): list of arguments to treat as if in log space, if "auto", then
                a variable is optimized in log space if

                - The lower bound on the variable is > 0
                - The ratio of the upper bound to lower bound is > 1000
                - The variable is not an integer variable

            flexible_bound_threshold (float): if to enlarge bounds if optimum is top or bottom
                ``flexible_bound_threshold`` quantile
            evaluations List[Tuple[Dict[str], float]]: list of tuples of x and y values
            maximize (bool): if to maximize or minimize (default ``True``)
        """
        self.function = function

        # check bounds
        assert len(lower_bounds) == len(upper_bounds), "Number of upper and lower bounds should be the same"
        for name in lower_bounds.keys():
            is_lower_integer = isinstance(lower_bounds[name], int)
            is_upper_integer = isinstance(upper_bounds[name], int)
            assert (is_lower_integer and is_upper_integer) or (
                not is_lower_integer and not is_upper_integer
            ), f"Argument {name} must be either integer or not integer"

        self.categories = categories
        self.arg_names = [list(upper_bounds.keys()) + list(self.categories.keys())]

        # set bounds
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        for name, cats in self.categories.items():
            self.lower_bounds[name] = 0
            self.upper_bounds[name] = len(cats) - 1

        # infer if variable is integer
        self.is_integer = {name: isinstance(self.lower_bounds.get(name, 0), int) for name in self.arg_names}

        # set arguemnts in log space
        if isinstance(log_args, str) and log_args == "auto":
            self.log_args = []
            for name in self.arg_names:
                if (
                    not self.is_integer[name]
                    and self.upper_bounds[name] / self.lower_bounds[name] > 1e3
                    and self.lower_bounds[name] > 0
                ):
                    self.log_args.append(name)
                    self.lower_bounds[name] = math.log(self.lower_bounds[name])
                    self.upper_bounds[name] = math.log(self.upper_bounds[name])
        else:
            self.log_args = log_args

        # check log args
        for name in self.log_args:
            assert not self.is_integer[name], f"Integer or categorical arguments such as {name} cannot be in log space"

        # convert initial evaluations
        self.init_evaluations = []
        for x, y in evaluations:
            e = {}
            for name, val in x.keys():
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
                [dlib.function_evaluation([x[0][name] for name in self.arg_names], x[1]) for x in self.evaluations]
            ],
            relative_noise_magnitude=0.001,
        )
        self.search.set_solver_epsilon(0.0)
        self.search.set_seed(214)

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

                    if (val - lower) / span <= self.flexible_bound_threshold:
                        # center value
                        self.lower_bounds[name] = val - (upper - val)
                        # redefine bounds
                        reinit = True
                    elif (upper - val) / span <= self.flexible_bound_threshold:
                        # center value
                        self.upper_bounds[name] = val + (val - lower)
                        # redefine bounds
                        reinit = True

                if reinit:  # reinitialize optimization with new bounds
                    self._init_search()

        return EvaluationCandidate(
            candidate=self.search.get_next_x(),
            arg_names=self.arg_names,
            categories=self.categories,
            maximize=self.maximize,
            log_args=self.log_args,
        )

    @property
    def evaluations(self):
        """
        evaluations (as initialized and carried out)

        Returns:
            List[Tuple[Dict[str], float]]]: list of x and y value pairs
        """
        if hasattr(self, "search"):
            evals = [
                ({name: e.x[name] for name in self.arg_names}, e.y)
                for e in self.search.get_function_evaluations()[1][0]
            ]
        else:
            evals = self.init_evaluations

        # convert log space and categories
        converted_evals = []
        for x, y in evals:
            e = {}
            for name, val in x.items():
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
