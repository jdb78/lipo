import logging

import numpy as np

from tqdm.autonotebook import tqdm
from sklearn.model_selection._search import BaseSearchCV
from sklearn.utils.validation import _deprecate_positional_args

from lipo.optimizer import GlobalOptimizer


logger = logging.getLogger(__name__)


class LIPOSearchCV(BaseSearchCV):
    """
    Global hyperparameter search that wraps the
    `MaxLIPO+TR algorithm <http://dlib.net/python/index.html#dlib.find_max_global>`_.
    A good explanation of how is works can be found in this
    `blog post <http://blog.dlib.net/2017/12/a-global-optimization-algorithm-worth.html>`_

    The class follows the standard scikit-learn API
    """

    _required_parameters = ["estimator", "param_space"]

    @_deprecate_positional_args
    def __init__(
        self,
        estimator,
        param_space,
        n_iter=10,
        flexible_bound_threshold=-1.0,
        flexible_bounds={},
        log_args="auto",
        tolerance=0.0,
        random_state=None,
        scoring=None,
        n_jobs=None,
        iid="deprecated",
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score=False,
    ):
        """
        Initialize self

        Args:
            estimator (BaseEstimator): estimator to tune
            param_space (Dict[str, List]): dictionary where keys are names of parameters to tune and
                values are either
                
                - List of form [minimum, maximum].
                  If minimum and maximum are integers, search will be limited to integers.
                  Optimization in log space happens automatically, if the bounds satisfy all these conditions:

                      - The lower bound on the variable is > 0
                      - The ratio of the upper bound to lower bound is > 1000
                      - The variable is not an integer variable

                - List of choices to test. List must either contain more than 2 elements or only strings.

            n_iter (int): number of iterations for fitting the estimator
            flexible_bounds (Dict[str, List[bool]]): dictionary of parameters and list of booleans indicating
                if parameters are deemed flexible or not. by default all parameters are deemed flexible
            flexible_bound_threshold (float): if to enlarge bounds if optimum is top or bottom
                ``flexible_bound_threshold`` quantile
            log_args (Union[str, List[str]]): list of parameters to be optimized in log space, defaults to rules
                as described in ``param_space``. Integer variables cannot be in log space.
            tolerance (float): Skip local search step if accuracy of found maximum is below tolerance.
                Continue with global search.
            scoring (Union[str, callable, List, Tuple, Dict, None]: as in sklearn.model_selection.GridSearchCV
            n_jobs (int): number of jobs for cross validation
            iid (bool): deprecated
            refit (bool): if to refit estimator with best parameters at the end
            cv (Union[int, iterable]): number of folds or iterable returning indices of (train, test)
            verbose (int): verbosity level
            pre_dispatch (Union[int, str]): number of jobs to dispatch for fitting (pre-dispatching can speed up
                code)
            error_score (Union[str, float]): value for score if it cannot be calculated
            return_train_score (bool): if to add training scores
        """
        self.param_space = param_space
        self.n_iter = n_iter
        self.tolerance = tolerance
        self.flexible_bounds = flexible_bounds
        self.flexible_bound_threshold = flexible_bound_threshold
        self.random_state = random_state
        self.log_args = log_args

        super().__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            iid=iid,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )

    def _run_search(self, evaluate_candidates):
        # initialize optimizer
        lower_bounds = {}
        upper_bounds = {}
        categories = {}
        for name, space in self.param_space.items():
            if len(space) == 2 and isinstance(space[0], (int, float)) and isinstance(space[1], (int, float)):
                lower_bounds[name] = space[0]
                upper_bounds[name] = space[1]
            else:
                categories[name] = space

        optimizer = GlobalOptimizer(
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            categories=categories,
            flexible_bound_threshold=self.flexible_bound_threshold,
            flexible_bounds=self.flexible_bounds,
            log_args=self.log_args,
            epsilon=self.tolerance,
            maximize=True,
            random_state=self.random_state,
        )
        if self.multimetric_:
            metric = self.refit
        else:
            metric = "score"

        iteration = 0
        t = tqdm(total=self.n_iter, desc="Best params at iteration --- with score ----", mininterval=10.0)
        try:
            while iteration < self.n_iter:
                candidate = optimizer.get_candidate()
                if self.verbose:
                    logger.debug(f"running iteration {iteration} with candidate {candidate.x}")

                scores = evaluate_candidates([candidate.x])["mean_test_%s" % metric]
                candidate.set(scores[-1])
                best_i_candidate = np.argmax(scores)
                if self.verbose:
                    logger.debug(
                        f"iteration {iteration} has score {scores[iteration]}"
                        f"({scores[iteration] / scores[best_i_candidate]:.2%} of best candidate at "
                        f"iteration {best_i_candidate} with score {scores[best_i_candidate]})"
                    )
                iteration += 1
                t.set_description(
                    f"Best params with score {scores[best_i_candidate]:.2g} at iteration {best_i_candidate}"
                )
                t.update()

        except KeyboardInterrupt:  # tuning can be stopped without stopping entire program
            message = "KeyboardInterrupt - stopping hyperparameter tuning - interrupt again to stop program"
            print(message)
            logger.warning(message)
