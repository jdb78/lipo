import math
import numpy as np
from sklearn.base import RegressorMixin, BaseEstimator

from lipo import LIPOSearchCV


def test_hyperparameter():
    class Estimator(BaseEstimator, RegressorMixin):
        def __init__(self, x=0, y=2, z="b"):
            self.zdict = {"a": 1, "b": 2}
            self.x = x
            self.y = y
            self.z = z

        def fit(self, X, y):
            return self

        def predict(self, X):
            return (-(abs(math.exp(self.x) - 1.23)) + -((self.y - 0.3) ** 4) * self.zdict[self.z]) * np.ones(len(X))

        def score(self, X, y):
            return self.predict(X).mean()  # find maximum of function

    estimator = Estimator()
    search = LIPOSearchCV(estimator, param_space={"x": [0.001, 3.0], "y": [-10, 3], "z": ["a", "b"]}, n_iter=1000)
    search.fit(np.random.rand(1000, 3), np.zeros(1000))
    assert 0.24 > search.best_params_["x"] > 0.2
    assert search.best_params_["y"] == 0
    assert search.best_params_["z"] == "a"
