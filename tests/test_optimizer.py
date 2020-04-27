import math
from lipo.optimizer import GlobalOptimizer


def test_global_optimization():
    def function(x, y, z):
        zdict = {"a": 1, "b": 2}
        return -(abs(math.exp(x) - 1.23)) + -((y - 0.3) ** 4) * zdict[z]

    test_x = dict(x=2.3, y=13, z="b")
    evaluations = [(test_x, function(**test_x))]

    search = GlobalOptimizer(
        function,
        lower_bounds={"x": 0.001, "y": -10},
        upper_bounds={"x": 3.0, "y": -3},  # require automatic extension of bounds
        categories={"z": ["a", "b"]},
        evaluations=evaluations,
        log_args=["x"],
        maximize=True,
    )

    num_function_calls = 1000
    search.run(num_function_calls)
    assert 0.24 > search.optimum[0]["x"] > 0.2
    assert search.optimum[0]["y"] == 0
    assert search.optimum[0]["z"] == "a"
