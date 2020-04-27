import numpy as np
from lipo.optimizer import GlobalOptimizer


def test_global_optimization():
    function = lambda x, y: -((x - 1.23) ** 6) + -(y ** 4)
    evaluations = [
        (x.tolist(), function(*x))
        for x in np.array(np.meshgrid(np.linspace(-10, 10, 4), np.linspace(-10, 10, 5))).T.reshape(-1, 2)
    ]

    search = GlobalOptimizer(
        function, [-10.0, -10.0], [10.0, 10.0], [False, False], evaluations=evaluations, maximize=True
    )

    num_function_calls = 1000
    search.run(num_function_calls)

    assert 1.25 > search.optimum[0][0] > 1.2
    assert 0.1 > search.optimum[0][1] > -0.1
