LIPO is a package for derivative-free, global optimization. Is based on
the `dlib` package and provides wrappers around its optimization routine.

Usage:

```python
from lipo import GlobalOptimizer

def function(x, y, z):
    zdict = {"a": 1, "b": 2}
    return -((x - 1.23) ** 6) + -((y - 0.3) ** 4) * zdict[z]

pre_eval_x = dict(x=2.3, y=13, z="b")
evaluations = [(pre_eval_x, function(**pre_eval_x))]

search = GlobalOptimizer(
    function,
    lower_bounds={"x": -10.0, "y": -10},
    upper_bounds={"x": 10.0, "y": -3},
    categories={"z": ["a", "b"]},
    evaluations=evaluations,
    maximize=True,
)

num_function_calls = 1000
search.run(num_function_calls)
```

The optimizer will automatically extend the search bounds if necessary.