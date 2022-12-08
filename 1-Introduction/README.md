# Introduction to Python



## Exercise 6 - Class 1

Given the base class:
```python
class Variable:
    def __init__(self, name):
        self.name = name

    def sample(self, size):
        raise NotImplementedError()
```

Implement an inherited `Normal` class which outputs a list of normal samples [mu=0, sigma=1] by overriding the `Variable.sample` method.

## Exercise 7 - Class 2

Construct a class which constructs and evaluates a 1D polynomial model with the following API:
- the class constructor must take the polynomial degree as argument.
- implement a `set_parameters` and `get_parameters` methods to update the parameter list.
- provide an `execute` method to access the polynomial prediction at a specific value in `x`.