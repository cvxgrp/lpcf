# LPCF
LPCF stands for learning parametrized convex functions.
A parametrized convex function or PCF depends on a variable and a parameter,
and is convex in the variable for any valid value of the parameter.  

LPCF is a framework for fitting a 
parametrized convex function that is compatible with disciplined programming,
to some given data.
This allows to fit a function arising in a convex optimization
formulation directly to observed or simulated data.

The PCF is represented as a simple
neural network whose architecture is designed
to ensure disciplined convexity in the variable, for any valid
parameter value. After fitting this neural network to triplets
of observed (or simulated) values of the function, the variable,
and the parameter, the learned PCF can be exported for use in optimization
frameworks like [CVXPY](https://www.cvxpy.org) or [JAX](https://docs.jax.dev/en/latest/index.html).

An overview of LPCF can be found in our [manuscript](XXX).

## Installation
LPCF is available on PyPI, and can be installed with
```
pip install lpcf
```

LPCF has the following dependencies:

- Python >= 3.9
- jax-sysid >= 1.0.6
- CVXPY >= 1.6.0
- NumPy >= 1.21.6

## Example
The following code fits a PCF to observed function values `Y`,
variable values `X`, and parameter values `Theta`, and
exports the result to CVXPY.

```python3
from lpcf.pcf import PCF

# observed data
Y = ...      # shape (N, d)
X = ...      # shape (N, n)
Theta = ...  # shape (N, p)

# fit PCF to data
pcf = PCF()
pcf.fit(Y, X, Theta)

# export PCF to CVXPY
x = cp.Variable((n, 1))
theta = cp.Parameter((p, 1))
pcf_cvxpy = pcf.tocvxpy(x=x, theta=theta)
```
The CVXPY expression `pcf_cvxpy`
might appear, for example, in the objective of a CVXPY problem.


## Settings

### Neural network architecture
When constructing the `PCF` object, we allow for a number of
customizations to the neural network architecture:

| Argument         | Description                                                            | Type       | Default         |
| ---------------- | ---------------------------------------------------------------------- | ---------- | --------------- |
| `widths`         | widths of the main network's hidden layers                             | array-like | `[2((n+d)//2), 2((n+d)//2)]` |
| `widths_psi`     | widths of the parameter network's hidden layers                        | array-like | `[2((p+m)//2), 2((p+m)//2)]` |
| `activation`     | activation function used in the main network                           | str        | `'relu'`        |
| `activation_psi` | activation function used in the parameter network                      | str        | `'relu'`        |
| `nonneg`         | PCF nonnegative?                                                       | Bool       | `False`         |
| `increasing`     | PCF increasing?                                                        | Bool       | `False`         |
| `decreasing`     | PCF decreasing?                                                        | Bool       | `False`         |
| `quadratic`      | PCF containing quadratic term?                                         | Bool       | `False`         |
| `quadratic_r`    | PCF containing quadratic term with low-rank + diagonal structure?      | Bool       | `False`         |
| `classification` | PCF used for classification?                                           | Bool       | `False`         |

Note that `m` is the number of outputs of the parameter network.

### Learning configuration
When fitting the `PCF` to data with its `.fit()` method, we provide
the following options:

| Argument         | Description                                                            | Type       | Default         |
| ---------------- | ---------------------------------------------------------------------- | ---------- | --------------- |
| `rho_th`         | regularization hyper-parameter that scales the sum of squared weights  | float      | `1e-8`          |
| `tau_th`         | regularization hyper-parameter that scales the sum of absolute weights | float      | `0`             |
| `zero_coeff`     | zero coefficient for fitting the PCF                                   | float      | `1e-4`          |
| `cores`          | number of cores for parallel training                                  | int        | `4`             |
| `seeds`          | random seeds for training from multiple initial guesses                | array-like | `max(10, cores)`|
| `adam_epochs`    | number of epochs for running ADAM                                      | int        | `200`           |
| `lbfgs_epochs`   | number of epochs for running L-BFGS-B                                  | int        | `2000`          |
| `tune`           | auto-tune `tau_th`?                                                    | Bool       | `False`         |
| `n_folds`        | number of cross-validation folds when auto-tuning `tau_th`             | int        | `5`             |
| `warm_start`     | warm-start training?                                                   | Bool       | `False`         |
