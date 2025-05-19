"""
Configuration for fitting a parametric convex function to data
and exporting it to cvxpy etc.

M. Schaller, A. Bemporad, March 19, 2025
"""

import jax
import jax.numpy as jnp
import cvxpy as cp
from utils import Activation, MakePositive


# registry of activation functions, with their jax and cvxpy implementations

ACTIVATIONS = {
    'relu': Activation(
        jax = lambda x: jnp.maximum(0., x),
        cvxpy = lambda x: cp.maximum(0., x),
        convex_increasing = True
    ),
    'logistic': Activation(
        jax = lambda x: jnp.logaddexp(0., x),
        cvxpy = lambda x: cp.logistic(x),
        convex_increasing = True
    ),
    'leaky-relu': Activation(
        jax = lambda x: jnp.maximum(0.1*x, x),
        cvxpy = lambda x: cp.maximum(0.1*x, x),
        convex_increasing = True
    ),
    'swish': Activation(
        jax = lambda x: jax.nn.swish(x),
        cvxpy = lambda x: x / (1. + cp.exp(cp.minimum(-x, 100.))),
        convex_increasing = False
    )
}


# functions to make an output positive

MAKE_POSITIVE = MakePositive(
    jax = lambda W: jnp.maximum(W, 0.),
    cvxpy = lambda W: cp.maximum(W, 0.)
)
