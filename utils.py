"""
Utility functions for fitting a parametric convex function to data
and exporting it to cvxpy etc.

M. Schaller, A. Bemporad, March 19, 2025
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax_sysid.utils import compute_scores
from dataclasses import dataclass
from typing import Callable
import warnings


@dataclass
class Activation:
    jax: Callable
    cvxpy: Callable
    convex_increasing: bool
    
    
@dataclass
class MakePositive:
    jax: Callable
    cvxpy: Callable


@dataclass
class Ind:
    start: int = 0
    end: int = 0


@dataclass
class WeightInd:
    w: Ind
    v: Ind
    o: Ind


@dataclass
class Section:
    start: int = 0
    end: int = 0
    shape: tuple = (0, 0)
    
    
def _append_section(section, offset, shape, size=None):
    """Add section in psi output p, such that p[start:end].reshape(shape)
    is a matrix or vector in main network; returns offset for next section"""
    
    if size is None:
        size = np.prod(shape)
    section.append(Section(start=offset, end=offset+size, shape=shape))
    return offset+size
    

def _rand(m, n):
    """Compute a ramdom array of shape (m, n) and entries in [-0.5, 0.5]"""
    return np.random.rand(m, n) - 0.5


def _unsqueeze(x):
    """If array has shape (n,), turn into array of shape (n, 1)"""
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x


def _compute_r2(Y, Yhat, return_msg=False):
    """Compute R2 score for true outputs Y and predicted outputs Yhat"""
    r2, _, msg = compute_scores(Y, Yhat, None, None, fit='R2')
    if return_msg:
        return r2, msg
    else:
        return r2


def _compute_acc(Y, Yhat, return_msg=False):
    """Compute accuracy for true labls Y and predicted labels Yhat"""
    acc, _, msg = compute_scores(Y==-1, Yhat<=0, None, None, fit='Accuracy')
    if return_msg:
        return acc, msg
    else:
        return acc


def _map_matmul(A, b):
    """Map matrix multiplication with JAX"""
    
    return jax.vmap(jnp.matmul)(A, b)


def _extract_activations(activation_registry, activation, activation_psi):
    """Extract activation functions from activation and activation_psi options"""
    
    # check that main activation is convex and increasing
    activation = activation.lower()
    if not activation_registry[activation].convex_increasing:
        raise ValueError('Activation function for variable network must'
                            'be convex and increasing.')
        
    # if not specified, make psi activation equal to main activation
    if activation_psi is None:
        activation_psi = activation
    else:
        activation_psi = activation_psi.lower()
    
    # extract main activations
    act_jax = activation_registry[activation].jax
    act_cvxpy = activation_registry[activation].cvxpy
    
    # extract psi activations
    act_psi_jax = activation_registry[activation_psi].jax
    act_psi_cvxpy = activation_registry[activation_psi].cvxpy
    
    return act_jax, act_cvxpy, act_psi_jax, act_psi_cvxpy


def _extract_monotonicity(increasing, decreasing):
    """Extract monotonicity multiplier from increasing/decreasing options"""
    
    if increasing and decreasing:
        warnings.warn("\033[1mFunction enforced to be both increasing and decreasing.\033[0m")
        return 0
    elif increasing:
        return 1
    elif decreasing:
        return -1
    else:
        return None
