"""
Testing utility functions of LPCF.

M. Schaller, A. Bemporad, March 23, 2025
"""

import pytest
import numpy as np
import jax.numpy as jnp
from lpcf.config import ACTIVATIONS
from lpcf.utils import _rand, _unsqueeze, _map_matmul, _append_section, \
    _compute_r2, _compute_acc, _extract_activations, _extract_monotonicity


def test_rand():
    """Test random array generation"""
    m, n = 3, 4
    A = _rand(m, n)
    assert A.shape == (m, n)
    assert (A >= -0.5).all() and (A <= 0.5).all()
    
def test_unsqueeze():
    """Test unsqueeze function"""
    np.random.seed(0)
    x = np.random.rand(5)
    x_unsqueezed = _unsqueeze(x)
    assert x_unsqueezed.shape == (5, 1)
    
    y = np.random.rand(5, 2)
    y_unsqueezed = _unsqueeze(y)
    assert y_unsqueezed.shape == (5, 2)
    
def test_map_matmul():
    """Test matrix multiplication with broadcasting"""
    A = jnp.array([
        [[1.0, 2.0, 3.0],
         [4.0, 5.0, 6.0]],
        [[7.0, 8.0, 9.0],
         [10.0, 11.0, 12.0]]
    ])
    b = jnp.array([
        [[1.0],
         [0.0],
         [-1.0]],
        [[0.5],
         [0.5],
         [0.5]]
    ])
    expected = jnp.stack([
        jnp.matmul(A[0], b[0]),
        jnp.matmul(A[1], b[1])
    ])
    result = _map_matmul(A, b)
    assert jnp.allclose(result, expected)
    
def test_append_section():
    """Test appending sections"""
    sections = []
    offset = 0
    offset = _append_section(sections, offset, shape=(2, 3), size=None)
    assert sections[0].start == 0
    assert sections[0].end == 6
    assert sections[0].shape == (2, 3)
    
    offset = _append_section(sections, offset, shape=(5, 5), size=15)
    assert sections[1].start == 6
    assert sections[1].end == 21
    assert sections[1].shape == (5, 5)
    
def test_compute_r2():
    """Test R2 score computation"""
    np.random.seed(0)
    Y = np.random.rand(10, 1)
    Yhat = 1*Y
    r2 = _compute_r2(Y, Yhat)
    assert r2 == 100
    
    Yhat *= 0.0
    r2, msg = _compute_r2(Y, Yhat, return_msg=True)
    assert r2 <= 0
    assert isinstance(msg, str)
    
def test_compute_acc():
    """Test accuracy computation"""
    np.random.seed(0)
    Y = np.random.rand(10, 1)
    Yhat = np.sign(Y)
    acc = _compute_acc(Y, Yhat)
    assert acc == 100
    
    Yhat[:5] = -Yhat[:5]
    acc, msg = _compute_acc(Y, Yhat, return_msg=True)
    assert acc == 50
    assert isinstance(msg, str)
    
def test_extract_activations():
    """Test activation extraction"""
    activation = 'relu'
    activation_psi = 'logistic'
    
    act_jax, act_cvxpy, act_psi_jax, act_psi_cvxpy = _extract_activations(
        ACTIVATIONS, activation, activation_psi)
    
    assert act_jax == ACTIVATIONS[activation].jax
    assert act_cvxpy == ACTIVATIONS[activation].cvxpy
    assert act_psi_jax == ACTIVATIONS[activation_psi].jax
    assert act_psi_cvxpy == ACTIVATIONS[activation_psi].cvxpy
    
    activation = 'swish'
    activation_psi = 'relu'
    
    with pytest.raises(ValueError):
        _extract_activations(ACTIVATIONS, activation, activation_psi)
        
    activation = 'leaky-relu'
    activation_psi = None
    
    act_jax, act_cvxpy, act_psi_jax, act_psi_cvxpy = _extract_activations(
        ACTIVATIONS, activation, activation_psi)
    
    assert act_jax == ACTIVATIONS[activation].jax
    assert act_cvxpy == ACTIVATIONS[activation].cvxpy
    assert act_psi_jax == ACTIVATIONS[activation].jax
    assert act_psi_cvxpy == ACTIVATIONS[activation].cvxpy
    
def test_extract_monotonicity():
    """Test monotonicity extraction"""
    increasing = True
    decreasing = False
    mono = _extract_monotonicity(increasing, decreasing)
    assert mono == 1
    
    increasing = False
    decreasing = True
    mono = _extract_monotonicity(increasing, decreasing)
    assert mono == -1
    
    increasing = True
    decreasing = True
    with pytest.warns(UserWarning):
        mono = _extract_monotonicity(increasing, decreasing)
        assert mono == 0
        
    increasing = False
    decreasing = False
    mono = _extract_monotonicity(increasing, decreasing)
    assert mono is None
