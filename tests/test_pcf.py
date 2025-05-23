"""
Testing PCF class of LPCF.

M. Schaller, A. Bemporad, March 23, 2025
"""

import pytest
import numpy as np
import cvxpy as cp
from lpcf.pcf import PCF
from lpcf.config import ACTIVATIONS


def test_pcf_initialization():
    """Test initialization of PCF class"""
    pcf = PCF()
    assert pcf.act_jax == ACTIVATIONS['relu'].jax
    assert pcf.act_cvxpy == ACTIVATIONS['relu'].cvxpy
    assert pcf.act_psi_jax == ACTIVATIONS['relu'].jax
    assert pcf.act_psi_cvxpy == ACTIVATIONS['relu'].cvxpy
    assert pcf.nonneg == False
    assert pcf.monotonicity == None
    assert pcf.quadratic == False
    assert pcf.quadratic_r == None
    assert pcf.classification == False
    assert pcf.force_argmin == False
    
    pcf = PCF(widths=[3, 2, 4], widths_psi=[2, 4, 3], activation='logistic', nonneg=True, increasing=True, quadratic=True)
    assert pcf.widths == [3, 2, 4]
    assert pcf.widths_psi == [2, 4, 3]
    assert pcf.act_jax == ACTIVATIONS['logistic'].jax
    assert pcf.act_cvxpy == ACTIVATIONS['logistic'].cvxpy
    assert pcf.act_psi_jax == ACTIVATIONS['logistic'].jax
    assert pcf.act_psi_cvxpy == ACTIVATIONS['logistic'].cvxpy
    assert pcf.nonneg == True
    assert pcf.monotonicity == 1
    assert pcf.quadratic == True
    assert pcf.quadratic_r == None
    assert pcf.classification == False
    assert pcf.force_argmin == False
    
def test_pcf_fit():
    """Test fit method of PCF class"""
    pcf = PCF()
    
    N = 100
    d, n, p = 2, 3, 2
    np.random.seed(0)
    Y = np.random.randn(N, d)
    X = np.random.randn(N, n)
    Theta = np.random.randn(N, p)
    
    with pytest.raises(ValueError):
        pcf.fit(Y, X, Theta, warm_start=True)
    
    result = pcf.fit(Y, X, Theta, adam_epochs=20, lbfgs_epochs=200)
    assert pcf.cache is not None
    assert result.keys() == {'time', 'R2', 'Accuracy', 'msg', 'lambda'}
    
def test_pcf_predict():
    """Test predict method of PCF class"""
    pcf = PCF()
    
    N = 100
    d, n, p = 2, 2, 2
    np.random.seed(0)
    Y = np.random.randn(N, d)
    X = np.random.randn(N, n)
    Theta = np.random.randn(N, p)
    
    pcf.fit(Y, X, Theta, adam_epochs=20, lbfgs_epochs=200)
    
    Yhat = pcf.predict(X, Theta)
    assert Yhat.shape == (N, d)
    
    pcf = PCF()
    
    N = 100
    d, n, p = 1, 3, 2
    np.random.seed(0)
    Y = np.random.randn(N, d)
    X = np.random.randn(N, n)
    Theta = np.random.randn(N, p)
    
    pcf.fit(Y, X, Theta, adam_epochs=20, lbfgs_epochs=200)
    
    Yhat = pcf.predict(X, Theta)
    assert Yhat.shape == (N, d)
    
    pcf = PCF()
    
    N = 100
    d, n, p = 2, 1, 1
    np.random.seed(0)
    Y = np.random.randn(N, d)
    X = np.random.randn(N, n)
    Theta = np.random.randn(N, p)
    
    pcf.fit(Y, X, Theta, adam_epochs=20, lbfgs_epochs=200)
    
    Yhat = pcf.predict(X, Theta)
    assert Yhat.shape == (N, d)
    
def test_pcf_tocvxpy():
    """Test tocvxpy method of PCF class"""
    pcf = PCF()
    
    N = 100
    d, n, p = 1, 3, 2
    np.random.seed(0)
    Y = np.random.randn(N, d)
    X = np.random.randn(N, n)
    Theta = np.random.randn(N, p)
    
    pcf.fit(Y, X, Theta, adam_epochs=20, lbfgs_epochs=200)
    
    x = cp.Variable((n, 1))
    theta = cp.Variable((p, 1))
    expr = pcf.tocvxpy(x, theta)
    assert expr.is_dcp()
    
def test_pcf_tojax():
    """Test tojax method of PCF class"""
    pcf = PCF()
    
    N = 100
    d, n, p = 1, 3, 2
    np.random.seed(0)
    Y = np.random.randn(N, d)
    X = np.random.randn(N, n)
    Theta = np.random.randn(N, p)
    
    pcf.fit(Y, X, Theta, adam_epochs=20, lbfgs_epochs=200)
    
    f = pcf.tojax()
    Yhat = f(X, Theta)
    assert Yhat.shape == (N, d)
