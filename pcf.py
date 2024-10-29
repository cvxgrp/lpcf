"""
Class for fitting a parametric convex function to data
and exporting it to cvxpy etc.

A. Bemporad, M. Schaller, October 15, 2024
"""

import time
import numpy as np
import cvxpy as cp
from typing import Callable
from jax_sysid.models import StaticModel
from jax_sysid.utils import compute_scores
import jax.numpy as jnp
import jax
from dataclasses import dataclass

jax.config.update('jax_platform_name', 'cpu')
if not jax.config.jax_enable_x64:
    jax.config.update('jax_enable_x64', True)  # enable 64-bit computations


# registry of activation functions, with their jax and cvxpy implementations
ACTIVATIONS = {
    'relu':         {'jax': lambda x: jnp.maximum(0.,x),    'cvxpy': lambda x: cp.maximum(0.,x),                        'convex_increasing': True},
    'logistic':     {'jax': lambda x: jnp.logaddexp(0.,x),  'cvxpy': lambda x: cp.logistic(x),                          'convex_increasing': True},
    'leaky-relu':   {'jax': lambda x: jnp.maximum(0.1*x,x), 'cvxpy': lambda x: cp.maximum(0.1*x,x),                     'convex_increasing': True},
    'swish':        {'jax': lambda x: jax.nn.swish(x),      'cvxpy': lambda x: x/(1. + cp.exp(cp.minimum(-x, 100.))),   'convex_increasing': False},
}

@dataclass
class Indices:
    W : int = 0
    V : int = 0
    W_psi : int = 0
    V_psi : int = 0
    b_psi : int = 0

class PCF:
    
    def __init__(self, widths=None, widths_psi=None, activation='relu', activation_psi=None):
        
        # initialize structure, None values are inferred later via data dimenions
        
        self.widths = widths
        self.widths_psi = widths_psi
        self.L = len(widths) + 1 if widths else None
        self.L_psi = len(widths_psi) + 1 if widths_psi else None
        
        self.num_W_V = None
        self.d, self.n, self.p = None, None, None
        
        if not ACTIVATIONS[activation]['convex_increasing']:
            raise ValueError('Activation function for variable network must be convex and increasing.')
        
        self.act_jax = self._get_act(activation, 'jax')
        self.act_cvxpy = self._get_act(activation, 'cvxpy')
        
        if activation_psi is None:
            activation_psi = activation
        self.act_psi_jax = self._get_act(activation_psi, 'jax')
        self.act_psi_cvxpy = self._get_act(activation_psi, 'cvxpy')
        
        self.model = None
        self.weights = None
        self.indices = None
        
        
    def _get_act(self, activation, interface):
        return ACTIVATIONS[activation.lower()][interface]
        
        
    def _init_weights(self, seed=0):
        
        np.random.seed(seed)
        
        self.num_W_V = 2 * self.L - 1
        
        W = []
        V = []
        for l in range(2, self.L + 1): # W1 does not exist
            W.append(np.random.rand(self.widths[l], self.widths[l - 1]))
        for l in range(1, self.L + 1):
            V.append(np.random.rand(self.widths[l], self.n))

        W_psi = []
        V_psi = []
        b_psi = []
        for l in range(2, self.L_psi + 1): # W_psi1 does not exist
            W_psi.append(np.random.randn(self.widths_psi[l], self.widths_psi[l - 1]))
        for l in range(1, self.L_psi + 1):
            V_psi.append(np.random.randn(self.widths_psi[l], self.p))
            b_psi.append(np.random.randn(self.widths_psi[l], 1))
        
        indices = [0]
        for list_ in [W, V, W_psi, V_psi]:
            indices.append(indices[-1] + len(list_))
        self.indices = Indices(*indices)
        
        self.weights = W + V + W_psi + V_psi + b_psi
        
        return self.weights


    def _init_lower_bounds(self):
        min_W = [np.zeros(w.shape) for w in self.weights[:self.indices.V]]
        min_other = [-np.inf*np.ones(w.shape) for w in self.weights[self.indices.V:]]
        return min_W + min_other


    def _setup_model(self, seed=0):
        """Initialize variable and parameter networks."""
        
        @jax.jit
        def _psi_fcn(theta, weights_psi):
            W_psi = weights_psi[:self.L_psi-1]
            V_psi = weights_psi[self.L_psi-1:2*self.L_psi-1]
            b_psi = weights_psi[2*self.L_psi-1:]
            omega = self.act_psi_jax(V_psi[0] @ theta.T + b_psi[0])
            for j in range(1, self.L_psi - 1):
                jW = j - 1 # because W_psi1 does not exist
                omega = self.act_psi_jax(W_psi[jW] @ omega + V_psi[j] @ theta.T + b_psi[j])
            omega = W_psi[-1] @ omega + V_psi[-1] @ theta.T + b_psi[-1]
            return omega.T
        
        @jax.jit
        def _fcn(xtheta, weights):
            x = xtheta[:, :self.n]
            theta = xtheta[:, self.n:]
            W = weights[:self.indices.V]
            V = weights[self.indices.V:self.indices.W_psi]
            omega = _psi_fcn(theta, weights[self.indices.W_psi:])
            i1 = self.widths[1]
            y = self.act_jax(V[0] @ x.T + omega[:, :i1].T)            
            for j in range(1, self.L - 1):
                i2 = self.widths[j + 1] + i1
                jW = j - 1 # because W1 does not exist
                y = self.act_jax(W[jW] @ y + V[j] @ x.T + omega[:, i1:i2].T)
                i1 = i2
            y = W[-1] @ y + V[-1] @ x.T + omega[:, i1:].T
            return y.T
        
        self.model = StaticModel(self.d, self.n + self.p, _fcn)
        self.model.init(params=self._init_weights(seed))


    def fit(self, Y, X, Theta, rho_th=1.e-8, tau_th=1.e-3, zero_coeff=1.e-4, seeds=0, cores=1, adam_epochs=1000, lbfgs_epochs=1000):
        
        if Y.ndim == 1:
            # single output
            Y = Y.reshape(-1, 1)
        if X.ndim == 1:
            # single input
            X = X.reshape(-1, 1)
        if Theta.ndim == 1:
            # single parameter
            Theta = Theta.reshape(-1, 1)
        
        if not isinstance(seeds, np.ndarray):
            seeds=np.array(seeds)
                
        N, self.d = Y.shape
        self.n = X.shape[1]
        self.p = Theta.shape[1]
        
        if self.widths is None:
            self.widths = [self.n, self.d, self.d]
        else:
            self.widths = [self.n] + self.widths + [self.d]
        
        self.m = sum(self.widths[1:])
        
        if self.widths_psi is None:
            self.widths_psi = [self.p, self.m, self.m]
        else:
            self.widths_psi = [self.p] + self.widths_psi + [self.m]

        self._setup_model(seeds[0])
        
        self.model.optimization(
            adam_epochs=adam_epochs, lbfgs_epochs=lbfgs_epochs,
            params_min=self._init_lower_bounds(), params_max=None
        )

        @jax.jit
        def output_loss(Yhat, Y): 
            return jnp.sum((Yhat[:, :self.d] - Y[:, :self.d])**2) / Y.shape[0]

        # TODO: cross-validate over tau_th
        self.model.loss(rho_th=rho_th, tau_th=tau_th, output_loss=output_loss, zero_coeff=zero_coeff)

        t = time.time()
        XTheta = np.hstack((X.reshape(N, self.n), Theta.reshape(N, self.p)))
        if cores > 1:
            models=self.model.parallel_fit(Y, XTheta, self._init_weights, seeds=seeds, n_jobs=cores)
            R2s = [np.sum(compute_scores(Y, m.predict(XTheta.reshape(-1, self.n + self.p)), None, None, fit='R2')[0]) for m in models]
            ibest = np.argmax(R2s)
            self.model.params = models[ibest].params
        else:
            self.model.fit(Y, XTheta)
        t = time.time()-t

        Yhat = self.model.predict(XTheta)
        R2, _, msg = compute_scores(Y, Yhat, None, None, fit='R2')

        self.weights = self.model.params
        return {'time': t, 'R2': R2, 'msg': msg}
    
    
    def generate_psi(self) -> Callable:
        
        @jax.jit
        def psi(theta):
            W_psi = self.weights[self.indices.W_psi:self.indices.V_psi]
            V_psi = self.weights[self.indices.V_psi:self.indices.b_psi]
            b_psi = self.weights[self.indices.b_psi:]
            
            out = self.act_psi_jax(V_psi[0] @ theta + b_psi[0])
            for j in range(1, self.L_psi - 1):
                jW = j - 1 # because W_psi1 does not exist
                out = self.act_psi_jax(W_psi[jW] @ out + V_psi[j] @ theta + b_psi[j])
            out = W_psi[-1] @ out + V_psi[-1] @ theta + b_psi[-1]
            return out
        
        return psi
    
    
    def _generate_psi_numpy_wrapper(self) -> Callable:
        
        psi_jnp = self.generate_psi()
        def psi(theta):
            return np.array(psi_jnp(jnp.array(theta)))
        return psi
        
    
    def tocvxpy(self, x: cp.Variable, theta: cp.Parameter) -> cp.Expression:
                
        psi = self._generate_psi_numpy_wrapper()
        omega = cp.CallbackParam(lambda: psi(theta.value), (self.m, 1))

        W = self.weights[:self.indices.V]
        V = self.weights[self.indices.V:self.indices.W_psi]

        # Evaluate convex objective function(s)
        n1 = self.widths[1]
        y = self.act_cvxpy(V[0] @ x + omega[:n1]) 
        for j in range(1, self.L - 1):
            n2 = self.widths[j + 1] + n1
            jW = j - 1 # because W1 does not exist
            y = self.act_cvxpy(W[jW] @ y + V[j] @ x + omega[n1:n2])
            n1 = n2
        y = W[-1] @ y + V[-1] @ x + omega[n1:]
        return y


    def tojax(self):
        @jax.jit
        def fcn_jax(x, theta, params):
            if x.ndim == 1:
                # single input
                x = x.reshape(-1, 1)
            if theta.ndim == 1:
                # single parameter
                theta = theta.reshape(-1, 1)
            xtheta = jnp.hstack((x, theta))
            return self.model.output_fcn(xtheta, params)
        return fcn_jax, self.model.params
