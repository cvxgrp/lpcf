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
    W_psi : int = 0
    V_psi : int = 0
    b_psi : int = 0


@dataclass
class Section:
    start : int = 0
    end : int = 0
    shape : tuple = (0, 0)


class PCF:
    
    def __init__(self, widths=None, widths_psi=None, activation='relu', activation_psi=None):
        
        # initialize structure, None values are inferred later via data dimenions
        
        self.widths = widths
        self.widths_psi = widths_psi
        self.L = len(widths) + 1 if widths else None
        self.L_psi = len(widths_psi) + 1 if widths_psi else None
        
        self.d, self.n, self.p, self.m = None, None, None, None
        self.section_W, self.section_V, self.section_omega = None, None, None
        
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
        self.weights_psi = None
        self.indices = None
        
        
    def _get_act(self, activation, interface):
        return ACTIVATIONS[activation.lower()][interface]
        
        
    def _init_weights(self, seed=0):
        
        np.random.seed(seed)
        
        W_psi = []
        V_psi = []
        b_psi = []
        for l in range(2, self.L_psi + 1): # W_psi1 does not exist
            W_psi.append(self._rand(self.widths_psi[l], self.widths_psi[l - 1]))
        for l in range(1, self.L_psi + 1):
            V_psi.append(self._rand(self.widths_psi[l], self.p))
            b_psi.append(self._rand(self.widths_psi[l], 1))
        
        indices = [0]
        for list_ in [W_psi, V_psi]:
            indices.append(indices[-1] + len(list_))
        self.indices = Indices(*indices)
        
        self.weights_psi = W_psi + V_psi + b_psi
        
        return self.weights_psi
    
    
    def _rand(self, first_dim, second_dim):
        return np.random.rand(first_dim, second_dim) - 0.5


    def _setup_model(self, seed=0):
        """Initialize variable and parameter networks."""
        
        @jax.jit
        def _psi_fcn(theta, weights_psi):
            W_psi = weights_psi[self.indices.W_psi:self.indices.V_psi]
            V_psi = weights_psi[self.indices.V_psi:self.indices.b_psi]
            b_psi = weights_psi[self.indices.b_psi:]
            out = self.act_psi_jax(V_psi[0] @ theta.T + b_psi[0])
            for j in range(1, self.L_psi - 1):
                jW = j - 1 # because W_psi1 does not exist
                out = self.act_psi_jax(W_psi[jW] @ out + V_psi[j] @ theta.T + b_psi[j])
            out = W_psi[-1] @ out + V_psi[-1] @ theta.T + b_psi[-1]
            start, end = self.section_W[0].start, self.section_W[-1].end
            out = out.at[start:end].set(jnp.maximum(out[start:end], 0))
            return out.T
        
        @jax.jit
        def _fcn(xtheta, weights_psi):
            x = xtheta[:, :self.n]
            theta = xtheta[:, self.n:]
            WVomega_flat = _psi_fcn(theta, weights_psi)
            W, V, omega = [], [], []
            for s in self.section_W:
                W.append(WVomega_flat[:, s.start:s.end].reshape((-1, *s.shape)))
            for s in self.section_V:
                V.append(WVomega_flat[:, s.start:s.end].reshape((-1, *s.shape)))
            for s in self.section_omega:
                omega.append(WVomega_flat[:, s.start:s.end].reshape((-1, *s.shape)))
            y = self.act_jax(jax.vmap(jnp.matmul)(V[0], x).T + jnp.squeeze(omega[0].T))
            for j in range(1, self.L - 1):
                jW = j - 1 # because W1 does not exist
                y = self.act_jax(jax.vmap(jnp.matmul)(W[jW], y.T).T + jax.vmap(jnp.matmul)(V[j], x).T + jnp.squeeze(omega[j].T))
            y = jax.vmap(jnp.matmul)(W[-1], y.T).T + jax.vmap(jnp.matmul)(V[-1], x).T + jnp.squeeze(omega[-1].T)
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
            seeds=np.atleast_1d(seeds)
                
        N, self.d = Y.shape
        self.n = X.shape[1]
        self.p = Theta.shape[1]
        
        if self.widths is None:
            self.widths = [self.n, self.d, self.d]
        else:
            self.widths = [self.n] + self.widths + [self.d]
        
        self.L = len(self.widths[1:])

        self.section_W = []
        self.section_V = []
        self.section_omega = []
        offset = 0
        for l in range(2, self.L + 1): # W_psi1 does not exist
            shape = (self.widths[l], self.widths[l - 1])
            size = np.prod(shape)
            self.section_W.append(Section(offset, offset + size, shape))
            offset += size
        for l in range(1, self.L + 1):
            shape = (self.widths[l], self.n)
            size = np.prod(shape)
            self.section_V.append(Section(offset, offset + size, shape))
            offset += size
        for l in range(1, self.L + 1):
            size = self.widths[l]
            self.section_omega.append(Section(offset, offset + size, (size, 1)))
            offset += size
        self.m = offset
        
        if self.widths_psi is None:
            self.widths_psi = [self.p, self.m, self.m]
        else:
            self.widths_psi = [self.p] + self.widths_psi + [self.m]
            
        self.L_psi = len(self.widths_psi[1:])

        self._setup_model(seeds[0])
        
        self.model.optimization(adam_epochs=adam_epochs, lbfgs_epochs=lbfgs_epochs)

        @jax.jit
        def output_loss(Yhat, Y):
            return jnp.sum((Yhat - Y)**2) / Y.shape[0]

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
        t = time.time() - t

        Yhat = self.model.predict(XTheta)
        R2, _, msg = compute_scores(Y, Yhat, None, None, fit='R2')

        self.weights_psi = self.model.params
        return {'time': t, 'R2': R2, 'msg': msg}
    
    
    def generate_psi(self) -> Callable:
        
        @jax.jit
        def psi(theta):
            W_psi = self.weights_psi[self.indices.W_psi:self.indices.V_psi]
            V_psi = self.weights_psi[self.indices.V_psi:self.indices.b_psi]
            b_psi = self.weights_psi[self.indices.b_psi:]
            
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
        WVomega_flat = cp.CallbackParam(lambda: psi(theta.value), (self.m, 1))
        W, V, omega = [], [], []
        for s in self.section_W:
            W.append(WVomega_flat[s.start:s.end].reshape(s.shape))
        for s in self.section_V:
            V.append(WVomega_flat[s.start:s.end].reshape(s.shape))
        for s in self.section_omega:
            omega.append(WVomega_flat[s.start:s.end].reshape(s.shape))

        # Evaluate convex objective function(s)
        y = self.act_cvxpy(V[0] @ x + omega[0]) 
        for j in range(1, self.L - 1):
            jW = j - 1 # because W1 does not exist
            y = self.act_cvxpy(W[jW] @ y + V[j] @ x + omega[j])
        y = W[-1] @ y + V[-1] @ x + omega[-1]
        return y


    def tojax(self):
        @jax.jit
        def fcn_jax(x, theta, params): # why do we need to pass params here?
            if x.ndim == 1:
                # single input
                x = x.reshape(-1, 1)
            if theta.ndim == 1:
                # single parameter
                theta = theta.reshape(-1, 1)
            xtheta = jnp.hstack((x, theta))
            return self.model.output_fcn(xtheta, params)
        return fcn_jax, self.model.params
