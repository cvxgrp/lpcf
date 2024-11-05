"""
Class for fitting a parametric convex function to data
and exporting it to cvxpy etc.

A. Bemporad, M. Schaller, October 15, 2024
"""

import time
import numpy as np
import cvxpy as cp
from typing import Callable, Dict, List, Tuple
from jax_sysid.models import StaticModel
from jax_sysid.utils import compute_scores
import jax.numpy as jnp
import jax
from dataclasses import dataclass

jax.config.update('jax_platform_name', 'cpu')
if not jax.config.jax_enable_x64:
    jax.config.update('jax_enable_x64', True)  # enable 64-bit computations
    
map_matmul = jax.vmap(jnp.matmul)


# registry of activation functions, with their jax and cvxpy implementations
ACTIVATIONS = {
    'relu': {
        'jax': lambda x: jnp.maximum(0., x),
        'cvxpy': lambda x: cp.maximum(0., x),
        'convex_increasing': True
        },
    'logistic': {
        'jax': lambda x: jnp.logaddexp(0., x),
        'cvxpy': lambda x: cp.logistic(x),
        'convex_increasing': True
        },
    'leaky-relu': {
        'jax': lambda x: jnp.maximum(0.1*x, x),
        'cvxpy': lambda x: cp.maximum(0.1*x, x),
        'convex_increasing': True
        },
    'swish': {
        'jax': lambda x: jax.nn.swish(x),
        'cvxpy': lambda x: x / (1. + cp.exp(cp.minimum(-x, 100.))),
        'convex_increasing': False
        },
}

MAKE_POSITIVE = {
    'jax': lambda W: jnp.maximum(W, 0.), #W**2,
    'cvxpy': lambda W: cp.maximum(W, 0.), #W**2,
}


@dataclass
class Indices:
    W_psi: int = 0
    V_psi: int = 0
    b_psi: int = 0


@dataclass
class Section:
    start: int = 0
    end: int = 0
    shape: tuple = (0, 0)


class PCF:

    def __init__(self, widths=None, widths_psi=None,
                 activation='relu', activation_psi=None) -> None:
        
        # initialize structure, None values inferred later via data dimenions
        
        self.widths = widths
        self.widths_psi = widths_psi
        self.L = len(widths) + 1 if widths else None
        self.L_psi = len(widths_psi) + 1 if widths_psi else None
        
        self.d, self.n, self.p, self.m, self.N = None, None, None, None, None
        self.section_W, self.section_V, self.section_omega = None, None, None
        
        if not ACTIVATIONS[activation]['convex_increasing']:
            raise ValueError('Activation function for variable network must'
                             'be convex and increasing.')
        
        self.act_jax = self._get_act(activation, 'jax')
        self.act_cvxpy = self._get_act(activation, 'cvxpy')
        
        if activation_psi is None:
            activation_psi = activation
        self.act_psi_jax = self._get_act(activation_psi, 'jax')
        self.act_psi_cvxpy = self._get_act(activation_psi, 'cvxpy')
        
        self.model = None
        self.weights = None
        self.indices = None
        
    def _get_act(self, activation, interface) -> Callable:
        return ACTIVATIONS[activation.lower()][interface]
        
    def _init_weights(self, seed=0) -> List[np.ndarray]:
        
        np.random.seed(seed)
        
        W_psi = []
        V_psi = []
        b_psi = []
        for l in range(2, self.L_psi + 1):  # W_psi1 does not exist
            W_psi.append(self._rand(self.widths_psi[l], self.widths_psi[l-1]))
        for l in range(1, self.L_psi + 1):
            V_psi.append(self._rand(self.widths_psi[l], self.p))
            b_psi.append(self._rand(self.widths_psi[l], 1))
        
        indices = [0]
        for list_ in [W_psi, V_psi]:
            indices.append(indices[-1] + len(list_))
        self.indices = Indices(*indices)
        
        self.weights = W_psi + V_psi + b_psi
        
        return self.weights
    
    def _rand(self, first_dim, second_dim) -> np.ndarray:
        return np.random.rand(first_dim, second_dim) - 0.5

    def _setup_model(self, seed=0) -> None:
        """Initialize variable and parameter networks."""
        
        @jax.jit
        def _make_positive(W):
            return MAKE_POSITIVE['jax'](W)
        
        @jax.jit
        def _psi_fcn(theta, weights):
            W_psi = weights[self.indices.W_psi:self.indices.V_psi]
            V_psi = weights[self.indices.V_psi:self.indices.b_psi]
            b_psi = weights[self.indices.b_psi:]
            out = self.act_psi_jax(V_psi[0] @ theta.T + b_psi[0])
            for j in range(1, self.L_psi - 1):
                jW = j - 1  # because W_psi1 does not exist
                out = self.act_psi_jax(W_psi[jW] @ out + V_psi[j] @ theta.T + b_psi[j])
            out = W_psi[-1] @ out + V_psi[-1] @ theta.T + b_psi[-1]
            W, V, omega = [], [], []
            for s in self.section_W:
                W.append(_make_positive(out[s.start:s.end].T.reshape((-1, *s.shape))))
            for s in self.section_V:
                V.append(out[s.start:s.end].T.reshape((-1, *s.shape)))
            for s in self.section_omega:
                omega.append(out[s.start:s.end].T.reshape((-1, *s.shape)))
            return W, V, omega

        @jax.jit
        def _fcn(xtheta, weights):
            x = xtheta[:, :self.n]
            theta = xtheta[:, self.n:]
            W, V, omega = _psi_fcn(theta, weights)
            y = self.act_jax(map_matmul(V[0], x) + omega[0])
            for j in range(1, self.L - 1):
                jW = j - 1  # because W1 does not exist
                y = self.act_jax(map_matmul(W[jW], y) + map_matmul(V[j], x) + omega[j])
            y = map_matmul(W[-1], y) + map_matmul(V[-1], x) + omega[-1]
            return y
        
        self.model = StaticModel(self.d, self.n + self.p, _fcn)
        self.model.init(params=self._init_weights(seed))

    def fit(self, Y, X, Theta, rho_th=1.e-8, tau_th=0., zero_coeff=1.e-4,
            seeds=None, cores=4, adam_epochs=200, lbfgs_epochs=2000,
            tune=False, n_folds=5) -> Dict[str, float]:
        
        if Y.ndim == 1:
            # single output
            Y = Y.reshape(-1, 1)
        if X.ndim == 1:
            # single input
            X = X.reshape(-1, 1)
        if Theta.ndim == 1:
            # single parameter
            Theta = Theta.reshape(-1, 1)
            
        if seeds is None:
            seeds = np.arange(max(10, cores))
        if not isinstance(seeds, np.ndarray):
            seeds = np.atleast_1d(seeds)
        
        self.N, self.d = Y.shape
        self.n = X.shape[1]
        self.p = Theta.shape[1]
        
        if self.widths is None:
            width_inner = 2 * ((self.n + self.d) // 2)
            self.widths = [self.n, width_inner, width_inner, self.d]
        else:
            self.widths = [self.n] + self.widths + [self.d]
        self.L = len(self.widths[1:])

        self.section_W = []
        self.section_V = []
        self.section_omega = []
        offset = 0
        for l in range(2, self.L + 1):  # W_psi1 does not exist
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
            self.section_omega.append(Section(offset, offset + size, (size,)))
            offset += size
        self.m = offset
        
        if self.widths_psi is None:
            width_inner = (self.p + self.m) // 2
            self.widths_psi = [self.p, width_inner, width_inner, self.m]
        else:
            self.widths_psi = [self.p] + self.widths_psi + [self.m]
        self.L_psi = len(self.widths_psi[1:])

        self._setup_model(seeds[0])
        self.model.optimization(adam_epochs=adam_epochs, lbfgs_epochs=lbfgs_epochs)

        @jax.jit
        def output_loss(Yhat, Y):
            return jnp.sum((Yhat - Y)**2) / Y.shape[0]
        
        XTheta = np.hstack((X.reshape(self.N, self.n), Theta.reshape(self.N, self.p)))
        
        t = time.time()
        if tune:
            tau_th_init = 1e-3 if tau_th == 0. else tau_th
            tau_th_candidates = [0.] + list(np.logspace(-2, 2, 5) * tau_th_init)
            f = int(np.ceil(self.N / n_folds))
            cv_scores = np.zeros_like(tau_th_candidates)
            for i, tau_th_candidate in enumerate(tau_th_candidates):
                self.model.loss(rho_th=rho_th, tau_th=tau_th_candidate, output_loss=output_loss, zero_coeff=zero_coeff)
                score = 0.
                for j in range(n_folds):
                    Y_train, XTheta_train = np.vstack((Y[:j*f], Y[(j+1)*f:])), np.vstack((XTheta[:j*f], XTheta[(j+1)*f:]))
                    Y_val, XTheta_val = Y[j*f:(j+1)*f], XTheta[j*f:(j+1)*f]
                    self._fit_data(Y_train, XTheta_train, seeds, cores)
                    score += self._compute_r2(Y_val, self.model.predict(XTheta_val.reshape(-1, self.n + self.p)))
                cv_scores[i] = score
            tau_th = tau_th_candidates[np.argmax(cv_scores)]
        self.model.loss(rho_th=rho_th, tau_th=tau_th, output_loss=output_loss, zero_coeff=zero_coeff)
        self._fit_data(Y, XTheta, seeds, cores)
        t = time.time() - t

        Yhat = self.model.predict(XTheta)
        R2, _, msg = compute_scores(Y, Yhat, None, None, fit='R2')

        self.weights = self.model.params
        return {'time': t, 'R2': R2, 'msg': msg, 'lambda': tau_th}
    
    def _fit_data(self, Y, XTheta, seeds, cores) -> None:
        if cores > 1:
            models = self.model.parallel_fit(Y, XTheta, self._init_weights, seeds=seeds, n_jobs=cores)
            R2s = [self._compute_r2(Y, m.predict(XTheta.reshape(-1, self.n + self.p))) for m in models]
            ibest = np.argmax(R2s)
            self.model.params = models[ibest].params
        else:
            self.model.fit(Y, XTheta)
    
    def _compute_r2(self, Y, Yhat) -> float:
        r2, _, _ = compute_scores(Y, Yhat, None, None, fit='R2')
        return r2
    
    def _generate_psi_flat(self) -> Callable:
        
        @jax.jit
        def psi(theta):
            W_psi = self.weights[self.indices.W_psi:self.indices.V_psi]
            V_psi = self.weights[self.indices.V_psi:self.indices.b_psi]
            b_psi = self.weights[self.indices.b_psi:]
            out = self.act_psi_jax(V_psi[0] @ theta + b_psi[0])
            for j in range(1, self.L_psi - 1):
                jW = j - 1  # because W_psi1 does not exist
                out = self.act_psi_jax(W_psi[jW] @ out + V_psi[j] @ theta + b_psi[j])
            out = W_psi[-1] @ out + V_psi[-1] @ theta + b_psi[-1]
            return out
        
        return psi
    
    def _generate_psi_flat_numpy_wrapper(self) -> Callable:
        
        psi_flat_jnp = self._generate_psi_flat()
        
        def psi_flat(theta):
            return np.array(psi_flat_jnp(jnp.array(theta)))
        
        return psi_flat
        
    def tocvxpy(self, x: cp.Variable, theta: cp.Parameter) -> cp.Expression:

        def _make_positive(W):
            return MAKE_POSITIVE['cvxpy'](W)
                
        psi_flat = self._generate_psi_flat_numpy_wrapper()
        WVomega_flat = cp.CallbackParam(lambda: psi_flat(theta.value), (self.m,))
        W, V, omega = [], [], []
        for s in self.section_W:
            W.append(_make_positive(WVomega_flat[s.start:s.end].reshape(s.shape)))  # enforce W weights to be nonnegative
        for s in self.section_V:
            V.append(WVomega_flat[s.start:s.end].reshape(s.shape))
        for s in self.section_omega:
            omega.append(WVomega_flat[s.start:s.end].reshape((-1, 1)))

        # Evaluate convex objective function(s)
        y = self.act_cvxpy(V[0] @ x + omega[0])
        for j in range(1, self.L - 1):
            jW = j - 1  # because W1 does not exist
            y = self.act_cvxpy(W[jW] @ y + V[j] @ x + omega[j])
        y = W[-1] @ y + V[-1] @ x + omega[-1]
        return y

    def tojax(self) -> Tuple[Callable, List[np.ndarray]]:
        @jax.jit
        def fcn_jax(x, theta, params):  # why do we need to pass params here?
            if x.ndim == 1:
                # single input
                x = x.reshape(-1, 1)
            if theta.ndim == 1:
                # single parameter
                theta = theta.reshape(-1, 1)
            xtheta = jnp.hstack((x, theta))
            return self.model.output_fcn(xtheta, params)
        return fcn_jax, self.model.params
