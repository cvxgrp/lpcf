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
    omega_psi: int = 0


@dataclass
class Section:
    start: int = 0
    end: int = 0
    shape: tuple = (0, 0)


class PCF:

    def __init__(self, widths=None, widths_psi=None,
                 activation='relu', activation_psi=None, nonneg=False) -> None:
        
        # initialize structure, None values inferred later via data dimenions
        
        self.widths, self.widths_psi = widths, widths_psi        
        self.w, self.w_psi = None, None
        self.L, self.M = None, None
        
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
        self.indices = None
        self.cache = None
        
        self.nonneg = nonneg

        self.force_argmin = False
        self.is_increasing=False
        self.is_decreasing=False
        self.is_monotonic=False
                
    def _get_act(self, activation, interface) -> Callable:
        return ACTIVATIONS[activation.lower()][interface]
    
    def _init_weights(self, seed=0, warm_start=False) -> List[np.ndarray]:
        if warm_start and self.cache is not None:
            return self.cache
        return self._rand_weights(seed)
        
    def _rand_weights(self, seed=0) -> List[np.ndarray]:
        
        np.random.seed(seed)
        
        W_psi = []
        V_psi = []
        omega_psi = []
        for l in range(2, self.M + 1):  # W_psi1 does not exist
            W_psi.append(self._rand(self.w_psi[l], self.w_psi[l-1]))
        for l in range(1, self.M + 1):
            V_psi.append(self._rand(self.w_psi[l], self.p))
            omega_psi.append(self._rand(self.w_psi[l], 1))
        
        indices = [0]
        for list_ in [W_psi, V_psi]:
            indices.append(indices[-1] + len(list_))
        self.indices = Indices(*indices)
                
        return W_psi + V_psi + omega_psi
    
    def _rand(self, first_dim, second_dim) -> np.ndarray:
        return np.random.rand(first_dim, second_dim) - 0.5

    def _setup_model(self, seed=0, warm_start=False) -> None:
        """Initialize variable and parameter networks."""
        
        @jax.jit
        def _make_positive(W):
            return MAKE_POSITIVE['jax'](W)
        
        @jax.jit
        def _psi_fcn(theta, weights):
            W_psi = weights[self.indices.W_psi:self.indices.V_psi]
            V_psi = weights[self.indices.V_psi:self.indices.omega_psi]
            omega_psi = weights[self.indices.omega_psi:]
            out = V_psi[0] @ theta.T + omega_psi[0]
            for j in range(1, self.M):
                jW = j - 1  # because W_psi1 does not exist
                out = self.act_psi_jax(out)
                out = W_psi[jW] @ out + V_psi[j] @ theta.T + omega_psi[j]
            W, V, omega = [], [], []
            for s in self.section_W:
                W.append(_make_positive(out[s.start:s.end].T.reshape((-1, *s.shape))))
            for s in self.section_V:
                if not self.is_monotonic:
                    V.append(out[s.start:s.end].T.reshape((-1, *s.shape)))
                else:
                    c = 1. if self.is_increasing else -1.
                    V.append(c*_make_positive(out[s.start:s.end].T.reshape((-1, *s.shape))))

            for s in self.section_omega:
                omega.append(out[s.start:s.end].T.reshape((-1, *s.shape)))

            return W, V, omega

        @jax.jit
        def _fcn(xtheta, weights):
            x = xtheta[:, :self.n]
            theta = xtheta[:, self.n:]
            W, V, omega = _psi_fcn(theta, weights)
            y = map_matmul(V[0], x) + omega[0]
            for j in range(1, self.L):
                jW = j - 1  # because W1 does not exist
                y = self.act_jax(y)
                y = map_matmul(W[jW], y) + map_matmul(V[j], x) + omega[j]
            if self.nonneg:
                y = jnp.maximum(y, 0.)
            return y
        
        self.model = StaticModel(self.d, self.n + self.p, _fcn)
        self.model.init(params=self._init_weights(seed, warm_start))

    def fit(self, Y, X, Theta, rho_th=1.e-8, tau_th=0., zero_coeff=1.e-4,
            seeds=None, cores=4, adam_epochs=200, lbfgs_epochs=2000,
            tune=False, n_folds=5, warm_start=None) -> Dict[str, float]:
        
        if Y.ndim == 1:
            # single output
            Y = Y.reshape(-1, 1)
        if X.ndim == 1:
            # single input
            X = X.reshape(-1, 1)
        if Theta.ndim == 1:
            # single parameter
            Theta = Theta.reshape(-1, 1)
            
        if warm_start and self.cache is None:
            raise ValueError('Trying to warm start before first training.')
        if warm_start is None:
            warm_start = self.cache is not None
            
        if seeds is None:
            seeds = 0 if warm_start else np.arange(max(10, cores))
        if not isinstance(seeds, np.ndarray):
            seeds = np.atleast_1d(seeds)
        
        self.N, self.d = Y.shape
        self.n = X.shape[1]
        self.p = Theta.shape[1]
        
        if self.widths is None:
            w_inner = 2 * ((self.n + self.d) // 2)
            self.w = [self.n, w_inner, w_inner, self.d]
        else:
            self.w = [self.n] + self.widths + [self.d]
        self.L = len(self.w[1:])

        self.section_W = []
        self.section_V = []
        self.section_omega = []
        offset = 0
        for l in range(2, self.L + 1):  # W_psi1 does not exist
            shape = (self.w[l], self.w[l - 1])
            size = np.prod(shape)
            self.section_W.append(Section(offset, offset + size, shape))
            offset += size
        for l in range(1, self.L + 1):
            shape = (self.w[l], self.n)
            size = np.prod(shape)
            self.section_V.append(Section(offset, offset + size, shape))
            offset += size
        for l in range(1, self.L + 1):
            size = self.w[l]
            self.section_omega.append(Section(offset, offset + size, (size,)))
            offset += size
        self.m = offset
        
        if self.widths_psi is None:
            w_inner = (self.p + self.m) // 2
            self.w_psi = [self.p, w_inner, w_inner, self.m]
        else:
            self.w_psi = [self.p] + self.widths_psi + [self.m]
        self.M = len(self.w_psi[1:])

        self._setup_model(seeds[0], warm_start)
        self.model.optimization(adam_epochs=adam_epochs, lbfgs_epochs=lbfgs_epochs)

        @jax.jit
        def output_loss(Yhat, Y):
            return jnp.sum((Yhat - Y)**2) / Y.shape[0]
        
        if self.force_argmin:
            @jax.jit
            def zero_grad_loss(params):
                return self.pcf_zero_grad_loss(params, Theta.reshape(self.N, self.p))
        else:
            zero_grad_loss = None
                            
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
                    self._fit_data(Y_train, XTheta_train, seeds, cores, warm_start)
                    score += self._compute_r2(Y_val, self.model.predict(XTheta_val.reshape(-1, self.n + self.p)))
                cv_scores[i] = score
            tau_th = tau_th_candidates[np.argmax(cv_scores)]
        
        self.model.loss(rho_th=rho_th, tau_th=tau_th, output_loss=output_loss, zero_coeff=zero_coeff, custom_regularization=zero_grad_loss)
        
        self._fit_data(Y, XTheta, seeds, cores, warm_start)
        t = time.time() - t
                
        Yhat = self.model.predict(XTheta)
        R2, _, msg = compute_scores(Y, Yhat, None, None, fit='R2')
        
        self.cache = self.model.params

        return {'time': t, 'R2': R2, 'msg': msg, 'lambda': tau_th}
    
    def _fit_data(self, Y, XTheta, seeds, cores, warm_start=False) -> None:
        if len(seeds) > 1:
            init_fun = lambda seed: self._init_weights(seed, warm_start)
            models = self.model.parallel_fit(Y, XTheta, init_fun, seeds=seeds, n_jobs=cores)
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
            W_psi = self.model.params[self.indices.W_psi:self.indices.V_psi]
            V_psi = self.model.params[self.indices.V_psi:self.indices.omega_psi]
            omega_psi = self.model.params[self.indices.omega_psi:]
            out = V_psi[0] @ theta + omega_psi[0]
            for j in range(1, self.M):
                jW = j - 1  # because W_psi1 does not exist
                out = self.act_psi_jax(out)
                out = W_psi[jW] @ out + V_psi[j] @ theta + omega_psi[j]
            return out.squeeze()
        
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
            W.append(_make_positive(WVomega_flat[s.start:s.end].reshape(s.shape, order='C')))  # enforce W weights to be nonnegative
        for s in self.section_V:
            if not self.is_monotonic:
                V.append(WVomega_flat[s.start:s.end].reshape(s.shape, order='C'))
            else:
                c = 1. if self.is_increasing else -1.
                V.append(c*_make_positive(WVomega_flat[s.start:s.end].reshape(s.shape, order='C')))   

        for s in self.section_omega:
            omega.append(WVomega_flat[s.start:s.end].reshape((-1, 1)))

        # Evaluate convex objective function(s)
        y = V[0] @ x + omega[0]
        for j in range(1, self.L):
            jW = j - 1  # because W1 does not exist
            y = self.act_cvxpy(y)
            y = W[jW] @ y + V[j] @ x + omega[j]
        if self.nonneg:
            y = cp.maximum(y, 0.)
        return y

    def tojax(self) -> Tuple[Callable, List[np.ndarray]]:
        @jax.jit
        def fcn_jax(x, theta):
            x = x.reshape(-1, self.n)
            theta = theta.reshape(-1, self.p)
            xtheta = jnp.hstack((x, theta))
            return self.model.output_fcn(xtheta, self.model.params)
        return fcn_jax

    def argmin(self, fun=None, penalty=1.e4):

        self.force_argmin = True
        
        if fun is None:
            @jax.jit
            def g(theta):
                return jnp.zeros(self.n)
        else:
            g = fun
        
        @jax.jit
        def pcf_model(x, theta, params):
            # Evaluate model output at x, theta
            y = self.model.output_fcn(jnp.hstack((x,theta)).reshape(1,self.n+self.p), params)[0][0]
            return y #penalty * jnp.sum(dY**2)            
        
        pcf_model_grad = jax.jit(jax.grad(pcf_model, argnums=0))
        @jax.jit
        def pcf_model_grad_g(theta,params):
            return pcf_model_grad(g(theta), theta, params)
        
        pcf_model_grad_g_vec = jax.vmap(pcf_model_grad_g, in_axes=(0,None))

        def pcf_zero_grad_loss(params, Theta):
            return penalty*jnp.sum(pcf_model_grad_g_vec(Theta, params)**2)/Theta.shape[0]
        
        self.pcf_zero_grad_loss = pcf_zero_grad_loss
        
    def increasing(self):
        self.is_increasing=True
        self.is_decreasing=False
        self.is_monotonic=True
        
    def decreasing(self):
        self.is_increasing=False
        self.is_decreasing=True
        self.is_monotonic=True
        