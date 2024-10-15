"""
Class for fitting a parametric convex function to data
and exporting it to cvxpy etc.

A. Bemporad, M. Schaller, October 15, 2024
"""

import time
import numpy as np
import cvxpy as cp
from jax_sysid.models import StaticModel
from jax_sysid.utils import compute_scores
import jax.numpy as jnp
import jax

jax.config.update('jax_platform_name', 'cpu')
if not jax.config.jax_enable_x64:
    jax.config.update('jax_enable_x64', True)  # enable 64-bit computations


# registry of activation functions, with their jax and cvxpy implementations
ACTIVATIONS = {
    'relu':     {'jax': lambda x: jnp.maximum(0.,x),    'cvxpy': lambda x: cp.maximum(0.,x)},
    'logistic': {'jax': lambda x: jnp.logaddexp(0.,x),  'cvxpy': lambda x: cp.logistic(x)},
}

# TODO: replace hard-coded dimensions with parametric dimensions
n1,n2 = 10,10  # number of neurons in convex function
n1w, n2w = 5, 5
n_convex = 5 # number of weights in convex fcn
n_bias = 8


class PCF:
    
    def __init__(self, L, n_, K, m_, activation_variable='relu', activation_parameter='relu'):
        
        # check that n_ has length L and m_ has length L
        if len(n_) != L:
            raise ValueError('list of layer widths of variable network must have length L')
        if len(m_) != K:
            raise ValueError('list of layer widths of parameter network must have length K')
        
        # initialize structure
        self.L = L
        self.n_ = n_
        self.K = K
        self.m_ = m_
        
        self.nx = n_[0]
        self.nt = m_[0]
        self.ny = n_[-1]
        
        self.bias_dims = [[n1w, self.nt], [n1w,1], [n2w, n1w], [n2w, self.nt], [n2w,1], [n1+n2+self.ny, n2w],  [n1+n2+self.ny, self.nt], [n1+n2+self.ny,1]]
        
        self.act_var_jax = ACTIVATIONS[activation_variable]['jax']
        self.act_var_cvxpy = ACTIVATIONS[activation_variable]['cvxpy']
        self.act_param_jax = ACTIVATIONS[activation_parameter]['jax']
        self.act_param_cvxpy = ACTIVATIONS[activation_parameter]['cvxpy']
        
        self.model = None
        self.model_weights = None
        self.model_weights_min = None
        self.model_weights_max = None
        
        
    def _init_weights(self, seed=0):
        
        np.random.seed(seed)
        
        weights_variable = [
            np.random.randn(n1, self.nx),  # W1 
            np.random.rand(n2, n1),  # W2z (constrained >= 0)
            np.random.randn(n2, self.nx),  # W2u 
            np.random.rand(self.ny, n2),  # W3z (constrained >= 0)
            np.random.randn(self.ny, self.nx)  # W3u (this is unconstrained, as the last layer 
            ]
        weights_parameter = [np.random.randn(d[0], d[1]) for d in self.bias_dims]
        self.model_weights = weights_variable + weights_parameter
        return self.model_weights
            
    
    def _setup_model(self, seed=0):
        """Initialize variable and parameter networks."""
                
        @jax.jit
        def _parameter_fcn(theta, weights_parameter):
            W1, b1, W2w, W2p, b2, W3w, W3p, b3 = weights_parameter
            z1 = self.act_param_jax(W1 @ theta.T + b1)
            z2 = self.act_param_jax(W2w @ z1 + W2p @ theta.T + b2)
            b = W3w @ z2 + W3p @ theta.T + b3
            return b.T
        
        @jax.jit
        def _variable_fcn(xtheta, weights):
            x = xtheta[:, :self.nx]
            theta = xtheta[:, self.nx:]
            W1, W2z, W2u, W3z, W3u = weights[:n_convex]
            weights_parameter = weights[n_convex:]
            omega = _parameter_fcn(theta, weights_parameter)
            z1 = self.act_var_jax(W1 @ x.T + omega[:,:n1].T)
            z2 = self.act_var_jax(W2z @ z1 + W2u @ x.T + omega[:,n1:n1+n2].T)
            y = W3z @ z2 + W3u @ x.T + omega[:,-self.ny:].T
            return y.T
        
        self.model = StaticModel(self.ny, self.nx + self.nt, _variable_fcn)
        self.model.init(params=self._init_weights(seed))
        
        params_convex_min = [-np.inf*np.ones((n1, self.nx)), np.zeros((n2,n1)), -np.inf*np.ones((n2,self.nx)), 
                    np.zeros((self.ny,n2)), -np.inf*np.ones((self.ny,self.nx))]
        params_bias_min = [-np.inf*np.ones((d[0], d[1])) for d in self.bias_dims]
        
        self.model_weights_min = params_convex_min + params_bias_min
        
    
    def fit(self, Y, X, Theta, rho_th=1.e-8, tau_th=1.e-3, zero_coeff=1.e-4, cores=1, adam_epochs=1000, lbfgs_epochs=2000, seed=0):
        
        N = Y.shape[0]
        
        self._setup_model(seed)
        
        self.model.optimization(
            adam_epochs=adam_epochs, lbfgs_epochs=lbfgs_epochs,
            params_min=self.model_weights_min, params_max=self.model_weights_max
        )

        @jax.jit
        def output_loss(Yhat,Y): 
            return jnp.sum((Yhat[:,:self.ny]-Y[:,:self.ny])**2)/Y.shape[0]

        # TODO: cross-validate over tau_th
        self.model.loss(rho_th=rho_th, tau_th=tau_th, output_loss=output_loss, zero_coeff=zero_coeff)

        t = time.time()
        XTheta = np.hstack((X.reshape(N, self.nx), Theta.reshape(N, self.nt)))
        if cores > 1:
            models=self.model.parallel_fit(Y, XTheta, self._init_weights, seeds=range(cores), n_jobs=cores)
            R2s = [np.sum(compute_scores(Y, m.predict(XTheta.reshape(-1, self.nx + self.nt)), None, None, fit='R2')[0]) for m in models]
            ibest = np.argmax(R2s)
            self.model.params = models[ibest].params
        else:
            self.model.fit(Y, XTheta)
        t = time.time()-t

        Yhat = self.model.predict(XTheta)
        R2, _, msg = compute_scores(Y, Yhat, None, None, fit='R2')

        self.model_weights = self.model.params
        return {'time': t, 'R2': R2, 'msg': msg}
        
    
    def tocvxpy(self, x: cp.Variable, theta: cp.Parameter) -> cp.Expression:
        
        W1b, b1b, W2wb, W2pb, b2b, W3wb, W3pb, b3b = self.model_weights[-n_bias:]
        z1 = self.act_param_cvxpy(W1b @ theta + b1b)
        z2 = self.act_param_cvxpy(W2wb @ z1 + W2pb @ theta + b2b)
        omega = W3wb @ z2 + W3pb @ theta + b3b
        
        W1, W2z, W2u, W3z, W3u = self.model_weights[:n_convex]
        z1 = self.act_var_cvxpy(W1 @ x + omega[:n1])
        z2 = self.act_var_cvxpy(W2z @ z1 + W2u @ x + omega[n1:n1+n2])
        y = W3z @ z2 + W3u @ x + omega[-self.ny:]
        
        return y

    
    def tojax(self):
        
        # return existing jax function
        pass
