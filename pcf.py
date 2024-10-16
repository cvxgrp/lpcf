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
    'leaky-relu': {'jax': lambda x: jnp.maximum(0.1*x,x),  'cvxpy': lambda x: cp.maximum(0.1*x,x)},
}

class PCF:
    
    def __init__(self, widths_variable=2, widths_parameter=2, activation_variable='relu', activation_parameter='relu'):
        
        
        # initialize structure
        self.L_variable = len(widths_variable)
        self.widths_variable = widths_variable
        self.L_parameter = len(widths_parameter)
        self.widths_parameter = widths_parameter
        self.n_convex = None
        self.n_bias = None
        
        self.nx = None # inferred later by dataset
        self.nt = None # inferred later by dataset
        self.ny = None # inferred later by dataset
        
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
        
        weights_variable=[np.random.randn(self.widths_variable[0], self.nx)] # W1
        for i in range(1,self.L_variable):
            weights_variable.append(np.random.rand(self.widths_variable[i], self.widths_variable[i-1]))
            weights_variable.append(np.random.randn(self.widths_variable[i], self.nx))
        weights_variable.append(np.random.rand(self.ny, self.widths_variable[-1]))
        weights_variable.append(np.random.randn(self.ny, self.nx))

        weights_parameter=[np.random.randn(self.widths_parameter[0], self.nt), np.random.randn(self.widths_parameter[0], 1)]
        for i in range(1,self.L_parameter):
            weights_parameter.append(np.random.randn(self.widths_parameter[i], self.widths_parameter[i-1]))
            weights_parameter.append(np.random.randn(self.widths_parameter[i], self.nt))
            weights_parameter.append(np.random.randn(self.widths_parameter[i], 1))
        weights_parameter.append(np.random.randn(self.nbias, self.widths_parameter[-1]))
        weights_parameter.append(np.random.randn(self.nbias, self.nt))
        weights_parameter.append(np.random.randn(self.nbias, 1))

        self.model_weights = weights_variable + weights_parameter
        self.n_convex = len(weights_variable)
        self.n_bias = len(weights_parameter)
        return self.model_weights
            
    def _init_lower_bounds(self):
        self.weights_min = [-np.inf*np.ones(w.shape) for w in self.model_weights]
        for i in range(1,self.L_variable+1):
            self.weights_min[2*i-1] = np.zeros(self.weights_min[2*i-1].shape)
        return self.weights_min
    
    def _setup_model(self, seed=0):
        """Initialize variable and parameter networks."""
        
        @jax.jit
        def _parameter_fcn(theta, weights_parameter):
            # weights_parameter = [W1p,b1,W2z,W2p,b2,...,WKz,WKp,bK]   K = int((len(weights_parameter)-2)/3+1)
            b = self.act_param_jax(weights_parameter[0]@theta.T+weights_parameter[1])
            for j in range(self.L_parameter-1):
                b = self.act_param_jax(weights_parameter[2+3*j]@b+weights_parameter[3+3*j]@theta.T+weights_parameter[4+3*j])
            b = weights_parameter[-3]@b+weights_parameter[-2]@theta.T+weights_parameter[-1]
            return b.T
        
        @jax.jit
        def _variable_fcn(xtheta, weights):
            x = xtheta[:, :self.nx]
            theta = xtheta[:, self.nx:]
            # weights[:n_convex] = [W1x,W2z,W2x,...,WLz,WLx]  L_variable = int((len(weights_convex)-2)/3+1)
            weights_parameter = weights[self.n_convex:]
            b = _parameter_fcn(theta, weights_parameter)

            n1 = self.widths_variable[0]
            y = self.act_var_jax(weights[0] @ x.T + b[:,:n1].T)            
            for j in range(self.L_variable-1):
                n2 = self.widths_variable[j+1] + n1
                y = self.act_var_jax(weights[1+2*j] @ y + weights[2+2*j] @ x.T + b[:,n1:n2].T)
                n1 = n2
            y = weights[self.n_convex-2]@y+weights[self.n_convex-1]@x.T+ b[:,n1:].T
            return y.T
        
        self.model = StaticModel(self.ny, self.nx + self.nt, _variable_fcn)
        self.model.init(params=self._init_weights(seed))        
        self.model_weights_min = self._init_lower_bounds()        
    
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
            
        N, self.ny = Y.shape
        self.nx = X.shape[1]
        self.nt = Theta.shape[1]
        self.nbias = sum(self.widths_variable)+self.ny

        self._setup_model(seeds[0])
        
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
            models=self.model.parallel_fit(Y, XTheta, self._init_weights, seeds=seeds, n_jobs=cores)
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
        # Evaluate bias terms
        weights_parameter = self.model_weights[self.n_convex:]
        b = self.act_param_cvxpy(weights_parameter[0]@theta+weights_parameter[1])
        for j in range(self.L_parameter-1):
            b = self.act_param_cvxpy(weights_parameter[2+3*j]@b+weights_parameter[3+3*j]@theta+weights_parameter[4+3*j])
        b = weights_parameter[-3]@b+weights_parameter[-2]@theta+weights_parameter[-1]

        # Evaluate convex objective function(s)
        n1 = self.widths_variable[0]
        y = self.act_var_cvxpy(self.model_weights[0] @ x + b[:n1])            
        for j in range(self.L_variable-1):
            n2 = self.widths_variable[j+1] + n1
            y = self.act_var_cvxpy(self.model_weights[1+2*j] @ y + self.model_weights[2+2*j] @ x + b[n1:n2])
            n1 = n2
        y = self.model_weights[self.n_convex-2]@y+self.model_weights[self.n_convex-1]@x+ b[n1:]
        return y

    def tojax(self):
        @jax.jit
        def fcn_jax(x,theta,params):
            if x.ndim == 1:
                # single input
                x = x.reshape(-1, 1)
            if theta.ndim == 1:
                # single parameter
                theta = theta.reshape(-1, 1)
            xtheta = jnp.hstack((x,theta))
            return self.model.output_fcn(xtheta, params)
        return fcn_jax, self.model.params
