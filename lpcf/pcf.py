"""
Class for fitting a parametric convex function to data
and exporting it to cvxpy etc.

A. Bemporad, M. Schaller, October 15, 2024
"""

import time
import numpy as np
import cvxpy as cp
from typing import Callable, Dict, List
from jax_sysid.models import StaticModel
import jax
import jax.numpy as jnp
from lpcf.config import ACTIVATIONS, MAKE_POSITIVE
from lpcf.utils import Ind, WeightInd, _extract_activations, _extract_monotonicity, _append_section, \
    _rand, _unsqueeze, _map_matmul, _compute_r2, _compute_acc

jax.config.update('jax_platform_name', 'cpu')
if not jax.config.jax_enable_x64:
    jax.config.update('jax_enable_x64', True)  # enable 64-bit computations
    

class PCF:
    """A parametric convex function.

    Arguments
    ---------
    widths : array-like
        The widths of the main networks hidden layers.
    widths_psi : array-like
        The widths of the psi networks hidden layers.
    activation : str
        The activation function used in the main network.
    activation_psi : str
        The activation function used in the psi network.
    nonneg : bool
        Flag that indicates whether the PCF is nonnegative.
    increasing : bool
        Flag that indicates whether the PCF is increasing.
    decreasing : bool
        Flag that indicates whether the PCF is decreasing.
    quadratic : bool
        Flag that indicates whether the PCF contains a quadratic term x^T Q x.
    quadratic_r : bool
        Flag that indicates whether the quadratic term has low-rank plus diagonal structure,
        i.e., if Q = F^T F + D, where F is a wide matrix and D is a diagonal matrix.
        Takes effect only if quadratic=True.
    classification : bool
        Flag that indicates whether PCF is used for classification.
    """

    def __init__(self, widths=None, widths_psi=None, activation='relu', activation_psi=None,
                 nonneg=False, increasing=False, decreasing=False, quadratic=False, quadratic_r=None,
                 classification=False) -> None:
        
        # architecture
        self.widths, self.widths_psi = widths, widths_psi
        self.act_jax, self.act_cvxpy, self.act_psi_jax, self.act_psi_cvxpy = \
            _extract_activations(ACTIVATIONS, activation, activation_psi)
        
        # extensions
        self.nonneg = nonneg
        self.monotonicity = _extract_monotonicity(increasing, decreasing)
        self.quadratic = quadratic
        self.quadratic_r = quadratic_r
        self.classification = classification
        self.force_argmin = False
        
        # populated when fitting
        self.w, self.w_psi = None, None
        self.L, self.M = None, None
        self.d, self.n, self.p, self.m, self.N = None, None, None, None, None
        self.section_W, self.section_V, self.section_omega, self.section_quadratic = [], [], [], []
        self.model, self.weight_ind, self.cache = None, None, None


    def _rand_weights(self, seed=0) -> List[np.ndarray]:
        """Assign random weights"""
        
        np.random.seed(seed)
        
        W_psi = []
        V_psi = []
        omega_psi = []
        for l in range(2, self.M + 1):  # W_psi1 does not exist
            W_psi.append(_rand(self.w_psi[l], self.w_psi[l-1]))
        for l in range(1, self.M + 1):
            V_psi.append(_rand(self.w_psi[l], self.p))
            omega_psi.append(_rand(self.w_psi[l], 1))
                
        return W_psi + V_psi + omega_psi


    def _init_weights(self, seed=0, warm_start=False) -> List[np.ndarray]:
        """Initialize weights from cache if not empty, otherwise randomly"""
        
        if warm_start and self.cache is not None:
            return self.cache
        return self._rand_weights(seed)


    def _setup_model(self, seed=0, warm_start=False) -> None:
        """Initialize variable and parameter networks."""
        
        @jax.jit
        def _make_positive(W):
            return MAKE_POSITIVE.jax(W)
        
        @jax.jit
        def _psi_fcn(theta, weights):
            W_psi = weights[self.weight_ind.w.start:self.weight_ind.w.end]
            V_psi = weights[self.weight_ind.v.start:self.weight_ind.v.end]
            omega_psi = weights[self.weight_ind.o.start:self.weight_ind.o.end]
            out = V_psi[0] @ theta.T + omega_psi[0]
            for j in range(1, self.M):
                jW = j - 1  # because W_psi1 does not exist
                out = self.act_psi_jax(out)
                out = W_psi[jW] @ out + V_psi[j] @ theta.T + omega_psi[j]
            W, V, omega = [], [], []
            for s in self.section_W:
                W.append(_make_positive(out[s.start:s.end].T.reshape((-1, *s.shape))))
            for s in self.section_V:
                Vl = out[s.start:s.end].T.reshape((-1, *s.shape))
                V.append(Vl if self.monotonicity is None else self.monotonicity*_make_positive(Vl))

            for s in self.section_omega:
                omega.append(out[s.start:s.end].T.reshape((-1, *s.shape)))
                
            if self.quadratic:
                s = self.section_quadratic[0]
                if self.quadratic_r is None:
                    triu_indices = np.triu_indices(self.n)
                    def reshape_chol(c):
                        res = jnp.zeros((self.n, self.n))
                        return res.at[triu_indices].set(c)
                    factor = jax.vmap(reshape_chol)(out[s.start:s.end].T)
                    diag_sqrt = None
                else:
                    factor = out[s.start:s.end].T.reshape((-1, *s.shape))
                    s = self.section_quadratic[1]
                    diag_sqrt = out[s.start:s.end].T.reshape((-1, *s.shape))
            else:
                factor = None
                diag_sqrt = None

            return W, V, omega, factor, diag_sqrt

        @jax.jit
        def _fcn(xtheta, weights):
            x = xtheta[:, :self.n]
            theta = xtheta[:, self.n:]
            W, V, omega, factor, diag_sqrt = _psi_fcn(theta, weights)
            y = _map_matmul(V[0], x) + omega[0]
            for j in range(1, self.L):
                jW = j - 1  # because W1 does not exist
                y = self.act_jax(y)
                y = _map_matmul(W[jW], y) + _map_matmul(V[j], x) + omega[j]
            if self.quadratic:
                if self.quadratic_r is None:
                    y += jax.vmap(lambda x, C: jnp.sum((C @ x)**2))(x, factor).reshape(-1, 1)
                else:
                    y += jax.vmap(lambda x, C, d: jnp.sum((C @ x)**2) + jnp.sum((d * x)**2))(x, factor, diag_sqrt).reshape(-1, 1)
            if self.nonneg:
                y = MAKE_POSITIVE.jax(y)
            return y
        
        self.model = StaticModel(self.d, self.n + self.p, _fcn)
        self.model.init(params=self._init_weights(seed, warm_start))


    def _fit_data(self, Y, XTheta, seeds, cores, warm_start=False) -> None:
        """Regress Y on XTheta, possibly with multiple initial guesses"""
        
        if len(seeds) > 1:
            init_fun = lambda seed: self._init_weights(seed, warm_start)
            models = self.model.parallel_fit(Y, XTheta, init_fun, seeds=seeds, n_jobs=cores)
            R2s = [_compute_r2(Y, m.predict(XTheta.reshape(-1, self.n + self.p))) for m in models]
            ibest = np.argmax(R2s)
            self.model.params = models[ibest].params
        else:
            self.model.fit(Y, XTheta)


    def _tune(self, tau_th, rho_th, output_loss, Y, XTheta, seeds, cores, warm_start, zero_coeff, n_folds):
        """Auto-tune hyper-parameter tau_th via CV with n_folds"""
        
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
                Yhat = self.model.predict(XTheta_val.reshape(-1, self.n + self.p))
                if not self.classification:
                    s = _compute_r2(Y_val, Yhat)
                else:
                    s = _compute_acc(Y_val, Yhat)
                score += s
            cv_scores[i] = score
        return tau_th_candidates[np.argmax(cv_scores)]
    
    
    def _generate_psi_flat(self) -> Callable:
        """Take fitted model and generate psi network with flat (vector) output, in JAX"""
        
        @jax.jit
        def psi(theta):
            W_psi = self.model.params[self.weight_ind.w.start:self.weight_ind.w.end]
            V_psi = self.model.params[self.weight_ind.v.start:self.weight_ind.v.end]
            omega_psi = self.model.params[self.weight_ind.o.start:self.weight_ind.o.end]
            out = V_psi[0] @ theta + omega_psi[0]
            for j in range(1, self.M):
                jW = j - 1  # because W_psi1 does not exist
                out = self.act_psi_jax(out)
                out = W_psi[jW] @ out + V_psi[j] @ theta + omega_psi[j]
            return out.squeeze()
        
        return psi


    def _generate_psi_flat_numpy_wrapper(self) -> Callable:
        """Take fitted model and generate psi network with flat (vector) output,
        as a numpy function wrapped around a JAX function"""
        
        psi_flat_jnp = self._generate_psi_flat()
        
        def psi_flat(theta):
            return np.array(psi_flat_jnp(jnp.array(theta)))
        
        return psi_flat


    def fit(self, Y, X, Theta, rho_th=1.e-8, tau_th=0., zero_coeff=1.e-4,
            seeds=None, cores=4, adam_epochs=200, lbfgs_epochs=2000,
            tune=False, n_folds=5, warm_start=None) -> Dict[str, float]:
        """Fit the PCF to data.

        Arguments
        ---------
        Y : array
            Function outputs of shape (N, d).
        X : array
            Variables of shape (N, n).
        Theta : array
            Parameters shape (N, p).
        rho_th : float
            Regularization parameter in r(w) = rho_th ||w||_2^2 + tau_th ||w||_1.
        tau_th : float
            Regularization parameter in r(w) = rho_th ||w||_2^2 + tau_th ||w||_1.
        zero_coeff : float
            Entries smaller than zero_coeff in absolute value are set to zero after training. Useful when tau_th>0
        seeds : array-like
            Random seeds for training from multiple initial guesses.
        cores : int
            Number of cores for parallel training.
        adam_epochs: int
            Number of epochs for running ADAM.
        lbfgs_epochs : int
            Number of epochs for running L-BFGS-B.
        tune : bool
            Flag that indicates whether to auto-tune tau_th.
        n_folds : bool
            Number of cross-validation folds when auto-tuning tau_th.
            Takes effect only if tune=True.
        warm_start : 
            Flag that indicates whether to warm-start training.
        
        Returns
        -------
        Dict[str, float]
            Summary of PCF fitting, including time, scores, and hyper-parameter choice.
        """
        
        Y = _unsqueeze(Y)
        X = _unsqueeze(X)
        Theta = _unsqueeze(Theta)
            
        # warm-start if possible
        if warm_start and self.cache is None:
            raise ValueError('Trying to warm start before first training.')
        if warm_start is None:
            warm_start = self.cache is not None
            
        # if not warm-starting, choose # of initial guesses as 10 or number of cores, whichever greater
        if seeds is None:
            seeds = 0 if warm_start else np.arange(max(10, cores))
        if not isinstance(seeds, np.ndarray):
            seeds = np.atleast_1d(seeds)
        
        # overall dimension
        self.N, self.d = Y.shape
        self.n = X.shape[1]
        self.p = Theta.shape[1]
        
        # main network depth and width
        if self.widths is None:
            w_inner = 2 * ((self.n + self.d) // 2)
            self.w = [self.n, w_inner, w_inner, self.d]
        else:
            self.w = [self.n] + self.widths + [self.d]
        self.L = len(self.w[1:])

        # construct sections in flat psi output for slicing and reshaping the main network W's, V's, and omega's
        offset = 0
        for l in range(2, self.L + 1):  # W_psi1 does not exist
            offset = _append_section(self.section_W, offset, shape=(self.w[l], self.w[l - 1]))
        for l in range(1, self.L + 1):
            offset = _append_section(self.section_V, offset, shape=(self.w[l], self.n))
        for l in range(1, self.L + 1):
            offset = _append_section(self.section_omega, offset, shape=(self.w[l],))
        
        # EXTENSION: adding a quadratic term
        if self.quadratic:
            if self.quadratic_r is None:
                offset = _append_section(self.section_quadratic, offset, shape=(self.n, self.n), size=self.n*(self.n+1)//2)
            else:
                offset = _append_section(self.section_quadratic, offset, shape=(self.quadratic_r, self.n))
                offset = _append_section(self.section_quadratic, offset, shape=(self.n,))
        
        # psi network depth and width
        self.m = offset
        if self.widths_psi is None:
            w_inner = (self.p + self.m) // 2
            self.w_psi = [self.p, w_inner, w_inner, self.m]
        else:
            self.w_psi = [self.p] + self.widths_psi + [self.m]
        self.M = len(self.w_psi[1:])
        
        # start and end indices of W_psi, V_psi, omega_psi in list of weights used by jax-sysid
        self.weight_ind = WeightInd(
            w = Ind(start=0, end=self.M-1),
            v = Ind(start=self.M-1, end=2*self.M-1),
            o = Ind(start=2*self.M-1, end=3*self.M-1)
        )

        # setup using jax-sysid
        self._setup_model(seeds[0], warm_start)
        self.model.optimization(adam_epochs=adam_epochs, lbfgs_epochs=lbfgs_epochs)
        if not self.classification:
            @jax.jit
            def output_loss(Yhat, Y):
                return jnp.sum((Yhat - Y)**2) / Y.shape[0]
        else:
            labels = np.unique(Y)
            if not list(labels)==[-1,1]:
                raise Exception("Target data must only contain values -1 (feasible) and 1 (infeasible)")
            @jax.jit
            def output_loss(Yhat, Y):
                return jnp.sum(jnp.logaddexp(0.,-Y*Yhat)) / Y.shape[0]
            
        # EXTENSION: specifying a subgradient
        if self.force_argmin:
            @jax.jit
            def zero_grad_loss(params):
                return self.pcf_zero_grad_loss(params, Theta.reshape(self.N, self.p))
        else:
            zero_grad_loss = None
        
        # Stack (X, Theta) and fit
        XTheta = np.hstack((X.reshape(self.N, self.n), Theta.reshape(self.N, self.p)))
        t = time.time()
        if tune:
            tau_th = self._tune(
                rho_th=rho_th, tau_th=tau_th, output_loss=output_loss, zero_coeff=zero_coeff,
                Y=Y, XTheta=XTheta, seeds=seeds, cores=cores, warm_start=warm_start, n_folds=n_folds
            )
        self.model.loss(rho_th=rho_th, tau_th=tau_th, output_loss=output_loss, zero_coeff=zero_coeff, custom_regularization=zero_grad_loss)
        self._fit_data(Y, XTheta, seeds, cores, warm_start)
        t = time.time() - t
        
        # report scores on training data
        Yhat = self.model.predict(XTheta)
        R2, ACC = None, None
        if self.classification:
            ACC, msg = _compute_acc(Y, Yhat, return_msg=True)
        else:
            R2, msg = _compute_r2(Y, Yhat, return_msg=True)
        self.cache = self.model.params

        return {'time': t, 'R2': R2, 'Accuracy': ACC, 'msg': msg, 'lambda': tau_th}


    def tocvxpy(self, x, theta) -> cp.Expression:
        """Export the PCF to a CVXPY expression.

        Arguments
        ---------
        x : cp.Variable
            The CVXPY variable that will enter the PCF as x in f(x, theta).
        theta : cp.Parameter
            The CVXPY parameter that will enter the PCF as theta in f(x, theta).
        
        Returns
        -------
        cp.Expression
            A CVXPY expression equivalent to f(x, theta).
        """
        
        # expression for psi network
        psi_flat = self._generate_psi_flat_numpy_wrapper()
        WVomega_flat = cp.CallbackParam(lambda: psi_flat(theta.value), (self.m,))
        W, V, omega = [], [], []
        for s in self.section_W:
            W.append(MAKE_POSITIVE.cvxpy(WVomega_flat[s.start:s.end].reshape(s.shape, order='C')))  # enforce W weights to be nonnegative
        for s in self.section_V:
            Vl = WVomega_flat[s.start:s.end].reshape(s.shape, order='C')
            V.append(Vl if self.monotonicity is None else self.monotonicity*MAKE_POSITIVE.cvxpy(Vl))
        for s in self.section_omega:
            omega.append(WVomega_flat[s.start:s.end].reshape((-1, 1), order='C'))

        # expression for main network, using expression for psi network
        y = V[0] @ x + omega[0]
        for j in range(1, self.L):
            jW = j - 1  # because W1 does not exist
            y = self.act_cvxpy(y)
            y = W[jW] @ y + V[j] @ x + omega[j]
        if self.quadratic:
            s = self.section_quadratic[0]
            if self.quadratic_r is None:  # full quadratic, represented as x^T Q x, with Q = U^T U
                U = cp.vec_to_upper_tri(WVomega_flat[s.start:s.end])
                y += cp.sum_squares(U @ x)
            else:  # low-rank plus diagonal, Q = F^T F + D, F wide and D diagonal
                F = WVomega_flat[s.start:s.end].reshape(s.shape, order='C')
                s = self.section_quadratic[1]
                diag_sqrt = WVomega_flat[s.start:s.end].reshape((-1, 1), order='C')
                y += cp.sum_squares(F @ x) + cp.sum_squares(cp.multiply(diag_sqrt, x))
        if self.nonneg:
            y = MAKE_POSITIVE.cvxpy(y)
        return y


    def tojax(self) -> Callable:
        """Export the PCF to a JAX function.
        
        Returns
        -------
        Callable
            A JAX function equivalent to f(x, theta).
        """
        
        @jax.jit
        def fcn_jax(x, theta):
            x = x.reshape(-1, self.n)
            theta = theta.reshape(-1, self.p)
            xtheta = jnp.hstack((x, theta))
            return self.model.output_fcn(xtheta, self.model.params)
        return fcn_jax


    def predict(self, X, Theta) -> jnp.array:
        """Evaluate f(X, Theta).

        Arguments
        ---------
        X : array
            Variables of shape (N, n).
        Theta : array
            Parameters shape (N, p).
        
        Returns
        -------
        array
            PCF outputs of shape (N, d).
        """
        
        XTheta = jnp.hstack((X.reshape(-1, self.n), Theta.reshape(-1, self.p)))
        return self.model.predict(XTheta).reshape(-1, self.d)


    def argmin(self, fun=None, penalty=1.e4) -> None:
        """EXTENSION: Specifying a subgradient.

        Arguments
        ---------
        fun : Callable
            The function g such that the subgradient at g(theta) should be zero. 
        penalty : float
            Scale of regularization term in objective.
        """

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
