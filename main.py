"""
Fit a parametric convex function to data.

A. Bemporad, October 13, 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax_sysid.utils import compute_scores
import cvxpy as cp
import jaxopt
from functools import partial
from pcf import PCF

jax.config.update('jax_platform_name', 'cpu')
if not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)  # Enable 64-bit computations

# ################################
plotfigs = True # set to True to plot figures
seeds = np.arange(10) # seeds to use for initial condition during training
cores=10 # number of cores to use
N = 5000 # number of training data points
# M=1000 # additional active learning samples (0=no active learning)
M=0 # no active learning
weight_a = 10. # weight on new active samples (only used if M>0)
tau_th = 0.*0.001 # L1-regularization term
zero_coeff = 1.e-4 # small coefficients are set to zero when L1-regularization is used
# input convex function model [Amos, Xu, Kolter, 2017]
def act(x):    
    return jnp.logaddexp(0.,x) # = log(1+exp(x)): activation function, must be convex and non decreasing on the domain of interest
ny = 1 # number of outputs
seed = 4 # for reproducibility of results
# ################################

np.random.seed(4)

# parametric function to approximate
@jax.jit
def f(U1,U2,P):
    Y = jnp.arctan(jnp.sin((U1**2+U2**2+P*U1-2*P*U2)/(4.+P)) + jnp.exp(-((U1-1.-P)**2+2.*(2.+P)*(U2-1.+P/2.)**2)))
    return Y

nu = 2
npar = 1

X = np.random.rand(N,nu+npar)*np.array([4.,4.,2.])+np.array([-2.,-2.,-1.]) # samples of variables + parameters 
U1, U2, P = X.T
Y = f(U1,U2,P)

#! TBD: add example where data are standard scaled first

U = np.vstack((U1,U2)).T

def oracle(U,P):
    return f(U[:,0],U[:,1],P)
   
nx = nu+npar  # number of inputs

pcf = PCF(widths_variable=[10,10], widths_parameter=[5,5], activation_variable='logistic', activation_parameter='swish')
stats = pcf.fit(Y, U, P, tau_th=tau_th, zero_coeff=zero_coeff, seeds=seeds, cores=cores, adam_epochs=100, lbfgs_epochs=100)

f_jax, weights = pcf.tojax() # get the jax function and parameters: y = f_jax(x,theta,params)
YHAT = f_jax(U, P, weights) # predict the output for the training data
# YHAT2 = pcf.model.output_fcn(X,pcf.model.params) = YHAT

print(f"Elapsed time: {stats['time']} s")
print(f"R2 score on (u,p) -> y mapping:         {stats['R2']}")
    
# #########################
# Convexity check in CVXPY
x_cvx = cp.Variable((nu, 1))
#theta_cvx = cp.Parameter((npar, 1))
f_cvx, param_cvx, transform_param = pcf.tocvxpy(x_cvx)
print(f'cvxpy expressions is {"DCP" if f_cvx.is_dcp() else "non-DCP"}')
print(f'cvxpy expressions is {"DPP" if f_cvx.is_dpp() else "non-DPP"}')
# #########################

# example evaluation of f_cvx
x_cvx.value = np.random.rand(nu, 1)
param_cvx.value = transform_param(np.random.rand(npar, 1))
print(f_cvx.value)

constr = [x_cvx<=2.*np.ones((nu,1)), 
            x_cvx>=-2.*np.ones((nu,1))]
cvx_prob = cp.Problem(cp.Minimize(f_cvx), constr)
def solve_cvx_problem(cvx_prob, p):
    param_cvx.value = transform_param(np.array(p).reshape(npar, 1))
    cvx_prob.solve(solver=cp.SCS)
    return x_cvx.value

if tau_th > 0:
    print(pcf.model.sparsity_analysis())

P=[-.8, 0., .3, .8]

# Solve random problems    
N_test=100
P_test = np.concatenate((np.array(P).reshape(len(P),npar),np.random.rand(N_test,npar)*2.-1.))

Uhat_test = np.block([solve_cvx_problem(cvx_prob,p) for p in P_test]).T

@jax.jit
def jaxopt_fun(u,p):
    return f(u[0],u[1],p)[0]

def solve_nl_problem(p, u0):
    solver = jaxopt.ScipyBoundedMinimize(fun=partial(jaxopt_fun, p=p),
                                tol=1.e-6, method="L-BFGS-B", maxiter=1000,
                                options={"disp": True})
    u, state = solver.run(u0, bounds=(-2.*np.ones((nu,1)),2.*np.ones((nu,1))))
    return np.array(u).reshape(-1)
# Solve NLP problem using the solution of the convex problem as initial guess
U_test = np.array([solve_nl_problem(P_test[i], Uhat_test[i]) for i in range(len(P_test))])

R2_test, _, msg = compute_scores(U_test, Uhat_test, None, None, fit='R2')
print("Comparing convex and nonlinear solutions:")
print(msg)

if plotfigs:
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Arial']
    plt.rcParams['font.size'] = 10

    def fhat(U1,U2,p):
        return f_jax(np.hstack((U1.reshape(-1,1),U2.reshape(-1,1))),p*np.ones((U1.size,1)), weights)
        
    U1, U2 = np.meshgrid(np.linspace(-2.,2.,100),np.linspace(-2.,2.,100))

    fig,ax = plt.subplots(2,len(P),figsize=(12,6))

    i=0
    for p in P:
        Ytrue = f(U1.reshape(-1,1),U2.reshape(-1,1),p*np.ones(U1.size).reshape(-1,1))
        Yhat = fhat(U1,U2,p)
        R2, _, msg = compute_scores(Ytrue, Yhat, None, None, fit='R2')
        print(f"p = {p}: {msg}")

        ax[0,i].contour(U1,U2,Ytrue.reshape(U1.shape))
        ax[0,i].plot(U_test[i][0],U_test[i][1],'*', color='blue')
        ax[0,i].plot(Uhat_test[i][0],Uhat_test[i][1],'d', color='orange')
        ax[0,i].grid()
        ax[0,i].set_title(f'True function (p={p})')
        ax[1,i].contour(U1,U2,Yhat.reshape(U1.shape))
        ax[1,i].plot(Uhat_test[i][0],Uhat_test[i][1],'d', color='orange')
        ax[1,i].grid()
        ax[1,i].set_title(f'Convex approximation (p={p})')
        
        i+=1
    plt.show()

# Check if CVXPY and JAX predictions are the same
assert np.abs(f_cvx.value-np.array(f_jax(x_cvx.value.reshape(1,-1),theta_cvx.value.reshape(1,-1), weights)))<=1.e-12, "Error in cvxpy and jax predictions"
