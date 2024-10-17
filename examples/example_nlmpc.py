"""
Fit a parametric convex function to a NL-MPC problem.

A. Bemporad, October 16, 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax_sysid.utils import standard_scale
import cvxpy as cp
import jaxopt
from pcf import PCF
import tqdm

jax.config.update('jax_platform_name', 'cpu')
if not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)  # Enable 64-bit computations

# ################################
plotfigs = True # set to True to plot figures
cores = 10 # number of parallel training sessions (cores = 1 means no parallel training)
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
use_nonlinear_model = True

if use_nonlinear_model:
    N = 10000 # number of training data points
    a1=1. # cost term for integral action
    a2=0.001 # cost term for input rate
    widths_variable=[10,10]
    widths_parameter=[30,30]
    rho_th=1.e-3
    seeds = np.arange(10) # seeds to use for initial condition during training
    adam_epochs=5000
    lbfgs_epochs=5000 
else:
    N = 10000 # number of training data points
    a1=.1
    a2=0.001
    widths_variable=[10,10]
    widths_parameter=[10,10]
    rho_th=1.e-3
    seeds = np.arange(10) # seeds to use for initial condition during training
    adam_epochs=1000
    lbfgs_epochs=2000 
# ################################

np.random.seed(seed)

@jax.jit
def prediction_model(x, u):
    if use_nonlinear_model:
        # Nonlinear system, cost function should be nonconvex
        xnext = jnp.array([.2*x[0]+.1*jnp.sin(x[0]/10.) + .3*u + .5*x[1],
                .2*x[2] -.4*u + .3*x[0]-.5*x[1],
                -.4*x[1] + .5*u*(1.+jnp.log(1+x[2]/10.)) -.3*x[0]])
        y = jnp.array([.5, -.5, 1])@(x+x**3/10.)
    else:    
        # Linear system, cost function should be quadratic
        xnext = jnp.array([.5*x[0] + .3*u + .5*x[1],
                .2*x[2] -.4*u + .3*x[0]-.5*x[1],
                -.4*x[1] + .5*u -.5*x[0]])
        y = jnp.array([.5, -.5, 1])@x
    return xnext, y
    
@jax.jit
def prediction(x, U):    
    _, Y = jax.lax.scan(prediction_model, jnp.array(x), U)
    return Y
@jax.jit
def loss_single(U, x0, r, uold, integral):
    Ypred = prediction(x0, U)
    Intpred = jnp.cumsum(Ypred-r)+integral
    loss = jnp.sum((Ypred-r)**2) + a1*jnp.sum(Intpred**2) + a2*((U[0]-uold)**2+jnp.sum(jnp.diff(U)**2))
    return loss[0]

loss = jax.jit(jax.vmap(loss_single, in_axes=(0,0,0,0,0)))

@jax.jit
def prediction_model_with_states(x, u):
    xnext, y = prediction_model(x, u)
    return xnext, jnp.concatenate((x.ravel(), y))
@jax.jit
def prediction_with_states(x, U):    
    _, XY = jax.lax.scan(prediction_model_with_states, jnp.array(x), U)
    return XY[:,:nx], XY[:,nx:]

umin = -10.
umax = 10.
nu = 1 # input dimension
N_mpc = 10 # MPC horizon length
nx = 3 # number of states
U = np.random.rand(N+N_mpc,nu)*(umax-umin)+umin # input excitation
UOLD = np.vstack((np.zeros((1,nu)),U[:-1-N_mpc,:])) # previous input
R = np.random.randn(N,1) # output references
x0 = np.zeros((nx,1))
X0, Y = prediction_with_states(x0, U) # get state trajectories and use them as initial states
X0 = np.vstack((np.zeros((1,nx)),X0[:-1,:]))[:N] # initial states
INT = np.random.randn(N,1) # integral action value @each MPC execution step
#INT = np.cumsum(Y[:N,:]-R).reshape(-1,1) # integral of tracking error
U = np.squeeze(np.array([U[i:i+N_mpc] for i in range(N)])) # input trajectories

Y = loss(U, X0, R, UOLD, INT)
#Y = np.arctan(Y) # take arctan to better approximate values where loss is small
Y, ymean, ygain = standard_scale(Y.reshape(-1,1))

P = np.hstack((X0,R,UOLD, INT))
npar = P.shape[1] # number of states + output references + previous input

pcf = PCF(widths_variable=widths_variable, widths_parameter=widths_parameter, activation_variable='logistic', activation_parameter='swish')
stats = pcf.fit(Y, U, P, rho_th=rho_th, tau_th=tau_th, zero_coeff=zero_coeff, cores=cores, seeds=seeds, adam_epochs=adam_epochs, lbfgs_epochs=lbfgs_epochs)

f_jax, weights = pcf.tojax() # get the jax function and parameters: y = f_jax(x,theta,params)
YHAT = f_jax(U, P, weights) # predict the output for the training data
# YHAT2 = pcf.model.output_fcn(X,pcf.model.params) = YHAT

print(f"Elapsed time: {stats['time']} s")
print(f"R2 score on (u,p) -> y mapping:         {stats['R2']}")
    
# #########################
# Convexity check in CVXPY
x_cvx = cp.Variable((N_mpc*nu, 1))
#theta_cvx = cp.Parameter((npar, 1))
f_cvx, param_cvx, transform_param = pcf.tocvxpy(x_cvx)
print(f'cvxpy expressions is {"DCP" if f_cvx.is_dcp() else "non-DCP"}')
print(f'cvxpy expressions is {"DPP" if f_cvx.is_dpp() else "non-DPP"}')
# #########################

constr = [x_cvx<=umax*np.ones((N_mpc*nu,1)), 
            x_cvx>=umin*np.ones((N_mpc*nu,1))]
cvx_prob = cp.Problem(cp.Minimize(f_cvx), constr)

def solve_cvx_problem(cvx_prob, p):
    param_cvx.value = transform_param(np.array(p).reshape(npar, 1))
    cvx_prob.solve(solver=cp.SCS)
    return x_cvx.value

if tau_th > 0:
    print(pcf.model.sparsity_analysis())

# Test original NLMPC and new convex-MPC controllers
x0 = np.random.randn(nx)
T=100 # number of closed-loop simulation steps

R = np.empty((T,1)) # output references
r=np.random.randn()
j=0
for k in range(T):
    j+=1
    if np.random.randn()>0.95 and j>10:
        r=np.random.randn()
        j=0
    R[k] = r

tol=1.e-6
options={'iprint': -1, 'maxls': 20, 'gtol': tol, 'eps': 1.e-8,
    'ftol': tol, 'maxfun': 1000, 'maxcor': 20}
Umin=umin*np.ones((N_mpc,nu))
Umax=umax*np.ones((N_mpc,nu))

def nlmpc(x,r,Uold, integral):
    uold=Uold[0]
    U0=np.vstack((Uold[1:],Uold[-1,:])) # initial guess
    solver=jaxopt.ScipyBoundedMinimize(fun=loss_single, tol=tol, maxiter=1000, options=options)
    Uopt, status = solver.run(U0.reshape(-1), bounds=(Umin.reshape(-1),Umax.reshape(-1)), x0=x, r=r, uold=uold, integral=integral)
    del options["maxiter"]
    return Uopt.reshape(N_mpc,nu)

def convexmpc(x,r,Uold, integral):
    Uopt = solve_cvx_problem(cvx_prob, np.hstack((x,r,Uold[0,:],integral)))
    return Uopt.reshape(N_mpc,nu)

def closedLoopSimulation(x0,R,controller):
    Uold=np.zeros((N_mpc,nu))
    T = R.shape[0]
    Y=np.empty((T,1))
    X=np.empty((T,nx))
    U=np.empty((T,nu))
    x=x0
    integral=0.
    for k in tqdm.tqdm(range(T)):
        Uopt=controller(x,R[k],Uold,integral)
        xnext, Y[k] = prediction_model(x, Uopt[0])
        U[k] = Uopt[0]
        if k<T-1:
            X[k+1] = xnext.ravel()
            Uold = Uopt
            x = xnext.ravel()
            integral += Y[k]-R[k]
    return Y,X,U

print("Simulating closed-loop NLMPC system:")
Y_nlmpc, X_nlmpc, U_nlmpc = closedLoopSimulation(x0, R, nlmpc)
print("Simulating closed-loop convex MPC system:")
Y_convexmpc, X_convexmpc, U_convexmpc = closedLoopSimulation(x0, R, convexmpc)

if plotfigs:
    fig,ax = plt.subplots(2,1,figsize=(8,10))
    ax[0].plot(R, label='Reference signal')
    ax[0].plot(Y_nlmpc, label='NLMPC')
    ax[0].plot(Y_convexmpc, label='Convex MPC')
    ax[0].set_title('Output signals')
    ax[0].legend()
    ax[1].plot(U_nlmpc, label='NLMPC')
    ax[1].plot(U_convexmpc, label='Convex MPC')
    ax[1].legend()
    ax[1].set_title('Input signals')
    plt.show()