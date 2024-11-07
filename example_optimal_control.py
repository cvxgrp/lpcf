"""
Fit performance index associated with state and input trajectories of a dynamical system.

The optimization vector is the sequence of control inputs, the parameter is the initial states.

A. Bemporad, November 5, 2024
"""

import pickle
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from pcf import PCF
import jax
import jax.numpy as jnp
import jaxopt
from jax_sysid.utils import lbfgs_options
from control import dlqr

TrainModel=1

seed = 0
np.random.seed(seed)

# Generate data from random initial states and random inputs
nx = 3 # number of states
nu = 1 # number of inputs

Q = np.random.randn(nx,nx)
Q=np.round(10.*Q.T@Q)/10.+np.eye(nx) # unknown weight on states
R = np.random.randn(nu,nu)
R=np.round(10.*R.T@R )/10.+np.eye(nu) # unknown weight on inputs

A = np.array([[.5,.3,0],[.3,-.5,.2],[.5,-.4,0]]) # unknown linear system matrix
B = np.array([[.3],[-.4],[.5]]) # unknown linear system matrix
A=A[:nx,:nx]
B=B[:nx,:nu]
@jax.jit
def dynamics(x, u):
    # unknown linear system dynamics
    xnext = (A@x.reshape(-1,1)+B@u.reshape(-1,1)).reshape(-1,1)
    return xnext, x

@jax.jit
def simulation(x0, U):    
    _, X = jax.lax.scan(dynamics, jnp.array(x0), U)
    return np.squeeze(X) # return state and input trajectories

@jax.jit
def stage_cost(X,U):
    loss = jnp.sum(X@Q*X,axis=1) + jnp.sum(U@R*U, axis=1)
    return loss

# dimension of optimization vector = number of entries of policy u=Kx
M = 10 # number of control steps optimized
n = M*nu # number of optimization variables
N=5000+M # length of experiment

# data generating (true) function

Y=list()
X=list()
Theta=list()

x0 = np.zeros((nx,1))
U = 2.*np.random.rand(N,nu)-1.
X = simulation(x0, U)
cost = stage_cost(X,U)
Y = np.array([np.sum(cost[k:k+M]) for k in range(N-M)])
Theta = X[:N-M,:]
X = np.array([U[k:k+M,:].reshape(-1) for k in range(N-M)])

Y = np.array(Y).reshape(-1,1)
Y = Y/np.max(Y) # normalize function values

pcf = PCF(activation='logistic', widths=[20,20], widths_psi=[nx])

if TrainModel:
    stats = pcf.fit(Y, X, Theta, rho_th=1.e-4, seeds=np.arange(10), cores=10)
    data = {"params": pcf.model.params, "stats": stats}
    pickle.dump(data, open('optimal_control.pkl', 'wb'))

else:
    data = pickle.load(open('optimal_control.pkl', 'rb'))
    pcf.fit(Y[0:10], X[0:10], Theta[0:10], seeds=0, adam_epochs=0, lbfgs_epochs=1) # dummy fit to initialize model
    pcf.model.params = data["params"]
    stats = data["stats"]
    
print(f"Elapsed time: {stats['time']} s")
print(f"R2 score on (u,p) -> y mapping:         {stats['R2']}")


if 0:
    # export to cvxpy
    U = cp.Variable((M*nu,1))
    x = cp.Parameter((nx,1))
    cvx_loss = pcf.tocvxpy(U,x)

    # Test
    x.value = np.random.randn(nx,1)
    U.value = np.random.randn(M*nu,1)
    cvx_loss.value

    #cvx_prob = cp.Problem(cp.Minimize(cvx_loss))
    #cvx_prob.solve()
    #U.value
else:
    pass
    
# export to jax 
f_jax = pcf.tojax() # get the jax function and parameters: y = f_jax(x,theta,params)
#ypred = f_jax(X[0].reshape(-1),Theta[0].reshape(-1))

options = lbfgs_options(iprint=-1, iters=1000, lbfgs_tol=1.e-6, memory=20)

@jax.jit
def closed_loop_dynamics(x, K):
    u=K.reshape(nu,nx)@x.reshape(nx,1)
    xnext = dynamics(x,u.ravel())[0]
    xu = jnp.vstack((x,u))
    return xnext, xu
    
@jax.jit
def closed_loop_simulation(x0, K):    
    _, XU = jax.lax.scan(closed_loop_dynamics, jnp.array(x0), jnp.tile(K,[M,1]))
    return XU[:,:nx].reshape(-1,nx),XU[:,nx:].reshape(-1,nu) # return state and input trajectories

KLQR = -dlqr(A,B,Q,R)[0].reshape(-1) # optimal feedback gain

@jax.jit
def simulation(x0, U):    
    _, X = jax.lax.scan(dynamics, jnp.array(x0), U)
    return np.squeeze(X) # return state and input trajectories

@jax.jit
def true_loss(U,x0):
    X=simulation(x0.reshape(-1,1),U.reshape(M,nu))
    loss = jnp.sum(X@Q*X) + jnp.sum(U.reshape(M,nu)@R*U.reshape(M,nu))
    return loss

for k in range(10):
    @jax.jit
    def fun(U):
        return f_jax(U, x0)[0][0]
    x0 = np.random.randn(nx)
    U0=jnp.zeros(M*nu)
    theoptions=options.copy()
    solver=jaxopt.ScipyMinimize(fun=fun, method="L-BFGS-B", options=theoptions, maxiter=1000)
    Uopt, status = solver.run(U0)
    Uopt=Uopt.reshape(M,nu)
    Xopt = simulation(x0.reshape(-1,1), Uopt)
    loss1 = np.sum(stage_cost(Xopt,Uopt))

    XX2,UU2 = closed_loop_simulation(x0.reshape(-1,1), KLQR.reshape(1,-1))
    loss2 = np.sum(stage_cost(XX2,UU2))

    theoptions=options.copy()
    solver=jaxopt.ScipyMinimize(fun=true_loss, method="L-BFGS-B", options=theoptions, maxiter=1000)
    Uopt3, status = solver.run(U0, x0=x0)
    Uopt3=Uopt3.reshape(M,nu)
    Xopt3 = simulation(x0.reshape(-1,1), Uopt3)
    loss3 = np.sum(stage_cost(Xopt3,Uopt3))

    print(f"k={k+1: 3d}, cost = {loss1: 12.8f} (learned), {loss3: 12.8f} (true), {loss2: 12.8f} (LQR)")
    