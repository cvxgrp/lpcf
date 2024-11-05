"""
Fit performance index associated with state and input trajectories of a dynamical system.

The optimization vector is a feedback gain K, u=Kx, the parameter is the current

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
from functools import partial
from jax_sysid.utils import lbfgs_options
from control import dlqr

TrainModel=1

seed = 0
np.random.seed(seed)

# Generate data from random initial states and random inputs
nx = 2 # number of states
nu = 1 # number of inputs

Q = np.random.randn(nx,nx)
Q=Q.T@Q # unknown weight on states
R = np.random.randn(nu,nu)
R=R.T@R # unknown weight on inputs

N1=100 # length of experiment to compute performance index
N2=1 # number of steps the performance index is evaluated along the state trajectory
N3=200 # number of experiments with different policies

A = np.array([[.5,.3,0],[.3,-.5,.2],[.5,-.4,0]]) # unknown linear system matrix
B = np.array([[.3],[-.4],[.5]]) # unknown linear system matrix
A=A[:nx,:nx]
B=B[:nx,:nu]
@jax.jit
def dynamics(x, u):
    # unknown linear system dynamics
    xnext = (A@x.reshape(-1,1)+B@u.reshape(-1,1)).reshape(-1,1)
    return xnext

@jax.jit
def closed_loop_dynamics(x, K):
    u=K.reshape(nu,nx)@x.reshape(nx,1)
    xnext = dynamics(x,u.ravel())
    xu = jnp.vstack((x,u))
    return xnext, xu
    
@jax.jit
def closed_loop_simulation(x0, K):    
    _, XU = jax.lax.scan(closed_loop_dynamics, jnp.array(x0), jnp.tile(K,[N1+N2,1]))
    return XU[:,:nx].reshape(-1,nx),XU[:,nx:].reshape(-1,nu) # return state and input trajectories

@jax.jit
def loss(X,U):
    loss = jnp.sum(X@Q*X) + jnp.sum(U@R*U)
    return loss

# dimension of optimization vector = number of entries of policy u=Kx
n = nx*nu

# data generating (true) function
N=N3*N2 # number of samples of the performance index

Y=list()
X=list()
Theta=list()

for k in range(N3):
    x0 = np.random.randn(nx,1)
    K = np.random.randn(nu,nx)
    XX,UU = closed_loop_simulation(x0, K) # closed-loop trajectories
    loss0 = loss(XX,UU)
    if loss0<1.e2:
        Y.append(np.array([loss(XX[i:i+N1,:],UU[i:i+N1,:]) for i in range(N2)]))
        Theta.append(XX[:N2,:])
        X.append(np.tile(K.reshape(-1),(N2,1))) # same policy for all samples

Y = np.array(Y).reshape(-1,1)
Y = Y/np.max(Y) # normalize function values
X = np.array(X).reshape(-1,nx*nu)
Theta = np.array(Theta).reshape(-1,nx)

# # Test:
# x=Theta[130].reshape(nx,1)
# K=X[130].reshape(nu,nx)
# XX,UU = closed_loop_simulation(x, K)
# print(loss(XX[:N1,:],UU[:N1,:]), Y[130])
    
# fit

pcf = PCF(activation='relu', widths=[30,30], widths_psi=[nx])

if TrainModel:
    stats = pcf.fit(Y, X, Theta, seeds=np.arange(10), cores=10)
    data = {"params": pcf.model.params, "stats": stats}
    pickle.dump(data, open('control_policy_data.pkl', 'wb'))

else:
    data = pickle.load(open('control_policy_data.pkl', 'rb'))
    pcf.fit(Y[0:10], X[0:10], Theta[0:10], seeds=0, adam_epochs=0, lbfgs_epochs=1) # dummy fit to initialize model
    pcf.model.params = data["params"]
    stats = data["stats"]
    
print(f"Elapsed time: {stats['time']} s")
print(f"R2 score on (u,p) -> y mapping:         {stats['R2']}")


if 0:
    # export to cvxpy
    K = cp.Variable((nx*nu,1))
    x = cp.Parameter((nx,1))
    cvx_loss = pcf.tocvxpy(K,x)

    # Test
    x.value = np.random.randn(nx,1)
    K.value = np.random.randn(nx,nu)
    cvx_loss.value

    #cvx_prob = cp.Problem(cp.Minimize(cvx_loss))
    #cvx_prob.solve()
    #K.value
else:
    pass
    
# export to jax 
f_jax = pcf.tojax() # get the jax function and parameters: y = f_jax(x,theta,params)
#ypred = f_jax(X[0].reshape(-1),Theta[0].reshape(-1))

options = lbfgs_options(iprint=-1, iters=1000, lbfgs_tol=1.e-6, memory=20)

KLQR = -dlqr(A,B,Q,R)[0].reshape(-1) # optimal feedback gain

for k in range(10):
    @jax.jit
    def fun(K):
        return f_jax(K, x0)[0][0]
    x0 = np.random.randn(nx)
    K0=jnp.zeros(nx*nu)
    theoptions=options.copy()
    solver=jaxopt.ScipyMinimize(fun=fun, method="L-BFGS-B", options=theoptions, maxiter=1000)
    Kopt, status = solver.run(K0)

    XX1,UU1 = closed_loop_simulation(x0.reshape(-1,1), Kopt.reshape(1,-1))
    loss1 = loss(XX1,UU1)
    XX2,UU2 = closed_loop_simulation(x0.reshape(-1,1), KLQR.reshape(1,-1))
    loss2 = loss(XX2,UU2)
    print(f"k={k+1}, cost = {loss1: 8.5f} (learned), {loss2: 8.5f} (LQR)")
    print(f"     gain = {Kopt} (learned)")
    print(f"            {KLQR} (LQR)\n")
    