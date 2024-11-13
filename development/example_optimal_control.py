"""
Fit performance index associated with output and input trajectories of a dynamical system.

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
ny = 1 # number of outputs

# Q = np.random.randn(nx,nx)
# Q=np.round(10.*Q.T@Q)/10.+np.eye(nx) # unknown weight on states
# R = np.random.randn(nu,nu)
# R=np.round(10.*R.T@R )/10.+np.eye(nu) # unknown weight on inputs

A = np.array([[.5,.3,0],[.3,-.5,.2],[.5,-.4,0]]) # unknown linear system matrix
B = np.array([[.3],[-.4],[.5]]) # unknown linear system matrix
C = np.array([[1.,.2,-.1]]) # unknown output matrix coefficients
D = np.array([[0]]) # unknown output matrix feedthrough coefficients
R=0.1 # input weight

M = 10 # number of control steps optimized
n = M*nu # number of optimization variables
N=5000+M # length of experiment

Nval = 1000 # number of test samples
make_pos_def = 1 # 1 = makes the surrogate positive definite
if make_pos_def:
    gamma = 1.e-2
else:
    gamma = 0.

betas = [0.,1., 2., 3.] # nonlinearity coefficients

A=A[:nx,:nx]
B=B[:nx,:nu]
C=C[:ny,:nx]
D=D[:ny,:nu]

@jax.jit
def output_nonlinearity(x,beta):
    return x+jnp.sin(beta*x)

@jax.jit
def stage_cost(Y,U):
    #loss = jnp.sum(X@Q*X,axis=1) + jnp.sum(U@R*U, axis=1)
    loss = jnp.sum(Y**2,axis=1) + R*jnp.sum(U**2,axis=1)
    return loss

#KLQR = -dlqr(A,B,Q,R)[0].reshape(-1) # optimal feedback gain
KLQR = -dlqr(A,B,C.T@C,R)[0].reshape(-1) # optimal feedback gain

nbeta=len(betas)

if TrainModel:
    PARAMS = list()
    STATS = list()
else:
    data = pickle.load(open('optimal_control.pkl', 'rb'))
    PARAMS = data["params"]
    STATS = data["stats"]

RMSE_fit = list()
RMSE_opt = list()
NZ_perc = list()

for beta in betas:
    
    @jax.jit
    def dynamics(x, u):
        # unknown system dynamics
        xnext = (A@x.reshape(-1,1)+B@u.reshape(-1,1)).reshape(-1,1)
        y = output_nonlinearity((C@x.reshape(-1,1)+D@u.reshape(-1,1)).reshape(-1,1),beta)
        return xnext, jnp.vstack((x,y))

    @jax.jit
    def simulation(x0, U):    
        _, XY = jax.lax.scan(dynamics, jnp.array(x0), U)
        return np.squeeze(XY)[:,:nx], np.squeeze(XY)[:,nx:] # return state and output trajectories

    @jax.jit
    def true_loss(U,x0):
        _,Y=simulation(x0.reshape(-1,1),U.reshape(M,nu))
        #loss = jnp.sum(X@Q*X) + jnp.sum(U.reshape(M,nu)@R*U.reshape(M,nu))
        loss = jnp.sum(Y**2) + R*jnp.sum(U**2)
        return loss

    # #####################
    # Functions required for comparison with infinite-horizon LQR

    @jax.jit
    def closed_loop_dynamics(x, K):
        u=K.reshape(nu,nx)@x.reshape(nx,1)
        xnext, xy = dynamics(x,u.ravel())
        xyu = jnp.vstack((xy,u))
        return xnext, xyu
        
    @jax.jit
    def closed_loop_simulation(x0, K):    
        _, XYU = jax.lax.scan(closed_loop_dynamics, jnp.array(x0), jnp.tile(K,[M,1]))
        return XYU[:,:nx].reshape(-1,nx),XYU[:,nx:nx+ny].reshape(-1,ny), XYU[:,nx+ny:].reshape(-1,nu)

    # data generating (true) function
    x0 = np.zeros((nx,1))
    U = 2.*np.random.rand(N,nu)-1.
    X,Y = simulation(x0, U)
    cost = stage_cost(Y,U)
    F = np.array([np.sum(cost[k:k+M]) for k in range(N-M)])
    Theta = X[:N-M,:]
    X = np.array([U[k:k+M,:].reshape(-1) for k in range(N-M)])

    F = np.array(F).reshape(-1,1)
    
    F = F - gamma*np.sum(X**2,axis=1).reshape(-1,1) # make the surrogate positive definite if gamma>0   
    F = F/np.max(F) # normalize function values

    pcf = PCF(activation='logistic', widths=[20,20], widths_psi=[10])

    if TrainModel:
        stats = pcf.fit(F, X, Theta, rho_th=1.e-8, tau_th=1.e-5, seeds=np.arange(10), cores=10)
        PARAMS.append(pcf.model.params)
        STATS.append(stats)

    else:
        pcf.fit(F[0:10], X[0:10], Theta[0:10], seeds=0, adam_epochs=0, lbfgs_epochs=1) # dummy fit to initialize model        
        ibeta = betas[betas==beta].item()
        pcf.model.params = PARAMS[ibeta]
        stats = STATS[ibeta]
        
    print(f"Elapsed time: {stats['time']} s")
    print(f"R2 score on (u,p) -> y mapping:         {stats['R2']}")

    # sparsity analysis
    w=np.block([pcf.model.params[i].ravel() for i in range(len(pcf.model.params))])
    nonzeros = np.sum(np.abs(w)>pcf.model.zero_coeff)
    nz_perc = 100-100*nonzeros/len(w)
    print(f"beta = {beta}: Number of non-zero parameters: {nonzeros} out of {len(w)} (zeros = {nz_perc:.2f}%)")
    NZ_perc.append(nz_perc)


    Fhat = pcf.model.predict(np.hstack((X, Theta)))
    RMS = np.sqrt(np.sum((F-Fhat)**2)/F.shape[0])
    print(f"beta = {beta}: RMS error on scaled training data: {RMS}")
    RMSE_fit.append(RMS)


    # export to cvxpy
    U = cp.Variable((M*nu,1))
    x = cp.Parameter((nx,1))
    cvx_loss = pcf.tocvxpy(U,x)
    # # Test
    #x.value = np.random.randn(nx,1)
    #U.value = np.random.randn(M*nu,1)
    #cvx_loss.value
   
    cvx_prob = cp.Problem(cp.Minimize(cvx_loss+gamma*cp.sum_squares(U))) 

    
    # export to jax 
    f_jax = pcf.tojax() # get the jax function and parameters: y = f_jax(x,theta,params)
    #ypred = f_jax(X[0].reshape(-1),Theta[0].reshape(-1))

    options = lbfgs_options(iprint=-1, iters=1000, lbfgs_tol=1.e-6, memory=20)

    xmin = np.array(jnp.min(Theta,axis=0))
    xmax = np.array(jnp.max(Theta,axis=0))

    X0 = list()
    LOSS_MIN = list()
    TRUE_LOSS_MIN = list()

    for k in range(Nval):
        x0 = (xmax-xmin)*np.random.rand(nx)+xmin
        if 0:
            # Minimize the surrogate via LBFGS
            @jax.jit
            def fun(U):
                return f_jax(U, x0)[0][0]+gamma*jnp.sum(U**2)
            U0=jnp.zeros(M*nu)
            theoptions=options.copy()
            solver=jaxopt.ScipyMinimize(fun=fun, method="L-BFGS-B", options=theoptions, maxiter=1000)
            Uopt, status = solver.run(U0)
        else:   
            # Minimize the surrogate via CVXPY
            x.value = x0.reshape(nx,1)
            cvx_prob.solve()    
            Uopt = U.value

        Uopt=Uopt.reshape(M,nu)
        Xopt,Yopt = simulation(x0.reshape(-1,1), Uopt)
        loss1 = np.sum(stage_cost(Yopt,Uopt))

        XX2,YY2,UU2 = closed_loop_simulation(x0.reshape(-1,1), KLQR.reshape(1,-1))
        loss2 = np.sum(stage_cost(YY2,UU2))

        # True loss
        theoptions=options.copy()
        solver=jaxopt.ScipyMinimize(fun=true_loss, method="L-BFGS-B", options=theoptions, maxiter=1000)
        U0=jnp.zeros(M*nu)
        Uopt3, status = solver.run(U0, x0=x0)
        Uopt3=Uopt3.reshape(M,nu)
        Xopt3,Yopt3 = simulation(x0.reshape(-1,1), Uopt3)
        loss3 = np.sum(stage_cost(Yopt3,Uopt3))

        print(f"k={k+1: 3d}, cost = {loss1: 12.8f} (learned), {loss3: 12.8f} (true), {loss2: 12.8f} (LQR)")

        X0.append(x0)
        LOSS_MIN.append(loss1)
        TRUE_LOSS_MIN.append(loss3)

    LOSS_MIN = np.array(LOSS_MIN)
    TRUE_LOSS_MIN = np.array(TRUE_LOSS_MIN)

    RMS_MIN = np.sqrt(np.sum((TRUE_LOSS_MIN-LOSS_MIN)**2)/Nval)
    print(f"beta = {beta}: RMS error of minima on scaled training data: {RMS_MIN}")

    RMSE_opt.append(RMS_MIN)

if TrainModel:
    data = {"params": PARAMS, "stats": STATS, "RMSE_fit": RMSE_fit, "RMSE_opt": RMSE_opt, "NZ_perc": NZ_perc}
    pickle.dump(data, open('optimal_control.pkl', 'wb'))
    
for k in range(len(betas)):
    print(f"{betas[k]} & {RMSE_fit[k]: .4f} & {RMSE_opt[k]: .4f} & {NZ_perc[k]: .2f} \\% \\\\")