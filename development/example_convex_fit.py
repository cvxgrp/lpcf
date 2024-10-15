"""
Fit a parametric convex function to data.

A. Bemporad, October 13, 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from jax_sysid.utils import compute_scores
from jax_sysid.models import StaticModel
import jax
import jax.numpy as jnp
import time
import cvxpy as cp
import flax.linen as nn
import jaxopt
from functools import partial

# ################################
plotfigs = True # set to True to plot figures
parallels_seeds = 10 # number of parallel training sessions (parallel_seeds = 1 means no parallel training)
N = 5000 # number of training data points
wy=1. # weight on fitting y (output function)
wu=0.1 # weight on fitting u (autoencoder) during training of the entire output + autoencoder functions
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

np.random.seed(seed)

jax.config.update('jax_platform_name', 'cpu')
if not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)  # Enable 64-bit computations

example = '2d'
#example = 'nlmpc (not_working_yet)'

if example == '2d':
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

    U = np.vstack((U1,U2)).T

    n1,n2 = 10,10  # number of neurons in convex function
    n1w, n2w = 5, 5 # number of neurons in convex function not dependent of optimization variables
    
    def oracle(U,P):
        return f(U[:,0],U[:,1],P)

else:
    # nonlinear model predictive control example
    @jax.jit
    def prediction(x, U):    
        @jax.jit
        def prediction_model(x, u):
            xnext = jnp.array([.5*jnp.sin(x[0]) + .3*u * jnp.cos(x[1]/2.),
                    .6*jnp.sin(x[0]+x[2]) -.4*u * jnp.arctan(x[0]+x[1]),
                    .4*jnp.exp(-x[1]) + .5*u * jnp.sin(-x[0]/2.)])
            y = jnp.arctan(jnp.array([.5*jnp.sin(x[0]), -.5, 1])@x**3)
            return xnext, y
        _, Y = jax.lax.scan(prediction_model, jnp.array(x), U)
        return Y
    @jax.jit
    def loss_single(x0, U, r, u1):
        Ypred = prediction(x0, U)
        return jnp.sum((Ypred-r)**2) + 0.1*((U[0]-u1)**2+jnp.sum(jnp.diff(U)**2))
    
    loss = jax.jit(jax.vmap(loss_single, in_axes=(0,0,0,0)))
    
    umin = -1.
    umax = 1.
    nu = 10*1 # horizon length * input dimension
    U = np.random.rand(N,nu)*(umax-umin)+umin # input sequences
    R = np.random.randn(N,1) # output references
    X0 = np.random.randn(N,3) # current states
    U1 = np.random.rand(N,1)*(umax-umin)+umin # previous input
    Y = loss(X0, U, R, U1)
    P = np.hstack((X0,R,U1))
    npar = P.shape[1] # number of states + output references + previous input
    
    X = np.hstack((U,P))
    
    n1,n2 = 10,10  # number of neurons in convex function
    n1w, n2w = 5, 5 # number of neurons in neural network generating bias terms from parameters
    
nx = nu+npar  # number of inputs

n_convex = 5 # number of weights in convex fcn
n_bias = 8 # number of weights in neural network generating bias terms from parameters

I_convex = range(n_convex)
I_bias = range(n_convex,n_convex+n_bias)

parallel_training = parallels_seeds > 1

@jax.jit
def bias_fcn(p,params_bias):
    W1, b1, W2w, W2p, b2, W3w, W3p, b3 = params_bias
    z1 = act(W1 @ p.T + b1)
    z2 = act(W2w @ z1 + W2p @ p.T + b2)
    b = W3w @ z2 + W3p @ p.T + b3
    return b.T

bias_dims = [[n1w, npar], [n1w,1], [n2w, n1w], [n2w, npar], [n2w,1], [n1+n2+ny, n2w],  [n1+n2+ny, npar], [n1+n2+ny,1]]
        
@jax.jit
def convex_fcn(x,params):
    W1, W2z, W2u, W3z, W3u = params[:n_convex]
    params_bias = params[n_convex:]
    u=x[:,:nu]
    p=x[:,nu:]
    b = bias_fcn(p,params_bias)
    z1 = act(W1 @ u.T + b[:,:n1].T)
    z2 = act(W2z @ z1 + W2u @ u.T + b[:,n1:n1+n2].T)
    y = W3z @ z2 + W3u @ u.T + b[:,-ny:].T
    return y.T

model = StaticModel(ny, nx, convex_fcn) 

def init_fcn(seed):
    np.random.seed(seed)
    params_convex = [
        np.random.randn(n1, nu),  # W1 
        np.random.rand(n2, n1),  # W2z (constrained >= 0)
        np.random.randn(n2, nu),  # W2u 
        np.random.rand(ny, n2),  # W3z (constrained >= 0)
        np.random.randn(ny, nu)  # W3u (this is unconstrained, as the last layer 
        ]
    params_bias = [np.random.randn(d[0], d[1]) for d in bias_dims]
    return params_convex + params_bias

model.init(params=init_fcn(4))

# ##############
# define lower bounds for parameters
params_convex_min = [-np.inf*np.ones((n1,nu)), np.zeros((n2,n1)), -np.inf*np.ones((n2,nu)), 
              np.zeros((ny,n2)), -np.inf*np.ones((ny,nu))]
params_bias_min = params_bias = [-np.inf*np.ones((d[0], d[1])) for d in bias_dims]

params_min = params_convex_min + params_bias_min
#params_min = None
params_max = None # no upper bounds

model.optimization(adam_epochs=1000, lbfgs_epochs=2000, params_min=params_min, params_max=params_max)

@jax.jit
def output_loss(Yhat,Y): 
    return jnp.sum((Yhat[:,:ny]-Y[:,:ny])**2)/Y.shape[0]

model.loss(rho_th=1.e-8, tau_th=tau_th, output_loss=output_loss, zero_coeff=zero_coeff)

t0 = time.time()
if parallel_training:
    models=model.parallel_fit(Y, X, init_fcn, seeds=range(parallels_seeds), n_jobs=10)
    R2s = [np.sum(compute_scores(Y, m.predict(X.reshape(-1,nx)), None, None, fit='R2')[0]) for m in models]
    ibest = np.argmax(R2s)
    model = models[ibest]
else:
    model.fit(Y, X)
t0 = time.time()-t0

YUhat = model.predict(X)
R2, _, msg = compute_scores(Y, YUhat, None, None, fit='R2')

print(f"Elapsed time: {t0} s")
print(f"R2 score on (u,p) -> y mapping:         {R2}")
    
# #########################
# Convexity check in CVXPY
u_cvx = cp.Variable((nu, 1))
p_cvx = cp.Parameter((npar, 1))
def create_convex_fcn(params):
    W1b, b1b, W2wb, W2pb, b2b, W3wb, W3pb, b3b = params[-n_bias:]
    z1 = cp.logistic(W1b @ p_cvx + b1b)
    z2 = cp.logistic(W2wb @ z1 + W2pb @ p_cvx + b2b)
    b = W3wb @ z2 + W3pb @ p_cvx + b3b
    
    W1, W2z, W2u, W3z, W3u = params[:n_convex]
    z1 = cp.logistic(W1 @ u_cvx + b[:n1])
    z2 = cp.logistic(W2z @ z1 + W2u @ u_cvx + b[n1:n1+n2])
    y = W3z @ z2 + W3u @ u_cvx + b[-ny:]
    return y
cvx_fun = create_convex_fcn(model.params)
print(f'cvxpy expressions is {"DCP" if cvx_fun.is_dcp() else "non-DCP"}')
print(f'cvxpy expressions is {"DPP" if cvx_fun.is_dpp() else "non-DPP"}')
# #########################

if example=='2d':
    constr = [u_cvx<=2.*np.ones((nu,1)), 
                u_cvx>=-2.*np.ones((nu,1))]
else:
    pass #! TODO
cvx_prob = cp.Problem(cp.Minimize(cvx_fun), constr)
def solve_cvx_problem(cvx_prob,p):
    p_cvx.value = np.array(p).reshape(npar,1)
    cvx_prob.solve()
    return u_cvx.value

useActiveSampling = M>0

if useActiveSampling:
    # Refine function by actively sampling at surrogate minimizers
    def active_sampling(P, oracle):
        U = np.block([solve_cvx_problem(cvx_prob,p) for p in P]).T
        Y = oracle(U,P)
        return U, Y

    Ua,Ya = active_sampling(P[0:M], oracle)
    model.optimization(adam_epochs=0, lbfgs_epochs=2000, params_min=params_min, params_max=params_max)
    @jax.jit
    def output_loss(Yhat,Y): 
        return jnp.sum((Yhat[:-M,:ny]-Y[:-M,:ny])**2)/(Y.shape[0]-M)+weight_a*jnp.sum((Yhat[-M:,:ny]-Y[-M:,:ny])**2)/M
    model.loss(rho_th=1.e-8, tau_th=tau_th, output_loss=output_loss, zero_coeff=zero_coeff)
    model.fit(np.vstack((Y.reshape(-1,ny),Ya.reshape(-1,ny))), np.vstack((X,np.hstack((Ua,P[0:M].reshape(-1,npar))))))
    cvx_fun = create_convex_fcn(model.params)
    cvx_prob = cp.Problem(cp.Minimize(cvx_fun), constr)

if tau_th > 0:
    print(model.sparsity_analysis())

if example=='2d':
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
            return model.predict(np.hstack([U1.reshape(-1,1),U2.reshape(-1,1),p*np.ones(U1.size).reshape(-1,1)]))[:,:ny]
            
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
    

   
if example=='nlmpc' and plotfigs:
    # Run closed-loop MPC
    pass

