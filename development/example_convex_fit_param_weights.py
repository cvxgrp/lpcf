"""
Fit a parametric convex function to data.

A. Bemporad, October 31, 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from jax_sysid.utils import compute_scores
from jax_sysid.models import StaticModel
import jax
import jax.numpy as jnp
import time
from itertools import product

# ###############################
parallels_seeds = 10 # number of parallel training sessions (parallel_seeds = 1 means no parallel training)
tau_th = 0*0.001 # L1-regularization term
zero_coeff = 1.e-4 # small coefficients are set to zero when L1-regularization is used
# input convex function model [Amos, Xu, Kolter, 2017]

def act(x):    
    #return jnp.logaddexp(0.,x) # = log(1+exp(x)): activation function, must be convex and non decreasing on the domain of interest
    return jnp.maximum(0.,x) # ReLU
    #return jnp.maximum(.1*x,x) # leaky-ReLU
ny = 1 # number of outputs
seed = 3 # for reproducibility of results
# ###############################
np.random.seed(seed)

jax.config.update('jax_platform_name', 'cpu')
if not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)  # Enable 64-bit computations

# ################################
# generate data
def plus(a):
    return np.maximum(a, 0)

def f_true(x, theta):
    splus, s_minus, m, v = theta
    return splus * plus(x - m) + s_minus * plus(m - x) + v

if 0:
    # generate data
    Nx = 50 # number of x-data per parameter value
    Nth = 2000 # number of th-data
    x_      = np.linspace(-1.,1.,Nx)
    theta_  = -1. + 2. * np.random.rand(Nth,4)
    XTH = list(product(x_, theta_))
    X = np.array([XTH[i][0] for i in range(len(XTH))]).reshape(-1,1)
    Theta = np.array([XTH[i][1] for i in range(len(XTH))])
    Y = np.array([f_true(X[i],Theta[i]) for i in range(X.shape[0])])

else:
    n_rand  = 10
    x_      = -1 + 2 * np.random.rand(n_rand)
    splus_  = -1 + 2 * np.random.rand(n_rand)
    sminus_ = -1 + 2 * np.random.rand(n_rand)
    m_      = -1 + 2 * np.random.rand(n_rand)
    v_      = -1 + 2 * np.random.rand(n_rand)

    x_      = np.linspace(-1.,1.,n_rand)

    IN = np.array(list(product(x_, splus_, sminus_, m_, v_)))
    X, Theta = IN[:, 0], IN[:, 1:]
    Y = f_true(X, Theta.T)
    X=X.reshape(-1,1)

# ################################
nx = X.shape[1]
npar = Theta.shape[1]
XTheta = np.hstack((X, Theta))

n1,n2 = 1,2  # number of neurons in convex function
n1psi, n2psi = 10, 10 # number of neurons in convex function not dependent of optimization variables
    
param_weights_dims = [[n1,nx],[n1,1],[n2,n1],[n2,nx],[n2,1],[ny,n2],[ny,nx],[ny,1]]
n_allweights = sum([d[0]*d[1] for d in param_weights_dims])

parallel_training = parallels_seeds > 1

weights_dims = [[n1psi, npar], [n1psi,1], [n2psi, n1psi], [n2psi, npar], [n2psi,1], [n_allweights, n2psi],  [n_allweights, npar], [n_allweights,1]]
        
@jax.jit
def convex_fcn(xth,params):
    x=xth[:,:nx]
    p=xth[:,nx:]
    
    # function generating weights and biases of convex NN from parameters
    W1psi, b1psi, W2wpsi, W2ppsi, b2psi, W3wpsi, W3ppsi, b3psi = params
    z1 = act(W1psi @ p.T + b1psi)
    z2 = act(W2wpsi @ z1 + W2ppsi @ p.T + b2psi)
    W = (W3wpsi @ z2 + W3ppsi @ p.T + b3psi).T # All weights, one per parameter entry
    
    j = 0
    weights = list()
    for i in range(len(param_weights_dims)):
        dims = param_weights_dims[i][0]*param_weights_dims[i][1]
        weights.append(W[:,j:j+dims].reshape([-1]+param_weights_dims[i]))
        j += dims
    W1, b1, W2z, W2x, b2, W3z, W3x, b3 = weights
    W2z = W2z**2 # makes weights on z non-negative
    W3z = W3z**2
    #W2z = jnp.exp(W2z) # makes weights on z non-negative
    #W3z = jnp.exp(W3z)

    z1 = act(jax.vmap(jnp.matmul)(W1,x).T + jnp.squeeze(b1.T))
    z2 = act(jax.vmap(jnp.matmul)(W2z,z1.T).T + jax.vmap(jnp.matmul)(W2x,x).T + jnp.squeeze(b2.T))
    y = jax.vmap(jnp.matmul)(W3z,z2.T).T + jax.vmap(jnp.matmul)(W3x,x).T + jnp.squeeze(b3.T)
    return y.T
    
model = StaticModel(ny, nx, convex_fcn) 

def init_fcn(seed):
    np.random.seed(seed)    
    params = [np.random.rand(d[0], d[1])-0.5 for d in weights_dims]
    return params

model.init(params=init_fcn(4))

model.optimization(adam_epochs=200, lbfgs_epochs=2000)

@jax.jit
def output_loss(Yhat,Y): 
    return jnp.sum((Yhat-Y)**2)/Y.shape[0]

model.loss(rho_th=1.e-8, tau_th=tau_th, output_loss=output_loss, zero_coeff=zero_coeff)

t0 = time.time()
if parallel_training:
    models=model.parallel_fit(Y, XTheta, init_fcn, seeds=range(parallels_seeds), n_jobs=10)
    R2s = [np.sum(compute_scores(Y, m.predict(XTheta), None, None, fit='R2')[0]) for m in models]
    ibest = np.argmax(R2s)
    model = models[ibest]
else:
    model.fit(Y, XTheta)
t0 = time.time()-t0

YUhat = model.predict(XTheta)
R2, _, msg = compute_scores(Y, YUhat, None, None, fit='R2')

print(f"Elapsed time: {t0} s")
print(f"R2 score on (u,p) -> y mapping:         {R2}")
    
x_ = np.linspace(-1, 1, 100)
@jax.jit
def f(x, theta):
    # Evaluate the model for a given theta and possibly multiple x
    return model.predict(jnp.concatenate([x.reshape(-1,1),jnp.tile(theta, (x.shape[0],1))], axis=1))

y_true_, y_, theta_ = [], [], []
for i in range(6):
    #theta = -1 + 2 * np.random.rand(4) # new test data
    theta = Theta[i] # training data
    y_true_.append(f_true(x_, theta))
    y_.append(f(x_,theta))
    theta_.append(theta)

# plot

if 0:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for i, ax in enumerate(axes.flat):
        ax.plot(x_, y_true_[i], label='y_true')
        ax.plot(x_, y_[i], label='y', linestyle='--')
        ax.set_title(f'theta = {np.round(theta_[i], 2)}')
        if i == 0:
            ax.legend()
else:
# evaluate f_true and f for 6 random points in Theta, over a grid of 100 points in X

    np.random.seed(0)

    x_ = np.linspace(-1, 1, 100)

    y_true_, y_, theta_ = [], [], []
    for _ in range(6):
        theta = -1 + 2 * np.random.rand(4)
        y_true_.append(f_true(x_, theta))
        y_.append(f(x_,theta))
        theta_.append(theta)

    # plot

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for i, ax in enumerate(axes.flat):
        ax.plot(x_, y_true_[i], label='y_true')
        ax.plot(x_, y_[i], label='y', linestyle='--')
        ax.set_title(f'theta = {np.round(theta_[i], 2)}')
        if i == 0:
            ax.legend()

           

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

print('done')