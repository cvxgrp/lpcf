"""
Fit a parametric static convex function to data, using an autoencoder as an input diffeomorphism.
The (possibly nonconvex) parametric function y=f(u,p) is approximated as follows:

v = encoder(u,p)    [this is a diffeomorphism]
y = convex(v,p)
u = decoder(v,p)

where u = original optimization variables, p = parameters, v = new optimization variables.

An alternative is to cascade the convex function by a monotonic function:

v = encoder(u,p)
h = convex(v,p)
y = monotonic(h,p)
u = decoder(v,p)

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

#! CLEAN THIS UP AND PUSH, REMOVE THE AUTOENCODER/MONOTONIC PART AND PUSH

plotfigs = True # set to True to plot figures
parallels_seeds = 10 # number of parallel training sessions (parallel_seeds = 1 means no parallel training)
useMonotonic = False # set to True to cascade the convex function with a monotonic function
useAutoencoder = False # set to True to use an autoencoder as input diffeomorphism
refitDecoder = True # if True, fit the decoder function again after fitting the convex function (only if useAutoencoder=True)
N = 5000 # number of training data points
wy=1. # weight on fitting y (output function)
wu=0.1 # weight on fitting u (autoencoder) during training of the entire output + autoencoder functions

seed = 4 # for reproducibility of results
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
    n3 = 30  # number of neurons in encoder 
    n4 = 30  # number of neurons in decoder
    n5 = 10  # number of neurons in monotonic function
    if useAutoencoder:
        dim_encoder = 10 # dimension of encoded input

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
    n3 = 30  # number of neurons in encoder 
    n4 = 30  # number of neurons in decoder
    n5 = 10  # number of neurons in monotonic function
    if useAutoencoder:
        dim_encoder = 50 # dimension of encoded input

    
nx = nu+npar  # number of inputs

n_convex = 11 # number of weights in convex fcn

if useAutoencoder:
    n_encoder = 5 # number of weights in encoder fcn
    n_decoder = 5 # number of weights in decoder fcn    
else:
    n_encoder = 0
    n_decoder = 0
    dim_encoder = nu
    refitDecoder = False

if useMonotonic:
    n_monotonic = 5 # number of weights in output monotonic fcn
    I_monotonic = range(n_convex+n_encoder+n_decoder,n_convex+n_encoder+n_decoder+n_monotonic)

parallel_training = parallels_seeds > 1

tau_th = 0.*0.001 # L1-regularization term
zero_coeff = 1.e-4 # small coefficients are set to zero when L1-regularization is used

ny = 1 # number of outputs

# input convex function model [Amos, Xu, Kolter, 2017]
def act(x):    
    return jnp.logaddexp(0.,x) # = log(1+exp(x)): activation function, must be convex and non decreasing on the domain of interest

I_convex = range(n_convex)
I_encoder = range(n_convex,n_convex+n_encoder)
I_decoder = range(n_convex+n_encoder,n_convex+n_encoder+n_decoder)

@jax.jit
def convex_fcn(x,convex_params):
    W1v, W1p, b1, W2z, W2v, W2p, b2, W3z, W3v, W3p, b3 = convex_params
    v=x[:,:dim_encoder]
    p=x[:,dim_encoder:]
    z1 = act(W1v @ v.T + W1p @ p.T + b1)
    z2 = act(W2z @ z1 + W2v @ v.T + W2p @ p.T + b2)
    y = W3z @ z2 + W3v @ v.T + W3p @ p.T + b3
    return y.T

if useAutoencoder:
    @jax.jit
    def encoder_fcn(x,encoder_params):
        W1u, W1p, b1, W2, b2 = encoder_params
        u=x[:,:nu]
        p=x[:,nu:]
        z1 = nn.swish(W1u @ u.T + W1p @ p.T + b1)
        v = W2 @ z1 + b2
        return v.T

    @jax.jit
    def decoder_fcn(x,decoder_params):
        W1v, W1p, b1, W2, b2 = decoder_params
        v=x[:,:dim_encoder]
        p=x[:,dim_encoder:]
        z1 = nn.swish(W1v @ v.T + W1p @ p.T + b1)
        u = W2 @ z1 + b2
        return u.T
else:
    def encoder_fcn(x,encoder_params):
        return x[:,:nu]
    decoder_fcn = encoder_fcn
    
if useMonotonic:
    @jax.jit
    def monotonic_fcn(x,monotonic_params):
        W1y, W1p, b1, W2, b2 = monotonic_params
        y=x[:,:ny]
        p=x[:,ny:]
        z1 = act(W1y @ y.T + W1p @ p.T + b1)
        f = W2 @ z1 + b2
        return f.T

@jax.jit
def output_fcn(x,params):
    v = encoder_fcn(x,[params[i] for i in I_encoder])
    vp = jnp.hstack((v,x[:,nu:]))
    y = convex_fcn(vp,[params[i] for i in I_convex])
    if useMonotonic:
        yp = jnp.hstack((y,x[:,nu:]))
        f = monotonic_fcn(yp,[params[i] for i in I_monotonic])
    else:
        f=y
    if useAutoencoder:
        u = decoder_fcn(vp,[params[i] for i in I_decoder])
        return jnp.hstack((f,u))
    else:
        return f

model = StaticModel(ny+nu*useAutoencoder, nx, output_fcn) 

def init_fcn(seed):
    np.random.seed(seed)
    params_convex = [
        np.random.randn(n1, dim_encoder),  # W1v 
        np.random.randn(n1, npar),  # W1p
        np.random.randn(n1, 1),  # b2
        np.random.rand(n2, n1),  # W2z (constrained >= 0)
        np.random.randn(n2, dim_encoder),  # W2v 
        np.random.randn(n2, npar),  # W2p
        np.random.randn(n2, 1),  # b2
        np.random.rand(ny, n2),  # W3z (constrained >= 0)
        np.random.randn(ny, dim_encoder),  # W3v (this is unconstrained, as the last layer is linear)
        np.random.randn(ny, npar),  # W3p
        np.random.randn(ny, 1)  # b3
    ]
    
    if useAutoencoder:
        params_encoder = [
            np.random.randn(n3, nu),  # W1u 
            np.random.randn(n3, npar),  # W1p
            np.random.randn(n3, 1),  # b1
            np.random.randn(dim_encoder, n3),  # W2
            np.random.randn(dim_encoder, 1),  # b2
        ]        
        
        params_decoder = [
            np.random.randn(n4, dim_encoder),  # W1v 
            np.random.randn(n4, npar),  # W1p
            np.random.randn(n4, 1),  # b1
            np.random.randn(nu, n4),  # W2
            np.random.randn(nu, 1),  # b2
        ]
    else:
        params_encoder = []
        params_decoder = []
            
    if useMonotonic:
        params_monotonic = [
            np.random.rand(n5, ny),  # W1y 
            np.random.randn(n5, npar),  # W1p
            np.random.randn(n5, 1),  # b1
            np.random.rand(ny, n5),  # W2
            np.random.randn(ny, 1),  # b2
        ]
    else:
        params_monotonic = []
    
    return params_convex + params_encoder + params_decoder + params_monotonic

model.init(params=init_fcn(4))

# ##############
# define lower bounds for parameters
#
# params_convex = W1v, W1p, b1, W2z, W2v, W2p, b2, W3z, W3v, W3p, b3 
params_convex_min = [-np.inf*np.ones((n1,dim_encoder)), -np.inf*np.ones((n1,npar)), -np.inf*np.ones((n1,1)),
              np.zeros((n2,n1)), -np.inf*np.ones((n2,dim_encoder)), -np.inf*np.ones((n2,npar)), -np.inf*np.ones((n2,1)),
              np.zeros((ny,n2)), -np.inf*np.ones((ny,dim_encoder)), -np.inf*np.ones((ny,npar)), -np.inf*np.ones((ny,1))]
#params_convex_min = None

if useAutoencoder:
    params_encoder_min = [-np.inf*np.ones((n3,nu)), -np.inf*np.ones((n3,npar)), -np.inf*np.ones((n3,1)), -np.inf*np.ones((dim_encoder,n3)), -np.inf*np.ones((dim_encoder,1))]

    params_decoder_min = [-np.inf*np.ones((n4,dim_encoder)), -np.inf*np.ones((n4,npar)), -np.inf*np.ones((n4,1)), -np.inf*np.ones((nu,n4)), -np.inf*np.ones((nu,1))]
else:
    params_encoder_min = []
    params_decoder_min = []

if useMonotonic:
    params_monotonic_min = [np.zeros((n5,ny)), -np.inf*np.ones((n5,npar)), -np.inf*np.ones((n5,1)), np.zeros((ny,n5)), -np.inf*np.ones((ny,1))]
else:
    params_monotonic_min = []
    
params_min = params_convex_min + params_encoder_min + params_decoder_min + params_monotonic_min
params_max = None # no upper bounds

model.optimization(adam_epochs=1000, lbfgs_epochs=2000, params_min=params_min, params_max=params_max)

if useAutoencoder:
    @jax.jit
    def output_loss(YUhat,YU): 
        # MSE loss on y and u, using different weights
        y_loss=jnp.sum((YUhat[:,:ny]-YU[:,:ny])**2)
        u_loss=jnp.sum((YUhat[:,ny:]-YU[:,ny:])**2)
        return (wy*y_loss + wu*u_loss)/YU.shape[0]
    YU = np.hstack((Y.reshape(-1,1),U))
else:
    @jax.jit
    def output_loss(Yhat,Y): 
        return jnp.sum((Yhat[:,:ny]-Y[:,:ny])**2)/Y.shape[0]
    YU = Y

model.loss(rho_th=1.e-8, tau_th=tau_th, output_loss=output_loss, zero_coeff=zero_coeff)

t0 = time.time()
if parallel_training:
    models=model.parallel_fit(YU, X, init_fcn, seeds=range(parallels_seeds), n_jobs=10)
    R2s = [np.sum(compute_scores(YU, m.predict(X.reshape(-1,nx)), None, None, fit='R2')[0]) for m in models]
    ibest = np.argmax(R2s)
    model = models[ibest]
else:
    model.fit(YU, X)
t0 = time.time()-t0

YUhat = model.predict(X)
R2, _, msg = compute_scores(YU, YUhat, None, None, fit='R2')

convex_params = [model.params[i] for i in I_convex]
encoder_params = [model.params[i] for i in I_encoder]
decoder_params = [model.params[i] for i in I_decoder]
if useMonotonic:
    monotonic_params = [model.params[i] for i in I_monotonic]

if refitDecoder:
    # refit the decoder function
    decoder_model = StaticModel(nu, nx, decoder_fcn)
    decoder_model.init(params=decoder_params)
    decoder_model.optimization(adam_epochs=3000, lbfgs_epochs=3000, params_min=params_decoder_min, params_max=None)
    decoder_model.loss(rho_th=1.e-8)
    Xdec = np.hstack((encoder_fcn(X,encoder_params),P.reshape(-1,npar))) # encoded inputs + params
    decoder_model.fit(U, Xdec)
    Uhat = decoder_model.predict(Xdec)
    R2_dec, _, msg = compute_scores(U, Uhat, None, None, fit='R2')
    decoder_params = decoder_model.params # update decoder params
    #for j in range(len(I_decoder)):
    #    model.params[I_decoder[j]] = decoder_params[j] # update overall model too. This doesn't change output function

print(f"Elapsed time: {t0} s")
if useAutoencoder:
    print(f"R2 score on (u,p) -> y mapping:         {R2[:ny]}")
    print(f"R2 score on (u,p) -> (u,p) autoencoder: {R2[ny:]}")
    if refitDecoder:
        print(f"R2 score on (u,p) -> (u,p) autoencoder: {R2_dec} (after retraining the decoder)")
else:
    print(f"R2 score on (u,p) -> y mapping:         {R2}")
    
# #########################
# Convexity check in CVXPY
v_cvx = cp.Variable((dim_encoder, 1))
p_cvx = cp.Parameter((npar, 1))
def create_convex_fcn(convex_params):
    W1v, W1p, b1, W2z, W2v, W2p, b2, W3z, W3v, W3p, b3 = convex_params
    z1 = cp.logistic(W1v @ v_cvx + W1p @ p_cvx + b1)
    z2 = cp.logistic(W2z @ z1 + W2v @ v_cvx + W2p @ p_cvx + b2)
    y = W3z @ z2 + W3v @ v_cvx + W3p @ p_cvx + b3
    return y
cvx_fun = create_convex_fcn(convex_params)
print(f'cvxpy expressions is {"DCP" if cvx_fun.is_dcp() else "non-DCP"}')
print(f'cvxpy expressions is {"DPP" if cvx_fun.is_dpp() else "non-DPP"}')
# #########################

if tau_th > 0:
    print(model.sparsity_analysis())

if example=='2d':
    if plotfigs:
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Arial']
        plt.rcParams['font.size'] = 10

        def fhat(U1,U2,p):
            return model.predict(np.hstack([U1.reshape(-1,1),U2.reshape(-1,1),p*np.ones(U1.size).reshape(-1,1)]))[:,:ny]
            
        if useMonotonic:
            @jax.jit
            def pre_output_fcn(x, encoder_params, convex_params):
                # output function before the monotonic transformation
                v = encoder_fcn(x,encoder_params)
                vp = jnp.hstack((v,x[:,nu:])) # add parameters
                y = convex_fcn(vp,convex_params)
                return y
            plt.rcParams['font.size'] = 8
        
        U1, U2 = np.meshgrid(np.linspace(-2.,2.,100),np.linspace(-2.,2.,100))
        P=[-.8, 0., .3, .8]

        if useMonotonic:
            fig,ax = plt.subplots(3,len(P),figsize=(12,8))
        else:
            fig,ax = plt.subplots(2,len(P),figsize=(12,6))

        i=0
        for p in P:
            Ytrue = f(U1.reshape(-1,1),U2.reshape(-1,1),p*np.ones(U1.size).reshape(-1,1))
            Yhat = fhat(U1,U2,p)
            R2, _, msg = compute_scores(Ytrue, Yhat, None, None, fit='R2')
            print(f"p = {p}: {msg}")

            ax[0,i].contour(U1,U2,Ytrue.reshape(U1.shape))
            ax[0,i].grid()
            ax[0,i].set_title(f'True function (p={p})')
            ax[1,i].contour(U1,U2,Yhat.reshape(U1.shape))
            ax[1,i].grid()
            ax[1,i].set_title(f'Convex approximation (p={p})')
            
            if useMonotonic:
                X = np.hstack([U1.reshape(-1,1),U2.reshape(-1,1),p*np.ones(U1.size).reshape(-1,1)])
                H = pre_output_fcn(X, encoder_params, convex_params)
                iH = np.argsort(H.ravel())
                ax[2,i].plot(H[iH],Yhat.reshape(-1,1)[iH])
                # linear approximation H->Yhat mapping
                ab = np.linalg.lstsq(np.hstack((H.reshape(-1,1),np.ones((H.size,1)))), Yhat.reshape(-1,1),rcond=None)[0]
                ax[2,i].plot(H[iH],ab[0]*H[iH]+ab[1])
                if all(np.diff(Yhat[iH].ravel())>=0.):
                    print(f"p = {p}: final layer seems monotonic")
                ax[2,i].set_title(f'Monotonic transformation (p={p})')
            i+=1
        plt.show()
    
    # Solve random problems    
    N_test=10
    P_test = np.random.rand(N_test,npar)*2.-1.
    constr = [v_cvx<=2.*np.ones((dim_encoder,1)), 
                v_cvx>=-2.*np.ones((dim_encoder,1))]
    cvx_prob = cp.Problem(cp.Minimize(cvx_fun), constr)

    def solve_cvx_problem(p):
        p_cvx.value = np.array(p).reshape(npar,1)
        cvx_prob.solve()
        return v_cvx.value
    Vhat_test = np.block([solve_cvx_problem(p) for p in P_test]).T
    Uhat_test = decoder_fcn(np.hstack((Vhat_test, P_test)), decoder_params)

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
    U_test = np.array([solve_nl_problem(P_test[i], decoder_fcn(np.concatenate((Vhat_test[i], P_test[i])).reshape(1,-1),decoder_params).reshape(nu)) for i in range(len(P_test))])
    
    R2_test, _, msg = compute_scores(U_test, Uhat_test, None, None, fit='R2')
    print(msg)

   
if example=='nlmpc' and plotfigs:
    # Run closed-loop MPC
    pass