"""
Approximate Dynamic Programming by fitting optimal costs for control of nonlinear system.

A. Bemporad, May 7, 2025
"""

import pickle
import numpy as np
import cvxpy as cp
from pcf import PCF
import jax
import jax.numpy as jnp
import jaxopt
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, cpu_count
from functools import partial
import control as ctrl

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
#plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

np.random.seed(0)

GenerateData=1
TrainModel=1

testset_size = .3 # fraction of data used for testing

# N_test = 10 # number of test cases for closed-loop simulation
# N_sim = 20 # number of time steps for each closed-loop simulation
beta = 0.1 # weight on control effort
p = 0.2 # model parameter

widths=[]
widths_psi=[]
rho_th = 1.e-8
tau_th = 0.e-8
n_seeds=cpu_count() # number of random seeds for training
adam_epochs=1000
lbfgs_epochs=1000

# Generate optimal control data from random initial states
M = 100 # number of initial states
H = 50 # finite-time optimal control horizon, large enough to approximate the infinite horizon problem
Qu = 1. # input weight

umin = -10000.
umax = 10000.
# range where initial states are generated
xmin = np.array([-2.,-2.])
xmax = -xmin

pmin = -0.5
pmax = 0.5

Ts = 1.0 # [sampling time

nx = 2 # number of states
nu = 1 # number of inputs

# ###################################
# Define stage cost functions

@jax.jit
def stage_cost(x,u):
    loss = jnp.sum(x**2) + beta*jnp.sum(u**2)
    return loss

def stage_cost_cvx(x,u):
    loss = cp.sum_squares(x) + beta*cp.sum_squares(u)
    return loss

# ###################################

# ###################################
# Define dynamics
A = np.array([[0.4, -0.5], [0.1, 0.7]])
B = np.array([[0.], [1.]])
Q_lqr = np.eye(2) # quadratic state cost
R_lqr = beta*np.eye(1) # quadratic input cost
K_lqr, P_lqr, E_lqr = ctrl.dlqr(A, B, Q_lqr, R_lqr) # LQR controller
K_lqr = -K_lqr

@jax.jit
def control_affine_dynamics(x):
    x1 = x[0]
    x2 = x[1]
    
    #f = jnp.hstack([.4*x1-0.1*jnp.sin(x1)-.5*x2, .1*x1+.7*x2-p*x2**3])
    #g = jnp.hstack([0.,1.+x1])
    f = jnp.hstack([A[0][0]*x1+A[0][1]*x2, A[1][0]*x1+A[1][1]*x2])
    g = jnp.hstack([B[0][0],B[1][0]])
    return f,g 

@jax.jit
def dynamics(x, u):    
    f, g = control_affine_dynamics(x)
    xnext = f + g*u
    return xnext, xnext

dynamics_p = dynamics
@jax.jit
def simulation(x0, U):
    #dynamics_p = partial(dynamics, p=p)    
    _, X = jax.lax.scan(dynamics_p, jnp.array(x0), U)
    return jnp.squeeze(X) # return state trajectory x(1),x(2),...,x(H)

# # Test
# p=0.
# uss = 1.
# xss = np.array([10.,10.])
# X=simulation(xss,jnp.tile(uss,(100,1)), p)
# plt.plot(X)

# ###################################
pcf = PCF(activation='logistic', activation_psi='logistic', widths=widths, widths_psi=widths_psi, quadratic=True)
pcf.argmin(fun=None, penalty=1.e3) # add regularization on gradient with respect to x at x=0

options = {'iprint': -1, 'maxls': 20, 'gtol': 1.e-12, 'eps': 1.e-12,
            'ftol': 1.e-12, 'maxfun': 1000, 'maxcor': 10}
Umin = jnp.tile(umin,(H,1))
Umax = jnp.tile(umax,(H,1))

stage_cost_all = jax.vmap(stage_cost, in_axes=(0, 0))

@jax.jit
def opt_control_loss(U, x0, p, beta):
    # performance index: \sum_{k=0}^{H-1} stage_cost(x(k),u(k)) starting from x(0)
    # Later on, this will be used (in surrogate form) as \sum_{k=1}^{H} stage_cost(x(k),u(k)) 
    # starting from x(1), so that by adding stage_cost(x(0),u(0)) we get the total cost starting 
    # from x(0). The optimal performance index only depends on x(0)
    
    X = simulation(x0, U) # state trajectory x(1),x(2),...,x(H)
    #loss = stage_cost_x(x0,q_ref)[0][0]
    loss = stage_cost(x0,U[0]) + jnp.sum(stage_cost_all(X[:-1],U[1:]))
    # loss += 1e3*(X[-1]-x_ref)**2 # terminal cost
    return loss

#raise ValueError("\U0001F7E5"*10 + " STOP HERE")

if GenerateData:
    n_jobs = cpu_count()

    def solve_optimal_control_problem(seed):
        if not jax.config.jax_enable_x64:
            jax.config.update("jax_enable_x64", True)  # Enable 64-bit computations

        np.random.seed(seed)

        # p = pmin + np.random.rand(1)*(pmax-pmin)                
        x0 = ((xmax-xmin)*np.random.rand(2)+xmin).reshape(nx) # random initial state
        
        theoptions=options.copy()
        solver=jaxopt.ScipyBoundedMinimize(fun=opt_control_loss, method="L-BFGS-B", options=theoptions, maxiter=options['maxfun'])
        U_guess=.5*(umin+umax)*np.ones(H)
        Uopt, status = solver.run(U_guess, bounds = (Umin,Umax), x0=x0, p=p, beta=beta)
        loss = status.fun_val
        #print(np.hstack((np.sum(K_lqr*x0),Uopt[0])))
        #print(np.hstack(((K_lqr@Xopt[:-1,:].T).T,Uopt[1:].reshape(-1,1))))
        failed = np.isnan(loss) # or loss>2000. # discard NaNs and outliers
        loss_lqr = (x0.reshape(1,nx)@P_lqr@x0.reshape(nx,1)).item()
        print(f"seed = {seed: 5d}, cost = {loss: 8.4f} " + ("(FAILED!)" if failed else "(OK)")
              + f"- LQR cost = {loss_lqr: 8.4f}")
        
        return {"x0": x0, "loss": loss, "failed": failed, "Uopt": Uopt}
                
    data = Parallel(n_jobs=cpu_count())(delayed(solve_optimal_control_problem)(seed) for seed in range(M))
    
    pickle.dump(data, open('optimal_control_samples_nonlinear.pkl', 'wb'))

else:
    data = pickle.load(open('optimal_control_samples_nonlinear.pkl', 'rb'))

X0 = [data[i]["x0"] for i in range(M)]
FOPT = [data[i]["loss"] for i in range(M)]
FAILED = [data[i]["failed"] for i in range(M)]
UOPT = [data[i]["Uopt"] for i in range(M)]

# Clean up data by removing possible NaNs
ii=np.where(np.array(FAILED))[0]
X0 = np.delete(X0,ii,axis=0)
UOPT = [Uopt for i, Uopt in enumerate (UOPT) if i not in ii]
FOPT = np.delete(FOPT,ii,axis=0)

X0 = np.array(X0)
UOPT = np.array(UOPT)
FOPT = np.array(FOPT)

X = X0
Theta = 0.*np.hstack(X0).reshape(-1,nx) # theta = x(0)
F = FOPT # function values
U = np.hstack([UOPT[:,0]]).reshape(-1,nu) # control inputs

# Normalize data
Fmax = np.max(F)
#Foff = 1.0
Foff = 0.
Fmax=1.
F = F/Fmax + Foff # normalize and shift function values (all values of F are nonnegative)

N=F.shape[0] # total number of data points
N_train = int(N*(1-testset_size)) # number of training data points
N_test = N-N_train # number of test data points

ii=np.random.permutation(N) # random permutation of data points

X_train = X[ii[0:N_train],:]
Theta_train = Theta[ii[0:N_train],:]
F_train = F[ii[0:N_train]]
U_train = U[ii[0:N_train],:]

X_test = X[ii[-N_test:],:]
Theta_test = Theta[ii[-N_test:],:]
F_test = F[ii[-N_test:]]
U_test = U[ii[-N_test:],:]

print(f"Set size: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")

if TrainModel:
    stats = pcf.fit(F_train, X_train, Theta_train, rho_th=rho_th, tau_th = tau_th, seeds=np.arange(n_seeds), cores=cpu_count(), adam_epochs=adam_epochs, lbfgs_epochs=lbfgs_epochs)

    data = {"stats": stats, "params": pcf.model.params}
    pickle.dump(data, open('optimal_control_fit_nonlinear.pkl', 'wb'))

    print(f"Elapsed time: {stats['time']} s")

else:
    data = pickle.load(open('optimal_control_fit_nonlinear.pkl', 'rb'))

    pcf.fit(F_train[0:10], X_train[0:10], Theta_train[0:10], seeds=0, adam_epochs=1, lbfgs_epochs=0) # dummy fit to initialize model        
    pcf.model.params = data["params"]
    stats = data["stats"]
    Fmax = data["Fmax"]
    Foff = data["Foff"]
    
#R2_train = pcf._compute_r2(F_train, pcf.model.predict(np.hstack((X_train, Theta_train)).reshape(-1, pcf.n + pcf.p)))
R2_train = stats['R2']
R2_test= pcf._compute_r2(F_test, pcf.model.predict(np.hstack((X_test, Theta_test)).reshape(-1, pcf.n + pcf.p)))

print(f"R2 score on (u,p) -> y mapping: {R2_train:.2f} (training data), {R2_test:.2f} (test data)")

# sparsity analysis
w=np.block([pcf.model.params[i].ravel() for i in range(len(pcf.model.params))])
nonzeros = np.sum(np.abs(w)>pcf.model.zero_coeff)
nz_perc = 100-100*nonzeros/len(w)
print(f"Number of non-zero parameters: {nonzeros} out of {len(w)} (zeros = {nz_perc:.2f}%)")

# if Theta.ndim == 1:
#     Theta = Theta.reshape(-1,1)
# Fhat = pcf.model.predict(np.hstack((X, Theta)))
# RMS = np.sqrt(np.sum((F-Fhat)**2)/F.shape[0])
# print(f"RMS error on scaled training data: {RMS}")

# export to cvxpy
u_cvx = cp.Variable((nu,1))
f0_cvx = cp.Parameter((nx,1)) # x1 = f(x0) + g(x0)*u
g0_cvx = cp.Parameter((nx,1)) 
x0_cvx = cp.Parameter((nx,1))
x1_cvx =  f0_cvx + g0_cvx@u_cvx

cvx_loss = Foff+Fmax*pcf.tocvxpy(x1_cvx, 0.*x0_cvx)+stage_cost_cvx(x0_cvx, u_cvx)

constr = [umin <= u_cvx, u_cvx <= umax]
cvx_prob = cp.Problem(cp.Minimize(cvx_loss), constr) 

# Evaluate control inputs on test data

Uhat_test = np.empty((N_test,nu))
for i in range(N_test):
    x0 = X_test[i]
    x0_cvx.value = x0.reshape(nx,1)
    fx,gx = control_affine_dynamics(x0.reshape(nx))
    f0_cvx.value = np.array(fx).reshape(nx,1)
    g0_cvx.value = np.array(gx).reshape(nx,1)
    #cvx_prob.solve(solver=cp.CLARABEL, tol_gap_abs=1.e-12)
    cvx_prob.solve()
    Uhat_test[i,:] = u_cvx.value.reshape(nu)
    
Uhat_train = np.empty((N_train,nu))
for i in range(N_train):
    x0 = X_train[i]
    x0_cvx.value = x0.reshape(nx,1)
    fx,gx = control_affine_dynamics(x0.reshape(nx))
    f0_cvx.value = np.array(fx).reshape(nx,1)
    g0_cvx.value = np.array(gx).reshape(nx,1)
    #cvx_prob.solve(solver=cp.CLARABEL, tol_gap_abs=1.e-12)
    cvx_prob.solve()
    Uhat_train[i,:] = u_cvx.value.reshape(nu)

plt.scatter(U_train,Uhat_train)
plt.scatter(U_test,Uhat_test)
plt.plot([-1.5, 1.5], [-1.5, 1.5], 'r--')
plt.xlabel('true value')
plt.ylabel('predicted value')
plt.grid()
plt.show()
plt.title('ADP')
    
