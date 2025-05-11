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

widths=[10,5]
widths_psi=[10,5]
rho_th = 1.e-8
tau_th = 0.e-8
n_seeds=cpu_count() # number of random seeds for training
adam_epochs=1000
lbfgs_epochs=1000

# Generate optimal control data from random initial states
M = 100 # number of initial states
H = 50 # finite-time optimal control horizon, large enough to approximate the infinite horizon problem

umin = -.5
umax = .5
# range where initial states are generated
xmin = np.array([-2.,-2.])
xmax = -xmin

log10_beta_min = np.log10(0.01)
log10_beta_max = np.log10(10.)

p_min = 0.
p_max = 1.

Ts = 1.0 # [sampling time

nx = 2 # number of states
nu = 1 # number of inputs

# ###################################
# Define stage cost functions

@jax.jit
def stage_cost(x,u, beta):
    loss = jnp.sum(x**2) + beta*jnp.sum(u**2)
    return loss

def stage_cost_cvx(x,u, beta):
    loss = cp.sum_squares(x) + beta*cp.sum_squares(u)
    return loss

# ###################################
# Define dynamics
A = np.array([[0.4, -0.5], [0.1, 0.7]])
B = np.array([[0.], [1.]])

@jax.jit
def control_affine_dynamics(x,p):
    x1 = x[0]
    x2 = x[1]
    
    #f = jnp.hstack([.4*x1-0.1*jnp.sin(x1)-.5*x2, .1*x1+.7*x2-p*x2**3])
    #g = jnp.hstack([0.,1.+x1])
    f = jnp.hstack([A[0][0]*x1+p*A[0][1]*x2, A[1][0]*x1+A[1][1]*x2])
    g = jnp.hstack([B[0][0],B[1][0]])
    return f,g 

@jax.jit
def dynamics(x, u, p):    
    f, g = control_affine_dynamics(x, p)
    xnext = f + g*u
    return xnext, xnext

dynamics_p = dynamics
@jax.jit
def simulation(x0, U, p):
    dynamics_p = partial(dynamics, p=p)    
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

stage_cost_all = jax.vmap(stage_cost, in_axes=(0, 0, None))

@jax.jit
def opt_control_loss(U, x0, beta ,p):
    # performance index: \sum_{k=0}^{H-1} stage_cost(x(k),u(k)) starting from x(0)
    # Later on, this will be used (in surrogate form) as \sum_{k=1}^{H} stage_cost(x(k),u(k)) 
    # starting from x(1), so that by adding stage_cost(x(0),u(0)) we get the total cost starting 
    # from x(0). The optimal performance index only depends on x(0)
    
    X = simulation(x0, U, p) # state trajectory x(1),x(2),...,x(H)
    #loss = stage_cost_x(x0,q_ref)[0][0]
    loss = stage_cost(x0,U[0],beta) + jnp.sum(stage_cost_all(X[:-1],U[1:],beta))
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
        x0 = ((xmax-xmin)*np.random.rand(nx)+xmin).reshape(nx) # random initial state
        log10_beta = log10_beta_min + np.random.rand(1)*(log10_beta_max-log10_beta_min) 
        p = p_min + np.random.rand(1)*(p_max-p_min) 
        beta = 10.**log10_beta.item() # random beta
        
        theoptions=options.copy()
        solver=jaxopt.ScipyBoundedMinimize(fun=opt_control_loss, method="L-BFGS-B", options=theoptions, maxiter=options['maxfun'])
        U_guess=np.tile(.5*(umin+umax), (H,1))
        Uopt, status = solver.run(U_guess, bounds = (Umin,Umax), x0=x0, beta=beta, p=p)
        loss = status.fun_val
        #print(np.hstack((np.sum(K_lqr*x0),Uopt[0])))
        #print(np.hstack(((K_lqr@Xopt[:-1,:].T).T,Uopt[1:].reshape(-1,1))))
        failed = np.isnan(loss) # or loss>2000. # discard NaNs and outliers
        print(f"seed = {seed: 5d}, cost = {loss: 8.4f} " + ("(FAILED!)" if failed else "(OK)"))        
        return {"x0": x0, "loss": loss, "failed": failed, "Uopt": Uopt, "log10_beta": log10_beta, "p": p}
                
    data = Parallel(n_jobs=cpu_count())(delayed(solve_optimal_control_problem)(seed) for seed in range(M))
    
    pickle.dump(data, open('optimal_control_samples_nonlinear.pkl', 'wb'))

else:
    data = pickle.load(open('optimal_control_samples_nonlinear.pkl', 'rb'))

X0 = [data[i]["x0"] for i in range(M)]
FOPT = [data[i]["loss"] for i in range(M)]
FAILED = [data[i]["failed"] for i in range(M)]
UOPT = [data[i]["Uopt"] for i in range(M)]
LOG10_BETA = [data[i]["log10_beta"] for i in range(M)]
P = [data[i]["p"] for i in range(M)]

# Clean up data by removing possible NaNs
ii=np.where(np.array(FAILED))[0]
X0 = np.delete(X0,ii,axis=0)
UOPT = [Uopt for i, Uopt in enumerate (UOPT) if i not in ii]
FOPT = np.delete(FOPT,ii,axis=0)
LOG10_BETA = np.delete(LOG10_BETA,ii,axis=0)
P = np.delete(P,ii,axis=0)

X0 = np.array(X0)
UOPT = np.array(UOPT)
FOPT = np.array(FOPT)
LOG10_BETA = np.array(LOG10_BETA)
P = np.array(P)

X = X0
Theta = np.hstack((LOG10_BETA.reshape(-1,1), P.reshape(-1,1))) # parameters
F = FOPT # function values
U = np.hstack([UOPT[:,0]]).reshape(-1,nu) # control inputs

# Normalize data
#Fmax = np.max(F)
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
LOG10_BETA_train = LOG10_BETA[ii[0:N_train]]
P_train = P[ii[0:N_train]]

X_test = X[ii[-N_test:],:]
Theta_test = Theta[ii[-N_test:],:]
F_test = F[ii[-N_test:]]
U_test = U[ii[-N_test:],:]
LOG10_BETA_test = LOG10_BETA[ii[-N_test:]]
P_test = P[ii[-N_test:]]

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
log10_beta_cvx = cp.Parameter((1,1))
p_cvx = cp.Parameter((1,1))
x1_cvx =  f0_cvx + g0_cvx@u_cvx

beta = cp.exp(cp.multiply(np.log(10.), log10_beta_cvx))
cvx_loss = Foff+Fmax*pcf.tocvxpy(x1_cvx, cp.vstack((log10_beta_cvx,p_cvx)))+stage_cost_cvx(x0_cvx, u_cvx, beta) # stage cost

constr = [umin <= u_cvx, u_cvx <= umax]
cvx_prob = cp.Problem(cp.Minimize(cvx_loss), constr) 

# Evaluate control inputs on test data
Uhat_test = np.empty((N_test,nu))
for i in range(N_test):
    x0 = X_test[i]
    log10_beta_cvx.value = LOG10_BETA_test[i].reshape(1,1)
    p_cvx.value = P_test[i].reshape(1,1)
    x0_cvx.value = x0.reshape(nx,1)
    fx,gx = control_affine_dynamics(x0.reshape(nx),p_cvx.value.item())
    f0_cvx.value = np.array(fx).reshape(nx,1)
    g0_cvx.value = np.array(gx).reshape(nx,1)
    #cvx_prob.solve(solver=cp.CLARABEL, tol_gap_abs=1.e-12)
    cvx_prob.solve()
    Uhat_test[i,:] = u_cvx.value.reshape(nu)

# Evaluate control inputs on training data    
Uhat_train = np.empty((N_train,nu))
for i in range(N_train):
    x0 = X_train[i]
    log10_beta_cvx.value = LOG10_BETA_train[i].reshape(1,1)
    p_cvx.value = P_train[i].reshape(1,1)
    x0_cvx.value = x0.reshape(nx,1)
    fx,gx = control_affine_dynamics(x0.reshape(nx), p_cvx.value.item())
    f0_cvx.value = np.array(fx).reshape(nx,1)
    g0_cvx.value = np.array(gx).reshape(nx,1)
    #cvx_prob.solve(solver=cp.CLARABEL, tol_gap_abs=1.e-12)
    cvx_prob.solve()
    Uhat_train[i,:] = u_cvx.value.reshape(nu)

plt.scatter(U_train,Uhat_train, label='training data')
plt.scatter(U_test,Uhat_test, label='test data')
plt.plot([umin,umax], [umin,umax], 'r--')
plt.xlabel('true value')
plt.ylabel('predicted value')
plt.grid()
plt.legend()
plt.show()
plt.title('ADP')
    


# Generate closed-loop test data using the optimal control policy (receding horizon)
def closed_loop_optimal(x0, N_sim, p, beta):        
    xk = x0.copy()
    X= [x0.reshape(nx)]
    U= list()

    Uopt=jnp.tile((umin+umax)/2.,(H,1))
    for k in range(N_sim):
        theoptions=options.copy()
        solver=jaxopt.ScipyBoundedMinimize(fun=opt_control_loss, method="L-BFGS-B", options=theoptions, maxiter=options['maxfun'])
        U_guess=np.vstack((Uopt[1:,:],Uopt[-1,:])) # shifted previous optimal sequence
        Uopt, status = solver.run(U_guess, bounds = (Umin,Umax), x0=xk.reshape(nx), p=p, beta=beta)
        uk = Uopt[0,:]
        fx,gx = control_affine_dynamics(xk.reshape(nx),p)
        xk = np.array(fx.reshape(nx,1)+gx.reshape(nx,1)@uk.reshape(1,1)).reshape(nx)
        X.append(xk)
        U.append(uk)
        
        print(f"k={k+1: 2d}"
            f", x = [{xk[0].item(): 5.4f}, {xk[1].item(): 5.4f}], u = {uk[0].item(): 5.4f}")
    return np.vstack(X), np.vstack(U)

# # ###################################
# # Closed-loop simulations for testing the ADP controller
# fig,ax = plt.subplots(1,1,figsize=(6,6))
# ax=[ax]

# #p = pmin + np.random.rand(1)*(pmax-pmin)
# N_test=10
# for i in range(N_test):
#     x0 = X_test[i]
#     xk = x0.copy()
#     X= [x0.reshape(nx)]
#     U= list()

#     Uopt=jnp.tile((umin+umax)/2.,(H,1))
#     for k in range(H):
#         # ADP solution
#         x0_cvx.value = xk.reshape(nx,1)
#         fx,gx = control_affine_dynamics(xk.reshape(nx),p)
#         f0_cvx.value = np.array(fx).reshape(nx,1)
#         g0_cvx.value = np.array(gx).reshape(nx,1)
#         cvx_prob.solve()
#         uk = u_cvx.value
#         xk = np.array((fx.reshape(nx,1)+gx.reshape(nx,1)@uk).reshape(nx))
#         X.append(xk)
#         U.append(uk)
        
#         # Optimal control solution
#         theoptions=options.copy()
#         solver=jaxopt.ScipyBoundedMinimize(fun=opt_control_loss, method="L-BFGS-B", options=theoptions, maxiter=options['maxfun'])
#         U_guess=np.vstack((Uopt[1:,:],Uopt[-1,:])) # shifted previous optimal sequence
#         Uopt, status = solver.run(Uopt, bounds = (Umin,Umax), x0=xk.reshape(nx), p=p, beta=beta)
#         uk = Uopt[0,:]
#         fx,gx = control_affine_dynamics(xk.reshape(nx),p)
#         xk = np.array(fx.reshape(nx,1)+gx.reshape(nx,1)@uk.reshape(1,1)).reshape(nx)
#         X.append(xk)
#         U.append(uk)
        
#         print(f"k={k+1: 2d}"
#                 f", x = [{xk[0].item(): 5.4f},{xk[1].item(): 5.4f}], u = {uk[0].item(): 5.4f}")

#     if i==0:
#         ax[0].plot(np.vstack(X_test[i])[:,0],np.vstack(X_test[i])[:,1],label='ADP', linestyle='--', color='black')
#         c1=ax[0].plot(np.vstack(X)[:,0],np.vstack(X)[:,1],label='Optimal')[0].get_color()
#     else:
#         ax[0].plot(np.vstack(X_test[i])[:,0],np.vstack(X_test[i])[:,1], linestyle='--', color='black')
#         ax[0].plot(np.vstack(X)[:,0],np.vstack(X)[:,1], color=c1)
#     if k==H-1:
#         ax[0].scatter(x0[0],x0[1],color=c1, s=50)
        

#     # if i==0:
#     #     ax[1].plot(np.vstack(U1),label='ADP')
#     #     ax[1].plot(np.vstack(U2),label='Optimal')
#     # else:
#     #     ax[1].plot(np.vstack(U1))
#     #     ax[1].plot(np.vstack(U2))
#     # ax[1].plot(np.ones(H)*umin, 'k--')
#     # ax[1].plot(np.ones(H)*umax, 'k--')
#     # ax[1].set_title("input", fontsize=12)
#     # ax[1].legend()
#     # ax[1].grid()
#     print(" ")

# ax[0].set_title("states", fontsize=12)
# ax[0].legend()
# ax[0].grid()