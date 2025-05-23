"""
Approximate Dynamic Programming by fitting optimal costs for control of a linear system.

A. Bemporad, November 13, 2024
"""

import pickle
import numpy as np
import cvxpy as cp
from pcf import PCF
import jax
import jax.numpy as jnp
import jaxopt
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

GenerateData=0
TrainModel=0

# Generate optimal control data from random initial states
M = 1000 # number of initial states
widths=[10,10]
widths_psi=[10,10]
H = 50 # finite-time optimal control horizon
Qu=0.1 # weight on deviations of inputs from steady state
gamma = 1.0 # discount factor

umin = np.array([-10])
umax = np.array([10])
y_ref_min=-1 
y_ref_max=1

nx = 3 # number of states
nu = 1 # number of inputs

A = np.array([[.5,.3,0],[.3,-.5,.2],[.5,-.4,0]]) # unknown linear system matrix
B = np.array([[.3],[-.4],[.5]]) # unknown linear system matrix
C = np.array([[1.,.2,-.1]]) # unknown output matrix coefficients
D = np.array([[0]]) # unknown output matrix feedthrough coefficients

# ###################################
# Define stage cost functions

@jax.jit
def stage_cost_x(x,y_ref):
    loss = (C@x-y_ref)**2
    return loss

@jax.jit
def stage_cost_u(u,u_ref):
    loss = Qu*(u-u_ref)**2 
    return loss

def stage_cost_x_cvx(x,y_ref):
    loss = cp.sum_squares(C@x-y_ref) 
    return loss

def stage_cost_u_cvx(u, u_ref):
    loss = Qu*cp.sum_squares(u-u_ref) 
    return loss

# ###################################

# ###################################
# Define dynamics
@jax.jit
def control_affine_dynamics(x):
    f = A@x
    g = B
    return f,g 

@jax.jit
def dynamics(x, u):    
    f, g = control_affine_dynamics(x)
    xnext = f + (g@u).reshape(nx,1)
    return xnext, xnext

@jax.jit
def simulation(x0, U):    
    _, X = jax.lax.scan(dynamics, jnp.array(x0), U)
    return X # return state trajectory x(1),x(2),...,x(H)

@jax.jit
def steady_state(y_ref):
    xu = jnp.linalg.solve(jnp.block([[np.eye(nx)-A, -B],[C, D]]), jnp.vstack((np.zeros((nx,1)),y_ref.reshape(1,1)))) 
    return xu[:nx], xu[nx:]

# # Test
# xss,uss = steady_state(np.array([1.]))
# X=simulation(xss,jnp.tile(uss,(100,1)))

# ###################################

pcf = PCF(activation='logistic', widths=widths, widths_psi=widths_psi)

options = {'iprint': -1, 'maxls': 20, 'gtol': 1.e-8, 'eps': 1.e-8,
            'ftol': 1.e-8, 'maxfun': 1000, 'maxcor': 10}
Umin = jnp.tile(umin.T,(H,1))
Umax = jnp.tile(umax.T,(H,1))
stage_cost_x_all = jax.vmap(stage_cost_x)
stage_cost_u_all = jax.vmap(stage_cost_u)

discount = gamma**jnp.arange(H)
@jax.jit
def opt_control_loss(U, x0, y_ref, u_ref):
    # performance index: \sum_{k=0}^{H-1} stage_cost(x(k+1),u(k)) starting from x(0)
    # Later on, this will be used (in surrogate form) as \sum_{k=1}^{H} stage_cost(x(k+1),u(k)) 
    # starting from x(1), so that by adding stage_cost(x(1),u(0)) we get the total cost starting 
    # from x(0). The optimal performance index only depends on x(0), y_ref, u_ref.
    
    X = simulation(x0, U)
    #loss = stage_cost_x(x0,y_ref)[0][0]
    loss = jnp.sum(discount*stage_cost_u_all(U,jnp.tile(u_ref,(H,1))).reshape(-1))
    loss += jnp.sum(discount*stage_cost_x_all(X,jnp.tile(y_ref,(H,1))).reshape(-1))
    # loss += 1e3*(X[-1,1]-y_ref)**2 # terminal cost
    return loss

if GenerateData:
    n_jobs = 10

    def solve_optimal_control_problem(seed):
        if not jax.config.jax_enable_x64:
            jax.config.update("jax_enable_x64", True)  # Enable 64-bit computations

        np.random.seed(seed)

        y_ref = (y_ref_max-y_ref_min)*np.random.rand(1)+y_ref_min # random reference signal
        xss, u_ref = steady_state(y_ref)
        
        x0 = xss + np.random.randn(nx,1) # perturb the initial steady state
        
        theoptions=options.copy()
        solver=jaxopt.ScipyBoundedMinimize(fun=opt_control_loss, method="L-BFGS-B", options=theoptions, maxiter=options['maxfun'])
        U_guess=jnp.tile(u_ref,(H,1))
        Uopt, status = solver.run(U_guess, bounds = (Umin,Umax), x0=x0, y_ref=y_ref, u_ref=u_ref)
        Xopt = simulation(x0,Uopt)
        loss = opt_control_loss(Uopt,x0,y_ref, u_ref)
        loss = loss.item()
        failed = np.isnan(loss) # or loss>30. # discard NaNs and outliers
        
        print(f"seed = {seed: 3d}, y_ref = {y_ref.item(): 5.4f}, y(end) = {(C@Xopt[-1,:]).item(): 5.4f}, cost = {loss: 5.4f}", end="") 
        if failed:
            print(" (FAILED!)")
        else:
            print(" (OK)")
        
        return {"y_ref": y_ref, "u_ref": u_ref, "x0": x0, "loss": loss, "failed": failed, "Uopt": Uopt, "Xopt": Xopt}
                
    results = Parallel(n_jobs=10)(delayed(solve_optimal_control_problem)(seed) for seed in range(M))
    
    pickle.dump(results, open('optimal_control_samples_linear.pkl', 'wb'))

else:
    results = pickle.load(open('optimal_control_samples_linear.pkl', 'rb'))

Y_REF = [results[i]["y_ref"] for i in range(M)]
U_REF = [results[i]["u_ref"] for i in range(M)]
X0 = [results[i]["x0"].T for i in range(M)]
FOPT = [results[i]["loss"] for i in range(M)]
FAILED = [results[i]["failed"] for i in range(M)]
UOPT = [results[i]["Uopt"] for i in range(M)]
XOPT = [results[i]["Xopt"] for i in range(M)]

# The value function fopt = V(x0,y_ref,u_ref)
F = np.array(FOPT).reshape(-1,1) # function values
X = np.vstack(X0) # variables
Theta = np.hstack((np.vstack(Y_REF),np.vstack(U_REF))) # parameters

# Clean up data by removing possible NaNs
ii=np.where(np.array(FAILED))[0]
F = np.delete(F,ii,axis=0)
X = np.delete(X,ii,axis=0)
Theta = np.delete(Theta,ii,axis=0)
Y_REF = np.delete(Y_REF,ii,axis=0)
U_REF = np.delete(U_REF,ii,axis=0)

UOPT = [Uopt for i, Uopt in enumerate (UOPT) if i not in ii]
XOPT = [Xopt for i, Xopt in enumerate (XOPT) if i not in ii]

# Normalize data
Fmax = np.max(F)
#Fmax=1.
F = F/Fmax # normalize function values (they are all nonnegative)

if TrainModel:
    stats = pcf.fit(F, X, Theta, rho_th=1.e-4, seeds=np.arange(10), cores=10)

    data = {"X": X, "Theta": Theta, "F": F, "y_ref": Y_REF, "u_ref": U_REF, "Fmax": Fmax, "stats": stats, "params": pcf.model.params}
    pickle.dump(data, open('optimal_control_fit_linear.pkl', 'wb'))

    print(f"Elapsed time: {stats['time']} s")

else:
    data = pickle.load(open('optimal_control_fit_linear.pkl', 'rb'))

    pcf.fit(data["F"][0:10], data["X"][0:10], data["Theta"][0:10], seeds=0, adam_epochs=1, lbfgs_epochs=0) # dummy fit to initialize model        
    pcf.model.params = data["params"]
    stats = data["stats"]
    Fmax = data["Fmax"]
    X = data["X"]
    Theta = data["Theta"]
    F = data["F"]
    Y_REF = data["y_ref"]
    U_REF = data["u_ref"]
    
print(f"R2 score on (u,p) -> y mapping:         {stats['R2']}")

# sparsity analysis
w=np.block([pcf.model.params[i].ravel() for i in range(len(pcf.model.params))])
nonzeros = np.sum(np.abs(w)>pcf.model.zero_coeff)
nz_perc = 100-100*nonzeros/len(w)
print(f"Number of non-zero parameters: {nonzeros} out of {len(w)} (zeros = {nz_perc:.2f}%)")

Fhat = pcf.model.predict(np.hstack((X, Theta)))
RMS = np.sqrt(np.sum((F-Fhat)**2)/F.shape[0])
print(f"RMS error on scaled training data: {RMS}")

# export to cvxpy
u_cvx = cp.Variable((nu,1))
f0_cvx = cp.Parameter((nx,1)) # x1 = f(x0) + g(x0)*u
g0_cvx = cp.Parameter((nx,1)) 
x1_cvx =  f0_cvx + g0_cvx@u_cvx
y_ref_cvx = cp.Parameter((1,1))
u_ref_cvx = cp.Parameter((1,1))

cvx_loss = gamma*Fmax*pcf.tocvxpy(x1_cvx,cp.vstack((y_ref_cvx,u_ref_cvx))) + stage_cost_u_cvx(u_cvx,u_ref_cvx) + stage_cost_x_cvx(x1_cvx,y_ref_cvx)

constr = [umin <= u_cvx, u_cvx <= umax]
cvx_prob = cp.Problem(cp.Minimize(cvx_loss), constr) 

# ###################################
# Closed-loop simulations for testing the ADP controller

seed = 0
np.random.seed(seed)

if 0:
    ind=12
    x0 = X0[ind]
    y_ref = np.array(Y_REF[ind])
    u_ref = np.array(U_REF[ind])

    xk = x0.copy()
    X= [x0]
    U= list()
    loss=0.
    gammak = 1.
    for k in range(H):
        y_ref_cvx.value = y_ref.reshape(1,1)
        u_ref_cvx.value = u_ref.reshape(1,1)
        fx,gx = control_affine_dynamics(xk.reshape(nx,1))
        f0_cvx.value = np.array(fx).reshape(nx,1)
        g0_cvx.value = np.array(gx).reshape(nx,1)
        cvx_prob.solve()
        uk = u_cvx.value
        xk = (fx.reshape(nx,1)+gx.reshape(nx,1)@uk).reshape(nx)
        X.append(xk)
        U.append(uk)
        loss += gammak*stage_cost_u(uk,u_ref)
        gammak *= gamma
        loss += gammak*stage_cost_x(xk,y_ref)
        # u0 = uk.copy()
        print(f"k={k+1: 3d}, y(k) = {(C@xk.reshape(nx,1)).item(): 5.4f}, "
            f"y_ref = {y_ref.item(): 5.4f}, "
            f"cost = {loss.item(): 5.4f}, "
            f"true cost = {(Fmax*F[ind]).item(): 5.4f}")
else:

    Nval = 4
    fig,ax = plt.subplots(2,2,figsize=(8,8))
    
    for h in range(Nval):
        y_init = (y_ref_max-y_ref_min)*np.random.rand(1)+y_ref_min # random reference signal
        x0, u0 = steady_state(y_init)

        y_ref = (y_ref_max-y_ref_min)*np.random.rand(1)+y_ref_min # random reference signal
        xss, u_ref = steady_state(y_ref)

        y_ref_cvx.value = y_ref.reshape(1,1)
        u_ref_cvx.value = np.array(u_ref.reshape(1,1))

        xk1 = x0.copy()
        xk2 = x0.copy()
        X1= [x0.reshape(nx)]
        X2= [x0.reshape(nx)]
        U1= list()
        U2= list()

        Uopt=jnp.tile(u_ref,(H,1))

        for k in range(H):
            # ADP solution
            fx,gx = control_affine_dynamics(xk1.reshape(nx,1))
            f0_cvx.value = np.array(fx).reshape(nx,1)
            g0_cvx.value = np.array(gx).reshape(nx,1)
            cvx_prob.solve()
            uk1 = u_cvx.value
            xk1 = (fx.reshape(nx,1)+gx.reshape(nx,1)@uk1).reshape(nx)
            X1.append(xk1)
            U1.append(uk1)
            
            # Optimal control solution
            theoptions=options.copy()
            solver=jaxopt.ScipyBoundedMinimize(fun=opt_control_loss, method="L-BFGS-B", options=theoptions, maxiter=options['maxfun'])
            U_guess=np.vstack((Uopt[1:,:],Uopt[-1,:])) # shifted previous optimal sequence
            Uopt, status = solver.run(Uopt, bounds = (Umin,Umax), x0=xk2.reshape(nx,1), y_ref=y_ref, u_ref=u_ref)
            uk2 = Uopt[0,:]
            fx,gx = control_affine_dynamics(xk2.reshape(nx,1))
            xk2 = (fx.reshape(nx)+gx.reshape(nx,1)@uk2).reshape(nx)
            X2.append(xk2)
            U2.append(uk2)
            
            print(f"h = {h+1: 3d}, k={k+1: 3d}")

        ax[h//2,h%2].plot(np.vstack(X1)@C.T,label='ADP')
        ax[h//2,h%2].plot(np.vstack(X2)@C.T,label='Optimal')
        ax[h//2,h%2].plot(np.ones(H)*y_ref,label='Reference')
        ax[h//2,h%2].set_title("output")
        ax[h//2,h%2].legend()
        ax[h//2,h%2].grid()
        print(" ")
        
    plt.show()
