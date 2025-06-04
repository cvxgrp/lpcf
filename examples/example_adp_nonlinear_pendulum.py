"""
Approximate Dynamic Programming by fitting optimal costs for control of nonlinear system.

A. Bemporad, May 7, 2025
"""

import pickle
import numpy as np
import cvxpy as cp
from lpcf.pcf import PCF
from lpcf.utils import _compute_r2
import jax
import jax.numpy as jnp
import jaxopt
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, cpu_count
from functools import partial

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
#plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
plt.rcParams['font.size'] = 20
np.random.seed(1)

GenerateData=0
TrainModel=0
CompareInputs=1
CompareClosedLoop=1
PlotValueFunction=0

testset_size = .3 # fraction of data used for testing

widths=[20,20]
widths_psi=[10,10]
rho_th = 1.e-8
tau_th = 1.e-1
n_seeds=cpu_count() # number of random seeds for training
adam_epochs=1000
lbfgs_epochs=5000

# Generate optimal control data from random initial states
M = 1000 # number of initial states
H = 150 # finite-time optimal control horizon, large enough to approximate the infinite horizon problem

umin = -50.*1000
umax = 50.*1000

# range where initial states are generated
xmin = np.array([-np.pi/6,-1.])
xmax = np.array([np.pi+np.pi/6,1.])

theta_ref = np.pi # reference angle (swing pendulum up)

mass_min = 0.5 # mass of pendulum [kg]
mass_max = 2.0

Ts = 0.02 # sampling time (s)
nx = 2 # number of states
nu = 1 # number of inputs

L=1.0 # length of pendulum [m]
g=9.81 # gravity [m/s^2]
b=0.05 # damping coefficient [kg*m^2/s]

model_params = [] # placeholder for extra model parameters to pass

# ###################################
# Define stage cost functions
beta=0.001 # weight on control input

@jax.jit
def stage_cost(x,u, beta):
    theta = x[0]  # [rad]
    loss = (theta-theta_ref)**2 + .01*x[1]**2 + beta*u**2
    return loss

def stage_cost_cvx(x,u, beta):
    loss = cp.sum_squares(x[0]-theta_ref) + .01*cp.sum_squares(x[1]) + beta*cp.sum_squares(u)
    return loss

# ###################################
# Define dynamics
c2=g/L

@jax.jit
def control_affine_dynamics(x, model_params):
    # input-affine pendulum model: ml^2 \ddot{\theta} + b\dot{\theta} + mgl \sin(\theta) = u
    #theta = x[0]  # [rad]
    #theta_dot = x[1] # [rad/s]
    # xdot = jnp.array([x[1], 
    #                   -b/(m*l**2)*x[1]-g/l*jnp.sin(theta)+u/(m*l**2)]) # [rad/s, rad/s^2] 
    # xnext = x + Ts*xdot

    m = model_params # mass of pendulum [kg]
    c3=1/(m*L**2)
    c1=b*c3
    
    f = x + Ts*jnp.array([x[1],
                          -c1*x[1]-c2*jnp.sin(x[0])]) 
    g = Ts*jnp.array([0.,c3])
    
    # f = x + Ts*jnp.array([x[1],
    #                       -c1*x[1]-x[0]]) # Model linearized around x=0
    # g = Ts*jnp.array([0.,c3])
    
    return f,g 

@jax.jit
def dynamics(x, u, model_params):    
    f, g = control_affine_dynamics(x, model_params)
    xnext = f + g*u
    return xnext, xnext

@jax.jit
def simulation(x0, U, model_params):
    dynamics_gain = partial(dynamics, model_params=model_params)    
    _, X = jax.lax.scan(dynamics_gain, jnp.array(x0), U)
    return jnp.squeeze(X) # return state trajectory x(1),x(2),...,x(H)

# # Test
# p=0.
# uss = 1.
# xss = np.array([10.,10.])
# X=simulation(xss,jnp.tile(uss,(100,1)), p)
# plt.plot(X)

options = {'iprint': -1, 'maxls': 20, 'gtol': 1.e-12, 'eps': 1.e-12,
            'ftol': 1.e-12, 'maxfun': 5000, 'maxcor': 20}
Umin = jnp.tile(umin,(H,1))
Umax = jnp.tile(umax,(H,1))

stage_cost_all = jax.vmap(stage_cost, in_axes=(0, 0, None))

@jax.jit
def opt_control_loss(U, x0, beta, model_params):
    # performance index: \sum_{k=0}^{H-1} stage_cost(x(k),u(k)) starting from x(0)
    # Later on, this will be used (in surrogate form) as \sum_{k=1}^{H} stage_cost(x(k),u(k)) 
    # starting from x(1), so that by adding stage_cost(x(0),u(0)) we get the total cost starting 
    # from x(0). The optimal performance index only depends on x(0)
    
    X = simulation(x0, U, model_params) # state trajectory x(1),x(2),...,x(H)
    
    #loss = stage_cost_x(x0,q_ref)[0][0]
    loss = stage_cost(x0,U[0,0],beta) + jnp.sum(stage_cost_all(X[:-1],U[1:],beta))
       
    # loss += 1e3*(X[-1]-x_ref)**2 # terminal cost
    return loss

if GenerateData:
    n_jobs = cpu_count()

    def solve_optimal_control_problem(seed):
        if not jax.config.jax_enable_x64:
            jax.config.update("jax_enable_x64", True)  # Enable 64-bit computations

        np.random.seed(seed)

        # p = pmin + np.random.rand(1)*(pmax-pmin)               
        
        # random initial state
        if 1: # np.random.rand() < 0.5: 
            x0 = ((xmax-xmin)*np.random.rand(nx)+xmin).reshape(nx) # full range
        else:
            # around the reference with small velocity
            x0 = np.array([theta_ref+np.random.randn()*np.pi/20., np.random.randn()*.01])
            
        mass = mass_min + np.random.rand()*(mass_max-mass_min) # random mass
        model_params = jnp.array(mass) # pass mass as model parameter
        theoptions=options.copy()
        solver=jaxopt.ScipyBoundedMinimize(fun=opt_control_loss, method="L-BFGS-B", options=theoptions, maxiter=5000)
        
        # initial guess for control inputs
        if 0:
            U_guess=np.tile(.5*(umin+umax), (H,1))
        elif 1:
            U_guess = np.zeros((H,nu)) 
        else:
            # solve smaller optimal control problem
            H1=50
            Uopt_short, status = solver.run(np.zeros((H1,nu)), bounds = (Umin[0:H1],Umax[0:H1]), x0=x0, beta=beta, model_params=model_params)
            U_guess = np.vstack((Uopt_short, np.tile(np.zeros(nu), (H-H1,1))))            
        
        Uopt, status = solver.run(U_guess, bounds = (Umin,Umax), x0=x0, beta=beta, model_params=model_params)
        loss = status.fun_val
        #print(np.hstack((np.sum(K_lqr*x0),Uopt[0])))
        #print(np.hstack(((K_lqr@Xopt[:-1,:].T).T,Uopt[1:].reshape(-1,1))))
        failed = np.isnan(loss) or not status.success # or loss>2000. # discard NaNs and outliers
        print(f"seed = {seed: 5d}, cost = {loss: 8.4f} " + ("(FAILED!)" if failed else "(OK)"))        
        return {"x0": x0, "loss": loss, "failed": failed, "Uopt": Uopt, "mass": mass}
                
    data = Parallel(n_jobs=cpu_count())(delayed(solve_optimal_control_problem)(seed) for seed in range(M))
    
    pickle.dump(data, open('optimal_control_samples_nonlinear.pkl', 'wb'))

else:
    data = pickle.load(open('optimal_control_samples_nonlinear.pkl', 'rb'))

X0 = [data[i]["x0"] for i in range(M)]
FOPT = [data[i]["loss"] for i in range(M)]
FAILED = [data[i]["failed"] for i in range(M)]
UOPT = [data[i]["Uopt"] for i in range(M)]
MASS = [data[i]["mass"] for i in range(M)]

# Clean up data by removing possible NaNs
ii=np.where(np.array(FAILED))[0]
X0 = np.delete(X0,ii,axis=0)
UOPT = [Uopt for i, Uopt in enumerate (UOPT) if i not in ii]
FOPT = np.delete(FOPT,ii,axis=0)
MASS = np.delete(MASS,ii,axis=0)

X0 = np.array(X0)
UOPT = np.array(UOPT)
FOPT = np.array(FOPT)
MASS = np.array(MASS)

X = X0
Theta = MASS.reshape(-1,1) # parameters
F = FOPT # function values
U = np.hstack([UOPT[:,0]]).reshape(-1,nu) # control inputs

# Check solution
def check_solution(i):
    x0=X0[i]
    Uopt = UOPT[i]
    model_params=np.array(MASS[i])
    Xopt = simulation(x0,Uopt,model_params=model_params)

    fig, ax = plt.subplots(2,1)
    ax[0].plot(np.arange(Xopt.shape[0])*Ts, Xopt[:,0], label='$\\theta$')
    ax[0].plot(np.arange(Xopt.shape[0])*Ts, np.tile(theta_ref,(Xopt.shape[0],1)), '--')
    ax[0].plot(0.,x0[0], 'o')
    ax[0].legend()
    ax[0].grid()
    ax[1].plot(np.arange(Xopt.shape[0]-1)*Ts, Uopt[:-1], label=f'$u$ ($m$={MASS[i]:.5f})')
    ax[1].legend()
    ax[1].grid()
    plt.show()
    return Xopt,Uopt

if 0:
    for i in range(10):
        check_solution(i)
    raise ValueError("\U0001F7E5"*10 + " STOP HERE")

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
MASS_train = MASS[ii[0:N_train]]

X_test = X[ii[-N_test:],:]
Theta_test = Theta[ii[-N_test:],:]
F_test = F[ii[-N_test:]]
U_test = U[ii[-N_test:],:]
MASS_test = MASS[ii[-N_test:]]

print(f"Set size: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")

# ###################################
pcf = PCF(activation='logistic', activation_psi='logistic', widths=widths, widths_psi=widths_psi, quadratic=True)
@jax.jit
def g(theta):
    # f should be minimized for x = g(theta)
    return jnp.array([theta_ref, 0.])
pcf.argmin(fun=g, penalty=1.e2) # add regularization on gradient with respect to x at x=0

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
    
#R2_train = _compute_r2(F_train, pcf.model.predict(np.hstack((X_train, Theta_train)).reshape(-1, pcf.n + pcf.p)))
R2_train = stats['R2']
R2_test= _compute_r2(F_test, pcf.model.predict(np.hstack((X_test, Theta_test)).reshape(-1, pcf.n + pcf.p)))

print(f"\U0001F7E5"*30)
print(f"R2 score on (x,th) -> y mapping: {R2_train:.6f} (training data), {R2_test:.6f} (test data)")

# sparsity analysis
w=np.block([pcf.model.params[i].ravel() for i in range(len(pcf.model.params))])
nonzeros = np.sum(np.abs(w)>pcf.model.zero_coeff)
nz_perc = 100-100*nonzeros/len(w)
print(f"Number of non-zero parameters:   {nonzeros} out of {len(w)} (zeros = {nz_perc:.2f}%)")

n_out_psi=pcf.w_psi[-1]
print(f"Number of outputs of the psi-network: {n_out_psi}")
print(f"Training time: {stats['time']:.2f} s")
print(f"\U0001F7E5"*30)

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
mass_cvx = cp.Parameter((1,1), nonneg=True)
x1_cvx =  f0_cvx + g0_cvx@u_cvx

cvx_loss = Foff+Fmax*pcf.tocvxpy(x1_cvx, mass_cvx)+stage_cost_cvx(x0_cvx, u_cvx, beta) # stage cost

constr = [umin <= u_cvx, u_cvx <= umax]
cvx_prob = cp.Problem(cp.Minimize(cvx_loss), constr) 

if CompareInputs:
    # Evaluate control inputs on test and training data

    # Evaluate control inputs on test data
    Uhat_test = np.empty((N_test,nu))
    for i in range(N_test):
        x0 = X_test[i]
        mass_cvx.value = MASS_test[i].reshape(1,1)
        x0_cvx.value = x0.reshape(nx,1)
        fx,gx = control_affine_dynamics(x0.reshape(nx), model_params=np.array(MASS_test[i]))
        f0_cvx.value = np.array(fx).reshape(nx,1)
        g0_cvx.value = np.array(gx).reshape(nx,1)
        #cvx_prob.solve(solver=cp.CLARABEL, tol_gap_abs=1.e-12)
        cvx_prob.solve()
        Uhat_test[i,:] = u_cvx.value.reshape(nu)

    # Evaluate control inputs on training data    
    Uhat_train = np.empty((N_train,nu))
    for i in range(N_train):
        x0 = X_train[i]
        mass_cvx.value = MASS_train[i].reshape(1,1)
        x0_cvx.value = x0.reshape(nx,1)
        fx,gx = control_affine_dynamics(x0.reshape(nx), model_params=np.array(MASS_train[i]))
        f0_cvx.value = np.array(fx).reshape(nx,1)
        g0_cvx.value = np.array(gx).reshape(nx,1)
        #cvx_prob.solve(solver=cp.CLARABEL, tol_gap_abs=1.e-12)
        cvx_prob.solve()
        Uhat_train[i,:] = u_cvx.value.reshape(nu)

    plt.figure()
    plt.scatter(U_train,Uhat_train, label='training data')
    plt.scatter(U_test,Uhat_test, label='test data')
    plt.plot([umin,umax], [umin,umax], 'r--')
    xumin = np.minimum(np.min(U_train),np.min(U_test))-10.
    xumax = np.maximum(np.max(U_train),np.max(U_test))+10.
    plt.xlim(xumin,xumax)
    plt.ylim(xumin,xumax)
    plt.xlabel('$u^*_0$ (nonlinear)')
    plt.ylabel('$\hat u_0$ (convex)')
    plt.grid()
    plt.legend()
    plt.show()
    plt.title('Convex ADP vs nonlinear optimal control', fontsize=20)
    plt.savefig('adp_inputs.pdf', bbox_inches='tight')

if CompareClosedLoop:
    # Generate closed-loop test data using the optimal control policy (receding horizon)
    def closed_loop_optimal(x0, method, N_sim, mass, model_params):        
        xk = x0.copy()
        X= [x0.reshape(nx)]
        U= list()
        model_params = np.array(mass)
        
        Uopt=np.zeros((H,1))
        for k in range(N_sim):            
            if method=='nonlinear':
                theoptions=options.copy()
                solver=jaxopt.ScipyBoundedMinimize(fun=opt_control_loss, method="L-BFGS-B", options=theoptions, maxiter=options['maxfun'])
                U_guess=np.vstack((Uopt[1:,:],Uopt[-1,:])) # shifted previous optimal sequence
                Uopt, status = solver.run(U_guess, bounds = (Umin,Umax), x0=xk.reshape(nx), beta=beta, model_params=model_params)
                uk = Uopt[0,:]
                fx,gx = control_affine_dynamics(xk.reshape(nx), model_params=model_params)
            elif method=='adp':
                # use the surrogate model to compute the optimal control input
                mass_cvx.value = np.array(mass).reshape(1,1)
                x0_cvx.value = xk.reshape(nx,1)
                fx,gx = control_affine_dynamics(xk.reshape(nx), model_params=model_params)
                f0_cvx.value = np.array(fx).reshape(nx,1)
                g0_cvx.value = np.array(gx).reshape(nx,1)
                cvx_prob.solve()
                uk = u_cvx.value.reshape(nu)
            else:
                raise ValueError("Unknown method")
            
            xk = np.array(fx.reshape(nx,1)+gx.reshape(nx,1)@uk.reshape(1,1)).reshape(nx)
            X.append(xk)
            U.append(uk)
            
            print(f"k={k+1: 2d}"
                  f", x = [{xk[0].item(): 5.4f}, {xk[1].item(): 5.4f}], u = {uk[0].item(): 5.4f}")
        return np.vstack(X), np.vstack(U)
    
    # Swing up the pendulum from the down position
    mass=1.
    x0 = np.array([0.,0.])
    Xn,Un = closed_loop_optimal(x0, 'nonlinear', H, mass, model_params)
    Xa,Ua = closed_loop_optimal(x0, 'adp', H, mass, model_params)
    
    fig,ax = plt.subplots(2,1)
    ax[0].plot(np.arange(Xn.shape[0])*Ts, Xn[:,0], label='nonlinear')
    ax[0].plot(np.arange(Xn.shape[0])*Ts, Xa[:,0], label='ADP')
    ax[0].grid()
    ax[0].legend()
    ax[0].set_ylabel('$\\delta$ [rad]')
    ax[1].set_xlabel('time [s]')
    ax[1].set_ylabel('$u$ [Nm]')    
    ax[1].plot(np.arange(Xn.shape[0]-1)*Ts, Un, label='nonlinear')
    ax[1].plot(np.arange(Xn.shape[0]-1)*Ts, Ua, label='ADP')
    ax[1].grid()
    ax[1].legend()
    plt.show()
    ax[0].set_title('Closed-loop control', fontsize=20)
    plt.savefig('adp_closed_loop.pdf', bbox_inches='tight')
    
    
if PlotValueFunction:
    # Plot value function
    nth=100
    nthdot = 10
    x1 = np.linspace(xmin[0],xmax[0],nth)
    x2 = np.linspace(xmin[1],xmax[1],nthdot)
    X1, X2 = np.meshgrid(x1,x2)
    mass = 1.
    Z = np.empty(X1.shape)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            x0 = np.array([X1[i,j],X2[i,j]])
            theoptions=options.copy()
            solver=jaxopt.ScipyBoundedMinimize(fun=opt_control_loss, method="L-BFGS-B", options=theoptions, maxiter=5000)
            U_guess = np.zeros((H,nu)) 
            Uopt, status = solver.run(U_guess, bounds = (Umin,Umax), x0=x0, beta=beta, model_params=jnp.array(mass))
            Z[i,j] = status.fun_val
            print(f"i={i: 2d}, j={j: 2d}, cost = {status.fun_val: 8.4f} " + ("(FAILED!)" if status.status != 0 else "(OK)"))
    
    # Plot the level sets of the optimal value function
    plt.figure(figsize=(8, 6))
    contour = plt.contour(X1, X2, Z, levels=10, cmap='viridis')
    #plt.colorbar(contour)
    plt.title("Optimal value function")
    plt.xlabel('$\\theta$ [rad]')
    plt.ylabel('$\\dot{\\theta}$ [rad/s]')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(8, 6))
    for i in range(nthdot):
        plt.plot(X1[i,:], Z[i,:], label=f'$\\dot{{\\theta}}$={X2[i,0]:.2f}')
    plt.grid()
    plt.title("Optimal value function")
    plt.xlabel('$\\theta$ [rad]')
    plt.ylabel('cost')
    plt.legend()
    plt.show()
            
    plt.figure(figsize=(8, 6))
    for i in range(nth):
        plt.plot(X2[:,i], Z[:,i])
    plt.grid()
    plt.title("Optimal value function")
    plt.xlabel('$\\dot\\theta$ [rad/s]')
    plt.ylabel('cost')
    plt.legend()
    plt.show()

    # Compute PCF
    Zhat = np.empty(X1.shape)
    for i in range(X1.shape[1]):
        Zhat[:,i] = pcf.model.predict(np.hstack((X1[:,i], X2[:,i],np.tile(mass,X1.shape[0]))).reshape(-1, pcf.n + pcf.p)).reshape(-1)
    Zhat = Foff + Fmax*Zhat

    # Approximate value function
    plt.figure(figsize=(8, 6))
    contour = plt.contour(X1, X2, Zhat, levels=10, cmap='viridis')
    #plt.colorbar(contour)
    plt.title("Parametric Convex Function")
    plt.xlabel('$\\theta$ [rad]')
    plt.ylabel('$\\dot{\\theta}$ [rad/s]')
    plt.legend()
    plt.grid()
    plt.show()
