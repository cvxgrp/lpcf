"""
Approximate Dynamic Programming by fitting optimal costs for control of a CSTR around
a given set-point.

Model and parameters from [1, Case 2, p. 563].

[1] B. Bequette, "Process Dynamics: Modeling, Analysis and Simulation", Prentice-Hall, 1998.

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

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
#plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

GenerateData=0
TrainModel=0

widths=[10,2]
widths_psi=[10,10]
rho_th = 1.e-8
tau_th = 0.e-5
n_seeds=10
adam_epochs=200
lbfgs_epochs=2000

# Generate optimal control data from random initial states
M = 100 # number of initial states
H = 200 # finite-time optimal control horizon
H1 = 20-1 # number of states along optimal trajectory included in the training dataset (number of samples N ~= (H1+1)*M, H1<=H-1)
Qu=0.01 # weight on deviations of inputs from steady state
gamma = 1.0 # discount factor

# Define model constants [1]
T0 = 273.15 # [K]
FoV = 1. # [hr^-1]
k0 = 9703*3600. # [hr^-1]
minusDeltaH = 5960. # [kcal/kgmol]
DeltaE = 11843. # [kcal/kgmol]
rhocp = 500. # [kcal/m^3/C]
Tf = 25. + T0 # [K]
CAf = 10. # [kmol/m^3]
UAoV = 150. # [kcal/(m^3 C hr)]
Tj = 25. + T0 # [C]
R = 1.985875 # [kcal/(kmol K)]

umin = np.array([285.15]) # [K]
umax = np.array([312.15]) # [K]
# range where initial states are generated
Tc_min = 273.15+5. # [K] 
Tc_max = 273.15+60. # [K]
CA_min = 7.0 # [kmol/m^3]
CA_max = 9.0 # [kmol/m^3]

CA_ref_min=7.5 # [kmol/m^3] CA setpoint range
CA_ref_max=8.5 # [kmol/m^3]

Ts = 0.5 # [hr] sampling time

nx = 2 # number of states
nu = 1 # number of inputs

# Scaling functions
def scale_T(T):
    return (T-300.)/10.
def unscale_T(T):
    return 10.*T+300.
def scale_CA(CA):
    return CA-7.75
def unscale_CA(CA):
    return CA+7.75
def scale_x(x):
    return jnp.hstack((scale_T(x[0]),scale_CA(x[1]))).reshape(x.shape)
def unscale_x(x):
    return jnp.hstack((unscale_T(x[0]),unscale_CA(x[1]))).reshape(x.shape)
def scale_x_cvx(x):
    return cp.hstack((scale_T(x[0]),scale_CA(x[1]))).reshape(x.shape)

umin = scale_T(umin)
umax = scale_T(umax)
CA_ref_min = scale_CA(CA_ref_min)
CA_ref_max = scale_CA(CA_ref_max)
# ###################################
# Define stage cost functions

@jax.jit
def stage_cost_x(x,x_ref):
    loss = jnp.sum((x[1]-x_ref[1])**2) # cost computed on scaled variables
    return loss

@jax.jit
def stage_cost_u(u,Tj_ref):
    loss = Qu*(u-Tj_ref)**2 # cost computed on scaled variables
    return loss

def stage_cost_x_cvx(x,x_ref):
    loss = cp.sum_squares(x[1]-x_ref[1]) # cost computed on scaled variables
    return loss

def stage_cost_u_cvx(u, Tj_ref):
    loss = Qu*cp.sum_squares(u-Tj_ref) # cost computed on scaled variables
    return loss
# ###################################

# ###################################
# Define dynamics
@jax.jit
def control_affine_dynamics(x):
    x=unscale_x(x)
    T = x[0]     # [K]
    CA = x[1]    # [kg mol/m^3]
    # T_c = u[0]  # manipulated input [K]

    r = k0 * jnp.exp(-DeltaE /R / T) * CA

    #xdot = jnp.array([FoV * (Tf - T) - UAoV/rhocp * (T - Tj) + minusDeltaH/rhocp * r,
    #                    FoV * (CAf - CA) - r])

    f = x + Ts * jnp.array([FoV * (Tf - T) - UAoV/rhocp * T + minusDeltaH/rhocp * r,
                        FoV * (CAf - CA) - r])
    g = Ts * jnp.array([UAoV/rhocp, 0.])
    return f,g # these are unscaled

@jax.jit
def dynamics(x, u):    
    # T_c = u[0]  # manipulated input [K]
    f, g = control_affine_dynamics(x)
    xnext = scale_x(f + g*unscale_T(u))
    return xnext, xnext

@jax.jit
def simulation(x0, U):    
    _, X = jax.lax.scan(dynamics, jnp.array(x0), U)
    return X # return state trajectory x(1),x(2),...,x(H)

@jax.jit
def ss_residual(xu,CA_ref):
    x = xu[:nx]
    u = xu[nx:]
    xnext,_ = dynamics(x,u)
    return jnp.hstack((xnext-x, xnext[1]-CA_ref))

@jax.jit
def steady_state(CA_ref):
    broyden  = jaxopt.Broyden(fun=ss_residual, tol=1.e-8, verbose=False)
    xu = jnp.array([scale_T(300.),scale_CA(8.),scale_T(300.)]) # initial guess
    xu = broyden.run(CA_ref=CA_ref, init_params=xu).params
    return xu[:nx], xu[nx:]

# # Test
# xss,uss = steady_state(jnp.array([scale_CA(8.)]))
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
def opt_control_loss(U, x0, x_ref, Tj_ref):
    # performance index: \sum_{k=0}^{H-1} stage_cost(x(k+1),u(k)) starting from x(0)
    # Later on, this will be used (in surrogate form) as \sum_{k=1}^{H} stage_cost(x(k+1),u(k)) 
    # starting from x(1), so that by adding stage_cost(x(1),u(0)) we get the total cost starting 
    # from x(0). The optimal performance index only depends on x(0), y_ref, u_ref.
    
    X = simulation(x0, U) # state trajectory x(1),x(2),...,x(H)
    #loss = stage_cost_x(x0,CA_ref)[0][0]
    loss = jnp.sum(discount*stage_cost_u_all(U,jnp.tile(Tj_ref,(H,1))).reshape(-1))
    loss += jnp.sum(discount*stage_cost_x_all(X,jnp.tile(x_ref,(H,1))).reshape(-1))
    # loss += 1e3*(X[-1]-x_ref)**2 # terminal cost
    return loss

if GenerateData:
    n_jobs = 10

    def solve_optimal_control_problem(seed):
        if not jax.config.jax_enable_x64:
            jax.config.update("jax_enable_x64", True)  # Enable 64-bit computations

        np.random.seed(seed)

        CA_ref = (CA_ref_max-CA_ref_min)*np.random.rand(1)+CA_ref_min # random reference signal
        x_ref, Tj_ref = steady_state(CA_ref)

        #CA_0 = (CA_ref_max-CA_ref_min)*np.random.rand(1)+CA_ref_min # random reference signal
        #x0, _ = steady_state(CA_0)
        
        #x0 = x_ref + np.array([np.random.rand()-5., 0.1*np.random.randn()]) # perturb the initial steady state
        xmin = scale_x(np.array([Tc_min, CA_min]))
        xmax = scale_x(np.array([Tc_max, CA_max]))
        x0 = jnp.array([(xmax[0]-xmin[0])*np.random.rand()+xmin[0], (xmax[1]-xmin[1])*np.random.rand()+xmin[1]]) # random initial state
        
        theoptions=options.copy()
        solver=jaxopt.ScipyBoundedMinimize(fun=opt_control_loss, method="L-BFGS-B", options=theoptions, maxiter=options['maxfun'])
        U_guess=jnp.tile(Tj_ref,(H,1))
        Uopt, status = solver.run(U_guess, bounds = (Umin,Umax), x0=x0, x_ref=x_ref, Tj_ref=Tj_ref)
        Xopt = simulation(x0,Uopt)
        loss = opt_control_loss(Uopt, x0, x_ref, Tj_ref) # total loss
        loss = loss.item()
        failed = np.isnan(loss) # or np.linalg.norm(Xopt[-1]-x_ref)>1.e-4 #loss>10. # discard NaNs and outliers

        loss_k = list()
        loss = 0.
        for k in range(H1,-1,-1): # k=H1-1,H1-1,...,0
            loss += gamma**k*(stage_cost_x(Xopt[k],x_ref)+stage_cost_u(Uopt[k],Tj_ref)).item()
            loss_k.append(loss)
        loss_k = loss_k[::-1]
        
        print(f"seed = {seed: 5d}"
            f", CA_ref = {unscale_CA(CA_ref.item()): 5.4f}"
			f", CA(end) = {unscale_CA(Xopt[-1,1]): 5.4f}"
			f", cost = {loss: 8.4f}", " (FAILED!)" if failed else " (OK)")
        
        return {"CA_ref": CA_ref, "Tj_ref": Tj_ref, "x_ref": x_ref, "x0": x0, "loss": loss, "failed": failed, "Uopt": Uopt, "Xopt": Xopt, "loss_k": loss_k}
                
    results = Parallel(n_jobs=10)(delayed(solve_optimal_control_problem)(seed) for seed in range(M))
    
    pickle.dump(results, open('optimal_control_samples_cstr.pkl', 'wb'))

else:
    results = pickle.load(open('optimal_control_samples_cstr.pkl', 'rb'))

CA_REF = [results[i]["CA_ref"] for i in range(M)]
TJ_REF = [results[i]["Tj_ref"] for i in range(M)]
X_REF = [results[i]["x_ref"] for i in range(M)]
X0 = [results[i]["x0"] for i in range(M)]
FOPT = [results[i]["loss"] for i in range(M)]
FAILED = [results[i]["failed"] for i in range(M)]
UOPT = [results[i]["Uopt"] for i in range(M)]
XOPT = [results[i]["Xopt"] for i in range(M)]
LOSS_K = [results[i]["loss_k"] for i in range(M)]

# Clean up data by removing possible NaNs
ii=np.where(np.array(FAILED))[0]
X0 = np.delete(X0,ii,axis=0)
X_REF = np.delete(X_REF,ii,axis=0)
CA_REF = np.delete(CA_REF,ii,axis=0)
TJ_REF = np.delete(TJ_REF,ii,axis=0)
UOPT = [Uopt for i, Uopt in enumerate (UOPT) if i not in ii]
XOPT = [Xopt for i, Xopt in enumerate (XOPT) if i not in ii]
LOSS_K = [loss_k for i, loss_k in enumerate (LOSS_K) if i not in ii]
NEXP = len(X0) # number of valid experiments

XOPT = np.array(XOPT)
X_REF = np.array(X_REF)
X0 = np.array(X0)

X = np.hstack([X0[:,np.newaxis,:]-X_REF[:,np.newaxis,:], XOPT[:,0:H1]-np.tile(X_REF[:,np.newaxis,:],(H1,1))]).reshape(-1,nx)
Theta =np.repeat(np.hstack((X_REF,TJ_REF)), repeats = H1+1, axis=0)
F = np.vstack(LOSS_K).reshape(-1,1) # function values

# Normalize data
#Fmax = np.max(F)
#Foff = 1.0
Foff = 0.
Fmax=1.
F = F/Fmax + Foff # normalize and shift function values (all values of F are nonnegative)

if TrainModel:
    pcf.argmin(fun=None, penalty=1.e4)

    stats = pcf.fit(F, X, Theta, rho_th=rho_th, tau_th = tau_th, seeds=np.arange(n_seeds), cores=10, adam_epochs=adam_epochs, lbfgs_epochs=lbfgs_epochs)

    data = {"X": X, "Theta": Theta, "F": F, "Fmax": Fmax, "Foff": Foff, "stats": stats, "params": pcf.model.params}
    pickle.dump(data, open('optimal_control_fit_cstr.pkl', 'wb'))

    print(f"Elapsed time: {stats['time']} s")

else:
    data = pickle.load(open('optimal_control_fit_cstr.pkl', 'rb'))

    pcf.fit(data["F"][0:10], data["X"][0:10], data["Theta"][0:10], seeds=0, adam_epochs=1, lbfgs_epochs=0) # dummy fit to initialize model        
    pcf.model.params = data["params"]
    stats = data["stats"]
    Fmax = data["Fmax"]
    Foff = data["Foff"]
    X = data["X"]
    Theta = data["Theta"]
    F = data["F"]
    
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
x_ref_cvx = cp.Parameter((nx,1))
x1_cvx =  scale_x_cvx(f0_cvx + g0_cvx@unscale_T(u_cvx))
CA_ref_cvx = cp.Parameter((1,1))
Tj_ref_cvx = cp.Parameter((1,1))

cvx_loss = gamma*(Foff+Fmax*pcf.tocvxpy(x1_cvx-x_ref_cvx,cp.vstack((x_ref_cvx,Tj_ref_cvx)))) + stage_cost_u_cvx(u_cvx,Tj_ref_cvx) + stage_cost_x_cvx(x1_cvx,x_ref_cvx)

constr = [umin <= u_cvx, u_cvx <= umax]
cvx_prob = cp.Problem(cp.Minimize(cvx_loss), constr) 

if 0:
    # Check argmin of the surrogate model
    dx = cp.Variable((nx,1))
    f_free = pcf.tocvxpy(dx,cp.vstack((x_ref_cvx,Tj_ref_cvx)))
    cvx_prob = cp.Problem(cp.Minimize(f_free))
    CAk = list()
    dxk = list()
    fk = list()
    Fmax = np.max(F)
    for ind in range(NEXP):
        x_ref = X_REF[ind]
        Tj_ref = TJ_REF[ind]
        x_ref_cvx.value = x_ref.reshape(nx,1)
        Tj_ref_cvx.value = Tj_ref.reshape(1,1)
        cvx_prob.solve()
        dxk.append(np.linalg.norm(dx.value))
        CAk.append(unscale_CA(x_ref[1].item()))
        fk.append(f_free.value.item())
        print(f"CA = {unscale_CA(x_ref[1].item()): 5.4f}, dx = {dxk[-1]: 5.4f}, f = {fk[-1]: 5.4f}")
    print(f"max(||dx||) = {np.max(np.abs(dxk)): 5.4f}, max(|f|) = {np.max(np.abs(fk)): 5.4f},  Fmax = {Fmax: 5.4f}")
seed = 0
np.random.seed(seed)

# ###################################
# Closed-loop simulations for testing the ADP controller
if 0:
    ind=1
    x0 = X0[ind]
    x_ref = X_REF[ind]
    CA_ref = CA_REF[ind]
    Tj_ref = TJ_REF[ind]

    xk = x0.copy()
    X= [x0]
    U= list()
    loss_x0=stage_cost_x(x0.reshape(nx,1),x_ref).item()

    loss=loss_x0
    gammak = 1.
    for k in range(H):
        x_ref_cvx.value = x_ref.reshape(nx,1)
        CA_ref_cvx.value = CA_ref.reshape(1,1)
        Tj_ref_cvx.value = Tj_ref.reshape(1,1)
        fx,gx = control_affine_dynamics(xk.reshape(nx,1))
        f0_cvx.value = np.array(fx).reshape(nx,1)
        g0_cvx.value = np.array(gx).reshape(nx,1)
        cvx_prob.solve()
        uk = u_cvx.value
        xk = scale_x(fx.reshape(nx,1)+gx.reshape(nx,1)@unscale_T(uk)).reshape(nx)
        X.append(xk)
        U.append(uk)
        loss += gammak*stage_cost_u(uk,Tj_ref)
        gammak *= gamma
        loss += gammak*stage_cost_x(xk,x_ref)
        #u0 = uk.copy()
        print(f"k={k+1: 3d}, CA = {unscale_CA(xk[1].item()): 5.4f}, "
            f"CA_ref = {unscale_CA(CA_ref.item()): 5.4f}, "
            f"cost = {loss.item(): 5.4f}, "
            f"true cost = {Fmax*F[ind].item()+Foff+loss_x0: 5.4f}")
else:

    Nval = 2
    fig,ax = plt.subplots(2,2,figsize=(8,6))

    for h in range(Nval):
        if h==0:
            CA_init = np.array(scale_CA(8.5))
            CA_ref = np.array(scale_CA(7.5))
        else:
            CA_init = np.array(scale_CA(7.5))
            CA_ref = np.array(scale_CA(8.5))

        x0, u0  = steady_state(CA_init)
        x_ref, Tj_ref = steady_state(CA_ref)

        x_ref_cvx.value = np.array(x_ref.reshape(nx,1))
        CA_ref_cvx.value = CA_ref.reshape(1,1)
        Tj_ref_cvx.value = np.array(Tj_ref.reshape(1,1))

        xk1 = x0.copy()
        xk2 = x0.copy()
        X1= [x0.reshape(nx)]
        X2= [x0.reshape(nx)]
        U1= list()
        U2= list()

        Uopt=jnp.tile(Tj_ref,(H,1))

        H2=50
        for k in range(H2):
            # ADP solution
            fx,gx = control_affine_dynamics(xk1.reshape(nx,1))
            f0_cvx.value = np.array(fx).reshape(nx,1)
            g0_cvx.value = np.array(gx).reshape(nx,1)
            cvx_prob.solve()
            uk1 = u_cvx.value
            xk1 = scale_x(fx.reshape(nx,1)+gx.reshape(nx,1)@unscale_T(uk1)).reshape(nx)
            X1.append(xk1)
            U1.append(uk1)
            
            # Optimal control solution
            theoptions=options.copy()
            solver=jaxopt.ScipyBoundedMinimize(fun=opt_control_loss, method="L-BFGS-B", options=theoptions, maxiter=options['maxfun'])
            U_guess=np.vstack((Uopt[1:,:],Uopt[-1,:])) # shifted previous optimal sequence
            Uopt, status = solver.run(Uopt, bounds = (Umin,Umax), x0=xk2.reshape(nx), x_ref=x_ref, Tj_ref=Tj_ref)
            uk2 = Uopt[0,:]
            fx,gx = control_affine_dynamics(xk2.reshape(nx,1))
            xk2 = scale_x(fx.reshape(nx,1)+gx.reshape(nx,1)@unscale_T(uk2.reshape(1,1))).reshape(nx)
            X2.append(xk2)
            U2.append(uk2)
            
            print(f"h={h+1: 2d}, k={k+1: 2d}"
                  f", CA = {unscale_CA(xk1[1].item()): 5.4f}"
                  f", CA_ref = {unscale_CA(x_ref[1].item()): 5.4f}"
                  f", T = {unscale_CA(xk1[0].item()): 5.4f}"
                  f", T_ref = {unscale_CA(x_ref[0].item()): 5.4f}")

        ax[0,h].plot(unscale_CA(np.vstack(X1)[:,1]),label='ADP')
        ax[0,h].plot(unscale_CA(np.vstack(X2)[:,1]),label='Optimal')
        ax[0,h].plot(np.ones(H2)*unscale_CA(CA_ref),label='Reference')
        ax[0,h].set_title("Concentration [kmol/m$^3$]", fontsize=12)
        ax[0,h].legend()
        ax[0,h].grid()

        ax[1,h].plot(unscale_T(np.vstack(U1)),label='ADP')
        ax[1,h].plot(unscale_T(np.vstack(U2)),label='Optimal')
        #ax[1,h].plot(np.ones(H)*unscale_T(Tj_ref),label='Reference')
        ax[1,h].plot(np.ones(H2)*unscale_T(umin), 'k--')
        ax[1,h].plot(np.ones(H2)*unscale_T(umax), 'k--')
        ax[1,h].set_title("Jacket Temperature [K]", fontsize=12)
        ax[1,h].legend()
        ax[1,h].grid()
        print(" ")
        
    plt.show()
    plt.savefig('example_control.pdf')