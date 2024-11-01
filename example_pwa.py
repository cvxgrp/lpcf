
import pickle
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from itertools import product
from pcf import PCF

seed = 3
np.random.seed(seed)

# data generating (true) function

def plus(a):
    return np.maximum(a, 0)


def f_true(x, theta):
    splus, s_minus, m, v = theta
    return splus * plus(x - m) + s_minus * plus(m - x) + v


# generate data
if 0:
    n_rand  = 10
    x_      = -1 + 2 * np.random.rand(n_rand)
    splus_  = -1 + 2 * np.random.rand(n_rand)
    sminus_ = -1 + 2 * np.random.rand(n_rand)
    m_      = -1 + 2 * np.random.rand(n_rand)
    v_      = -1 + 2 * np.random.rand(n_rand)

    IN = np.array(list(product(x_, splus_, sminus_, m_, v_)))
    X, Theta = IN[:, 0], IN[:, 1:]
    Y = f_true(X, Theta.T)
    X = X.reshape(-1, 1)

else:

    Nx = 50 # number of x-data per parameter value
    Nth = 2000 # number of th-data
    x_      = np.linspace(-1.,1.,Nx)
    theta_  = -1. + 2. * np.random.rand(Nth,4)
    XTH = list(product(x_, theta_))
    X = np.array([XTH[i][0] for i in range(len(XTH))]).reshape(-1,1)
    Theta = np.array([XTH[i][1] for i in range(len(XTH))])
    Y = np.array([f_true(X[i],Theta[i]) for i in range(X.shape[0])])


# fit

pcf = PCF(widths=[1, 2], widths_psi=[10, 10], activation='relu')
stats = pcf.fit(Y, X, Theta, rho_th=1.e-8, tau_th=0., seeds=np.arange(10), cores=10, adam_epochs=200, lbfgs_epochs=2000)

print(f"Elapsed time: {stats['time']} s")
print(f"R2 score on (u,p) -> y mapping:         {stats['R2']}")

# export to jax

f, weights = pcf.tojax()

# evaluate f_true and f for 6 random points in Theta, over a grid of 100 points in X

np.random.seed(0)

x_ = np.linspace(-1, 1, 100)

y_true_, y_, theta_ = [], [], []
for _ in range(6):
    theta = -1 + 2 * np.random.rand(4)
    y_true_.append(f_true(x_, theta))
    y_.append(f(x_, np.tile(theta, (len(x_), 1)), weights))
    theta_.append(theta)
    
# pickle x_, y_true_, y_, theta_
with open('example_pwa.pkl', 'wb') as f:
    pickle.dump([x_, y_true_, y_, theta_], f)

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

x = cp.Variable((1, 1))
theta = cp.Parameter((4, 1))
cvxpy_model = pcf.tocvxpy(x=x, theta=theta)
