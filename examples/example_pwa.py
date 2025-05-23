
import pickle
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from itertools import product
from lpcf.pcf import PCF

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

pcf = PCF()
stats = pcf.fit(Y, X, Theta, cores=10)

print(f"Elapsed time: {stats['time']} s")
print(f"R2 score on (u,p) -> y mapping:         {stats['R2']}")
print(f"lambda value: {stats['lambda']}")

# export to jax

f = pcf.tojax()

# evaluate f_true and f for 6 random points in Theta, over a grid of 100 points in X

np.random.seed(0)

x_ = np.linspace(-1, 1, 100)

y_true_, y_, theta_ = [], [], []
for _ in range(1000):
    theta = -1 + 2 * np.random.rand(4)
    y_true_.append(f_true(x_, theta))
    y_.append(f(x_, np.tile(theta, (len(x_), 1))))
    theta_.append(theta)
    
# pickle x_, y_true_, y_, theta_
with open('example_pwa.pkl', 'wb') as f:
    pickle.dump([x_, y_true_, y_, theta_], f)

# plot

fig, axes = plt.subplots(3, 3, figsize=(15, 12))

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


# load [x_, y_true_, y_, theta_] that was saved to example_pwa.pkl
with open('example_pwa.pkl', 'rb') as f:
    x_, y_true_, y_, theta_ = pickle.load(f)
    

M = np.vstack((x_, np.ones_like(x_))).T
    
    
sum_squares = 0
sum_squares_convex = 0
sum_squares_nonconvex = 0
sum_squares_affine = 0

num_convex = 0
num_nonconvex = 0

for i in range(len(y_true_)):
    sum_squares += np.sum((y_true_[i] - y_[i].squeeze())**2)
    splus, s_minus, _, _ = theta_[i]
    if splus >= -s_minus:
        sum_squares_convex += np.sum((y_true_[i] - y_[i].squeeze())**2)
        num_convex += len(y_true_[i])
    else:
        sum_squares_nonconvex += np.sum((y_true_[i] - y_[i].squeeze())**2)
        coeff = np.linalg.lstsq(M, y_true_[i])[0]
        linear = M @ coeff
        sum_squares_affine += np.sum((linear - y_[i].squeeze())**2)
        num_nonconvex += len(y_true_[i])
    
num = num_convex + num_nonconvex

print(f"Number of samples = {num}")
print(f"Number of convex samples = {num_convex}")
print(f"Number of nonconvex samples = {num_nonconvex}")
    
RMS = np.sqrt(sum_squares / num)

RMS_convex = np.sqrt(sum_squares_convex / num_convex)
RMS_nonconvex = np.sqrt(sum_squares_nonconvex / num_nonconvex)
RMS_affine = np.sqrt(sum_squares_affine / num_nonconvex)

print(f"RMS = {RMS}")
print(f"RMS convex = {RMS_convex}")
print(f"RMS nonconvex = {RMS_nonconvex}")
print(f"RMS affine = {RMS_affine}")
    

indices = [0, 3, 6, 7]

# use latex for text rendering
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 20

fig, axes = plt.subplots(2, 2, figsize=(15, 8))

for i, ax in enumerate(axes.flat):
    
    mse = np.mean((y_true_[indices[i]] - y_[indices[i]])**2)
    print(f"Case {i}: MSE = {mse}")
    
    ax.plot(x_, y_true_[indices[i]], 'b', label=r'$f^{\mathrm{true}}$')
    
    if i in [2, 3]:
        coeff = np.linalg.lstsq(M, y_true_[indices[i]])[0]
        linear = M @ coeff
        ax.plot(x_, linear, 'gray', linestyle='dashdot', label=r'$f^{\mathrm{linear}}$')
        
    ax.plot(x_, y_[indices[i]], 'r--', label='$f$')
    
    ax.set_xlabel('$x$')
    ax.set_title(r'$\theta^' + str(i + 1) + r'$')
    if i in [0, 2]:
        ax.set_ylabel('$y$')
        ax.legend()

# Adjust layout to prevent overlap
plt.tight_layout()
#plt.savefig('example_pwa.pdf', bbox_inches='tight')
plt.show()
