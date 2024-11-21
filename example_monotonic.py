""" Testing monotonic PCF functions

A. Bemporad, November 19, 2024
"""

import pickle
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from itertools import product
from pcf import PCF
import jax
import jax.numpy as jnp

seed = 3
np.random.seed(seed)

# Generate data
@jax.jit
def f_true(x, theta):
    y = (2.*(x-theta)+1.)**2
    return y

f_true_vec = jax.vmap(f_true, in_axes=(0,0))

X, Theta = np.meshgrid(np.linspace(-1, 1, 50), np.linspace(-1, 1, 50))
X = X.flatten()[:,None]
Theta = Theta.flatten()[:,None]
Y = f_true_vec(X,Theta)

# fit
pcf = PCF(activation='logistic', widths=[5,5], widths_psi=[10,10])
if 1:
    pcf.increasing()
elif 0:
    pcf.decreasing()
    
stats = pcf.fit(Y, X, Theta, cores=10)

print(f"Elapsed time: {stats['time']} s")
print(f"R2 score on (u,p) -> y mapping:         {stats['R2']}")
print(f"lambda value: {stats['lambda']}")

# export to jax
f = pcf.tojax()

# evaluate f_true and f over a grid of 100 points in X for each theta
x_ = np.linspace(-1, 1, 100)
y_true_, y_, theta_ = [], [], []
for theta in [-1., 0., .5, 1.]:
    y_true_.append(f_true(x_, theta))
    y_.append(f(x_, np.tile(theta, (len(x_), 1))))
    theta_.append(theta)

fig, axes = plt.subplots(2, 2, figsize=(6,6), sharex=True, sharey=True)

for i, ax in enumerate(axes.flat):
    ax.plot(x_, y_true_[i], label='y_true')
    ax.plot(x_, y_[i], label='y', linestyle='--')
    ax.set_title(f'theta = {np.round(theta_[i], 2)}')
    ax.set_xlabel('x')
    ax.grid()
    if i == 0:
        ax.legend()

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

print('done')

