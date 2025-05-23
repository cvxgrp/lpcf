""" Testing convex classification problem to fit PCF function to feasible/infeasible datapoints.

A. Bemporad, November 22, 2024
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
    y = (-.5*theta[0]+1.)*x[0]**2 + (5.*theta[0]+1.)*x[1]**2 + .5*theta[0] -.9
    return y
f_true_vec = jax.vmap(f_true, in_axes=(0,0))

N=10000
xmin = -1.*np.ones(2)
xmax = 1.*np.ones(2)
thmin = 0.*np.ones(1)
thmax = 1.*np.ones(1)
X = (xmax-xmin)*np.random.rand(N,2)+xmin
Theta = (thmax-thmin)*np.random.rand(N,1)+thmin
Y = 1.-2.*(f_true_vec(X,Theta)<=0)

# fit
pcf = PCF(activation='logistic', widths=[], widths_psi=[5], classification=True)
    
stats = pcf.fit(Y, X, Theta, tau_th=0.e-5, zero_coeff=1.e-4, seeds=range(10), cores=10,
                adam_epochs=100, lbfgs_epochs=2000)

print(f"Elapsed time: {stats['time']} s")
print(f"Accuracy on (u,p) -> y mapping:         {stats['Accuracy']} %")
print(f"lambda value: {stats['lambda']}")

np.random.seed(1)
x1, x2 = np.meshgrid(np.linspace(xmin[0],xmax[0],100),np.linspace(xmin[1],xmax[1],100))
XX = np.hstack((x1.reshape(-1,1),x2.reshape(-1,1)))

#xval1, xval2 = np.meshgrid(np.linspace(xmin[0],xmax[0],20),np.linspace(xmin[1],xmax[1],20))
xval1 = X[:,0]
xval2 = X[:,1]
xval = np.hstack((xval1.reshape(-1,1),xval2.reshape(-1,1)))

y_=list()
theta_=list()
yval_=list()

for theta in list(np.array([0.,.25,.75,1.])):
    TTheta = np.tile(theta,(x1.size,1))
    y_.append(pcf.predict(XX,TTheta).reshape(x1.shape)<=0.)
    yval_.append((1.-2.*(f_true_vec(xval,np.tile(theta,(xval.shape[0],1)))<=0.)).reshape(xval.shape[0]))
    theta_.append(theta)

fig, axes = plt.subplots(2, 2, figsize=(6,6), sharex=True, sharey=True)

for i, ax in enumerate(axes.flat):
    ax.contour(x1,x2,y_[i])
    ax.set_title(f'theta = {np.round(theta_[i], 2)}')
    ax.set_xlabel('x_1')
    ax.set_ylabel('x_2')
    ax.grid()
    ii=(yval_[i]==-1)
    ax.scatter(xval[~ii,0],xval[~ii,1], marker='o', alpha=0.2, s=10)
    ax.scatter(xval[ii,0],xval[ii,1], marker='o', alpha=0.7, s=10)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

print('done')

