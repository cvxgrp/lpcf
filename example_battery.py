
import pickle
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from itertools import product
from pcf import PCF

seed = 3
np.random.seed(seed)

# data generating (true) function
# x = (E, I)
# theta = (A, T)

E_A = 31500.
R_g = 8.3145

z = 0.57
alpha, beta, eta = 2800., 6700., 150.


def f_true(x, theta):
    E, I = x
    A, T = theta
    return (alpha * E + beta) * np.exp((-E_A + eta * I) / (R_g * (273.15 + T))) * A**z


# generate data

def gen_theta():
    return np.array([0, 10]) + np.array([50, 40]) * np.random.rand(2)

K                   = 10
X1_train            = np.linspace(20, 80, K)
X2_train            = np.linspace(0, 30, K)
X1_grid, X2_grid    = np.meshgrid(X1_train, X2_train)

y_true_, theta_, x_ = [], [], []
for _ in range(1000):
    th = gen_theta()
    y_true = np.zeros((K, K))
    for i, x1 in enumerate(X1_train):
        for j, x2 in enumerate(X2_train):
            x = np.array([x1, x2])
            x_.append(x)
            theta_.append(th)
            y_true_.append(f_true(x, th))
    
Y = np.array(y_true_)
X = np.array(x_)

Theta = np.array(theta_)

# fit

pcf = PCF(widths=[5, 5], widths_psi=[10], activation='logistic')
stats = pcf.fit(Y, X, Theta, rho_th=0., cores=10, adam_epochs=1000, lbfgs_epochs=4000)

print(f"Elapsed time: {stats['time']} s")
print(f"R2 score on (u,p) -> y mapping:         {stats['R2']}")
print(f"lambda value: {stats['lambda']}")

# export to jax

f = pcf.tojax()

# evaluate f_true and f for 6 random points in Theta, over a grid of 100 points in X

np.random.seed(0)

K1, K2              = 100, 100
X1_test             = np.linspace(20, 80, K1)
X2_test             = np.linspace(0, 30, K2)
X1_grid, X2_grid    = np.meshgrid(X1_test, X2_test)

y_true_, y_, theta_ = [], [], []
for _ in range(10):
    th = gen_theta()
    y_true = np.zeros((K2, K1))
    y = np.zeros((K2, K1))
    for i, x1 in enumerate(X1_test):
        for j, x2 in enumerate(X2_test):
            x = np.array([x1, x2])
            y_true[j, i] = f_true(x, th)
            y[j, i] = f(x.reshape(1, 2), th.reshape(1, 2))[0, 0]
    y_true_.append(y_true)
    y_.append(y)
    
# pickle X1_grid, X2_grid, y_true_, y_, a_
with open('example_battery.pkl', 'wb') as f:
    pickle.dump([X1_grid, X2_grid, y_true_, y_, theta_], f)

# plot

fig, axes = plt.subplots(3, 3, subplot_kw={'projection': '3d'}, figsize=(15, 8))

for i, ax in enumerate(axes.flat):
    
    # Plot the true surface
    ax.plot_surface(X1_grid, X2_grid, y_true_[i], color='blue', alpha=0.6, label='y_true')
    
    # Plot the predicted surface
    ax.plot_surface(X1_grid, X2_grid, y_[i], color='orange', alpha=0.4, label='y')
    
    mse = np.mean((y_true_[i] - y_[i])**2)
    ax.set_title(r'$\theta^' + str(i + 1) + r'$')
    
    ax.set_zlim(0, 40)
    
    if i == 0:
        ax.legend()

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

print('done')
