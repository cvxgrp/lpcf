
import pickle
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pcf import PCF

seed = 0
np.random.seed(seed)

# dimension

n = 2

# data generating (true) function

def f_true(x, theta_flat):
    theta = theta_flat.reshape(n, n)
    return x @ theta @ x


# generate data, half random symmetric, half random psd

n_theta, n_x_per_theta = 1000, 100
N = n_theta * n_x_per_theta

factors = [np.random.rand(n, n) * np.tri(n) for _ in range(n_theta // 2)]
summands = [np.random.rand(n, n) for _ in range(n_theta // 2)]
for i in range(len(summands)):
    summands[i] -= 0.5 * np.diag(np.diag(summands[i]))
theta_list = [factor @ factor.T for factor in factors] + [summand + summand.T for summand in summands]

X = - 1 + 2 * np.random.rand(N, n)
X /= np.maximum(np.linalg.norm(X, axis=1)[:, None], 1)

Theta = []
for theta in theta_list:
    theta_flat = theta.flatten()
    for _ in range(n_x_per_theta):
        Theta.append(theta_flat)
Theta = np.array(Theta)

Y = np.zeros((N, 1))
for i in range(N):
    Y[i] = f_true(X[i], Theta[i])
    
# fit

pcf = PCF(activation='logistic')
stats = pcf.fit(Y, X, Theta, seeds=np.arange(10), cores=10)

print(f"Elapsed time: {stats['time']} s")
print(f"R2 score on (u,p) -> y mapping:         {stats['R2']}")

# export to jax

f = pcf.tojax()

# evaluate f_true and f for for theta = a M1 + (1-a) M2, with M1, M2 random psd matrices and a in [0, 1]

np.random.seed(0)

K                   = 100
X1_test             = np.linspace(-1, 1, K)
X2_test             = np.linspace(-1, 1, K)
X1_grid, X2_grid    = np.meshgrid(X1_test, X2_test)

factors = [np.random.rand(n, n) * np.tri(n) for _ in range(2)]
summands = [np.random.rand(n, n) for _ in range(1)]
for i in range(len(summands)):
    summands[i] -= 0.5 * np.diag(np.diag(summands[i]))
theta_ = [factor @ factor.T for factor in factors] + [summand + summand.T for summand in summands]

y_true_ = []
y_ = []
for th in theta_:
    y_true = np.zeros((K, K))
    y = np.zeros((K, K))
    for i, x1 in enumerate(X1_test):
        for j, x2 in enumerate(X2_test):
            x = np.array([x1, x2])
            y_true[i, j] = f_true(x, th.flatten())
            y[i, j] = f(x.reshape(1, n), th.flatten().reshape(1, n**2))[0, 0]
    y_true_.append(y_true)
    y_.append(y)
    
# pickle X1_grid, X2_grid, y_true_, y_, a_
with open('example_quadratic.pkl', 'wb') as f:
    pickle.dump([X1_grid, X2_grid, y_true_, y_, theta_], f)

# plot

fig, axes = plt.subplots(1, 3, subplot_kw={'projection': '3d'}, figsize=(15, 8))

for i, ax in enumerate(axes.flat):
    
    # Plot the true surface
    ax.plot_surface(X1_grid, X2_grid, y_true_[i], color='blue', alpha=0.6, label='y_true')
    
    # Plot the predicted surface
    ax.plot_surface(X1_grid, X2_grid, y_[i], color='orange', alpha=0.4, label='y')
    
    mse = np.mean((y_true_[i] - y_[i])**2)
    ax.set_title(r'$\theta^' + str(i + 1) + r'$')
    
    ax.set_zlim(0, 2.5)
    
    if i == 0:
        ax.legend()

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

print('done')
