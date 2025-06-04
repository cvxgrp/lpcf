
import pickle
import numpy as np
from lpcf.pcf import PCF
import matplotlib.pyplot as plt

np.random.seed(3)

# data generating (true) function
# x = (E, I)
# theta = (A, T)

E_A = 31500.
R_g = 8.3145
T_0 = 273.15

z = 0.60
alpha, beta, eta = 28.966, 74.112, 152.5


def f_true(x, theta, scale=1000):
    q, b = x
    A, T = theta
    return scale * z * A**(z-1) * np.abs(b) * (alpha * q + beta) * np.exp((-E_A + eta * b) / (R_g * (T_0 + T)))


# generate data

def gen_theta():
    return np.array([0, 10]) + np.array([50, 40]) * np.random.rand(2)

K                   = 10
X1_train            = np.linspace(0.2, 0.8, K)
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

print(f'Elapsed time: {stats['time']} s')
print(f'R2 score: {stats['R2']}')

# export to jax

f = pcf.tojax()

# evaluate

np.random.seed(0)

K1, K2              = 100, 100
X1_test             = np.linspace(0.2, 0.8, K1)
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
    theta_.append(th)
    
with open('example_battery.pkl', 'wb') as f:
    pickle.dump([X1_grid, X2_grid, X1_test, X2_test, y_true_, y_, theta_], f)

# plot

fig, axes = plt.subplots(3, 3, subplot_kw={'projection': '3d'}, figsize=(15, 8))

for i, ax in enumerate(axes.flat):
    
    ax.plot_surface(X1_grid, X2_grid, y_true_[i], color='blue', alpha=0.6, label='y_true')
    ax.plot_surface(X1_grid, X2_grid, y_[i], color='orange', alpha=0.4, label='y')
    
    mse = np.mean((y_true_[i] - y_[i])**2)
    ax.set_title(r'$\theta^' + str(i + 1) + r'$')
        
    if i == 0:
        ax.legend()

plt.tight_layout()
plt.show()

with open('example_battery.pkl', 'rb') as f:
    X1_grid, X2_grid, X1_test, X2_test, y_true_, y_, theta_ = pickle.load(f)

scale = 1000

def f_short(x, theta, scale=scale):
    q, b = x
    A, T = theta
    mu = beta * np.exp(-E_A / (R_g * (T_0 + T))) * z * A**(z-1)
    nu = alpha / beta
    return scale * mu * (1 + nu / 2) * b
    
y_short_ = []
for k in range(10):
    y_short = np.zeros((len(X2_test), len(X1_test)))
    for i, x1 in enumerate(X1_test):
        for j, x2 in enumerate(X2_test):
            x = np.array([x1, x2])
            y_short[j, i] = f_short(x, theta_[k])
    y_short_.append(y_short)
    
sum_squares = 0
sum_squares_short = 0
num = 0

mi = np.inf
ma = -np.inf

for i in range(len(y_true_)):
    
    num += y_true_[i].size
    y_true_[i] /= scale
    
    mi = min(mi, np.min(y_true_[i]))
    ma = max(ma, np.max(y_true_[i]))
    
    y_[i] /= scale
    sum_squares += np.sum((y_true_[i] - y_[i])**2)
    
    y_short_[i] /= scale
    sum_squares_short += np.sum((y_true_[i] - y_short_[i])**2)
    

RMS = np.sqrt(sum_squares / num)
RMS_short = np.sqrt(sum_squares_short / num)

print(f'Ranging from {mi} to {ma}')
print(f'Number of samples = {num}')
print(f"RMS = {RMS}")
print(f"RMS short = {RMS_short}")

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 20

indices = [6, 1, 3]

fig, axes = plt.subplots(1, 3, subplot_kw={'projection': '3d'}, figsize=(14, 8))

for idx, ax in enumerate(axes.flat):
    
    i = indices[idx]
    
    ax.plot_surface(X1_grid, X2_grid, y_true_[i], color='blue', alpha=0.6, label=r'$f^{\mathrm{true}}$')
    ax.plot_surface(X1_grid, X2_grid, y_[i], color='red', alpha=0.4, label='$f$')
    
    ax.view_init(elev=20, azim=220)
    ax.set_title(r'$\theta^' + str(idx + 1) + r'$')
    
    ax.set_xlabel('$q$')
    ax.set_ylabel('$b$')
    
    ax.set_zlim(0, 0.02)
    ax.set_zticks([0, 0.01, 0.02])
    ax.set_xlim(0.1, 0.9)
    
    if idx == 0:
        ax.set_zlabel('$y$')
        ax.legend()

plt.tight_layout()
plt.subplots_adjust(left=0.05, right=1.0, top=1.0, bottom=0.)
plt.show()
