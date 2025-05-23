
import pickle
import numpy as np
from lpcf.pcf import PCF

seed = 0
np.random.seed(seed)

# dimension

n = 3

# data generating (true) function

def f_true(x, theta_flat):
    theta = theta_flat.reshape(n, n)
    return x @ theta @ x


# generate data, half random symmetric, half random psd

def gen_Y_X_Theta():

    n_theta, n_x_per_theta = 1000, 100
    N = n_theta * n_x_per_theta

    factors = [-1 + 2 * np.random.rand(n, n) for _ in range(n_theta)]
    theta_list = [factor @ factor.T / n for factor in factors]

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
        
    return Y, X, Theta
    
# fit

Y, X, Theta = gen_Y_X_Theta()

pcf = PCF(activation='logistic')
stats = pcf.fit(Y, X, Theta, cores=10)

print(f"Elapsed time: {stats['time']} s")
print(f"R2 score on (u,p) -> y mapping:         {stats['R2']}")

# export to jax

f = pcf.tojax()

# evaluate f_true and f for for theta = a M1 + (1-a) M2, with M1, M2 random psd matrices and a in [0, 1]

np.random.seed(0)

Y_test, X_test, Theta_test = gen_Y_X_Theta()
Y_hat = f(X_test, Theta_test)
print(Y_hat[0])

import cvxpy as cp

x = cp.Variable((n, 1))
theta = cp.Parameter((n**2, 1))
cvxpy_model = pcf.tocvxpy(x=x, theta=theta)

x.value = X_test[0].reshape(n, 1)
theta.value = Theta_test[0].reshape(n**2, 1)
print(cvxpy_model.value)

print("Done")

# pickle Y_test, X_test, Theta_test, Y_hat
with open('example_quadratic.pkl', 'wb') as f:
    pickle.dump((Y_test, X_test, Theta_test, Y_hat), f)


with open('example_quadratic.pkl', 'rb') as f:
    Y_test, X_test, Theta_test, Y_hat = pickle.load(f)

rmse = np.sqrt(np.mean((Y_test - Y_hat)**2))

print(f"range of Y_test: {np.max(Y_test)}, {np.min(Y_test)}")
print(f"RMSE: {rmse}")
print('done')
