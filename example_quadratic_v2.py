
import numpy as np
from pcf import PCF

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
    
sum_squares = 0
sum_squares_convex = 0
sum_squares_nonconvex = 0
sum_squares_proj = 0

num_convex = 0
num_nonconvex = 0

for i in range(len(Y_test)):
    sum_squares += (Y_test[i] - Y_hat[i])**2
    w, v = np.linalg.eigh(Theta_test[i].reshape(n, n))
    if all(w >= 0):
        sum_squares_convex += (Y_test[i] - Y_hat[i])**2
        num_convex += 1
    else:
        sum_squares_nonconvex += (Y_test[i] - Y_hat[i])**2
        w_proj = np.maximum(w, 0)
        theta_proj = v @ np.diag(w_proj) @ v.T
        y_proj = np.dot(X_test[i], theta_proj @ X_test[i])
        sum_squares_proj += (y_proj - Y_hat[i])**2
        num_nonconvex += 1
    
num = num_convex + num_nonconvex

print(f"Number of samples = {num}")
print(f"Number of convex samples = {num_convex}")
print(f"Number of nonconvex samples = {num_nonconvex}")
    
RMS = np.sqrt(sum_squares / num)

RMS_convex = np.sqrt(sum_squares_convex / num_convex)
RMS_nonconvex = np.sqrt(sum_squares_nonconvex / num_nonconvex)
RMS_proj = np.sqrt(sum_squares_proj / num_nonconvex)

print(f"RMS = {RMS}")
print(f"RMS convex = {RMS_convex}")
print(f"RMS nonconvex = {RMS_nonconvex}")
print(f"RMS proj = {RMS_proj}")

print('done')
