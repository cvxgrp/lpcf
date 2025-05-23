
import pickle
import numpy as np
from lpcf.pcf import PCF

seed = 0
np.random.seed(seed)

# dimension

n = 2

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

pcf = PCF(activation='logistic', quadratic=True, quadratic_r=1)
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
with open('example_quadratic_v2_n2.pkl', 'wb') as f:
    pickle.dump((Y_test, X_test, Theta_test, Y_hat), f)


# load [X1_grid, X2_grid, y_true_, y_, a_] that was saved to example_quadratic.pkl
with open('example_quadratic.pkl', 'rb') as f:
    X1_grid, X2_grid, y_true_, y_, theta_ = pickle.load(f)
    
    
sum_squares = 0
sum_squares_convex = 0
sum_squares_nonconvex = 0
sum_squares_proj = 0

num_convex = 0
num_nonconvex = 0

for i in range(len(y_true_)):
    sum_squares += np.sum((y_true_[i] - y_[i])**2)
    w, v = np.linalg.eigh(theta_[i])
    if all(w >= 0):
        sum_squares_convex += np.sum((y_true_[i] - y_[i])**2)
        num_convex += y_true_[i].size
    else:
        sum_squares_nonconvex += np.sum((y_true_[i] - y_[i])**2)
        w_proj = np.maximum(w, 0)
        theta_proj = v @ np.diag(w_proj) @ v.T
        y_proj = np.zeros_like(y_true_[i])
        for j, x1 in enumerate(X1_grid[0,:]):
            for k, x2 in enumerate(X2_grid[:,0]):
                x = np.array([x1, x2])
                y_proj[j, k] = np.dot(x, theta_proj @ x)
        sum_squares_proj += np.sum((y_proj - y_[i])**2)
        num_nonconvex += y_true_[i].size
    
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


# use latex for text rendering
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 15


indices = [0, 1, 5, 6]


fig, axes = plt.subplots(2, 2, subplot_kw={'projection': '3d'}, figsize=(12, 8))

for idx, ax in enumerate(axes.flat):
    
    i = indices[idx]
    
    mse = np.mean((y_true_[i] - y_[i])**2)
    print(f"Case {i}: MSE = {mse}")
    
    # Plot the true surface
    ax.plot_surface(X1_grid, X2_grid, y_true_[i], color='blue', alpha=0.6, label=r'$f^{\mathrm{true}}$')
    
    if idx in [2, 3]:
        # eigenvalue decomposition of theta_[i]
        w, v = np.linalg.eigh(theta_[i])
        # project eigenvalues to nonnegative
        w_proj = np.maximum(w, 0)
        # reconstruct matrix
        theta_proj = v @ np.diag(w_proj) @ v.T
        y_proj = np.zeros_like(y_true_[i])
        for j, x1 in enumerate(X1_grid[0,:]):
            for k, x2 in enumerate(X2_grid[:,0]):
                x = np.array([x1, x2])
                y_proj[j, k] = np.dot(x, theta_proj @ x)
        ax.plot_surface(X1_grid, X2_grid, y_proj, color='gray', alpha=0.6, label=r'$f^{\mathrm{proj}}$')
    
    # set rotation of plot around vertical axis
    ax.view_init(elev=20, azim=285)
    
    # Plot the predicted surface
    ax.plot_surface(X1_grid, X2_grid, y_[i], color='red', alpha=0.4, label='$f$')
    
    ax.set_title(r'$\theta^' + str(idx + 1) + r'$')
    
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$y$')
    
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    
    #ax.set_zlim(0, 2.5)
    
    if idx in [0, 2]:
        ax.legend(loc='upper left')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.subplots_adjust(wspace=-0.5, hspace=0.3)
plt.savefig('example_quadratic.pdf', bbox_inches='tight')
plt.show()