# Standard library imports
import math

# 3rd party imports
import numpy as np
import matplotlib.pyplot as plt

# local imports
import lava.core as lava

# Setup
np.random.seed(0)

# Options
A = np.array([[.95, .2],
              [-.2, .95]])
B = np.array([[1, 0],
              [0, 0.1]])
N = 200

# generate data
N_fit = int(math.ceil(N / 2))
N_test = N - N_fit
U = np.random.normal(0, 1, size=(2, N))
Y = np.zeros((2, N))
Y[:,0] = np.array([2,-2]).T
for t in range(N - 1):  Y[:, t + 1] = A @ Y[:, t] + B @ U[:, t]

# train lava model
arx_regressor = lava.ARXRegressor(y_lag_max=1, u_lag_max=1)
intercept_regressor = lava.InterceptRegressor()
lb = lava.Lava(nominal_model=arx_regressor, latent_model=intercept_regressor)
for t in range(N_fit):
    lb.step(y=Y[:, t], u=U[:, t])

# predict with lava model
Y_hat, Theta_phi, Z_gamma = lb.simulate(u=U[:, N_fit:N_fit + N_test])

# plot results
t_all = np.arange(0,N)
t_pred = np.arange(N_test,N)

plt.subplot(2, 1, 1)
plt.plot(t_all,Y[0,:])
plt.plot(t_pred,Y_hat[0, 0:N_test],".")
plt.legend(["y", "y_hat"])
plt.subplot(2, 1, 2)
plt.plot(t_all,Y[1,:])
plt.plot(t_pred,Y_hat[1, 0:N_test],".")
plt.legend(["y", "y_hat"])
plt.suptitle("Result for AR(1) system")
plt.show()