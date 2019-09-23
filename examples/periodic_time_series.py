"""
Time series
-----------
Example of periodic time series identified using LAVA-R and Fourier basis function.
Developed in a previous version of LAVA, with special handling of binary inputs.
This version shows some high frequency issues, that could be addressed with other RegressorModel subclasses.
"""

# Imports.
import math
import numpy as np
import matplotlib.pyplot as plt
import lava.core as lava

# Generate U
N = 1000
N_fit = int(math.ceil(N / 2))
N_test = N - N_fit
Y = np.zeros((1, N))
U = np.zeros((2, N))

# Generate synthetic data
for t in range(N):
    U[0, t] = np.remainder(t, 24)
    U[1, t] = t + 4 * np.random.randn(1, 1)
    p = 24
    r = np.remainder(t + 5, p)
    if r > (p / 2):
        Y[:, t] = 100 + t
    else:
        Y[:, t] = 200 + t

# Create lava object
ar = lava.ARXRegressor(y_lag_max=0, u_lag_max=2)
four = lava.FourierRegressor(fourier_order=20, periodicity_y=500, periodicity_u=[24, 24], lags_y=0, lags_u=2)
lava_obj = lava.Lava(nominal_model=ar, latent_model=four)

# Identify system parameters
for t in range(N_fit):
    lava_obj.step(y=Y[:, t], u=U[:, t])

# Forecast
Y_hat, Theta_phi, Z_gamma = lava_obj.simulate(U[:, N_fit:N_fit + N_test])

# Plot results
plt.plot(Y[0, N_fit:N_fit + N_test])
plt.plot(Y_hat[0, 0:N_test])
plt.plot(Z_gamma[0, 0:N_test])
plt.plot(Theta_phi[0, 0:N_test])
plt.legend(["y", "y_hat", "Z_gamma", "Theta_phi"])
plt.show()

plt.boxplot(lava_obj.Z.flatten())
plt.title("Boxplot of coefficients in latent model.")
plt.show()

plt.hist(lava_obj.Z.flatten(), bins=50)
plt.title("Histogram of coefficients in latent model.")
plt.show()
