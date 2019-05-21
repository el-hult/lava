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
B = np.array([[.3, 0],
              [0, 0.1]])
N = 400

# generate data
N_fit = int(math.ceil(N / 2))
N_test = N - N_fit
U = np.random.normal(0, 1, size=(2, N))
Y = np.zeros((2, N))
Y[:, 0:2] = np.array([[2, -2], [2, -2]]).T
for t in range(N - 2):
    Y[:, t + 2] = A @ Y[:, t] + B @ U[:, t + 1] + np.random.normal(0, .2)

# train lava model and plot results
# do this twice - once with ARX overparametrized as latent, and once as nominal model
for k in [0, 1]:
    ar_order = 3
    arx_regressor = lava.ARXRegressor(y_lag_max=ar_order, u_lag_max=ar_order)
    intercept_regressor = lava.InterceptRegressor()

    if k == 0:
        lb = lava.Lava(nominal_model=intercept_regressor, latent_model=arx_regressor)
    else:
        lb = lava.Lava(nominal_model=arx_regressor, latent_model=intercept_regressor)

    for t in range(N_fit):
        lb.step(y=Y[:, t], u=U[:, t])

    # predict with lava model
    Y_hat, _, _ = lb.simulate(u=U[:, N_fit:N_fit + N_test])

    # plot results
    t_all = np.arange(0, N)
    t_pred = np.arange(N_test, N)

    plt.subplot(2, 1, 1)
    plt.plot(t_all, Y[0, :])
    plt.plot(t_pred, Y_hat[0, 0:N_test], ":")
    plt.legend(["y", "y_hat"])
    plt.subplot(2, 1, 2)
    plt.plot(t_all, Y[1, :])
    plt.plot(t_pred, Y_hat[1, 0:N_test], ":")
    plt.legend(["y", "y_hat"])
    plt.suptitle(
        f"Predictions, using {lb.nominal_model.__class__.__name__} as nominal, "
        f"and {lb.latent_model.__class__.__name__} as latent", fontsize=10)
    plt.show()

    plt.figure()
    ar_matrix = lb.Theta if k == 1 else lb.Z

    affects_names = ['y0', 'y1']
    affecting_names_y = [f"{p} Lag{l}" for l in range(1, ar_order + 1) for p in affects_names]
    affecting_names_u = [f"{p} Lag{l}" for l in range(1, ar_order + 1) for p in ['u0', 'u1']]
    affecting_names = [*affecting_names_y, *affecting_names_u, 'Intercept']

    fig = plt.figure()
    ax = plt.gca()
    im = plt.imshow(ar_matrix)

    # We want to show all ticks...
    ax.set_yticks(np.arange(len(affects_names)))
    ax.set_xticks(np.arange(len(affecting_names)))
    # ... and label them with the respective list entries
    ax.set_yticklabels(affects_names)
    ax.set_xticklabels(affecting_names)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(affects_names)):
        for j in range(len(affecting_names)):
            text = ax.text(j, i, ar_matrix[i, j].round(2),
                           ha="center", va="center", color="w")

    ax.set_title(
        f"AR-matrices, using {lb.nominal_model.__class__.__name__} as nominal, and"
        f" {lb.latent_model.__class__.__name__} as latent", fontsize=10)

    fig.tight_layout()
    plt.show()
