import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# User settings
# -----------------------------
g4bl_file = "particles_after.txt"  # adjust path to your G4BL output
Np_plot   = 20000                  # max number of particles to plot for clarity

# -----------------------------
# Twiss parameters from MAD-X (example values)
# -----------------------------
beta_x = 20.0   # m
alpha_x = -0.0025
emit_x = 1.0    # pi mm-mrad

beta_y = 105.0
alpha_y = -5.0
emit_y = 1.0

D  = 0        # m
Dp = 0       # rad

# MAD-X longitudinal sigma (fractional momentum deviation)
sig_delta = 1e-3

# -----------------------------
# Read G4BL distribution
# -----------------------------
data_g4bl = np.loadtxt(g4bl_file)
x_g4bl, xp_g4bl = data_g4bl[:,0], data_g4bl[:,1]
y_g4bl, yp_g4bl = data_g4bl[:,2], data_g4bl[:,3]
delta_g4bl      = data_g4bl[:,5] if data_g4bl.shape[1] > 5 else np.zeros_like(x_g4bl)

Np = len(x_g4bl)  # same number of particles as G4BL

# -----------------------------
# Generate MAD-X synthetic distribution using covariance matrices
# -----------------------------
def generate_distribution(beta, alpha, emit, D, Dp, sig_delta, Np):
    gamma = (1 + alpha**2) / beta
    Sigma = emit * np.array([[beta, -alpha],
                             [-alpha, gamma]])
    
    # Draw correlated (x, x')
    x, xp = np.random.multivariate_normal([0, 0], Sigma, Np).T
    
    # Add dispersion effects
    deltas = np.random.normal(0, sig_delta, Np)
    x  += D * deltas
    xp += Dp * deltas
    
    return x, xp

# Horizontal
x_madx, xp_madx = generate_distribution(beta_x, alpha_x, emit_x, D, Dp, sig_delta, Np)

# Vertical
y_madx, yp_madx = generate_distribution(beta_y, alpha_y, emit_y, 0, 0, sig_delta, Np)

# -----------------------------
# Plot comparison
# -----------------------------
fig, axs = plt.subplots(1, 2, figsize=(12,5))

# Horizontal phase space
axs[0].scatter(x_g4bl[:Np_plot], xp_g4bl[:Np_plot], s=1, alpha=0.5, label="G4BL")
axs[0].scatter(x_madx[:Np_plot], xp_madx[:Np_plot], s=1, alpha=0.5, label="MAD-X")
axs[0].set_xlabel("x [mm]")
axs[0].set_ylabel("x' [mrad]")
axs[0].set_title("Horizontal phase space")
axs[0].legend()

# Vertical phase space
axs[1].scatter(y_g4bl[:Np_plot], yp_g4bl[:Np_plot], s=1, alpha=0.5, label="G4BL", color='green')
axs[1].scatter(y_madx[:Np_plot], yp_madx[:Np_plot], s=1, alpha=0.5, label="MAD-X", color='orange')
axs[1].set_xlabel("y [mm]")
axs[1].set_ylabel("y' [mrad]")
axs[1].set_title("Vertical phase space")
axs[1].legend()

plt.tight_layout()
plt.show()