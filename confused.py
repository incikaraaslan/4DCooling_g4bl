import numpy as np
import matplotlib.pyplot as plt

# Currently this is looking at whether the particle distribution post-wedge in g4bl can be simulated by
# assuming a Gaussian distribution. If the distributions are matched, then I can feed these to the input
# beam of MadX and compare calculations a bit better ig.

# --- Python and MadX Analysis Settings ---
g4bl_file = "particles_after.txt"
madx_target_file = "G4BLtwiss.txt"  
Np_plot   = 50000                   # Number of particles            

# --- Helper Functions ---
def cov(a, b):
    return np.mean((a - np.mean(a)) * (b - np.mean(b)))

def compute_dispersion_from_samples(coord, slope, delta):
    # Return (D, Dp) = (cov(coord,delta)/var(delta), cov(slope,delta)/var(delta))
    var_d = np.var(delta)
    if var_d <= 0:
        return 0.0, 0.0
    D = cov(coord, delta) / var_d
    Dp = cov(slope, delta) / var_d
    return float(D), float(Dp)

# ---------------------------
# Read G4BL distribution
# ---------------------------
# assume columns (x, y, z, Px, Py, Pz)
data = np.loadtxt(g4bl_file)
if data.shape[1] >= 6:
    # many G4BL "extended" formats: assume columns [x y z Px Py Pz ...]
    # We will treat x,y in mm, convert to m for Twiss arithmetic; slopes from Px/Pz etc may need unit care.
    # BUT your earlier pipeline used columns: x px y py t pt  -> we handle that case below
    # Heuristic: if column 1 looks like a small angle (|val| < 1e-1) maybe it's xp not y
    col1 = data[:,1]
    if np.all(np.abs(col1) < 1.0) and np.mean(np.abs(col1)) < 0.1:
        # interpret as [x, xp, y, yp, t, pt]
        x_g4bl = data[:,0]   # mm
        xp_g4bl = data[:,1]  # rad or mrad? (we'll assume mrad-like scale, check units)
        y_g4bl = data[:,2]
        yp_g4bl = data[:,3]
        delta_g4bl = data[:,5]  # fractional dp/p
    else:
        # interpret as [x y z Px Py Pz ...]
        # We'll compute slopes xp = Px/Pz, yp = Py/Pz, and delta = (|P|-P0)/P0 approx
        x_g4bl = data[:,0]   # mm
        y_g4bl = data[:,1]
        Px = data[:,3]
        Py = data[:,4]
        Pz = data[:,5]
        # slopes xp, yp:
        xp_g4bl = Px / Pz    # unitless (rad)
        yp_g4bl = Py / Pz
        # approximate momentum deviation delta (relative)
        P = np.sqrt(Px**2 + Py**2 + Pz**2)
        delta_g4bl = (P - np.mean(P)) / np.mean(P)
        # convert slopes to mrad and positions to mm to match plotting earlier if desired
        xp_g4bl = xp_g4bl * 1e3   # rad -> mrad
        yp_g4bl = yp_g4bl * 1e3
else:
    raise RuntimeError("Unexpected G4BL file shape: need at least 6 columns or the known x xp y yp t pt format.")

# convert to numpy arrays
x_g4bl = np.asarray(x_g4bl)
xp_g4bl = np.asarray(xp_g4bl)
y_g4bl = np.asarray(y_g4bl)
yp_g4bl = np.asarray(yp_g4bl)
delta_g4bl = np.asarray(delta_g4bl)

# If positional units are in mm (likely), convert to meters for Twiss math below
# We'll keep the stored copies for plotting in mm (so no conversion for plotting),
# but the Twiss generation expects positions in meters when using beta*emit formulas.
# Decide: assume x_g4bl in mm -> convert to m for computing covariances and dispersions
x_m = x_g4bl * 1e-3
y_m = y_g4bl * 1e-3
# slopes: if xp_g4bl was in mrad convert to rad

# Heuristic: if mean(abs(xp)) > 1 => probably mrad; convert
if np.mean(np.abs(xp_g4bl)) > 1.0:
    xp_rad = xp_g4bl * 1e-3
    yp_rad = yp_g4bl * 1e-3
else:
    xp_rad = xp_g4bl
    yp_rad = yp_g4bl


# --- Target Twiss parameters to generate distribution ---
data = np.loadtxt(madx_target_file)

# --- Assignments ---
beta_x, alpha_x, emit_x, D_x, Dp_x = data[0]
beta_y, alpha_y, emit_y, D_y, Dp_y = data[1]

# --- Verify ---
print(f"Horizontal (x): beta={beta_x}, alpha={alpha_x}, emit={emit_x}, D={D_x}, Dp={Dp_x}")
print(f"Vertical   (y): beta={beta_y}, alpha={alpha_y}, emit={emit_y}, D={D_y}, Dp={Dp_y}")

# --- Generation of Distribution ---
sigx = np.sqrt(beta_x * beta_x)          # rms beam size [mm·mrad scale]
sigp = 11.0
x0   = np.random.normal(0, 1, Np_plot) * sigx
px0   = np.random.normal(0, 1, Np_plot) * sigx

sigy = np.sqrt(beta_y * beta_y)          # rms beam size [mm·mrad scale]
y0  = np.random.normal(0, 1, Np_plot) * sigy
py0   = np.random.normal(0, 1, Np_plot) * sigy

# Compute slope x'/y' for each particle
xp0  = (px0 - alpha_x * x0) / beta_x
delta = np.random.normal(0, 1, Np_plot) * sigp  # Δp/p distribution
x     = x0  + D_x  * delta                   # [mm]
xp    = xp0 + Dp_x * delta                   # [mrad]
px     = alpha_x * x + beta_x * xp              # conjugate variable, for consistency

yp0  = (py0 - alpha_y * y0) / beta_y
y     = y0  + D_y  * delta                   # [mm]
yp    = yp0 + Dp_y * delta                   # [mrad]
py     = alpha_y * y + beta_y * yp              # conjugate variable, for consistency

# Convert synthetic to plotting units (mm, mrad) to match G4BL earlier plots
x_madx = x
y_madx = y
xp_madx = xp
yp_madx = yp

# ---------------------------
# Generate MAD-X-style synthetic distribution (consistent δ for each particle)
# ---------------------------
"""Np = len(x_g4bl)

# Horizontal emittance & derived sigmas (units in meters and radians for generation)
sigx = np.sqrt(beta_x * emit_x)            # meters
sig_xp_rms = np.sqrt((1 + alpha_x**2) * emit_x / beta_x)   # radians
# Generate Courant-Snyder circular coordinates then form xp properly
# Approach: generate two standard normals for the ellipse, scale correctly
u = np.random.normal(0, 1, Np)
v = np.random.normal(0, 1, Np)
x0 = u * sigx
p0 = v * sigx
# convert (x0, p0) circular coords to physical x and xp using alpha,beta mapping:
# xp0 = (p0 - alpha*x0)/beta  (units: rad)
xp0 = (p0 - alpha_x * x0) / beta_x

# draw delta once per particle (same delta applied to both x and xp)
deltas = np.random.normal(0, delta, Np)

# apply dispersion (D in meters, Dp dimensionless)
x_madx_m = x0 + D_x * deltas
xp_madx_rad = xp0 + Dp_x * deltas

# Vertical
sigy = np.sqrt(beta_y * emit_y)
u = np.random.normal(0, 1, Np)
v = np.random.normal(0, 1, Np)
y0 = u * sigy
p0 = v * sigy
yp0 = (p0 - alpha_y * y0) / beta_y
y_madx_m = y0 + D_y * deltas
yp_madx_rad = yp0 + Dp_y * deltas

# Convert synthetic to plotting units (mm, mrad) to match G4BL earlier plots
x_madx = x_madx_m * 1e3
y_madx = y_madx_m * 1e3
xp_madx = xp_madx_rad * 1e3
yp_madx = yp_madx_rad * 1e3
"""
# --- Plot comparison ---
fig, axs = plt.subplots(1, 2, figsize=(12,5))

# Horizontal
axs[0].scatter(x_madx[:Np_plot], xp_madx[:Np_plot], s=1, alpha=0.5, label="MAD-X synthetic", color='orange')
axs[0].scatter(x_g4bl[:Np_plot], xp_g4bl[:Np_plot], s=1, alpha=0.5, label="G4BL")
axs[0].set_xlabel("x [mm]")
axs[0].set_ylabel("x' [mrad]")
axs[0].set_title("Horizontal phase space")
axs[0].legend()
axs[0].grid(True)

# Vertical
axs[1].scatter(y_madx[:Np_plot], yp_madx[:Np_plot], s=1, alpha=0.5, label="MAD-X synthetic", color='orange')
axs[1].scatter(y_g4bl[:Np_plot], yp_g4bl[:Np_plot], s=1, alpha=0.5, label="G4BL", color='green')
axs[1].set_xlabel("y [mm]")
axs[1].set_ylabel("y' [mrad]")
axs[1].set_title("Vertical phase space")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()