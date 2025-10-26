import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
# ---- LaTeX Setup for Matplotlib ----
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",  # Uses Computer Modern by default
    "axes.labelsize": 18,
    "font.size": 16,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "figure.dpi": 300,
    "text.latex.preamble": r"\usepackage{amsmath}"  # DON'T add newtxtext/newtxmath!
})
# --- Interactive File Input ---
filename_inputs = []
while True:
    a = input("File name (include .txt): (press Enter if None): ")
    if a == '':
        break
    filename_inputs.append(a)

# --- Predefine plot containers ---
x_all = []
y_all = []
px_all = []
py_all = []
pz_all = []
file_labels = []

# --- Define plot colors (cycled) ---
colors = itertools.cycle(['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple'])

# --- Start plotting ---
fig, ax = plt.subplots(3, 2, figsize=(13.0, 16.0))
count = 0
# --- Data extraction and plotting ---
for filename_input in tqdm(filename_inputs, desc="Processing files"):
    data = np.loadtxt(filename_input, usecols=[0, 1, 2, 3, 4, 5]) # ASCII, [0, 6, 7, 8, 9, 10, 11] for for009 x,y,z,px,py,pz
    data = data[:7000]
    color = next(colors)
    if count == 0:
        label = "Initial Input Beam (Pre-Wedge)"
    else:
        label = "Achromat Beam Output"
    # Temp storage
    x_fin, y_fin, px_fin, py_fin, pz_fin = [], [], [], [], []

    # Collect particles starting at x == 0
    for i in data:
        x_fin.append(i[0])     # mm 
        y_fin.append(i[1])
        px_fin.append(i[3])   # MeV/c
        py_fin.append(i[4])
        pz_fin.append(i[5])

    if len(x_fin) == 0:
        continue

    # Plot X phase space
    ax[0, 0].scatter(x_fin, px_fin, s=2, alpha=0.5, label=label, color=color)
    ax[1, 0].hist(px_fin, bins=64, alpha=0.5, label=label, color=color)
    ax[2, 0].hist(x_fin, bins=64, alpha=0.5, label=label, color=color)

    # Plot Y phase space
    ax[0, 1].scatter(y_fin, py_fin, s=2, alpha=0.5, label=label, color=color)
    ax[1, 1].hist(py_fin, bins=64, alpha=0.5, label=label, color=color)
    ax[2, 1].hist(y_fin, bins=64, alpha=0.5, label=label, color=color)
    
    count += 1

# --- Label & Style Plots ---
# X side
ax[0, 0].set_xlabel("x [mm]")
ax[0, 0].set_ylabel(r"$p_x$ [MeV/c]")
# ax[0, 0].set_title("Phase Space in x")
ax[1, 0].set_xlabel(r"$p_x$ [MeV/c]")
ax[1, 0].set_ylabel("Count")
ax[1, 0].set_title(r"Histogram of $p_x$")
ax[2, 0].set_xlabel("x [mm]")
ax[2, 0].set_ylabel("Count")
ax[2, 0].set_title("Histogram of x")

# Y side
ax[0, 1].set_xlabel("y [mm]")
ax[0, 1].set_ylabel(r"$p_y$ [MeV/c]")
# ax[0, 1].set_title("Phase Space in y")
ax[1, 1].set_xlabel(r"$p_y$ [MeV/c]")
ax[1, 1].set_ylabel("Count")
ax[1, 1].set_title(r"Histogram of $p_y$")
ax[2, 1].set_xlabel("y [mm]")
ax[2, 1].set_ylabel("Count")
ax[2, 1].set_title("Histogram of y")

# --- Legends ---
for axis in ax.flatten():
    axis.legend()

plt.tight_layout()
plt.savefig(f"combined_phase_space.png")
# plt.show()
