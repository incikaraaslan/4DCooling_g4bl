import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from g4beam import *

def plot_p_vs_z(df, label, ax):
    """Compute total momentum (MeV/c) vs. z position (mm) and plot."""
    p_tot = np.sqrt(df["Px"]**2 + df["Py"]**2 + df["Pz"]**2)
    ax.scatter(df["z"], p_tot, s=4, alpha=0.5, label=label)
    ax.set_xlabel("z position (mm)")
    ax.set_ylabel("Total momentum (MeV/c)")
    ax.legend()
    ax.grid(True)

# ---------- USER CONFIG ----------
post_wedge_file  = "out_1760039204_1614938.txt"
post_dipole_file = "vd_B1_achromat.txt"

# read files
df_wedge  = read_trackfile(post_wedge_file)
df_dipole = read_trackfile(post_dipole_file)

# plot
fig, axs = plt.subplots(2, 1, figsize=(8, 10), sharex=False)
plot_p_vs_z(df_wedge,  "Post-Wedge",  axs[0])
plot_p_vs_z(df_dipole, "Post-Dipole", axs[1])

axs[0].set_title("Total Momentum vs. Position (Post-Wedge)")
# axs[0].set_xlim(-0.0001, 0.0001)
axs[1].set_title("Total Momentum vs. Position (Post-Dipole)")
# axs[1].set_xlim(311.60, 311.62)
plt.tight_layout()
plt.show()
