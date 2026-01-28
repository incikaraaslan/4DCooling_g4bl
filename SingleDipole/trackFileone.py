import numpy as np
import matplotlib.pyplot as plt
import os

from g4beam import *
from scan import *

# -------------------------------
# Plasma color palette
# -------------------------------
cmap = plt.cm.plasma
colors = cmap(np.linspace(0.1, 0.9, 9))

# -------------------------------
# Configuration
# -------------------------------
TRACKFILE = "AllTracks.txt"
OUTDIR = "figs"
POINTSIZE = 1

os.makedirs(OUTDIR, exist_ok=True)



# -------------------------------
# Load data
# -------------------------------
df = read_trackfile(TRACKFILE)
P = p_total(df)
Pmean = np.mean(P)
delta = (P - Pmean) / Pmean

dfs = df[np.abs(df["z"] - df["z"].max()) < 0.01]
print(f"Particles in slice: {len(dfs)}")
print("Unique track IDs in slice:", dfs["TrackID"].nunique())
print("Rows per particle:", len(dfs) / dfs["TrackID"].nunique())

# -------------------------------
# 1. Horizontal phase space (x, x')
# -------------------------------
plt.figure()
plt.scatter(df["x"], df["Px"] / P, s=POINTSIZE, color=colors[0])
plt.xlabel("x [mm]")
plt.ylabel("x' = Px / P")
plt.title("Horizontal phase space")
plt.savefig(f"{OUTDIR}/x_xp.png", dpi=300)
plt.close()

# -------------------------------
# 2. Vertical phase space (y, y')
# -------------------------------
plt.figure()
plt.scatter(df["y"], df["Py"] / P, s=POINTSIZE, color=colors[1])
plt.xlabel("y [mm]")
plt.ylabel("y' = Py / P")
plt.title("Vertical phase space")
plt.savefig(f"{OUTDIR}/y_yp.png", dpi=300)
plt.close()

# -------------------------------
# 3. Raw horizontal momentum phase space
# -------------------------------
plt.figure()
plt.scatter(df["x"], df["Px"], s=POINTSIZE, color=colors[2])
plt.xlabel("x [mm]")
plt.ylabel("Px [MeV/c]")
plt.title("x–Px phase space")
plt.savefig(f"{OUTDIR}/x_Px.png", dpi=300)
plt.close()

# -------------------------------
# 4. Longitudinal phase space (z, Pz)
# -------------------------------
plt.figure()
plt.scatter(df["z"], df["Pz"], s=POINTSIZE, color=colors[3])
plt.xlabel("z [mm]")
plt.ylabel("Pz [MeV/c]")
plt.title("Longitudinal phase space")
plt.savefig(f"{OUTDIR}/z_Pz.png", dpi=300)
plt.close()

# -------------------------------
# 5. Longitudinal optics phase space (z, δ)
# -------------------------------
plt.figure()
plt.scatter(df["z"], delta, s=POINTSIZE, color=colors[4])
plt.xlabel("z [mm]")
plt.ylabel("δ = Δp / p")
plt.title("Longitudinal phase space (z–δ)")
plt.savefig(f"{OUTDIR}/z_delta.png", dpi=300)
plt.close()

# -------------------------------
# 6. Transverse spatial distribution
# -------------------------------
plt.figure()
plt.scatter(df["x"], df["y"], s=POINTSIZE, color=colors[5])
plt.xlabel("x [mm]")
plt.ylabel("y [mm]")
plt.title("Transverse spatial distribution")
plt.savefig(f"{OUTDIR}/x_y.png", dpi=300)
plt.close()

# -------------------------------
# 7. Dispersion plot (x vs δ)
# -------------------------------
plt.figure()
plt.scatter(df["x"], delta, s=POINTSIZE, color=colors[6])
plt.ylabel("δ = Δp / p")
plt.xlabel("x [mm]")
plt.title("Horizontal dispersion")
plt.savefig(f"{OUTDIR}/x_delta.png", dpi=300)
plt.close()

# -------------------------------
# 8. Momentum-space correlation (Px, Pz)
# -------------------------------
plt.figure()
plt.scatter(df["Px"], df["Pz"], s=POINTSIZE, color=colors[7])
plt.xlabel("Px [MeV/c]")
plt.ylabel("Pz [MeV/c]")
plt.title("Momentum-space projection")
plt.savefig(f"{OUTDIR}/Px_Pz.png", dpi=300)
plt.close()

# -------------------------------
# 9. Approximate transverse action histogram
# -------------------------------
x = df["x"] - np.mean(df["x"])
xp = df["Px"] / P
Jx = x**2 + xp**2

plt.figure()
plt.hist(Jx, bins=100, color=colors[8])
plt.xlabel("Approximate action Jx")
plt.ylabel("Counts")
plt.title("Horizontal action distribution")
plt.savefig(f"{OUTDIR}/Jx_hist.png", dpi=300)
plt.close()

print(f"Saved tracking plots to ./{OUTDIR}/")
