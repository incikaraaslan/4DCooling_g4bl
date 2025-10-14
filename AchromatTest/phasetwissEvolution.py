import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# CONFIGURATION
# -------------------------
data_path = "."  # folder where vd_*.txt files are
file_pattern = "vd_*.txt"  # match your ascii files
output_folder = "plots"

# Make output folder if missing
os.makedirs(output_folder, exist_ok=True)

# -------------------------
# LOOP OVER FILES
# -------------------------
files = sorted(glob.glob(os.path.join(data_path, file_pattern)))

for filename in files:
    print(f"Processing {filename} ...")

    # Skip commented lines (those starting with '#')
    df = pd.read_csv(filename, comment='#', delim_whitespace=True,
                     names=["x", "y", "z", "Px", "Py", "Pz", "t", "PDGid", "EventID", "TrackID", "ParentID", "Weight"])

    # Compute normalized momenta
    df["xp"] = df["Px"] / df["Pz"]   # x' = Px/Pz
    df["yp"] = df["Py"] / df["Pz"]   # y' = Py/Pz

    # Extract z-position for plot title
    z_value = df["z"].mean() if not df.empty else 0
    label = os.path.basename(filename).replace(".txt", "")

    # -------------------------
    # Plot X–X' phase space
    # -------------------------
    plt.figure(figsize=(6, 5))
    plt.scatter(df["x"], df["xp"], s=5, alpha=0.5)
    plt.xlabel("x [mm]")
    plt.ylabel("x' = Px/Pz [rad]")
    plt.title(f"Horizontal Phase Space @ z={z_value:.1f} mm ({label})")
    plt.grid(True)
    plt.tight_layout()
    plt.ylim(-1.0, 1.0)
    plt.xlim(-200.0, 200.0)
    plt.savefig(os.path.join(output_folder, f"{label}_xphase.png"))
    plt.close()

    # -------------------------
    # Plot Y–Y' phase space
    # -------------------------
    plt.figure(figsize=(6, 5))
    plt.scatter(df["y"], df["yp"], s=5, alpha=0.5, color='orange')
    plt.xlabel("y [mm]")
    plt.ylabel("y' = Py/Pz [rad]")
    plt.title(f"Vertical Phase Space @ z={z_value:.1f} mm ({label})")
    plt.grid(True)
    plt.tight_layout()
    plt.ylim(-1.0, 1.0)
    plt.xlim(-200.0, 200.0)
    plt.savefig(os.path.join(output_folder, f"{label}_yphase.png"))
    plt.close()

print(f"Plots saved in folder: {output_folder}")
