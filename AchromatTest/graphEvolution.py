import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from g4beam import *
from scan import *
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# -------------------------------------------------------------------
# Your existing functions and constants must be imported or defined:
# from your_module import calc_all_params, p_total
# -------------------------------------------------------------------

# Example constants (if not imported already)
MUON_MASS = 105.658  # MeV/c^2
C = 299792458  # m/s

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
data_path = "."
file_pattern = "vd_*.txt"
output_folder1 = "plots"
output_folder2 = "plots_optics"
os.makedirs(output_folder1, exist_ok=True)
os.makedirs(output_folder2, exist_ok=True)

# -------------------------------------------------------------------
# PLOTTING FXNS
# -------------------------------------------------------------------

def phaseEvolution(df, filename, output_folder = output_folder1):
    
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
    
    print(f"All plots saved in '{output_folder}'")

def twissComp(df, z_positions, emit_x, emit_y, emit_z, beta_x, beta_y, alpha_x, alpha_y, D_x, Dp_x, D_y, Dp_y):
    
    # Compute Twiss + dispersion parameters
    (x_params, y_params, z_emit_val) = calc_all_params(df)

    # Unpack: (emit, beta, gamma, alpha, D, D')
    ex, bx, gx, ax, Dx, Dxp = x_params
    ey, by, gy, ay, Dy, Dyp = y_params

    z_positions.append(np.mean(df["z"]))
    emit_x.append(ex)
    beta_x.append(bx)
    alpha_x.append(ax)
    D_x.append(Dx)
    Dp_x.append(Dxp)

    emit_y.append(ey)
    beta_y.append(by)
    alpha_y.append(ay)
    D_y.append(Dy)
    Dp_y.append(Dyp)

    emit_z.append(z_emit_val)

def twissPlot(z_positions, emit_x, emit_y, emit_z, beta_x, beta_y, alpha_x, alpha_y, D_x, Dp_x, D_y, Dp_y, output_folder=output_folder2):
    
    # -------------------------------------------------------------------
    # CONVERT TO ARRAYS FOR PLOTTING
    # -------------------------------------------------------------------
    z_positions = np.array(z_positions)
    order = np.argsort(z_positions)

    z_positions = z_positions[order]
    emit_x = np.array(emit_x)[order]
    emit_y = np.array(emit_y)[order]
    beta_x = np.array(beta_x)[order]
    beta_y = np.array(beta_y)[order]
    alpha_x = np.array(alpha_x)[order]
    alpha_y = np.array(alpha_y)[order]
    D_x = np.array(D_x)[order]
    Dp_x = np.array(Dp_x)[order]
    D_y = np.array(D_y)[order]
    Dp_y = np.array(Dp_y)[order]
    emit_z = np.array(emit_z)[order]

    # -------------------------------------------------------------------
    # PLOTTING
    # -------------------------------------------------------------------

    def make_plot(yvals, ylabel, title, filename, labels=("x", "y")):
        plt.figure(figsize=(7,5))
        plt.plot(z_positions, yvals[0], 'o-', label=f"{labels[0]}-plane")
        plt.plot(z_positions, yvals[1], 's--', label=f"{labels[1]}-plane")
        plt.xlabel("z [mm]")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, filename))
        plt.close()

    make_plot((emit_x, emit_y), "Emittance [mm·mrad]",
            "Emittance Evolution", "emittance_evolution.png")

    make_plot((beta_x, beta_y), "β-function [mm/rad]",
            "Beta Function Evolution", "beta_evolution.png")

    make_plot((alpha_x, alpha_y), "α-function [–]",
            "Alpha Function Evolution", "alpha_evolution.png")

    make_plot((D_x, D_y), "Dispersion D [mm]",
            "Dispersion Function", "dispersion_D.png")

    make_plot((Dp_x, Dp_y), "Dispersion Derivative D' [rad]",
            "Dispersion Derivative", "dispersion_Dprime.png")

    # Optional: longitudinal emittance
    plt.figure(figsize=(7,5))
    plt.plot(z_positions, emit_z, 'o-', color='purple')
    plt.xlabel("z [mm]")
    plt.ylabel("Longitudinal Emittance [arb.]")
    plt.title("Longitudinal Emittance Evolution")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "emittance_longitudinal.png"))
    plt.close()

    print(f"All plots saved in '{output_folder}'")

# -------------------------------------------------------------------
# LOOP OVER FILES AND COMPUTE PARAMETERS
# -------------------------------------------------------------------
if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(data_path, file_pattern)))
    
    z_positions = []
    emit_x, beta_x, alpha_x, D_x, Dp_x = [], [], [], [], []
    emit_y, beta_y, alpha_y, D_y, Dp_y = [], [], [], [], []
    emit_z = []
    
    for f in files:
        print(f"Processing {f} ...")
        df = pd.read_csv(f, comment="#", sep='\s+',
                        names=["x", "y", "z", "Px", "Py", "Pz", "t",
                                "PDGid", "EventID", "TrackID", "ParentID", "Weight"])
        
        # Plot phase space evolution
        phaseEvolution(df, f)
        
        # Compute TWISS parameters for each file
        twissComp(df, z_positions, emit_x, emit_y, emit_z, beta_x, beta_y, alpha_x, alpha_y, D_x, Dp_x, D_y, Dp_y)
    
    # Plot TWISS parameter evolution
    twissPlot(z_positions, emit_x, emit_y, emit_z, beta_x, beta_y, alpha_x, alpha_y, D_x, Dp_x, D_y, Dp_y)
    
    
