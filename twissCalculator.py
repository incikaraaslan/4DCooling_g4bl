"""
Inci's Script to calculate the Twiss Parameters of a given particle distribution. 
The calculations are made using the linear optics assumption and 
https://nicadd.niu.edu/~syphers/tutorials/analyzeTrack.html#analyzing-a-distribution.

"""
import numpy as np
import matplotlib.pyplot as plt
# -----------------
# Helper Functions
# -----------------

# --- Helper: covariance ---
def covar(a, b):
    return np.mean((a - np.mean(a)) * (b - np.mean(b)))

# --- Helper: basic twiss calculations ---
def compute_twiss(x, xp, dell):
    Axx = np.var(x)
    Axpxp = np.var(xp)
    Add = np.var(dell)
    Axxp = covar(x, xp)
    Axd = covar(x, dell)
    Axpd = covar(xp, dell)

    # Dispersion
    D = Axd / Add
    Dp = Axpd / Add

    # Twiss reconstruction
    ebet = Axx - Axd**2 / Add
    egam = Axpxp - Axpd**2 / Add
    ealf = -Axxp + (Axpd * Axd) / Add
    emit = np.sqrt(max(ebet * egam - ealf**2, 0))

    beta = ebet / emit if emit > 0 else np.nan
    alpha = ealf / emit if emit > 0 else np.nan

    return {
        "beta": beta,
        "alpha": alpha,
        "emit": emit,
        "D": D,
        "Dp": Dp
    }
    
def analyze_g4bl_trackfile(filename, P0=None):
    """
    Read a G4Beamline #BLTrackFile2 and compute Courant-Snyder parameters & dispersion
    for both x and y planes.

    Parameters:
        filename : str, path to the track file
        P0       : reference momentum [MeV/c]. Since there's none in Daniel's wedge file, we take the mean.

    Returns:
        Dictionary with computed Twiss parameters and dispersions for x and y.
    """

    # --- Load file, skip header lines starting with '#'
    data = np.loadtxt(filename, comments="#")

    # Columns (see file header): x y z Px Py Pz ...
    x = data[:, 0]   # mm
    y = data[:, 1]   # mm
    z = data[:, 2]   # mm (longitudinal coordinate)
    Px = data[:, 3]  # MeV/c
    Py = data[:, 4]
    Pz = data[:, 5]

    # Reference momentum
    if P0 is None:
        P0 = np.mean(np.sqrt(Px**2 + Py**2 + Pz**2))

    # Transverse slopes
    xp = Px / Pz
    yp = Py / Pz

    # Relative momentum deviation δ = (P - P0)/P0
    P = np.sqrt(Px**2 + Py**2 + Pz**2)
    delta = (P - P0) / P0
    
    return {
        "twiss_x": compute_twiss(x, xp, delta),
        "twiss_y": compute_twiss(y, yp, delta)}


def g4bl_to_madx_ptc(infile, outfile, P0=None):
    """
    Convert a G4Beamline #BLTrackFile2 distribution to MAD-X PTC external track format.

    Parameters:
        infile  : str, path to G4BL track file
        outfile : str, path to write MAD-X PTC track file
        P0      : reference momentum [MeV/c]. Since there's none in Daniel's wedge file, we take the mean.

    Output format (per line for each particle):
        x [m]   px [GeV/c]   y [m]   py [GeV/c]   t [m]   pt [Δp/p]
    """

    # Load data, skipping header lines
    data = np.loadtxt(infile, comments="#")
    
    
    # Extract columns
    x_mm, y_mm, z_mm = data[:, 0], data[:, 1], data[:, 2]
    Px, Py, Pz = data[:, 3], data[:, 4], data[:, 5]  # MeV/c

    # Convert positions from mm to m
    x = x_mm * 1e-3
    y = y_mm * 1e-3
    z = z_mm * 1e-3

    # Absolute momentum
    P = np.sqrt(Px**2 + Py**2 + Pz**2)

    # Reference momentum
    if P0 is None:
        P0 = np.mean(P)

    # Slopes -- here we need to consider relativistic beta due to mad-X calculations that are found
    # in the manual : https://cds.cern.ch/record/2296385/files/madphys.pdf 
    xp = Px / Pz
    yp = Py / Pz

    # MAD-X transverse momenta calculation --- if muon momentum less than approx 0.2 GeV/c
    # Important! In Mad-X convention, the momenta are dimensionless! Scaled by p0!
    mass = 105.6583755
    h = 0.0                                                         # h is the curvature of the reference orbit in the mid-plane, 0 for drift
    E = np.sqrt(P0**2 + mass**2)
    beta_rel = P0 / E                                               # Relativistic beta for reference particle
    delta = (P - P0) / P0 * beta_rel                                # Relative momentum deviation δ = Δp/p0 * β
    px_madx = xp / (1.0 + h * x - delta / beta_rel) * (P0 / 1000.0) # P0 gets converted to GeV/c 
    py_madx = yp  / (1.0 + h * x - delta / beta_rel) * (P0 / 1000.0)
    
    # Time coordinate (can be zeroed or from z)
    # Here we just use z/c ~ z, since MAD-X uses length units for t
    ct = np.zeros_like(z)  # simple choice, ignore longitudinal spread

    # Stack columns into MAD-X format
    madx_data = np.column_stack([x, px_madx, y, py_madx, ct, delta])

    # Save to file
    header = "x[m] px[GeV/c] y[m] py[GeV/c] ct[m] pt[dp/p]"
    np.savetxt(outfile, madx_data, header=header)

    print(f"Converted {len(madx_data)} particles to {outfile}, using P0={P0:.3f} MeV/c")
    
    # Plot particle distributions
    fig, axs = plt.subplots(1, 2, figsize=(12,5))
    Np_plot = 500000
    # Horizontal
    axs[0].scatter(x_mm[:Np_plot], px_madx[:Np_plot]*1e3, s=1, alpha=0.5, label="MAD-X synthetic", color='orange')
    axs[0].scatter(x_mm[:Np_plot], Px[:Np_plot], s=1, alpha=0.5, label="G4BL")
    axs[0].set_xlabel("x [mm]")
    axs[0].set_ylabel("px [MeV/c]")
    axs[0].set_xlim(-30,30)
    axs[0].set_ylim(-150,150)
    axs[0].set_title("Horizontal phase space")
    axs[0].legend()
    axs[0].grid(True)

    # Vertical
    axs[1].scatter(y_mm[:Np_plot], py_madx[:Np_plot]*1e3, s=1, alpha=0.5, label="MAD-X synthetic", color='orange')
    axs[1].scatter(y_mm[:Np_plot], Py[:Np_plot], s=1, alpha=0.5, label="G4BL")
    axs[1].set_xlabel("y [mm]")
    axs[1].set_ylabel("py [MeV/c]")
    axs[1].set_xlim(-30,30)
    axs[1].set_ylim(-150,150)
    axs[1].set_title("Vertical phase space")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig("madxrelcheck.png")
    plt.show()


if __name__ == "__main__":
    g4blfilename = "particles_after"
    
    results = analyze_g4bl_trackfile(g4blfilename+".txt")

    # Extract into rows
    row_x = [results["twiss_x"]["beta"],
            results["twiss_x"]["alpha"],
            results["twiss_x"]["emit"],
            results["twiss_x"]["D"],
            results["twiss_x"]["Dp"]]

    row_y = [results["twiss_y"]["beta"],
            results["twiss_y"]["alpha"],
            results["twiss_y"]["emit"],
            results["twiss_y"]["D"],
            results["twiss_y"]["Dp"]]

    header = "beta alpha emit D Dp"
    np.savetxt(f"G4BL_{g4blfilename}_twiss.txt", [row_x, row_y], header=header)
    g4bl_to_madx_ptc(g4blfilename+".txt", f"{g4blfilename}_2madx.dat")