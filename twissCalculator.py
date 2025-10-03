"""
Inci's Script to calculate the Twiss Parameters of a given particle distribution. 
The calculations are made using the linear optics assumption and 
https://nicadd.niu.edu/~syphers/tutorials/analyzeTrack.html#analyzing-a-distribution.

"""
import numpy as np


def analyze_g4bl_trackfile(filename, P0=None):
    """
    Read a G4Beamline #BLTrackFile2 and compute Courant-Snyder parameters & dispersion
    for both x and y planes.

    Parameters:
        filename : str, path to the track file
        P0       : reference momentum [MeV/c]. If None, mean(P) is used.

    Returns:
        dict with computed Twiss parameters and dispersions for x and y.
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

    # --- Helper: covariance
    def covar(a, b):
        return np.mean((a - np.mean(a)) * (b - np.mean(b)))

    def compute_twiss(coord, slope, delta):
        Auu = np.var(coord)
        Aspsp = np.var(slope)
        Add = np.var(delta)
        Au_sp = covar(coord, slope)
        Aud = covar(coord, delta)
        Aspd = covar(slope, delta)

        # Dispersion
        D = Aud / Add
        Dp = Aspd / Add

        # Twiss reconstruction
        ebet = Auu - Aud**2 / Add
        egam = Aspsp - Aspd**2 / Add
        ealf = -Au_sp + (Aspd * Aud) / Add
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

    return {
        "twiss_x": compute_twiss(x, xp, delta),
        "twiss_y": compute_twiss(y, yp, delta)
    }


def g4bl_to_madx_ptc(infile, outfile, P0=None):
    """
    Convert a G4Beamline #BLTrackFile2 distribution to MAD-X PTC external track format.

    Parameters:
        infile  : str, path to G4BL track file
        outfile : str, path to write MAD-X PTC track file
        P0      : reference momentum [MeV/c]. If None, mean(P) is used.

    Output format (per line for each particle):
        x [m]   px [GeV/c]   y [m]   py [GeV/c]   t [m]   pt [Δp/p]
    """

    # Load data, skipping header lines
    data = np.loadtxt(infile, comments="#")

    # Extract G4BL columns
    x_mm = data[:, 0]
    y_mm = data[:, 1]
    z_mm = data[:, 2]
    Px = data[:, 3]  # MeV/c
    Py = data[:, 4]
    Pz = data[:, 5]

    # Convert mm -> m
    x = x_mm * 1e-3
    y = y_mm * 1e-3
    z = z_mm * 1e-3

    # Reference momentum
    P = np.sqrt(Px**2 + Py**2 + Pz**2)
    if P0 is None:
        P0 = np.mean(P)

    # Slopes
    xp = Px / Pz
    yp = Py / Pz

    # Canonical transverse momenta in MAD-X convention
    # MAD-X expects units of GeV/c for px, py
    px = xp * (P0 / 1000.0)  # convert MeV/c -> GeV/c
    py = yp * (P0 / 1000.0)

    # Relative momentum deviation pt = δ
    pt = (P - P0) / P0

    # Time coordinate (can be zeroed or from z)
    # Here we just use z/c ~ z, since MAD-X uses length units for t
    t = np.zeros_like(z)  # simple choice, ignore longitudinal spread

    # Stack columns into MAD-X format
    madx_data = np.column_stack([x, px, y, py, t, pt])

    # Save to file
    header = "x[m] px[GeV/c] y[m] py[GeV/c] t[m] pt[dp/p]"
    np.savetxt(outfile, madx_data, header=header)

    print(f"Converted {len(madx_data)} particles to {outfile}, using P0={P0:.3f} MeV/c")


if __name__ == "__main__":
    results = analyze_g4bl_trackfile("particles_after.txt")

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
    np.savetxt("G4BLtwiss.txt", [row_x, row_y], header=header)
    g4bl_to_madx_ptc("particles_after.txt", "g4bl2madx.dat")