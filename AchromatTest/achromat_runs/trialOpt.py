#!/usr/bin/env python3
# ------------------------------------------------------------
# Achromat Dispersion Optimizer for G4beamline
# Author: I. Karaaslan
# Date: 2025-10-13
# ------------------------------------------------------------

import os
import subprocess
import numpy as np
from scipy.optimize import differential_evolution
from pathlib import Path
import matplotlib.pyplot as plt
from g4beam import *
from scan import *

import math
import matplotlib.animation as animation
from matplotlib import cm
import numpy as np
import pandas as pd
from tqdm import *
import pandas as pd
import pickle
import itertools
from tabulate import tabulate
import tempfile
import glob
import json

# ------------------------------------------------------------
# USER CONFIGURATION
# ------------------------------------------------------------
G4BEAMLINE_CMD = "g4bl"
TEMPLATE_FILE = "Achromat.g4bl"
OUTPUT_DIR = "achromat_runs"
Path(OUTPUT_DIR).mkdir(exist_ok=True)

VD_FILENAME = "vd_achromat.txt"
N_PARTICLES = 5000
"""G4BLFILE = os.path.join(OUTPUT_DIR, "run.g4bl")
G4BLOUTPUT = os.path.join(OUTPUT_DIR, VD_FILENAME)"""
G4BLFILE = f"/home/incik/Cooling_4D/AchromatTest/{OUTPUT_DIR}/run.g4bl"
G4BLOUTPUT =f"/home/incik/Cooling_4D/AchromatTest/{OUTPUT_DIR}/{VD_FILENAME}"

# ------------------------------------------------------------
# VARIABLE NAMES & BOUNDS (18 params)
# ------------------------------------------------------------
opt_var_names = [
    "B1_field", "B1_width", "B1_height", "B1_length", "B1_z",
    "Q1_gradient", "Q1_length", "radius_q",
    "B2_field", "B2_width", "B2_height", "B2_length",
    "Drift1_width", "Drift1_height", "Drift1_length",
    "Drift2_width", "Drift2_height", "Drift2_length"
]

opt_bounds = [
    (-2.0, 2.0),      # B1_field [T]
    (50.0, 200.0),    # B1_width
    (50.0, 200.0),    # B1_height
    (50.0, 400.0),    # B1_length
    (0.0, 200.0),     # B1_z
    (-150.0, 150.0),  # Q1_gradient [T/m]
    (50.0, 400.0),    # Q1_length
    (5.0, 30.0),      # radius_q
    (-2.0, 2.0),      # B2_field [T]
    (50.0, 200.0),    # B2_width
    (50.0, 200.0),    # B2_height
    (50.0, 400.0),    # B2_length
    (50.0, 200.0),    # Drift1_width
    (50.0, 200.0),    # Drift1_height
    (50.0, 1000.0),   # Drift1_length
    (50.0, 200.0),    # Drift2_width
    (50.0, 200.0),    # Drift2_height
    (50.0, 1000.0)    # Drift2_length
]

# ------------------------------------------------------------
# BASIC FILE UTILITIES
# ------------------------------------------------------------
def write_input_from_template(template_path, out_path, replacements):
    """Write G4BL input file from template using Python string formatting."""
    with open(template_path, 'r') as f:
        txt = f.read()
    try:
        txt = txt.format(**replacements)
    except KeyError as e:
        raise RuntimeError(f"Template substitution failed; missing placeholder: {e}")
    with open(out_path, 'w') as f:
        f.write(txt)

# ------------------------------------------------------------
# PARAMETER BUILDING FUNCTIONS
# ------------------------------------------------------------
def make_params_for_g4bl(achromat_params):
    """Compute derived z-positions and merge constants."""
    GAP = 0.1
    B1_z = float(achromat_params["B1_z"])
    L_B1 = float(achromat_params["B1_length"])
    L_D1 = float(achromat_params["Drift1_length"])
    L_Q1 = float(achromat_params["Q1_length"])
    L_D2 = float(achromat_params["Drift2_length"])
    L_B2 = float(achromat_params["B2_length"])

    Drift1_z = B1_z + (L_B1/2) + (L_D1/2) + GAP
    Q1_z     = Drift1_z + (L_D1/2) + (L_Q1/2) + GAP
    Drift2_z = Q1_z + (L_Q1/2) + (L_D2/2) + GAP
    B2_z     = Drift2_z + (L_D2/2) + (L_B2/2) + GAP
    VD_z     = B2_z + (L_B2/2) + 10.0 + GAP

    add_params = {
        "Drift1_z": Drift1_z,
        "Q1_z": Q1_z,
        "Drift2_z": Drift2_z,
        "B2_z": B2_z,
        "VD_z": VD_z,
        "N_PARTICLES": N_PARTICLES,
        "VD_FILENAME": VD_FILENAME
    }
    return achromat_params | add_params

# ------------------------------------------------------------
# SIMULATION RUNNERS
# ------------------------------------------------------------
def run_g4beamline(achromat_params):
    """Build, run, and wait for G4beamline to finish."""
    merged_params = make_params_for_g4bl(achromat_params)
    write_input_from_template(TEMPLATE_FILE, G4BLFILE, merged_params)
    subprocess.run([G4BEAMLINE_CMD, G4BLFILE], capture_output=True, text=True, check=True)

def calculate_D_for_df(output=G4BLOUTPUT):
    """Compute dispersion quantities from G4BL output."""
    df = read_trackfile(output)
    x_params, y_params, z_emit = calc_all_params(df)
    return {"D_x": x_params[4], "D'_x": x_params[5],
            "D_y": y_params[4], "D'_y": y_params[5]}

# ------------------------------------------------------------
# OPTIMIZATION MAPPING
# ------------------------------------------------------------
def xvec_to_achromat_params(xvec):
    """Convert optimizer vector to parameter dict."""
    return {name: float(val) for name, val in zip(opt_var_names, xvec)}

def cost_fn_from_xvec(xvec):
    """Objective function — run simulation and compute dispersion cost."""
    achromat_params = xvec_to_achromat_params(xvec)

    try:
        run_g4beamline(achromat_params)
        D_dict = calculate_D_for_df(G4BLOUTPUT)
    except Exception as e:
        print("Run failed:", e)
        return 1e10

    Dx, Dpx = D_dict["D_x"], D_dict["D'_x"]
    Dy, Dpy = D_dict["D_y"], D_dict["D'_y"]
    cost = (Dx**2 + Dpx**2) + (Dy**2 + Dpy**2)

    print(f"Trial cost={cost:.3e} | Dx={Dx:.3e} Dpx={Dpx:.3e} Dy={Dy:.3e} Dpy={Dpy:.3e}")
    print("DEBUG params:", achromat_params)
    """with open(G4BLFILE) as f:
        print("First few lines of generated .g4bl:")
        print("".join(f.readlines()[:10]))"""
    
    if not os.path.exists(G4BLOUTPUT) or os.path.getsize(G4BLOUTPUT) < 100:
        print("Empty vd file — penalizing")
        return 1e10

    return cost

# ------------------------------------------------------------
# MAIN OPTIMIZER
# ------------------------------------------------------------
def differentialOptimizer():
    print("Starting global optimization (Differential Evolution)...")
    res = differential_evolution(cost_fn_from_xvec, bounds=opt_bounds, maxiter=8, popsize=6, disp=True, polish=False)
    print("\nBest result:")
    for k, v in zip(opt_var_names, res.x):
        print(f"  {k:20s} = {v:10.4f}")
    print(f"Final cost = {res.fun:.3e}")
    return res

# ------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------
if __name__ == "__main__":
    print(">>> Running Achromat Dispersion Optimization <<<\n")
    test_x = [np.mean(b) for b in opt_bounds]
    print("Testing one midpoint configuration...")
    cost_fn_from_xvec(test_x)
    print("\nLaunching optimizer...")
    result = differentialOptimizer()
    print("\nOptimization complete.")
