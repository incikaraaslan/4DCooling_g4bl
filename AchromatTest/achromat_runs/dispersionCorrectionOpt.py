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
import traceback

# ------------------------------------------------------------
# USER CONFIGURATION
# ------------------------------------------------------------
G4BEAMLINE_CMD = "g4bl"
TEMPLATE_FILE = "Achromat.g4bl"
OUTPUT_DIR = "achromat_runs"
# Path(OUTPUT_DIR).mkdir(exist_ok=True)

VD_FILENAME = "vd_achromat.txt"
N_PARTICLES = 1000
G4BLFILE = f"/home/incik/Cooling_4D/AchromatTest/{OUTPUT_DIR}/run.g4bl"
G4BLOUTPUT = f"/home/incik/Cooling_4D/AchromatTest/{OUTPUT_DIR}/{VD_FILENAME}"

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
    (-2.0, 2.0),
    (50.0, 200.0),
    (50.0, 200.0),
    (50.0, 400.0),
    (0.0, 200.0),
    (-150.0, 150.0),
    (50.0, 400.0),
    (5.0, 30.0),
    (-2.0, 2.0),
    (50.0, 200.0),
    (50.0, 200.0),
    (50.0, 400.0),
    (50.0, 200.0),
    (50.0, 200.0),
    (50.0, 1000.0),
    (50.0, 200.0),
    (50.0, 200.0),
    (50.0, 1000.0)
]

# ------------------------------------------------------------
# BASIC FILE UTILITIES
# ------------------------------------------------------------
def write_input_from_template(template_path, out_path, replacements):
    
    """
    Takes a g4bl template with parameters that will be changed denoted with {} around them, and
    changes those parameter values in the template, outputting a functioning .g4bl file.

    Args:
        template_path (str): Location of the template .g4bl file.
        out_path (str): Where the output .g4bl file will be.
        replacements (dict): Label and name of the replaced params. The label should be the same as the parameter names
        specified in g4bl.

    Raises:
        RuntimeError: If there is a missing value for a parameter substitution, this will raise an error.
    """
    
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
    """
    Calculates lattice element positions to be put in .g4bl.

    Args:
        achromat_params (dict): The parameters optimized for/variable parameters in g4bl.

    Returns:
        dict: Adds the calculated positions to the parameters that will be substituted into the .g4bl parameter
        to run it.
    """
    
    # This gap is needed to get rid of any geometry errors in .g4bl to prevent an overlap between elements.
    GAP = 0.1
    
    # Take in relevant values to calculate lattice element positions.
    B1_z = float(achromat_params["B1_z"])
    L_B1 = float(achromat_params["B1_length"])
    L_D1 = float(achromat_params["Drift1_length"])
    L_Q1 = float(achromat_params["Q1_length"])
    L_D2 = float(achromat_params["Drift2_length"])
    L_B2 = float(achromat_params["B2_length"])

    Drift1_z = B1_z + (L_B1 / 2) + (L_D1 / 2) + GAP
    Q1_z     = Drift1_z + (L_D1 / 2) + (L_Q1 / 2) + GAP
    Drift2_z = Q1_z + (L_Q1 / 2) + (L_D2 / 2) + GAP
    B2_z     = Drift2_z + (L_D2 / 2) + (L_B2 / 2) + GAP
    VD_z     = B2_z + (L_B2 / 2) + 10.0 + GAP

    # Make a dictionary out of calculated parameters.
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
    
    """
    Run g4bl safely, catching runtime errors and deleting stale outputs.
    
    Args:
        achromat_params (dict): The parameters optimized for/variable parameters in g4bl.
    
    """
    
    # 18 Parameters + 4 Calculated + 2 String filenames etc = 26 Parameters
    merged_params = make_params_for_g4bl(achromat_params)
    write_input_from_template(TEMPLATE_FILE, G4BLFILE, merged_params)

    # Remove previous detector file to avoid reusing old data
    if os.path.exists(G4BLOUTPUT):
        os.remove(G4BLOUTPUT)

    # Run g4bl for the run.g4bl file
    try:
        result = subprocess.run(
            [G4BEAMLINE_CMD, G4BLFILE],
            capture_output=True, text=True, check=False
        )
    except Exception as e:
        print("[ERROR] Failed to launch G4beamline:", e)
        return False
    if result.returncode != 0:
        print("[WARN] G4beamline exited with nonzero code:", result.returncode)
        print(result.stderr[:300])
        return False

    return True

def calculate_D_for_df(output=G4BLOUTPUT):
    """
    Extracts dispersion for the pandas dataframe file generated from Daniel Fu's read_trackfile() function, which
    reads the run.g4bl particle distribution output gathered from the run_g4beamline() function. Daniel Fu's 
    calc_all_params() then takes this data frame and gives x_params and y_params in the form of 
    (emittance, beta, gamma, alpha, dispersion, derivative of dispersion), which we then extract.

    Args:
        output (_type_, optional): Output file created from run.g4bl. Defaults to G4BLOUTPUT.

    Returns:
        dict: dispersion, derivative of dispersion values for x and y.
    """
    df = read_trackfile(output)
    x_params, y_params, z_emit = calc_all_params(df)
    return {
        "D_x": x_params[4], "D'_x": x_params[5],
        "D_y": y_params[4], "D'_y": y_params[5]
    }

# ------------------------------------------------------------
# OPTIMIZATION MAPPING
# ------------------------------------------------------------
def xvec_to_achromat_params(xvec):
    """
    Combines values generated/optimized for each of the parameters with globally defined parameter names
    under the array opt_var_names.

    Args:
        xvec (array): Optimized parameter values.

    Returns:
        dict: Optimized dict of values for each lattice object.
    """
    return {name: float(val) for name, val in zip(opt_var_names, xvec)}

def cost_fn_from_xvec(xvec):
    """
    Objective: run simulation and compute dispersion cost robustly. This will be the optimized function. 
    The parameters tried by the optimization algorithm will first be combined with names for each parameter using
    xvec_to_achromat_params(). Then the 4 calculated parameters will be calculated, added to a dictionary, 
    run in new g4bl file created from this dictionary for these values within run_g4beamline(). Any weird value with 
    huge blowup or nans will be indicative of bad geometry and will be penalized by returning a huge cost, disincentivizing
    the algorithm to go further in that direction.
    
    The cost is calculated as (Dx ** 2 + Dpx ** 2) + (Dy ** 2 + Dpy ** 2).
    
    Args:
        xvec (array): Parameter values we'd like to optimize for.

    Returns:
        cost
    """
    achromat_params = xvec_to_achromat_params(xvec)
    PENALTY = 1e10

    try:
        ok = run_g4beamline(achromat_params)
        if not ok:
            print("Simulation failed to start/run — penalizing.")
            return PENALTY

        # Check detector file existence and non-emptiness
        if not os.path.exists(G4BLOUTPUT):
            print("Detector file missing — penalizing.")
            return PENALTY

        if os.path.getsize(G4BLOUTPUT) < 100:
            print(f"Empty or tiny detector file ({G4BLOUTPUT}), penalizing.")
            return PENALTY

        # Compute dispersion
        D_dict = calculate_D_for_df(G4BLOUTPUT)
        Dx, Dpx = D_dict["D_x"], D_dict["D'_x"]
        Dy, Dpy = D_dict["D_y"], D_dict["D'_y"]

        cost = (Dx ** 2 + Dpx ** 2) + (Dy ** 2 + Dpy ** 2)
        if not np.isfinite(cost):
            print("Non-finite cost — penalizing.")
            return PENALTY

        print(f"Trial cost={cost:.3e} | Dx={Dx:.3e} Dpx={Dpx:.3e} Dy={Dy:.3e} Dpy={Dpy:.3e}")
        print("DEBUG params:", achromat_params)
        return cost

    except Exception as e:
        print("[ERROR] Exception in cost function:", e)
        traceback.print_exc()
        return PENALTY

# ------------------------------------------------------------
# MAIN OPTIMIZER
# ------------------------------------------------------------
def differentialOptimizer():
    """
    Function that allows differential_evolution() to directly call cost_fn_from_xvec(xvec) many times with 
    different xvec values, and uses those return values to guide the optimization.
    
    Internally differential_evolution() handles the following:
    - Initializes a population of random vectors xvec within the bounds. 
    (So there will be e.g. 3*(number of parameters) random guesses at first.)
    - Evaluates cost_fn_from_xvec(xvec) for each of them.
    - Then, it repeatedly mutates, crosses over, and selects new xvecs based on the cost values returned, 
    to minimize the objective.
    - It continues until maxiter or convergence.

    Returns:
        result (scipy.optimize.OptimizeResult): looks sth like:
        
        fun: 2.348726e-05
        message: 'Optimization terminated successfully.'
        nfev: 93
        nit: 6
        success: True
        x: [ -0.0215,  1.3051,  243.7820,  ... ]
        and we take res.x to get the values that got us the result.
    
    """
    
    print("Starting global optimization (Differential Evolution)...")
    res = differential_evolution(
        cost_fn_from_xvec,
        bounds=opt_bounds,
        maxiter=2,
        popsize=3,
        disp=True,
        polish=False
    )
    print("\nBest result:")
    for k, v in zip(opt_var_names, res.x):
        print(f"  {k:20s} = {v:10.4f}")
    print(f"Final cost = {res.fun:.3e}")
    return res

# ------------------------------------------------------------
# START
# ------------------------------------------------------------
if __name__ == "__main__":
    print(">>> Running Achromat Dispersion Optimization <<<\n")
    print("Testing one midpoint configuration...")
    # Our test in the mean of these bounds
    test_x = [np.mean(b) for b in opt_bounds]
    cost_fn_from_xvec(test_x)
    
    print("\nLaunching optimizer...")
    result = differentialOptimizer()
    print("\nOptimization complete.")
