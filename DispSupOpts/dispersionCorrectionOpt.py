#!/usr/bin/env python3
# ------------------------------------------------------------
# Dispersion Optimizer for G4beamline
# Author: I. Karaaslan
# Date: 2025-12-2
# ------------------------------------------------------------

import os
import subprocess
import numpy as np
from scipy.optimize import differential_evolution
from pathlib import Path
import matplotlib.pyplot as plt
from g4beam import *
from scan import *
import re
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
TEMPLATE_FILE = "G4_FinalCooling_dispsup_Template.g4bl"
OUTPUT_DIR = "dispsup_runs"
# Path(OUTPUT_DIR).mkdir(exist_ok=True)

VD_FILENAME = "vd_dispsup.txt"
N_PARTICLES = 2000
G4BLFILE = f"/home/incik/Cooling_4D/DispSupOpts/{OUTPUT_DIR}/run.g4bl"
G4BLOUTPUT = f"/home/incik/Cooling_4D/DispSupOpts/{VD_FILENAME}"

# ------------------------------------------------------------
# VARIABLE NAMES & BOUNDS (18 params)
# ------------------------------------------------------------
opt_var_names = [
    "B1_field", "B1_width", "B1_height", "B1_length", 
    "Q1_gradient", "Q1_length", "radius_q", "Q1_z", "thickness",
    "Q2_gradient", "Q2_width", "Q2_height", "Q2_length",
    "Drift1_width", "Drift1_height", "Drift1_length",  
    "Drift2_width", "Drift2_height","Drift2_length"]

opt_bounds = [
    (-2.0, 2.0),
    (10.0, 300.0),
    (10.0, 300.0),
    (10.0, 300.0),
    (-2.0, 2.0),
    (50.0, 300.0),
    (50.0, 300.0),
    (0.0, 300.0),
    (10.0, 500.0),
    (-2.0, 2.0),
    (50.0, 300.0),
    (50.0, 300.0),
    (50.0, 300.0),
    (10.0, 300.0),
    (10.0, 300.0),
    (10.0, 300.0),
    (10.0, 300.0),
    (10.0, 300.0),
    (10.0, 300.0)
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
        
def convertZ(input_file, output_file):
    """
    Ensure that the g4bl output starts from z=0.

    Args:
        input_file (str): output with non-zero z.
        output_file (str): output file with zero z.

    """
    event_id_counter = 1
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            # Skip header lines (those starting with #)
            if line.strip().startswith("#"):
                outfile.write(line)
                continue

            # Split the line into columns
            parts = line.strip().split()
            if len(parts) >= 12:
                if parts[2] == "0":
                    return None
                parts[2] = "0"  # Set the 3rd column (z) to 0
                # Replace event ID (assuming it's the 9th column, zero-based index 8)
                # Adjust if your event ID is in a different column
                parts[8] = str(event_id_counter)
                event_id_counter += 1
                new_line = " ".join(parts)
                outfile.write(new_line + "\n")
            else:
                # Handle lines that don't match expected format
                outfile.write(line)
    print(f"Updated file saved as '{output_file}'")
    os.remove(input_file)
    return None

def read_g4bl_params(TEMPLATE_FILE=TEMPLATE_FILE):
    """
    Reads 'param' definitions from a .g4bl file and returns a dict of parameter names and values.
    """
    params = {}
    pattern = re.compile(r"param\s+(?:-unset\s+)?(\w+)=([^\s#]+)")
    with open(TEMPLATE_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("param"):
                continue
            m = pattern.search(line)
            if m:
                key, val = m.groups()
                try:
                    params[key] = float(eval(val, {"__builtins__": None, "pi": math.pi}))
                except Exception:
                    params[key] = val  # keep as string if not numeric
    return params
# ------------------------------------------------------------
# PARAMETER BUILDING FUNCTIONS
# ------------------------------------------------------------
# Compute wedge elements
def compute_wedge_geometry():
    """
    Given parameters like absLEN3, abshgt, abswidth, abshalfangle3, compute geometry info.
    """
    params = read_g4bl_params()
    W = params.get("absLEN3")
    H = params.get("abshgt")
    L = params.get("abswidth")
    half_angle_deg = params.get("abshalfangle3")
    offset = params.get("absoffset3", 0)

    if L and half_angle_deg:
        half_angle_rad = math.radians(half_angle_deg)
        L_centerline = W / math.sin(half_angle_rad)

    return {
        "Length_Wedge": L,
        "Height_Base": H,
        "Width_Base": W,
        "Half-angle": half_angle_deg,
        "Centerline_Coords": L_centerline,
        "Offset": offset
    }

def make_params_for_g4bl(dispsup_params):
    """
    Calculates lattice element positions to be put in .g4bl.

    Args:
        dispsup_params (dict): The parameters optimized for/variable parameters in g4bl.

    Returns:
        dict: Adds the calculated positions to the parameters that will be substituted into the .g4bl parameter
        to run it.
    """
    geom = compute_wedge_geometry()
    GAP = 0.1  # in mm (can be up to 1.0 safely)
    A_GAP = 5 # mm
    
    required_keys = ["Q1_length", "Drift1_length", "Q2_length", "Drift2_length", "B1_length"]
    missing = [k for k in required_keys if k not in dispsup_params]
    if missing:
        raise KeyError(f"make_params_for_g4bl: missing required parameters: {missing}. "
                    "Check opt_var_names and optimizer mapping.")
    
    L_Q1 = float(dispsup_params["Q1_length"])
    L_D1 = float(dispsup_params["Drift1_length"])
    L_Q2 = float(dispsup_params["Q2_length"])
    L_D2 = float(dispsup_params["Drift2_length"])
    L_B1 = float(dispsup_params["B1_length"])

    wedge_end = geom["Centerline_Coords"]
    Q1_z      = geom["Centerline_Coords"] + (L_Q1/2)
    Drift1_z = Q1_z + (L_Q1/2) + (L_D1/2) + GAP
    Q2_z     = Drift1_z + (L_D1/2) + (L_Q2/2) + GAP
    Drift2_z = Q2_z + (L_Q2/2) + (L_D2/2)+ GAP
    B1_z     = Drift2_z + (L_D2/2) + (L_B1/2) + GAP
    VD_z     = B1_z + (L_B1/2) + 10.0 + GAP

    Q1_end = Q1_z + (L_Q1/2)
    D1_end = Drift1_z + (L_D1/2)
    Q2_end = Q2_z + (L_Q2/2)
    D2_end = Drift2_z + (L_D2/2)
    B1_end = B1_z + (L_B1/2)

    add_params = {"wedge_end": wedge_end, "Q1_z": Q1_z, "Drift1_z": Drift1_z, "Q2_z": Q2_z, "Drift2_z":Drift2_z, "B1_z": B1_z,"VD_z": VD_z, 
            "B1_end": B1_end, "Q2_end": Q2_end,  "D1_end": D1_end, "Q1_end": Q1_end,  "D2_end": D2_end, "N_PARTICLES": N_PARTICLES,
            "VD_FILENAME": VD_FILENAME}

    return dispsup_params | add_params

# ------------------------------------------------------------
# SIMULATION RUNNERS
# ------------------------------------------------------------
def run_g4beamline(dispsup_params):
    
    """
    Run g4bl safely, catching runtime errors and deleting stale outputs.
    
    Args:
        dispsup_params (dict): The parameters optimized for/variable parameters in g4bl.
    
    """
    
    # 18 Parameters + 4 Calculated + 2 String filenames etc = 26 Parameters
    merged_params = make_params_for_g4bl(dispsup_params)
    write_input_from_template(TEMPLATE_FILE, G4BLFILE, merged_params)
    print("Parameter writing successful.")

    # Remove previous detector file to avoid reusing old data
    if os.path.exists(G4BLOUTPUT):
        os.remove(G4BLOUTPUT)
    
    if os.path.exists("field_cell.dat"):
        os.remove("field_cell.dat")

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

def calculate_D_trans_for_df(output=G4BLOUTPUT):
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
    with open(output) as f:
        N_out = sum(1 for line in f if not line.startswith("#") and line.strip())

    # Compute transmission (%)
    trans_percent = 100.0 * N_out / int(N_PARTICLES)
    
    return {
        "D_x": x_params[4], "D'_x": x_params[5],
        "D_y": y_params[4], "D'_y": y_params[5],
        "transmission": trans_percent
    }

# ------------------------------------------------------------
# OPTIMIZATION MAPPING
# ------------------------------------------------------------
def xvec_to_dispsup_params(xvec):
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
    xvec_to_dispsup_params(). Then the 4 calculated parameters will be calculated, added to a dictionary, 
    run in new g4bl file created from this dictionary for these values within run_g4beamline(). Any weird value with 
    huge blowup or nans will be indicative of bad geometry and will be penalized by returning a huge cost, disincentivizing
    the algorithm to go further in that direction.
    
    The cost is calculated as (Dx ** 2 + Dpx ** 2) + (Dy ** 2 + Dpy ** 2).
    
    Args:
        xvec (array): Parameter values we'd like to optimize for.

    Returns:
        cost
    """
    dispsup_params = xvec_to_dispsup_params(xvec)
    PENALTY = 1e10

    try:
        ok = run_g4beamline(dispsup_params)
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
        D_dict = calculate_D_trans_for_df(G4BLOUTPUT)
        Dx, Dpx = D_dict["D_x"], D_dict["D'_x"]
        Dy, Dpy = D_dict["D_y"], D_dict["D'_y"]
        transmission = D_dict["transmission"]

        # ---------------------------
        # USER TARGETS
        # ---------------------------
        DISP_TARGET = 1e-3      # m
        TRANS_MIN   = 85.0      # % required survival

        # ---------------------------
        # WEIGHTS / PENALTIES
        # ---------------------------
        # Reasonable because your Dx ~ 10^-2 mm at test point
        W_DISP   = 1.0
        W_SLOPE  = 1.0

        # VERY large penalties for violating constraints
        PENALTY_DISP = 1e10     # penalty when |Dx| or |Dy| > DISP_TARGET
        PENALTY_TRANS = 1e10    # penalty when transmission < TRANS_MIN

        # ---------------------------
        # BASIC COST (dispersion + slopes)
        # ---------------------------
        dispersion_cost = (Dx**2 + Dy**2)       # mm^2
        slope_cost      = (Dpx**2 + Dpy**2)     # dimensionless

        cost = W_DISP * dispersion_cost + W_SLOPE * slope_cost

        # ---------------------------
        # APPLY HARD CONSTRAINTS
        # ---------------------------
        # penalty: too large dispersion
        maxD = max(abs(Dx), abs(Dy))
        if maxD > DISP_TARGET:
            exceed_factor = (maxD / DISP_TARGET)**2    # how much worse than target
            cost += PENALTY_DISP * exceed_factor

        # penalty: too low transmission
        if transmission < TRANS_MIN:
            deficit = (TRANS_MIN - transmission) / TRANS_MIN
            cost += PENALTY_TRANS * deficit**2
        
        # Safety: huge penalty for non-finite or NaN values
        if not np.isfinite(cost):
            print("Non-finite cost — penalizing.")
            return PENALTY

        print(f"Trial cost={cost:.3e} | Dx={Dx:.3e} Dpx={Dpx:.3e} Dy={Dy:.3e} Dpy={Dpy:.3e}")
        print("DEBUG params:", dispsup_params)
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
    # print("Making sure z=0 in input file (do this once)...")
    # convertZ("particles_after.txt", "particles_after_upt.txt")
    print(">>> Running Dispersion Suppressor Optimization <<<\n")
    print("Testing one midpoint configuration...")
    # Our test in the mean of these bounds
    test_x = [np.mean(b) for b in opt_bounds]
    cost_fn_from_xvec(test_x)
    
    print("\nLaunching optimizer...")
    result = differentialOptimizer()
    print("\nOptimization complete.")
