#!/usr/bin/env python3
"""
QQD_full_opt_Br.py

Propagate Twiss (beta, alpha) and dispersion through a QQD optics.
Optimizes quadrupoles by directly varying (B_pole, r_aperture) pairs,
and supports "box" dipoles as ("dipole", L, B_field, width, height).

Quadrupole format in optics_template: ("quad", L, B_pole, r_aperture)
Dipole format in optics_template:    ("dipole", L, B_field, width, height)

Optimizer parameter vector x: [B1, r1, B2, r2, ...] (one pair per quad)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
import math
from g4beam import *
from scan import *

# ------------------------
# Constants & helpers
# ------------------------
MUON_MASS_MeV = 105.6583755  # MeV/c^2
C = 299792458.0  # m/s
e_SI = 1.602176634e-19

def rel_beta(p_MeV):
    """Relativistic beta for momentum in MeV/c (approx)."""
    return np.sqrt(1 - 1 / (1 + (p_MeV / MUON_MASS_MeV) ** 2))

def p_total_from_df(df):
    """Return total momentum per particle (MeV/c) from columns Px,Py,Pz in MeV/c."""
    return np.sqrt(np.square(df["Px"]) + np.square(df["Py"]) + np.square(df["Pz"]))

def Brho_from_p_MeV(p_MeV):
    """
    Convert momentum in MeV/c to magnetic rigidity Bρ in T·m.
    Uses: p [GeV/c] / 0.299792458 = Bρ [T·m] from https://uspas.fnal.gov/materials/14Knoxville/Lecture5_Transverse_Beam_Optics_1.pdf
    
    Another way to do this:
    p_SI = p_MeV * (1.6*10**(-13)/(3*10**8)) # kg m/s
    Brho = p_SI / (1.6 * 10**(-19)) # p/q of the particle, q in Coulombs
    """
    p_GeV = p_MeV * 1e-3
    return p_GeV / 0.299792458  # T·m

# ------------------------
# Trackfile parsing and initial beam params
# ------------------------
def read_for009_trackfile(filepath):
    """Read a for009-like track file produced by G4BL (looks for header '#x ')."""
    cols = None
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("#x "):
                cols = line.strip().lstrip("#").split()
                break
    if cols is None:
        # fallback: common columns
        cols = ["x","y","z","Px","Py","Pz","t","PDGid","EventID","TrackID","ParentID","Weight"]
    df = pd.read_csv(filepath, comment="#", sep=r"\s+", names=cols, header=None, index_col=False)
    return df

# ------------------------
# Piecewise optics builder
# ------------------------
def build_piecewise_optics(optics, Brho, ds=1e-3):
    """
    Build arrays s_positions, Kx(s), Ky(s), h(s) from optics list.

    Element formats:
      quad:   ("quad", L, B_pole, r_aperture)        # B_pole in T, r in m
      drift:  ("drift", L)
      dipole: ("dipole", L, B_field, width, height) # B_field in T ; width/height used as metadata
    """
    s_positions = []
    Kx = []
    Ky = []
    h = []
    s_cursor = 0.0

    for elem in optics:
        typ = elem[0].lower()
        if typ == "quad":
            _, L, B_pole, r_ap = elem
            # gradient G (T/m) and k1 (1/m^2)
            if r_ap == 0:
                raise ValueError("quad with zero aperture radius")
            G = B_pole / r_ap
            k1 = G / Brho  # focusing strength (1/m^2)
            n = max(1, int(np.ceil(L / ds)))
            for i in range(n):
                s_positions.append(s_cursor + (i + 0.5) * (L / n))
                # convention: x-plane focusing sign depends on B_pole sign
                Kx.append(-k1)   # assuming B_pole>0 focuses y, defocuses x -> keep sign as original approach
                Ky.append(k1)
                h.append(0.0)
            s_cursor += L

        elif typ == "drift":
            _, L = elem
            n = max(1, int(np.ceil(L / ds)))
            for i in range(n):
                s_positions.append(s_cursor + (i + 0.5) * (L / n))
                Kx.append(0.0); Ky.append(0.0); h.append(0.0)
            s_cursor += L
        
        elif typ == "dipole":
            # Option C: dipole carries geometry (width,height) but bending radius from field
            _, L, B_dip, width, height = elem
            if abs(B_dip) < 1e-30:
                rho = 1e30  # effectively straight
            else:
                rho = Brho / B_dip
            n = max(1, int(np.ceil(L / ds)))
            for i in range(n):
                s_positions.append(s_cursor + (i + 0.5) * (L / n))
                Kx.append(1.0 / (rho ** 2) if rho != 0 else 0.0)
                Ky.append(0.0)
                h.append(1.0 / rho if rho != 0 else 0.0)
            s_cursor += L
            
        else:
            raise ValueError(f"Unknown element type: {elem}")
        """elif typ == "dipole":
        # New: support bending in either plane
        # Format: ("dipole", L, B_field, width, height, plane)
            _, L, B_dip, width, height, plane = elem

            if abs(B_dip) < 1e-30:
                rho = 1e30  # effectively straight
            else:
                rho = Brho / B_dip

            n = max(1, int(np.ceil(L / ds)))
            for i in range(n):
                s_positions.append(s_cursor + (i + 0.5) * (L / n))
                if plane.lower() == 'x':
                    Kx.append(1.0 / (rho ** 2) if rho != 0 else 0.0)
                    Ky.append(0.0)
                    h.append(1.0 / rho if rho != 0 else 0.0)
                elif plane.lower() == 'y':
                    Kx.append(0.0)
                    Ky.append(1.0 / (rho ** 2) if rho != 0 else 0.0)
                    h.append(0.0)
                else:
                    raise ValueError(f"Unknown dipole plane: {plane}")
                s_cursor += L"""

    return np.array(s_positions), np.array(Kx), np.array(Ky), np.array(h)

# ------------------------
# ODE RHS functions
# ------------------------
def beta_rhs(s, y, s_positions, K_vals):
    beta, bp = y
    K = np.interp(s, s_positions, K_vals)
    """if beta < 0:
        raise RuntimeError("Beta function became non-positive")
    else:"""
    bpp = bp * bp / (2.0 * beta +1e-30) - 2.0 * K * beta + 2.0 / (beta+1e-30)
    return [bp, bpp]

def disp_rhs(s, y, s_positions, K_vals, h_vals):
    eta, etap = y
    K = np.interp(s, s_positions, K_vals)
    h = np.interp(s, s_positions, h_vals)
    return [etap, -K * eta + h]

# ------------------------
# Fundamental matrix integration (Phi)
# ------------------------
def fundamental_matrix(s_positions, K_vals):
    def rhs(s, y):
        phi11, phi21, phi12, phi22 = y
        K = np.interp(s, s_positions, K_vals)
        return [phi21, -K * phi11, phi22, -K * phi12]

    sN = s_positions[-1]
    t_eval = np.concatenate(([0.0], s_positions))
    y0 = [1.0, 0.0, 0.0, 1.0]
    sol = solve_ivp(rhs, (0.0, sN), y0, t_eval=t_eval, rtol=1e-8, atol=1e-9)
    phi11, phi21, phi12, phi22 = sol.y[:, -1]
    M = np.array([[phi11, phi12], [phi21, phi22]])
    return M, sol.t, sol.y

# ------------------------
# Propagation helpers
# ------------------------
def propagate_beta(beta0, alpha0, s_positions, K_vals):
    bp0 = -2.0 * alpha0 # Accelerator physics software convention, with diffeqs being defines as x'' + Kx and not -Kx.
    sN = s_positions[-1]
    t_eval = np.concatenate(([0.0], s_positions))
    sol = solve_ivp(beta_rhs, (0.0, sN), [beta0, bp0], args=(s_positions, K_vals), t_eval=t_eval, rtol=1e-8, atol=1e-9)
    return sol.t, sol.y[0], sol.y[1]

def propagate_dispersion(D0, Dp0, s_positions, K_vals, h_vals):
    sN = s_positions[-1]
    t_eval = np.concatenate(([0.0], s_positions))
    sol = solve_ivp(disp_rhs, (0.0, sN), [D0, Dp0], args=(s_positions, K_vals, h_vals), t_eval=t_eval, rtol=1e-8, atol=1e-9)
    return sol.t, sol.y[0], sol.y[1]

# ------------------------
# Sigma / Twiss helpers
# ------------------------
def sigma_from_twiss(beta, alpha, emit_m):
    gamma = (1.0 + alpha * alpha) / beta
    Sigma = emit_m * np.array([[beta, -alpha], [-alpha, gamma]])
    return Sigma

def twiss_from_sigma(Sigma, emit_m):
    beta = Sigma[0, 0] / (emit_m)
    alpha = -Sigma[0, 1] / (emit_m)
    return beta, alpha, (1.0 + alpha * alpha) / beta

# ------------------------
# Cost function: optimizer varies (B_pole, r) pairs
# ------------------------
def dispersion_cost_br(x_vec, optics_template, x_params, y_params, Brho, ds=1e-3, w=0.5, reg=1e-6, w_beta=5.0, w_betay=10.0):
    """
    x_vec: flat vector [B1, r1, B2, r2, ...] for each quad in optics_template order
    """
    # rebuild optics: replace each quad in template with (quad, L, B_pole, r_ap)
    optics = []
    qi = 0
    for e in optics_template:
        if e[0].lower() == "quad":
            _, L, B_init, r_init = e
            B_new = x_vec[qi]
            r_new = x_vec[qi + 1]
            # sanity: ensure radius positive
            if r_new <= 0:
                return 1e12 + 1e8 * abs(r_new)  # punish invalid radii
            optics.append(("quad", L, B_new, r_new))
            qi += 2
        else:
            optics.append(e)

    # build piecewise arrays
    s_pos, Kx, Ky, h = build_piecewise_optics(optics, Brho, ds=ds)

    # initial Twiss / dispersion
    beta_x0, alpha_x0 = x_params[1], x_params[3]  # note: calc_params format (e, beta, gamma, alpha, D, Dp)
    beta_y0, alpha_y0 = y_params[1], y_params[3]
    Dx0, Dxp0 = x_params[4], x_params[5]

    # propagate envelopes
    _, beta_x_s, alpha_x_s = propagate_beta(beta_x0, alpha_x0, s_pos, Kx)
    _, beta_y_s, alpha_y_s = propagate_beta(beta_y0, alpha_y0, s_pos, Ky)

    # propagate dispersion
    _, Dx_s, Dxp_s = propagate_dispersion(Dx0, Dxp0, s_pos, Kx, h)
    Dx_end = Dx_s[-1]; Dxp_end = Dxp_s[-1]

    # cost: dispersion + beta-preservation + regularization to keep B/r near initial 
    cost = Dx_end ** 2 + w * (Dxp_end ** 2) + w_beta * ((beta_x_s[-1] - beta_x0) ** 2) + w_betay * ((beta_y_s[-1] - beta_y0) ** 2)

    # regularization: keep B/r (i.e., G) near initial
    # build initial G vector
    initial_Gs = []
    qi = 0
    for e in optics_template:
        if e[0].lower() == "quad":
            _, L, B_init, r_init = e
            G_init = B_init / r_init
            initial_Gs.append(G_init)
    # current Gs
    current_Gs = []
    qi = 0
    for e in optics_template:
        if e[0].lower() == "quad":
            Bn = x_vec[qi]; rn = x_vec[qi + 1]
            current_Gs.append(Bn / rn)
            qi += 2
    initial_Gs = np.array(initial_Gs)
    current_Gs = np.array(current_Gs)
    cost += reg * np.sum((current_Gs - initial_Gs) ** 2)

    return float(cost)
def region_cost_br_mixed(
    x_vec,
    elem_map,
    optics_template,
    x_params,
    y_params,
    Brho,
    ds=1e-3,
    w_beta_qdq=5.0,
    w_disp_qdq=0.1,
    w_disp_db=10.0,
    w_beta_db=0.1,
    reg=1e-6
):
    """
    Region-aware cost for mixed optimization of:
      - quad: (B_pole, r_ap)
      - drift: L
      - dipole: B_field

    elem_map: list of ("quad"/"drift"/"dipole", original_element)
    """

    # --------------------
    # rebuild optics from x_vec
    # --------------------
    optics = []
    xi = 0

    for typ, e in elem_map:
        if typ == "quad":
            _, L, B_init, r_init = e
            B_new = x_vec[xi]
            r_new = x_vec[xi + 1]
            xi += 2

            if r_new <= 0:
                return 1e12 + 1e8 * abs(r_new)

            optics.append(("quad", L, B_new, r_new))

        elif typ == "drift":
            _, L_init = e
            L_new = x_vec[xi]
            xi += 1

            if L_new <= 0:
                return 1e12 + 1e8 * abs(L_new)

            optics.append(("drift", L_new))

        elif typ == "dipole":
            _, L, B_init, w, h = e
            B_new = x_vec[xi]
            xi += 1

            optics.append(("dipole", L, B_new, w, h))

        else:
            raise ValueError(f"Unknown element type: {typ}")

    # --------------------
    # build piecewise lattice
    # --------------------
    s_pos, Kx, Ky, h = build_piecewise_optics(optics, Brho, ds=ds)

    # --------------------
    # propagate Twiss & dispersion
    # --------------------
    beta_x0, alpha_x0 = x_params[1], x_params[3]
    beta_y0, alpha_y0 = y_params[1], y_params[3]
    Dx0, Dxp0 = x_params[4], x_params[5]

    _, beta_x_s, _ = propagate_beta(beta_x0, alpha_x0, s_pos, Kx)
    _, beta_y_s, _ = propagate_beta(beta_y0, alpha_y0, s_pos, Ky)
    _, Dx_s, _     = propagate_dispersion(Dx0, Dxp0, s_pos, Kx, h)

    # --------------------
    # identify QDQ and DB slices
    # --------------------
    spans = get_lattice_spans(optics)
    qdq_idx = []
    db_idx  = []

    for sp in spans:
        idx = np.where(
            (s_pos >= sp["s_start"]) &
            (s_pos <= sp["s_end"])
        )[0]

        if sp["type"] == "quad":
            qdq_idx.extend(idx)
        elif sp["type"] == "dipole":
            db_idx.extend(idx)

    qdq_idx = np.array(qdq_idx, dtype=int)
    db_idx  = np.array(db_idx, dtype=int)

    # --------------------
    # QDQ region cost
    # --------------------
    beta_max_qdq = 2.0  # meters

    cost_beta_qdq = np.mean(
        np.maximum(beta_x_s[qdq_idx] - beta_max_qdq, 0.0)**2 +
        np.maximum(beta_y_s[qdq_idx] - beta_max_qdq, 0.0)**2
    )

    cost_disp_qdq = np.mean(Dx_s[qdq_idx]**2)

    cost_qdq = (
        w_beta_qdq * cost_beta_qdq +
        w_disp_qdq * cost_disp_qdq
    )

    # --------------------
    # DB region cost
    # --------------------
    beta_tol_db = 1e-2

    cost_disp_db = np.mean(Dx_s[db_idx]**2)

    cost_beta_db = np.mean(
        ((beta_x_s[db_idx] - beta_tol_db) / beta_tol_db)**2 +
        ((beta_y_s[db_idx] - beta_tol_db) / beta_tol_db)**2
    )

    cost_db = (
        w_disp_db * cost_disp_db +
        w_beta_db * cost_beta_db
    )

    # --------------------
    # regularization: quad gradients only
    # --------------------
    initial_G = []
    current_G = []

    xi = 0
    for typ, e in elem_map:
        if typ == "quad":
            _, _, B0, r0 = e
            Bn = x_vec[xi]
            rn = x_vec[xi + 1]
            xi += 2

            initial_G.append(B0 / r0)
            current_G.append(Bn / rn)

        elif typ == "drift":
            xi += 1
        elif typ == "dipole":
            xi += 1

    initial_G = np.array(initial_G)
    current_G = np.array(current_G)

    cost_reg = reg * np.sum((current_G - initial_G)**2)

    # --------------------
    # total cost
    # --------------------
    return float(cost_qdq + cost_db + cost_reg)

def region_cost_br(x_vec, optics_template, x_params, y_params, Brho, ds=1e-3,
                    w_beta_qdq=5.0, w_disp_qdq=0.1,
                    w_disp_db=10.0, w_beta_db=0.1, reg=1e-6):
    """
    Region-aware cost for QQD lattice optimization.

    QDQ region: maximize beta, small weight on dispersion.
    DB region: minimize dispersion, tolerate beta ~ 1e-2.

    x_vec: flat vector [B1,r1,B2,r2,...] for each quad
    """
    # --------------------
    # rebuild optics
    # --------------------
    optics = []
    qi = 0
    for e in optics_template:
        if e[0].lower() == "quad":
            _, L, B_init, r_init = e
            B_new = x_vec[qi]
            r_new = x_vec[qi + 1]
            if r_new <= 0:  # punish invalid radii
                return 1e12 + 1e8 * abs(r_new)
            optics.append(("quad", L, B_new, r_new))
            qi += 2
        else:
            optics.append(e)

    # build piecewise lattice
    s_pos, Kx, Ky, h = build_piecewise_optics(optics, Brho, ds=ds)

    # --------------------
    # propagate Twiss & dispersion
    # --------------------
    beta_x0, alpha_x0 = x_params[1], x_params[3]
    beta_y0, alpha_y0 = y_params[1], y_params[3]
    Dx0, Dxp0 = x_params[4], x_params[5]

    _, beta_x_s, _ = propagate_beta(beta_x0, alpha_x0, s_pos, Kx)
    _, beta_y_s, _ = propagate_beta(beta_y0, alpha_y0, s_pos, Ky)
    _, Dx_s, _     = propagate_dispersion(Dx0, Dxp0, s_pos, Kx, h)

    # --------------------
    # identify QDQ and DB slices
    # --------------------
    spans = get_lattice_spans(optics_template)
    qdq_slices = []
    db_slices  = []
    for e, span in zip(optics_template, spans):
        idxs = np.where((s_pos >= span["s_start"]) & (s_pos <= span["s_end"]))[0]
        if e[0].lower() == "quad":
            qdq_slices.extend(idxs)
        elif e[0].lower() == "dipole":
            db_slices.extend(idxs)

    # --------------------
    # QDQ region cost
    # --------------------
    beta_max_qdq = 2.0  # meters
    beta_x_qdq = beta_x_s[qdq_slices]
    beta_y_qdq = beta_y_s[qdq_slices]

    # Penalize beta above the limit strongly
    cost_beta_qdq = np.mean( np.maximum(beta_x_qdq - beta_max_qdq, 0.0)**2 +
                            np.maximum(beta_y_qdq - beta_max_qdq, 0.0)**2 )
    cost_disp_qdq = np.mean(Dx_s[qdq_slices]**2)
    cost_qdq = w_beta_qdq * cost_beta_qdq + w_disp_qdq * cost_disp_qdq


    # --------------------
    # DB region cost
    # --------------------
    beta_tol_db = 1e-2  # allow small beta here
    cost_disp_db = np.mean(Dx_s[db_slices]**2)
    cost_beta_db = np.mean( ((beta_x_s[db_slices]-beta_tol_db)/beta_tol_db)**2 +
                            ((beta_y_s[db_slices]-beta_tol_db)/beta_tol_db)**2 )
    cost_db = w_disp_db * cost_disp_db + w_beta_db * cost_beta_db

    # --------------------
    # regularization: keep B/r (G) near initial
    # --------------------
    initial_Gs = []
    current_Gs = []
    qi = 0
    for e in optics_template:
        if e[0].lower() == "quad":
            _, _, B_init, r_init = e
            initial_Gs.append(B_init / r_init)
            current_Gs.append(x_vec[qi] / x_vec[qi+1])
            qi += 2
    initial_Gs = np.array(initial_Gs)
    current_Gs = np.array(current_Gs)
    cost_reg = reg * np.sum((current_Gs - initial_Gs)**2)

    # --------------------
    # total cost
    # --------------------
    total_cost = cost_qdq + cost_db + cost_reg
    return float(total_cost)

# For plotting where things are
def get_lattice_spans(optics):
    """
    Returns a list of dicts with element type and (s_start, s_end).
    """
    spans = []
    s_cursor = 0.0

    for elem in optics:
        typ = elem[0].lower()
        L = elem[1]
        spans.append({
            "type": typ,
            "s_start": s_cursor,
            "s_end": s_cursor + L
        })
        s_cursor += L

    return spans
def shade_lattice(ax, spans, alpha=0.15):
    """
    Shades lattice elements on an existing matplotlib axis.
    """
    colors = {
        "quad": "red",
        "dipole": "blue",
        "drift": "gray"
    }

    labeled = set()

    for sp in spans:
        c = colors.get(sp["type"], "black")
        label = sp["type"].capitalize() if sp["type"] not in labeled else None

        ax.axvspan(
            sp["s_start"],
            sp["s_end"],
            color=c,
            alpha=alpha,
            label=label
        )

        labeled.add(sp["type"])

# ------------------------
# Main runner
# ------------------------
def run_all(particle_file=None, do_plots=True, do_optimize=True, ds=1e-3):
    # Example optics template: quads have (L, B_pole, r) and dipole is (L, B, width, height)
    optics_template = [
        ("quad", 0.2, 3.0, 0.29),  # L=0.03 m, B_pole=0.02965 T, r=0.1 m
        ("drift", 0.1),
        ("quad", 0.2, -2.0, 0.29), # L=0.025 m, B_pole=-0.014825 T, r=0.1 m
        ("drift", 0.2),
        ("quad", 0.2, 2.0, 0.29),
        ("drift", 0.2),
        ("dipole", 0.320, 1.5, 0.1, 0.1)  # L=0.03 m, B=1.5 T, width=0.1 m , height=0.1 m
    ]
    
    # Plotting where things are
    lattice_spans = get_lattice_spans(optics_template)
    
    # read particle file to get initial twiss & Brho
    if particle_file:
        df = read_for009_trackfile(particle_file)
        x_params, y_params, z_emit = calc_all_params(df)
        p_ref_mean = np.mean(p_total_from_df(df))
        # p_ref_mean in MeV/c; compute Brho
        Brho = Brho_from_p_MeV(p_ref_mean)
    else:
        # fallback defaults
        print("No particle file provided; using default Twiss and Brho from 88 MeV/c muon.")
        p_ref_mean = 88.961842  # MeV/c
        Brho = Brho_from_p_MeV(p_ref_mean)
        # default Twiss tuple format returned by calc_params: (emit_mm, beta_m, gamma, alpha, D, Dp)
        x_params = (0.032314888033942994, 0.04352034430332993, 255.79365628939377, -3.183116082131164, 0.015274752174883286, -0.16276468456961618)
        y_params = (0.11683931480995156, 0.022433027940989454, 73.66612099940427, -0.8078082388066774, -0.0003564703042096456, -0.01884340044268174)

    print(f"Using Bρ = {Brho:.6g} T·m (from p_ref ~ {p_ref_mean:.3f} MeV/c)")

    # Build initial piecewise optics & propagate for baseline
    s_pos, Kx_vals, Ky_vals, h_vals = build_piecewise_optics(optics_template, Brho, ds=ds)

    tx, betax_s, betax_p = propagate_beta(x_params[1], x_params[3], s_pos, Kx_vals)
    ty, betay_s, betay_p = propagate_beta(y_params[1], y_params[3], s_pos, Ky_vals)
    tdx, Dx_s, Dxp_s = propagate_dispersion(x_params[4], x_params[5], s_pos, Kx_vals, h_vals)

    # prepare results dict
    results = {
        "s": s_pos,
        "beta_x": betax_s, "beta_y": betay_s,
        "Dx": Dx_s, "Dy": None, "Dxp": Dxp_s,
        "Kx": Kx_vals, "Ky": Ky_vals, "h": h_vals,
        "Brho": Brho
    }

    # Show baseline end values
    print("Baseline Dx_end =", Dx_s[-1], "Dxp_end =", Dxp_s[-1])

    # ------------------------
    # Optimization
    # ------------------------
    if do_optimize:
        x0 = []
        bounds = []
        elem_map = []  # track element type for reconstruction

        for e in optics_template:
            typ = e[0].lower()
            if typ == "quad":
                _, L, B_init, r_init = e
                x0.extend([B_init, r_init])
                bounds.extend([(-3.0, 3.0), (0.05, 0.5)])
                elem_map.append(("quad", e))  # consume 2 entries in x_vec
            elif typ == "drift":
                _, L_init = e
                x0.append(L_init)
                bounds.append((0.001, 0.2))
                elem_map.append(("drift", e))
            elif typ == "dipole":
                _, L, B_opt, w, h = e
                x0.append(B_init)
                bounds.append((0.0, 4.0))
                elem_map.append(("dipole", e))

        # cost wrapper
        def total_cost(x_vec):
            return region_cost_br_mixed(
                x_vec,
                elem_map,
                optics_template,
                x_params,
                y_params,
                Brho,
                ds=ds,
                reg=1e-6
            )

        """def total_cost(x_vec):
            optics = []
            qi = 0
            for typ, e in elem_map:
                if typ == "quad":
                    L = e[1]
                    B_new = x_vec[qi]
                    r_new = x_vec[qi + 1]
                    qi += 2
                    optics.append(("quad", L, B_new, r_new))
                elif typ == "drift":
                    L_new = x_vec[qi]; qi += 1
                    optics.append(("drift", L_new))
                elif typ == "dipole":
                    L, _, _, w, h = e
                    B_new = x_vec[qi]; qi += 1
                    optics.append(("dipole", L, B_new, w, h))
            return region_cost_br(x_vec, optics_template, x_params, y_params, Brho, ds=ds, reg=1e-6)"""

        print("Starting optimization of quads, drifts, and dipoles...")
        res = minimize(
            lambda x: total_cost(x),
            x0,
            method="Nelder-Mead",
            bounds=bounds,
            options={"ftol": 1e-12},
        )
        """res = differential_evolution(
            func=total_cost,
            bounds=bounds,
            maxiter=200,
            popsize=10,
            tol=1e-8,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=True,
            updating="deferred",
            workers=1
        )"""

        print("Optimization finished:", res.success, res.message)
        x_opt = res.x

        # rebuild optimized optics
        optics_opt = []
        qi = 0
        for typ, e in elem_map:
            if typ == "quad":
                L = e[1]
                B_opt = x_opt[qi]; r_opt = x_opt[qi + 1]; qi += 2
                optics_opt.append(("quad", L, B_opt, r_opt))
            elif typ == "drift":
                L_opt = x_opt[qi]; qi += 1
                optics_opt.append(("drift", L_opt))
            elif typ == "dipole":
                _, L, B_opt, w, h = e
                B_opt = x_opt[qi]; qi += 1
                optics_opt.append(("dipole", L, B_opt, w, h))


        print("\n===== OPTIMIZED MAGNET PARAMETERS =====")
        quad_index = 1
        dip_index = 1
        drift_index = 1
        for e in optics_opt:
            typ = e[0].lower()
            print(e)
            if typ == "quad":
                _, L, B_opt, r_opt = e
                G_opt = B_opt / r_opt
                k1_opt = G_opt / Brho
                print(f"Quad {quad_index}: L={L:.6f} m, B={B_opt:.6f} T, r={r_opt:.6f} m, G={G_opt:.6f} T/m, k1={k1_opt:.6f} 1/m^2")
                quad_index += 1
            elif typ == "dipole":
                _, L, B_opt, w, h = e
                rho = Brho / B_opt if B_opt != 0 else np.inf
                print(f"Dipole {dip_index}: L={L:.6f} m, B={B_opt:.6f} T, w={w:.6f} m, h={h:.6f} m, rho={rho:.6f} m")
                dip_index += 1
            elif typ == "drift":
                _, L_opt = e
                print(f"Drift {drift_index}: L={L_opt:.6f} m")
                drift_index += 1
        print("========================================\n")

        # propagate optimized lattice for plots
        s_pos_o, Kx_o, Ky_o, h_o = build_piecewise_optics(optics_opt, Brho, ds=ds)
        _, betax_s_o, _ = propagate_beta(x_params[1], x_params[3], s_pos_o, Kx_o)
        _, betay_s_o, _ = propagate_beta(y_params[1], y_params[3], s_pos_o, Ky_o)
        _, Dx_s_o, Dxp_s_o = propagate_dispersion(x_params[4], x_params[5], s_pos_o, Kx_o, h_o)

        results["opt"] = {
            "x_opt": x_opt,
            "s_opt": s_pos_o,
            "Dx_opt": Dx_s_o,
            "Dxp_opt": Dxp_s_o,
            "beta_x_opt": betax_s_o,
            "beta_y_opt": betay_s_o,
            "Kx_opt": Kx_o,
            "Ky_opt": Ky_o,
            "h_opt": h_o
        }


    # Plotting
    if do_plots:
        s_plot = results["s"]
        minlen = min(len(s_plot), len(results["beta_x"]))
        
        fig, ax = plt.subplots(figsize=(10,6))

        ax.plot(s_plot[:minlen], results["beta_x"][:minlen], label=r'$\beta_x$')
        ax.plot(s_plot[:minlen], results["beta_y"][:minlen], label=r'$\beta_y$')

        shade_lattice(ax, lattice_spans)

        if "opt" in results:
            m = min(len(results["opt"]["s_opt"]), minlen)
            ax.plot(results["opt"]["s_opt"][:m], results["opt"]["beta_x_opt"][:m], '--', label=r'$\beta_x$ (opt)')
            ax.plot(results["opt"]["s_opt"][:m], results["opt"]["beta_y_opt"][:m], '--', label=r'$\beta_y$ (opt)')

        ax.set_xlabel("s [m]")
        ax.set_ylabel(r"$\beta$ [m]")
        ax.set_title("Beta functions")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.savefig("QQD_trialBeta_17.png")

        
        fig, ax = plt.subplots(figsize=(10,6))

        ax.plot(s_plot[:minlen], results["Dx"][:minlen], label=r'$D_x$')
        
        shade_lattice(ax, lattice_spans)

        if "opt" in results:
            ax.plot(results["opt"]["s_opt"][:minlen], results["opt"]["Dx_opt"][:minlen], '--', label=r'$D_x$ (opt)')
        
        ax.set_xlabel("s [m]")
        ax.set_ylabel("Dispersion [m]"); 
        ax.legend()
        ax.grid(True)
        ax.set_title("Dispersion functions")
        plt.tight_layout()
        plt.savefig("QQD_trialDisp_17.png")
        
        fig, ax = plt.subplots(figsize=(10,6))
        
        ax.plot(s_plot[:minlen], results["Dxp"][:minlen], label=r'$D_xp$')
        
        shade_lattice(ax, lattice_spans)
        
        if "opt" in results:
            ax.plot(results["opt"]["s_opt"][:minlen], results["opt"]["Dxp_opt"][:minlen], '--', label=r'$D_xp$ (opt)')
            
        ax.set_xlabel("s [m]")
        ax.set_ylabel("Dispersion Derivative")
        ax.grid(True)
        ax.set_title("Dispersion Derivative functions")
        ax.legend()
        plt.tight_layout()
        plt.savefig("QQD_trialDispDeriv_17.png")
        
        fig, ax = plt.subplots(figsize=(10,6))
        
        ax.plot(s_plot[:minlen], results["Kx"][:minlen], label=r'$K_x$')
        ax.plot(s_plot[:minlen], results["Ky"][:minlen], label=r'$K_y$')
        
        shade_lattice(ax, lattice_spans)
        
        if "opt" in results:
            ax.plot(results["opt"]["s_opt"][:minlen], results["opt"]["Kx_opt"][:minlen], '--', label=r'$K_x$ (opt)')
            ax.plot(results["opt"]["s_opt"][:minlen], results["opt"]["Ky_opt"][:minlen], '--', label=r'$K_y$ (opt)')
        ax.set_xlabel("s [m]")
        ax.set_ylabel(r"$K\;[1/m^2]$")
        ax.grid(True)
        ax.set_title("Focusing functions")
        ax.legend()
        plt.tight_layout()
        plt.savefig("QQD_trialFocusing_17.png")
        plt.show()

    return results

# ------------------------
# CLI entry
# ------------------------
if __name__ == "__main__":
    # change this to your post-wedge output if needed
    post_wedge_filename = "particles_afterupt.txt"
    res = run_all(particle_file=post_wedge_filename, do_plots=True, do_optimize=True, ds=1e-3)
