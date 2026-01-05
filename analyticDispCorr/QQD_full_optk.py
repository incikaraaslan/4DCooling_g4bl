
#!/usr/bin/env python3
"""
QQD_full.py

Propagate Twiss (beta, alpha) and dispersion through a QQD optics.
Features:
 - Reads G4Beamline for009-style track files to extract initial Twiss/dispersion using your calc_params logic.
 - Builds piecewise Kx(s), Ky(s) and h(s)=1/rho on a grid.
 - Propagates beta(s) by integrating the envelope ODE.
 - Propagates dispersion(s) by integrating the driven Hill equation.
 - Integrates the fundamental matrix Phi(s) and uses it to compute Sigma_f = M Sigma_i M^T
   and extracts Twiss from that as a consistency check.
 - Adds an optimizer that varies the two quadrupole strengths to minimize the final
   dispersion cost = Dx_end^2 + w * Dxp_end^2 + reg*(k-k_ref)^2.
 - Produces plots of beta_x/beta_y, dispersion, and Kx/Ky.

"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import differential_evolution


# ------------------------
# Constants & helpers
# ------------------------
MUON_MASS = 105.6583755  # MeV
C = 299792458.0  # m/s

def rel_beta(p):
    return np.sqrt(1 - 1 / (1 + (p / MUON_MASS) ** 2))

def p_total(df):
    return np.sqrt(np.square(df["Px"]) + np.square(df["Py"]) + np.square(df["Pz"]))

# ------------------------
# calc_params & calc_all_params (adapted)
# ------------------------
def calc_params(x, xp, delta, normalization=1.0):
    x = np.asarray(x) - np.mean(x)
    xp = np.asarray(xp) - np.mean(xp)

    mean_x2 = np.mean(np.square(x))
    mean_xp2 = np.mean(np.square(xp))
    mean_xxp = np.mean(x * xp)
    mean_d2 = np.mean(np.square(delta))
    mean_xd = np.mean(x * delta)
    mean_xpd = np.mean(xp * delta)

    eb = mean_x2 - (mean_xd ** 2) / (mean_d2 + 1e-30)  # mm^2
    ey = mean_xp2 - (mean_xpd ** 2) / (mean_d2 + 1e-30)
    ea = -mean_xxp + (mean_xpd * mean_xd) / (mean_d2 + 1e-30)
    e = np.sqrt(max(eb * ey - ea ** 2, 0.0))  # mm

    d = mean_xd / (mean_d2 + 1e-30)  # mm
    dp = mean_xpd / (mean_d2 + 1e-30)

    return (
        e * normalization,         # emittance in mm (as original)
        eb / (e + 1e-30) / 1000.0, # beta in m
        ey / (e + 1e-30) * 1000.0, # gamma in 1/m
        ea / (e + 1e-30),          # alpha
        d / 1000.0,                # D in m
        dp                        # D' dimensionless
    )

def calc_all_params(df):
    total_momentum = p_total(df)  # MeV/c
    mean_total_momentum = np.mean(total_momentum)
    delta = (total_momentum - mean_total_momentum) / mean_total_momentum

    x_params = calc_params(df["x"].to_numpy(), (df["Px"] / total_momentum).to_numpy(), delta,
                    normalization=mean_total_momentum / MUON_MASS)
    y_params = calc_params(df["y"].to_numpy(), (df["Py"] / total_momentum).to_numpy(), delta,
                    normalization=mean_total_momentum / MUON_MASS)

    beta_rel = rel_beta(total_momentum)
    z_emit = np.mean(beta_rel) * np.std(df["z"]) * np.std(df["Pz"]) / MUON_MASS

    return x_params, y_params, z_emit

# ------------------------
# Read for009-like trackfile
# ------------------------
def read_for009_trackfile(filepath):
    cols = None
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("#x "):
                cols = line.strip().lstrip("#").split()
                break
    if cols is None:
        cols = ["x","y","z","Px","Py","Pz","t","PDGid","EventID","TrackID","ParentID","Weight"]
    df = pd.read_csv(filepath, comment="#", sep=r"\s+", names=cols, header=None, index_col=False)
    return df

# ------------------------
# Build piecewise optics function
# ------------------------
def build_piecewise_optics(optics, ds=1e-3):
    s_positions = []
    Kx = []
    Ky = []
    h = []
    s_cursor = 0.0
    for elem in optics:
        typ = elem[0].lower()
        if typ == "quad":
            _, L, k1 = elem
            n = max(1, int(np.ceil(L / ds)))
            for i in range(n):
                s_positions.append(s_cursor + (i+0.5)*(L/n))
                Kx.append(-k1)
                Ky.append(k1)
                h.append(0.0)
            s_cursor += L
        elif typ == "drift":
            _, L = elem
            n = max(1, int(np.ceil(L / ds)))
            for i in range(n):
                s_positions.append(s_cursor + (i+0.5)*(L/n))
                Kx.append(0.0)
                Ky.append(0.0)
                h.append(0.0)
            s_cursor += L
        elif typ == "dipole":
            _, L, rho = elem
            n = max(1, int(np.ceil(L / ds)))
            for i in range(n):
                s_positions.append(s_cursor + (i+0.5)*(L/n))
                Kx.append(1.0/(rho**2))
                Ky.append(0.0)
                h.append(1.0/rho)
            s_cursor += L
        else:
            raise ValueError("Unknown element type: "+str(typ))
    return np.array(s_positions), np.array(Kx), np.array(Ky), np.array(h)

# ------------------------
# ODE RHS
# ------------------------
def beta_rhs(s, y, s_positions, K_vals):
    beta, bp = y
    K = np.interp(s, s_positions, K_vals)
    bpp = bp*bp/(2.0*beta + 1e-30) - 2.0*K*beta + 2.0/(beta + 1e-30)
    return [bp, bpp]

def disp_rhs(s, y, s_positions, K_vals, h_vals):
    eta, etap = y
    K = np.interp(s, s_positions, K_vals)
    h = np.interp(s, s_positions, h_vals)
    return [etap, -K*eta + h]

# ------------------------
# Fundamental matrix integration (Phi)
# ------------------------
def fundamental_matrix(s_positions, K_vals):
    # integrate phi11,phi21,phi12,phi22 as y = [phi11,phi21,phi12,phi22]
    def rhs(s, y):
        phi11, phi21, phi12, phi22 = y
        K = np.interp(s, s_positions, K_vals)
        return [phi21, -K*phi11, phi22, -K*phi12]
    s0 = s_positions[0]
    sN = s_positions[-1]
    t_eval = np.concatenate(([0.0], s_positions))
    y0 = [1.0, 0.0, 0.0, 1.0]
    sol = solve_ivp(rhs, (0.0, sN), y0, t_eval=t_eval, rtol=1e-8, atol=1e-9)
    # final matrix
    phi11, phi21, phi12, phi22 = sol.y[:, -1]
    M = np.array([[phi11, phi12],[phi21, phi22]])
    return M, sol.t, sol.y

# ------------------------
# Propagation wrappers
# ------------------------
def propagate_beta(beta0, alpha0, s_positions, K_vals):
    bp0 = -2.0*alpha0
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
# Sigma from Twiss
# ------------------------
def sigma_from_twiss(beta, alpha, emit_m):
    # emit_m is geometric emittance in meters
    gamma = (1.0 + alpha*alpha)/beta
    Sigma = emit_m * np.array([[beta, -alpha],[-alpha, gamma]])
    return Sigma

def twiss_from_sigma(Sigma, emit_m):
    beta = Sigma[0,0]/(emit_m + 1e-30)
    alpha = -Sigma[0,1]/(emit_m + 1e-30)
    gamma = Sigma[1,1]/(emit_m + 1e-30)
    return beta, alpha, gamma

# ------------------------
# Cost function for optimizer: vary k1,k2
# ------------------------
def dispersion_cost(kvec, optics_template, x_params, y_params,
                    ds=1e-3, w=1.0, reg=1e-6, w_beta=0.1):
    """
    Cost = Dx(L)^2 + w*Dx'(L)^2 + w_beta * (beta_x(L)-beta_x0)^2
        + regularization term
    """

    # -----------------------------
    # 1. rebuild optics with new k's
    # -----------------------------
    optics = []
    qi = 0
    for e in optics_template:
        if e[0].lower() == "quad":
            _, L, _ = e
            optics.append(("quad", L, kvec[qi]))
            qi += 1
        else:
            optics.append(e)

    # -----------------------------
    # 2. build K(s)
    # -----------------------------
    s_pos, Kx, Ky, h = build_piecewise_optics(optics, ds=ds)

    # -----------------------------
    # 3. unpack initial Twiss
    # -----------------------------
    beta_x0, alpha_x0 = x_params[0], x_params[1]
    beta_y0, alpha_y0 = y_params[0], y_params[1]
    Dx0, Dxp0 = x_params[4], x_params[5]

    # -----------------------------
    # 4. propagate beta functions
    # -----------------------------
    _, beta_x_s, alpha_x_s = propagate_beta(beta_x0, alpha_x0, s_pos, Kx)
    _, beta_y_s, alpha_y_s = propagate_beta(beta_y0, alpha_y0, s_pos, Ky)

    beta_x_end = beta_x_s[-1]
    beta_y_end = beta_y_s[-1]

    # -----------------------------
    # 5. propagate dispersion
    # -----------------------------
    _, Dx_s, Dxp_s = propagate_dispersion(Dx0, Dxp0, s_pos, Kx, h)
    Dx_end = Dx_s[-1]
    Dxp_end = Dxp_s[-1]

    # -----------------------------
    # 6. cost function
    # -----------------------------
    cost = (
        Dx_end**2
        + w * (Dxp_end**2)
        + w_beta * ((beta_x_end - beta_x0)**2 + (beta_y_end - beta_y0)**2)
    )

    # -----------------------------
    # 7. regularization
    # -----------------------------
    initial_ks = np.array([e[2] for e in optics_template if e[0].lower() == "quad"])
    cost += reg * np.sum((kvec - initial_ks)**2)

    return cost

"""def dispersion_cost(kvec, optics_template, x_params, y_params, ds=1e-3, w=1.0, reg=1e-6):
    # build optics with new k's replacing quad entries in template
    optics = []
    qi = 0
    for e in optics_template:
        if e[0].lower()=="quad":
            _, L, _ = e
            k = kvec[qi]
            optics.append(("quad", L, k))
            qi += 1
        else:
            optics.append(e)
    s_pos, Kx, Ky, h = build_piecewise_optics(optics, ds=ds)
    # initial dispersion from x_params
    Dx0 = x_params[4]; Dxp0 = x_params[5]
    # propagate dispersion
    _, Dx_s, Dxp_s = propagate_dispersion(Dx0, Dxp0, s_pos, Kx, h)
    Dx_end = Dx_s[-1]; Dxp_end = Dxp_s[-1]
    # cost
    cost = Dx_end**2 + w*(Dxp_end**2)
    # regularization to keep k near initial
    cost += reg*np.sum((kvec - np.array([e[2] for e in optics_template if e[0].lower()=="quad"]))**2)
    return cost"""

# ------------------------
# Main runner that assembles everything and optionally runs the optimizer
# ------------------------
def run_all(particle_file=None, do_plots=True, do_optimize=True, ds=1e-3):
    # Option A optics template
    optics_template = [
        ("quad", 0.20,  1.0),
        ("quad", 0.20, -0.5),
        ("dipole", 0.50, 0.25)
    ]

    # initial Twiss from particle file or defaults
    if particle_file is not None:
        df = read_for009_trackfile(particle_file)
        x_params, y_params, z_emit = calc_all_params(df)
    else:
        # default example params: (emit_mm, beta_m, gamma, alpha, D, D')
        print("Particle File not found. Going to default...")
        x_params = (0.2, 8.0, 0.2, -1.0, 0.3, 0.0)
        y_params = (0.2, 7.0, 0.15, 0.2, 0.0, 0.0)

    emit_x_mm = x_params[0]
    emit_x_m = emit_x_mm / 1000.0  # convert mm -> m (geometric)
    beta_x0 = x_params[1]; alpha_x0 = x_params[3]
    Dx0 = x_params[4]; Dxp0 = x_params[5]

    emit_y_mm = y_params[0]
    emit_y_m = emit_y_mm / 1000.0
    beta_y0 = y_params[1]; alpha_y0 = y_params[3]
    Dy0 = y_params[4]; Dyp0 = y_params[5]

    # build optics arrays
    s_pos, Kx_vals, Ky_vals, h_vals = build_piecewise_optics(optics_template, ds=ds)

    # propagate betas and dispersions (initial optics)
    tx, betax_s, betax_p = propagate_beta(beta_x0, alpha_x0, s_pos, Kx_vals)
    ty, betay_s, betay_p = propagate_beta(beta_y0, alpha_y0, s_pos, Ky_vals)
    tdx, Dx_s, Dxp_s = propagate_dispersion(Dx0, Dxp0, s_pos, Kx_vals, h_vals)
    tdy, Dy_s, Dyp_s = propagate_dispersion(Dy0, Dyp0, s_pos, Ky_vals, np.zeros_like(h_vals))

    # fundamental matrix and Sigma propagation check (x-plane)
    Mx, tphi_x, phi_x = fundamental_matrix(s_pos, Kx_vals)
    # construct Sigma_i for x
    Sigma_x_i = sigma_from_twiss(beta_x0, alpha_x0, emit_x_m)
    Sigma_x_f = Mx @ Sigma_x_i @ Mx.T
    beta_x_from_M, alpha_x_from_M, _ = twiss_from_sigma(Sigma_x_f, emit_x_m)

    # same for y
    My, tphi_y, phi_y = fundamental_matrix(s_pos, Ky_vals)
    Sigma_y_i = sigma_from_twiss(beta_y0, alpha_y0, emit_y_m)
    Sigma_y_f = My @ Sigma_y_i @ My.T
    beta_y_from_M, alpha_y_from_M, _ = twiss_from_sigma(Sigma_y_f, emit_y_m)

    results = dict()
    results['s'] = s_pos
    results['beta_x'] = betax_s; results['beta_y'] = betay_s
    results['Dx'] = Dx_s; results['Dy'] = Dy_s
    results['Kx'] = Kx_vals; results['Ky'] = Ky_vals
    results['final'] = {
        'beta_x_end': betax_s[-1],
        'beta_y_end': betay_s[-1],
        'Dx_end': Dx_s[-1],
        'Dy_end': Dy_s[-1],
        'Dxp_end': Dxp_s[-1],
        'Dyp_end': Dyp_s[-1],
        'beta_x_from_M': beta_x_from_M,
        'beta_y_from_M': beta_y_from_M,
        'alpha_x_from_M': alpha_x_from_M,
        'alpha_y_from_M': alpha_y_from_M
    }

    print("Initial beta_x, alpha_x:", beta_x0, alpha_x0)
    print("Initial beta_y, alpha_y:", beta_y0, alpha_y0)
    print("Final beta_x (envelope) =", results['final']['beta_x_end'])
    print("Final beta_x (from MΣM^T) =", results['final']['beta_x_from_M'])
    print("Final beta_y (envelope) =", results['final']['beta_y_end'])
    print("Final beta_y (from MΣM^T) =", results['final']['beta_y_from_M'])
    print("Final Dx =", results['final']['Dx_end'], " D'x =", results['final']['Dxp_end'])

    # Optimization: vary the two quad strengths
        # Optimization: vary the two quad strengths
    if do_optimize:
        # initial k vector from template
        k_init = np.array([e[2] for e in optics_template if e[0].lower() == "quad"])
        bounds = [(-20.0, 20.0)] * len(k_init)
        res = differential_evolution(
                func=lambda k: dispersion_cost(k, optics_template, x_params, y_params,
                                            ds=ds, w=1.0, reg=1e-6),
                bounds=bounds,
                maxiter=200,
                popsize=8,
                tol=1e-8,
                mutation=(0.5, 1.0),
                recombination=0.7,
                polish=True,
                updating="deferred",
                workers=1   # set -1 for parallel, but only if thread-safe
            )

        """res = minimize(
            lambda k: dispersion_cost(k, optics_template, x_params, y_params, ds=ds, w=1.0, reg=1e-6),
            k_init,
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1e-12},
        )"""
        k_opt = res.x
        print("Optimizer result:", res.message)
        print("Initial k:", k_init, "Optimized k:", k_opt)

        # build optics with optimized k and recompute final dispersion AND betas
        optics_opt = []
        qi = 0
        for e in optics_template:
            if e[0].lower() == "quad":
                optics_opt.append(("quad", e[1], k_opt[qi]))
                qi += 1
            else:
                optics_opt.append(e)

        # piecewise optics for optimized lattice
        s_pos_o, Kx_o, Ky_o, h_o = build_piecewise_optics(optics_opt, ds=ds)

        # recompute betas for optimized optics (important!)
        _, betax_s_o, betax_p_o = propagate_beta(beta_x0, alpha_x0, s_pos_o, Kx_o)
        _, betay_s_o, betay_p_o = propagate_beta(beta_y0, alpha_y0, s_pos_o, Ky_o)

        # recompute dispersion for optimized optics
        _, Dx_s_o, Dxp_s_o = propagate_dispersion(Dx0, Dxp0, s_pos_o, Kx_o, h_o)

        print("Optimized final Dx, D'x:", Dx_s_o[-1], Dxp_s_o[-1])
        # save optimized results so plotting can use them
        results["opt"] = {
            "k_init": k_init,
            "k_opt": k_opt,
            "Dx_opt": Dx_s_o,
            "Dxp_opt": Dxp_s_o,
            "s_opt": s_pos_o,
            "beta_x": betax_s_o,
            "beta_xp": betax_p_o,
            "beta_y": betay_s_o,
            "beta_yp": betay_p_o,
            "Kx": Kx_o,
            "Ky": Ky_o,
            "h": h_o,
        }


    # Plotting
    if do_plots:
        s_plot = s_pos
        minlen = min(len(s_plot), len(results['beta_x']))
        plt.figure(figsize=(10,6))
        plt.plot(s_plot[:minlen], results['beta_x'][:minlen], label=r'$\beta_x$ (env)')
        plt.plot(s_plot[:minlen], results['beta_y'][:minlen], label=r'$\beta_y$ (env)')
        # plot M-derived betas as markers at end
        plt.scatter([s_plot[-1]],[results['final']['beta_x_from_M']], marker='x', color='C0', label=r'$\beta_x$ from M')
        plt.scatter([s_plot[-1]],[results['final']['beta_y_from_M']], marker='x', color='C1', label=r'$\beta_y$ from M')
        plt.xlabel("s [m]"); plt.ylabel(r"$\beta$ [m]"); plt.legend(); plt.grid(True)
        plt.title("Beta functions")
        # if optimized lattice exists, plot its betas too
        if 'opt' in results:
            s_opt = results['opt']['s_opt']
            # align lengths for plotting
            m = min(len(s_plot), len(s_opt), len(results['opt']['beta_x']))
            plt.plot(s_opt[:m], results['opt']['beta_x'][:m], '--', label=r'$\beta_x$ (opt)')
            plt.plot(s_opt[:m], results['opt']['beta_y'][:m], '--', label=r'$\beta_y$ (opt)')
        plt.tight_layout()
        
        plt.figure(figsize=(10,6))
        plt.plot(s_plot[:minlen], results['Dx'][:minlen], label=r'$D_x$')
        plt.plot(s_plot[:minlen], results['Dy'][:minlen], label=r'$D_y$')
        if 'opt' in results:
            plt.plot(results['opt']['s_opt'][:minlen], results['opt']['Dx_opt'][:minlen], '--', label=r'$D_x$ (opt)')
        plt.xlabel("s [m]"); plt.ylabel("Dispersion [m]"); plt.legend(); plt.grid(True)
        plt.title("Dispersion functions")
        plt.tight_layout()

        plt.figure(figsize=(10,6))
        plt.plot(s_plot[:minlen], results['Kx'][:minlen], label=r'$K_x$')
        plt.plot(s_plot[:minlen], results['Ky'][:minlen], label=r'$K_y$')
        plt.xlabel("s [m]"); plt.ylabel(r"$K\;[1/m^2]$"); plt.legend(); plt.grid(True)
        plt.title("Focusing functions")
        plt.tight_layout()

        plt.show()

    return results

# ------------------------
# Run as script
# ------------------------
if __name__ == "__main__":
    res = run_all(particle_file="out_1760039204_1614938.txt", do_plots=True, do_optimize=True, ds=1e-3)
