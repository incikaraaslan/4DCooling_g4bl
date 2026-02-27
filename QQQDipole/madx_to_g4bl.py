#!/usr/bin/env python3
"""
Build a full G4BL file using:
- User-approved static G4BL boilerplate
- Quadrupoles generated from MAD-X-optimized K1 values

"""

import re

# ============================================================
# USER SETTINGS
# ============================================================

MADX_OUT = "madx.out"
OUTPUT = "G4_FinalCooling_dispsup_drift_run.g4bl"

P_GEV = 0.08872   # reference momentum [GeV/c]

# ============================================================
# STATIC G4BL CONTENT (KEEP THIS AS-IS)
# ============================================================

G4BL_STATIC = r"""
# ============================================================
# Simulates Final Cooling Drift Channel post Dispersion Suppressor
# Created by: Inci Karaaslan
# Auto-generated quad strengths from MAD-X
# ============================================================

param zbegin=0.0
start initialZ=$zbegin  y=0.000   x=0.00   z=0.00    radiusCut=300
param steppingFormat=N,GLOBAL,CL,STEP,VOL,PROCESS,P,KE,POLAR,B
param fieldVoxels=400,400,400 
param maxStep=0.5

param -unset minRangeCut=1

physics QGSP_BERT doStochastics=1 spinTracking=1 synchrotronRadiation=1 minRangeCut=$minRangeCut

trackcuts keep=mu+

######################### DEFINE INPUT BEAM

param nparticles=12000
param -unset beamfile=particles_after_bupt.txt
param -unset outname=particles_after_drift.txt

trace nTrace=1 format=ascii oneNTuple=1
beam ascii filename=$beamfile nEvents=$nparticles beamZ=0  

######################### DEFINE ELEMENTS

zntuple vd_start z=0.0 file=vd_start_dispsup.txt format=ascii require=PDGid==-13
"""

# ============================================================
# QUAD GEOMETRY (MATCHES YOUR TEMPLATE)
# ============================================================

QUADS = [
    ("Q1",  "k_q1"),
    ("Q2",  "k_q2"),
    ("Q3",  "k_q3"),
    ("Q4",  "k_q4"),
    ("Q5",  "k_q5"),
    ("Q6",  "k_q6"),
    ("Q7",  "k_q7"),
    ("Q8",  "k_q8"),
    ("QF1", "k_qf1"),
    ("QD1", "k_qd1"),
]

THICKNESS = "{thickness}"
MAXSTEP = 0.5

# ============================================================
# STATIC FOOTER (DRIFTS, WALLS, OUTPUT)
# ============================================================

G4BL_FOOTER = r"""
zntuple vd_end z=$zEnd file=$outname format=ascii require=PDGid==-13
"""

# ============================================================
# PHYSICS
# ============================================================

def brho(p):
    return p / 0.299792458

def k1_to_gradient(k1):
    return k1 * brho(P_GEV)

# ============================================================
# MAD-X PARSING
# ============================================================

def parse_madx_kvals(fname):
    pat = re.compile(r'^\s*(k_q\w+)\s*=\s*([-+0-9.eE]+)', re.MULTILINE)
    with open(fname) as f:
        txt = f.read()
    kvals = {k: float(v) for k, v in pat.findall(txt)}
    if not kvals:
        raise RuntimeError("NO k_q* VALUES FOUND — WRONG MADX.OUT")
    return kvals

# ============================================================
# QUAD GENERATION
# ============================================================

def generate_quads(kvals):
    out = []
    out.append("\n### BEGIN AUTO-GENERATED MAD-X QUADS ###\n")

    for name, kname in QUADS:
        if kname not in kvals:
            raise RuntimeError(f"MISSING {kname} IN MAD-X OUTPUT")

        grad = k1_to_gradient(kvals[kname])

        out.append(f"""
# --- {name}
genericquad {name} \\
    gradient={grad:.6f} \\
    fieldLength={{{name}_length}} \\
    ironLength={{{name}_length}} \\
    ironRadius={{radius_{name.lower()}}}+{{thickness}} \\
    apertureRadius={{radius_{name.lower()}}} \\
    maxStep=0.5 \\
    ironMaterial=Fe fieldMaterial=Vacuum \\
    kill=1 \\
    fringe=0

place {name} z={{{name}_z}}
""")

    out.append("\n### END AUTO-GENERATED MAD-X QUADS ###\n")
    return "".join(out)

# ============================================================
# MAIN
# ============================================================

def main():
    kvals = parse_madx_kvals(MADX_OUT)
    quad_block = generate_quads(kvals)

    with open(OUTPUT, "w") as f:
        f.write(G4BL_STATIC)
        f.write(quad_block)
        f.write(G4BL_FOOTER)

    print(f"[OK] Wrote runnable G4BL file: {OUTPUT}")

if __name__ == "__main__":
    main()