import numpy as np

# Load G4BL trackfile
data = np.loadtxt("particles_after.txt", comments="#")
madx_file = "sigma_from_g4bl.madx"  # Output MAD-X file

x = data[:, 0]   # mm
y = data[:, 1]   # mm
Px = data[:, 3]  # MeV/c
Py = data[:, 4]
Pz = data[:, 5]

# Reference momentum
P0 = np.mean(np.sqrt(Px**2 + Py**2 + Pz**2))

# Transverse slopes
xp = Px / Pz
yp = Py / Pz

# Momentum deviation
P = np.sqrt(Px**2 + Py**2 + Pz**2)
delta = (P - P0)/P0

# Build 6D array: [x, xp, y, yp, δ, 0]  (MAD-X SIGMA expects 6x6)
X = np.vstack([x, xp, y, yp, delta, np.zeros_like(x)]).T

# Covariance matrix
SIGMA = np.cov(X, rowvar=False, bias=True)  # bias=True gives population covariance, 6x6

# Save for inspection
np.savetxt("G4BL_SIGMA.txt", SIGMA)
# --- Flatten matrix in MAD-X order ---
madx_entries = []
for i in range(6):
    for j in range(6):
        if i == 5 or j == 5:  # Last row/col: 0 except SIGMA[5,5] = 1e-12
            madx_entries.append("1e-12" if i == j == 5 else "0")
        else:
            madx_entries.append(f"{SIGMA[i,j]:.12e}")

# Generate MAD-X snippet
madx_text = f"""! Define a minimal "dummy" lattice to test TWISS calculation
qf: quadrupole, l=0.025, k1=0.1;

beamline: sequence, l=1;
  drift, l=0.01;
  qf, at=0.45;
  drift, l=0.45;
endsequence;

! Define muon + the beam
BEAM, PARTICLE=POSMUON, ENERGY={P0:.6f};  ! P0=64.219 MeV/c in G4BL

! Automatically generated SIGMA from G4BL
USE, SEQUENCE=beamline;

SIGMA = MATRIX(6,6, [
{', '.join(madx_entries)}
]);

PTC_CREATE_UNIVERSE;
PTC_CREATE_LAYOUT, MODEL=6D;
PTC_TWISS, SIGMA=SIGMA, METHOD=6D, ICASE=2;
"""

# --- Write to file ---
with open(madx_file, "w") as f:
    f.write(madx_text)

print(f"MAD-X SIGMA snippet written to '{madx_file}'")