import pandas as pd
from g4beam import * 
from scan import *
#  L={2*l_drift + 2*l_quad + l_bend}
"""DQDQB: SEQUENCE, L={ATs[-1]+lengths[-1]};
    DR1: DR, L={l_drift}, AT={ATs[0]};
    QF1: QUADRUPOLE, L={l_quad}, K1={k1_init}, AT={ATs[1]};
    DR2: DR, L={l_drift}, AT={ATs[2]};
    QD1: QUADRUPOLE, L={l_quad}, K1={-k1_init}, AT={ATs[3]};
    B01: SBEND, L={l_bend}, ANGLE={angle}, K0={angle/l_bend}, AT={ATs[4]};
ENDSEQUENCE;"""
def generate_madx_dqdqb_jacobian_optimization(
    betx_target,
    bety_target,
    alfx0, alfy0, Dx0, Dpx0,
    filename="dqdqb_opt.madx",
    l_drift=0.1, #m
    l_quad=0.8,
    l_bend=1.25,
    angle=0.1,
    k1_init=0.5
):
    lengths = [l_drift, l_quad, l_drift, l_quad, l_bend]
    ATs = [0.0]  # start at zero
    for L in lengths[:-1]:
        ATs.append(ATs[-1] + L)

    
    madx_script = f"""
TITLE, "D-Q-D-Q-B Lattice Optimization";

BEAM, PARTICLE=POSMUON, PC=0.88961842;

! ---- Elements ----
DR: DRIFT, L={l_drift};

QF: QUADRUPOLE, L={l_quad}, K1={k1_init};
QD: QUADRUPOLE, L={l_quad}, K1={-k1_init};
D: DRIFT, L={l_drift};
B: SBEND, L={l_bend}, ANGLE={angle}, K0={angle/l_bend};

! ---- Sequence ----
DQDQB: LINE=(QF, D, QD, D, B);


USE, SEQUENCE=DQDQB;

! ---- Initial optics (from POST-WEDGE distribution) ----
TWISS,
    BETX={betx_target}, BETY={bety_target},
    ALFX={alfx0}, ALFY={alfy0},
    DX={Dx0}, DPX={Dpx0};


! ---- Optimization ----
MATCH, SEQUENCE=DQDQB;

    ! ---- Variables to vary ----
    VARY, NAME=QF->K1, STEP=1e-3;
    VARY, NAME=QD->K1, STEP=1e-3;
    VARY, NAME=B->L, STEP=1e-4;
    VARY, NAME=B->ANGLE, STEP=1e-4;
    

    ! ---- Constraints ----
    ! Strongly prioritize dispersion at the end (must be ≥ 1e-3)
    CONSTRAINT, RANGE=#E, DX=0.001, WEIGHT=5000;
    CONSTRAINT, RANGE=#E, DPX=0.01, WEIGHT=500;

    ! Lightly match beta functions at the end
    CONSTRAINT, RANGE=#E, BETX={betx_target}, BETY={bety_target}, WEIGHT=1;

    ! Use Jacobian optimizer
    JACOBIAN, CALLS=300, TOLERANCE=1e-12;

ENDMATCH;

! ---- Final optics ----
TWISS, FILE="dqdqb_optimized.tfs";
"""

    with open(filename, "w") as f:
        f.write(madx_script)

    print(f"MAD-X optimization script written to {filename}")


def plot_twiss_parameters(nameofmadxfile, nameofoutputfile, remove=False):
    madx_path = '/usr/local/bin/madx' 

    subprocess.run([madx_path, nameofmadxfile])

    myex=0.1e-3; 
    dpp=1e-3; 



    S=[]
    BETX = []
    BETY = []
    ALPHAX = []
    ALPHAY = []
    disp_x = []
    disp_y = []
    with open(nameofoutputfile) as f:
        for line in f:
            if line.startswith('@') or line.startswith('*') or line.startswith('$'):
                continue
            values = line.split()
            S.append(float(values[2]))
            BETX.append(float(values[3]))
            BETY.append(float(values[6]))
            ALPHAX.append(float(values[4]))
            ALPHAY.append(float(values[7]))
            disp_x.append(float(values[15]))
            disp_y.append(float(values[17]))


    sxb = []
    sxp = []
    syb = []
    syp = []

    ## horizontal

    for i in range(len(disp_x)):
        sxb.append((BETX[i] * myex) ** 0.5)
        sxp.append(disp_x[i] * dpp)
        syb.append((BETY[i] * myex) ** 0.5)
        syp.append(disp_y[i] * dpp)



    beam_size_x = []
    beam_size_y = []
    for i in range(len(sxb)):
        beam_size_x.append((BETX[i] * myex + sxp[i] ** 2))
        beam_size_y.append((BETY[i] * myex + syp[i] ** 2))


    ## vertical

    # Plot BETX and BETY
    plt.figure(figsize=(12, 8))
    plt.plot(S, BETX, label='BETX')
    plt.plot(S, BETY, label='BETY')
    plt.xlabel('Position S (m)')
    plt.ylabel('Beta (m)')
    plt.title(f'Beta Functions vs. Position, {nameofmadxfile}')
    plt.legend()
    plt.grid(True)

    plt.savefig("Betas.png")


    # Plot ALPHAX and ALPHAY
    plt.figure(figsize=(12, 8))
    plt.plot(S, ALPHAX, label='ALPHAX')
    plt.plot(S, ALPHAY, label='ALPHAY')
    plt.xlabel('Position S (m)')
    plt.ylabel('Alpha')
    plt.title(f'Alpha Functions vs. Position, {nameofmadxfile}')
    plt.legend()
    plt.grid(True)

    plt.savefig("Alphas.png")

    plt.figure(figsize=(12, 8))
    plt.plot(S, beam_size_x, label='BEAMSIZE_X')
    plt.plot(S, beam_size_y, label='BEAMSIZE_Y')

    plt.xlabel('Position S (m)')
    plt.ylabel('BEAMSIZE_X')
    plt.ylabel('BEAMSIZE_Y')
    plt.title(f'BEAMSIZE vs. Position, {nameofmadxfile}')
    plt.legend()
    plt.grid(True)
    plt.savefig("Beamsizes.png")
    
    plt.figure(figsize=(12, 8))
    plt.plot(S, disp_x, label='Dx')
    # plt.plot(S, disp_y, label='Dy')

    plt.xlabel('Position S (m)')
    plt.ylabel('Dx')
    #plt.ylabel('Dy')
    plt.title(f'D vs. Position, {nameofmadxfile}')
    plt.legend()
    plt.grid(True)
    plt.savefig("Dispersion.png")
    

    if remove == True:
        os.remove(nameofmadxfile)
        os.remove(nameofoutputfile)


if __name__ == "__main__":
    filename = "analyticDispCorr/out_1760039204_1614938upt.txt"

    with open(filename) as f:
        for line in f:
            if line.startswith("#x "):
                columns = line.strip().lstrip("#").split()
                break

    df = pd.read_csv(filename, comment="#", sep=r"\s+", names=columns)

    x_params, y_params, z_emit = calc_all_params(df)

    betx_target = x_params[1]
    alfx_0 = x_params[3]
    Dx0 = x_params[4]
    Dpx0 =  x_params[5]
    bety_target = y_params[1]
    alfy_0 = y_params[3]

    print(f"Target BETX = {betx_target:.4f} m")
    print(f"Target BETY = {bety_target:.4f} m")
    
    print(f"Initial DX = {Dx0:.4f} m")
    print(f"Initial DpX = {Dpx0:.4f} m")
    
    generate_madx_dqdqb_jacobian_optimization(betx_target, bety_target,alfx_0,alfy_0, Dx0, Dpx0)
    plot_twiss_parameters("dqdqb_opt.madx", "dqdqb_optimized.tfs")
    
    
