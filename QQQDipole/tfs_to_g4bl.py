import re
import sys

MADX_OUT = "madx.out"
G4BL_IN  = "G4_FinalCooling_dispsup_driftm_run.g4bl"
G4BL_OUT = "G4_FinalCooling_dispsup_driftm_run_filled.g4bl"

# Beam momentum [GeV/c]
P_GEV = 0.08857
BRHO = P_GEV / 0.299792458  # T·m
IRON_THICKNESS = 5.0  # mm 

# Quadrupoles we expect
K_NAMES = [
    "k_q1","k_q2","k_q3","k_q4",
    "k_q5","k_q6","k_q7","k_q8",
    "k_qf1","k_qd1"
]

# ------------------------------------------------------------
# 1) Parse MAD-X MATCH table
# ------------------------------------------------------------
kvals = {}

with open(MADX_OUT) as f:
    for line in f:
        m = re.match(
            r'\s*(k_q\w+)\s+([-+0-9.eE]+)\s+[-+0-9.eE]+\s+[-+0-9.eE]+\s+[-+0-9.eE]+',
            line
        )
        if m:
            name = m.group(1)
            val  = float(m.group(2))
            kvals[name] = val

# Sanity check
missing = [k for k in K_NAMES if k not in kvals]
if missing:
    print("ERROR: Missing variables in madx.out:")
    for k in missing:
        print("  ", k)
    sys.exit(1)

# ------------------------------------------------------------
# 2) Convert K1 → gradient (T/m)
# ------------------------------------------------------------
grads = {}
for k, v in kvals.items():
    grads[k] = v * BRHO

# ------------------------------------------------------------
# 3) Read G4BL template
# ------------------------------------------------------------
with open(G4BL_IN) as f:
    text = f.read()

# ------------------------------------------------------------
# 3.5) Shift lattice so MF0 is at z = 0
# ------------------------------------------------------------

MF0_Z = 2775.0  # mm (from MAD-X MF0 position)

def shift_z(match):
    expr = match.group(1)   # e.g. "2812.5" or "75+250"
    return f"z=({expr})-{MF0_Z}"

text = re.sub(
    r'z=([0-9.+\-*/ ]+)',
    shift_z,
    text
)


# ------------------------------------------------------------
# 4) Inject K1 values
# ------------------------------------------------------------
for k, v in kvals.items():
    text = re.sub(
        rf'param {k}\s*=.*',
        f'param {k}={v:.10g}',
        text
    )

# ------------------------------------------------------------
# 5) Inject gradients
# ------------------------------------------------------------
for k, g in grads.items():
    gname = "g_" + k   # g_k_q1, g_k_q2, ...
    text = re.sub(
        rf'param {gname}\s*=.*',
        f'param {gname}={g:.10g}',
        text
    )

# ------------------------------------------------------------
# 5.5) Auto-rename repeated QF1h / QD1h placements
# ------------------------------------------------------------

qf_count = 0
qd_count = 0

def rename_quads(match):
    global qf_count, qd_count
    name = match.group(1)

    if name == "QF1h":
        qf_count += 1
        return f"place QF1h rename=QF1h_{qf_count:02d} z="
    elif name == "QD1h":
        qd_count += 1
        return f"place QD1h rename=QD1h_{qd_count:02d} z="

# Replace "place QF1h z=" and "place QD1h z="
text = re.sub(
    r'place\s+(QF1h|QD1h)\s+z=',
    rename_quads,
    text
)
# ------------------------------------------------------------
# 5.55) Ensure all genericquads have ironRadius > apertureRadius
# ------------------------------------------------------------

lines = text.splitlines()
out = []
i = 0

while i < len(lines):
    line = lines[i]

    # Start of a genericquad block
    if line.strip().startswith("genericquad"):
        block = [line]
        i += 1

        # Collect indented lines
        while i < len(lines) and lines[i].startswith((" ", "\t")):
            block.append(lines[i])
            i += 1

        # Extract apertureRadius and check for ironRadius
        aperture = None
        has_iron = False

        for b in block:
            m = re.search(r'apertureRadius\s*=\s*([0-9.+\-eE]+)', b)
            if m:
                aperture = float(m.group(1))
            if "ironRadius" in b:
                has_iron = True

        # Insert ironRadius if missing and apertureRadius exists
        if aperture is not None and not has_iron:
            iron_radius = aperture + IRON_THICKNESS

            # Insert right after apertureRadius line
            new_block = []
            for b in block:
                new_block.append(b)
                if "apertureRadius" in b:
                    indent = re.match(r'(\s*)', b).group(1)
                    new_block.append(
                        f"{indent}ironRadius={iron_radius:.10g}"
                    )
            block = new_block

        out.extend(block)
        continue

    out.append(line)
    i += 1

text = "\n".join(out)

# ------------------------------------------------------------
# 5.6) Add "\" continuation to simple G4BL element blocks
# ------------------------------------------------------------

lines = text.splitlines()
out = []
i = 0

G4BL_KEYWORDS = (
    "genericquad", "solenoid", "box", "tubs", "trap", "coil",
    "fieldmap", "pipe", "absorber"
)

while i < len(lines):
    line = lines[i]

    # Element definition start: e.g. "genericquad Q5"
    if line.strip().startswith(G4BL_KEYWORDS):
        block = [line]
        i += 1

        # Collect indented parameter lines
        while i < len(lines) and lines[i].startswith((" ", "\t")):
            block.append(lines[i])
            i += 1

        # Add "\" to all but the last line
        for j, b in enumerate(block):
            if j < len(block) - 1:
                out.append(b + " \\")
            else:
                out.append(b)

        continue

    # Everything else untouched
    out.append(line)
    i += 1

text = "\n".join(out)

# ------------------------------------------------------------
# 6) Write output
# ------------------------------------------------------------
with open(G4BL_OUT, "w") as f:
    f.write(text)

print("OK: wrote", G4BL_OUT)
print("\nFinal values:")
for k in K_NAMES:
    print(f"{k:6s} = {kvals[k]: .6e}   G = {grads[k]: .6e} T/m")