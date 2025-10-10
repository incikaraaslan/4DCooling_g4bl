# Put this in your script (replace your old read_tfs)
from io import StringIO
import shlex
import pandas as pd
import matplotlib.pyplot as plt

def read_tfs(filepath):
    """
    Robust reader for MAD-X .tfs files.
    Returns a pandas DataFrame with column names from the '*' header
    and numeric columns coerced where appropriate.
    """
    with open(filepath, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    # find '*' header line
    star_idx = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("*"):
            star_idx = i
            break
    if star_idx is None:
        raise ValueError("No '*' header line found in TFS file. Can't determine column names.")

    colnames = lines[star_idx].lstrip()[1:].strip().split()

    # find where data starts (after the $ line if present)
    data_start = None
    for i in range(star_idx + 1, len(lines)):
        if lines[i].lstrip().startswith("$"):
            data_start = i + 1
            break
    if data_start is None:
        data_start = star_idx + 1

    # collect data lines (skip metadata and empty lines)
    data_lines = [ln for ln in lines[data_start:] if not ln.lstrip().startswith(("@", "*", "$")) and ln.strip() != ""]

    # Try fast parse with pandas
    try:
        df = pd.read_csv(StringIO("".join(data_lines)),
                         sep=r'\s+',
                         header=None,
                         names=colnames,
                         engine='python',
                         quoting=0)   # quoting=0 -> csv.QUOTE_MINIMAL as integer; works for numeric whitespace-separated input
        if df.shape[1] != len(colnames):
            raise ValueError("column count mismatch")
    except Exception:
        # Fallback: shlex split + pad/trim tokens to match header length
        rows = []
        for ln in data_lines:
            toks = shlex.split(ln.strip())
            if len(toks) < len(colnames):
                toks += [''] * (len(colnames) - len(toks))
            elif len(toks) > len(colnames):
                toks = toks[:len(colnames)]
            rows.append(toks)
        df = pd.DataFrame(rows, columns=colnames)

    # Coerce columns to numeric where appropriate
    for col in df.columns:
        coerced = pd.to_numeric(df[col], errors='coerce')
        # if at least half entries are numeric (or at least 1), convert the column
        if coerced.notna().sum() >= max(1, len(coerced) // 2):
            df[col] = coerced

    return df

# Example usage (use your real path here)
twiss = read_tfs("./RyanMichaud_Work/whole.tfs")   # <-- adjust path to your file

# Check columns and datatypes
print(twiss.columns.tolist())
print(twiss.dtypes.head(20))

# Plot (if columns exist)
if set(["S","BETX","BETY"]).issubset(twiss.columns):
    plt.figure(figsize=(10,4.5))
    plt.plot(twiss["S"], twiss["BETX"], label=r"$\beta_x$")
    plt.plot(twiss["S"], twiss["BETY"], label=r"$\beta_y$")
    plt.xlabel("s [m]")
    plt.ylabel(r"$\beta$ [m]")
    plt.title("Beta Function Evolution Along the Lattice")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("Missing one of S/BETX/BETY - can't plot beta.")

if set(["S","ALFX","ALFY"]).issubset(twiss.columns):
    plt.figure(figsize=(10,4.5))
    plt.plot(twiss["S"], twiss["ALFX"], label=r"$\alpha_x$")
    plt.plot(twiss["S"], twiss["ALFY"], label=r"$\alpha_y$")
    plt.xlabel("s [m]")
    plt.ylabel(r"$\alpha$")
    plt.title("Alpha Function Evolution Along the Lattice")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("Missing one of S/ALFX/ALFY - can't plot alpha.")
