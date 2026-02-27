import numpy as np
import pandas as pd
import math

def read_for009_trackfile(filepath):
    """
    
    Reads a for009-like track file produced by G4BL (looks for header '#x '). 
    It should produce a pandas (pd) table.
    
    Usage should be simply read_for009_trackfile("particles_after_wedge.txt")
    
    """
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

def p_to_SI(p_MeV_c):
    """
    The momentum in g4bl is given in this weird pseudo-natural units unit MeV/c. But since B is in
    Tesla and q is in Coulombs, we'd like to make sure the momenta are also SI.
    
    MeV = 1.6 * 10^(-13) C, c = 2.9 *10^8 m/s
    
    """
    p_SI = p_MeV_c * (1.6*10**(-13)/(3*10**8)) # kg m/s
    
    return p_SI    