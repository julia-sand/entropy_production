"""Compute underdamped drift and distribution predicted by multiscale perturbation theory"""

import numpy as np

from ep.perturbation.functions import Perturbation
from ep.utils.datafetch import open_array
from ep.utils.parser import fetch_results_folder_from_cmd
from ep.utils.misc import append_param

def compute_corrections(data, g, p):
    """
    Compute underdamped PDF and drift corrections and append as new columns.
    Returns a new structured array with UDpdf and UDdrift fields appended.
    """

    n = len(data)
    UDpdf  = np.ones(n)
    UDdrift = np.ones(n)

    for t0 in p.times_t0:
        mask = data["t0"] == t0
        UDpdf[mask]   = p.calculate_distribution(t0, g, T=p.Tf2/(p.epsilon**2))
        UDdrift[mask] = p.calculate_optimal_drift(t0, g, T=p.Tf2/(p.epsilon**2))

    # Replace inf/nan with 0
    UDpdf[~np.isfinite(UDpdf)]   = 0
    UDdrift[~np.isfinite(UDdrift)] = 0

    # Append new fields to structured array
    new_dtype = np.dtype(data.dtype.descr + [("UDpdf", float), ("UDdrift", float)])
    out = np.empty(n, dtype=new_dtype)
    for field in data.dtype.names:
        out[field] = data[field]
    out["UDpdf"]   = UDpdf
    out["UDdrift"] = UDdrift

    return out

def save_g(g, folder=None):
    """Save structured array to space-delimited CSV with header."""
    if folder is None: 
      folder= fetch_results_folder_from_cmd()
    
    append_param(folder + "/results", "g", g)

def save_array(data, folder=None):
    """Save structured array to space-delimited CSV with header."""
    if folder is None: 
      folder= fetch_results_folder_from_cmd()
    
    header = " ".join(data.dtype.names)
    np.savetxt(folder + "/results_corrected.csv", np.column_stack([data[f] for f in data.dtype.names]),
               header=header, comments="")

def compute_and_save_corrections(data, g, p):
    out = compute_corrections(p.data, g, p)
    save_array(out)
    save_g(g)

if __name__ == "__main__":
    p = Perturbation()
    data = open_array()
    g = 0.01
    compute_and_save_corrections(data, g, p)

