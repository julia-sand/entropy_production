import numpy as np

from ep.utils.misc import *
from ep.cell.sinkhorn import solve_cell
from ep.perturbation.functions import Perturbation
from ep.utils.datafetch import open_array
from ep.perturbation.distributionanddrift import compute_and_save_corrections
from ep.perturbation.cumulantscalculated import compute_and_save_predicted_cumulants 

folder = make_results_dir()

solve_cell(folder+"/results")

bridgetype="Full"

if bridgetype =="Full":

    p = FullBridgePerturbation()
else: 
    p = HalfBridgePerturbation()


g = 0.01

compute_and_save_corrections(p.data, g, p)

gs = np.logspace(-1, -4, 4)
compute_and_save_predicted_cumulants(gs, bridge_type=bridgetype, folder=folder)
