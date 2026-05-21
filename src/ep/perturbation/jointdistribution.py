"""
Compute the joint distribution using the perturbative drift and Girsanov's theorem.
See: On the numerical integration of the Fokker-Planck equation driven by a mechanical
force and the Bismut-Elworthy-Li formula
Julia Sanders, Paolo Muratore-Ginanneschi, arxiv 2411.08518

Saves results to a space-delimited CSV.
"""

import numpy as np

from ep.perturbation.functions import Perturbation
from ep.compute.sdeintegrate import euler_maruyama
from ep.utils.parser import fetch_results_folder_from_cmd
from ep.utils.boundary import Boundary

def ud_pinitial(mom, q, pert):
    """Initial underdamped joint distribution pi(p, q) — update to match your model."""
    
    boundary = Boundary()
    #update the params
    pert.peakloc = boundary.peak_center
    pert.denom = boundary.denom

    return boundary.ud_pinitial(mom, q)

def compute_girsanov_joint(pert, g, plot_times, mc_samples, p_samples, q_samples, folder=None):
    """
    For each time in plot_times, run backward EM trajectories with Girsanov
    accumulation and compute the reweighted joint distribution. Saves to CSV.
    """
    if folder is None:
      folder=fetch_results_folder_from_cmd()

    p_init = np.linspace(-10, 10, p_samples)
    q_init = np.linspace(-3, 3, q_samples)
    P, Q = np.meshgrid(p_init, q_init)

    filename_out = folder+f"/ep_girsanovjoint.csv"
    header = "t P Q ptx"

    # Write header
    with open(filename_out, "w") as f:
        f.write(header + "\n")

    for t in plot_times:
        print("t", t)

        start_index = np.where(pert.times_t0 == t)[0][0]
        print("start_index", start_index)

        # Initialise trajectories — must copy since broadcast_to returns read-only view
        q_init_traj   = np.broadcast_to(Q, (mc_samples, p_samples, q_samples)).copy()
        mom_init_traj = np.broadcast_to(P, (mc_samples, p_samples, q_samples)).copy()

        # Run EM integration with Girsanov accumulation
        q_evo, mom_evo, girsanov = euler_maruyama(
            pert, g, start_index, q_init_traj, mom_init_traj
        )

        # Compute Girsanov-reweighted joint distribution
        joint_out = np.nanmean(
            ud_pinitial(mom_evo, q_evo, pert) * np.exp(-girsanov), axis=0
        )

        data = np.column_stack((
            t * np.ones(p_samples * q_samples),
            P.flatten(),
            Q.flatten(),
            joint_out.flatten(),
        ))
        np.nan_to_num(data, copy=False, nan=0, posinf=0, neginf=0)

        with open(filename_out, "a") as f:
            np.savetxt(f, data)


if __name__ == "__main__":
    pert = Perturbation()

    g          = pert.g
    mc_samples = 100
    p_samples  = 21
    q_samples  = 21
    folder     = fetch_results_folder_from_cmd()
    plot_times = np.array([2, 1.5, 1.0, 0.5, 0.25, 0])

    compute_girsanov_joint(pert, g, plot_times, mc_samples, p_samples, q_samples, folder)