import numpy as np


def euler_maruyama(pert, g, start_index, q_init, mom_init):
    """
    Euler-Maruyama integration of the underdamped SDE backward from
    times_t0[start_index] to times_t0[0], with Girsanov factor accumulation.
 
    Parameters
    ----------
    pert        : Perturbation instance
    g           : coupling constant 
    start_index : index in pert.times_t0 to start backward integration from
    q_init      : initial position array, shape (mc_samples, p_samples, q_samples)
    mom_init    : initial momentum array, same shape as q_init
 
    Returns
    -------
    q        : position array at t0=0
    mom      : momentum array at t0=0
    girsanov : accumulated Girsanov exponent, same shape as q
    """
    h0_step = pert.times_t0[1] - pert.times_t0[0]
 
    q        = q_init.copy()
    mom      = mom_init.copy()
    girsanov = np.zeros_like(q)
 
    for i in range(start_index):
 
        curr_time = pert.times_t0[start_index - i]
        print("curr_time", curr_time)
 
        innovation = np.random.standard_normal(q.shape)
        drift = pert.underdamped_drift_interp(curr_time, q, g)
 
        q   = q   - pert.epsilon * h0_step * (mom - g * pert.epsilon * drift) \
                  - pert.epsilon * np.sqrt(2 * g * h0_step) * np.random.standard_normal(q.shape)
 
        mom = mom + pert.epsilon * drift * h0_step \
                  - np.sqrt(2 * h0_step) * innovation
 
        # Girsanov factor accumulation
        girsanov = girsanov \
                   + np.sqrt(h0_step / 2) * mom * innovation \
                   + (h0_step / 4) * np.square(mom)
 
    return q, mom, girsanov
