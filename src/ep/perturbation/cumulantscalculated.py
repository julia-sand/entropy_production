"""Compute cumulants predicted by perturbation theory"""
import numpy as np

from ep.perturbation.functions import HalfBridgePerturbation,FullBridgePerturbation
from ep.utils.parser import fetch_results_folder_from_cmd

class FullBridgeCumulants:

    def __init__(self):
        self.p = FullBridgePerturbation()
        self.T = self.p.Tf2/(self.p.epsilon**2)

    def mom_mean(self, t0, g):
        A_minus_B = self.p.A_minus_B(self.T, g)
        return self.p.epsilon * self.p.a_minus_b2(t0, g, self.T) * self.p.kappa(t0) / A_minus_B

    def momentum_variance(self, t0, g):
        Ag = self.p.A_fun(self.T, g)

        term1 = 1 - ((self.p.kappa(t0) * self.p.epsilon / Ag) * self.p.a(t0, g, self.T)) ** 2

        int1 = np.trapz(self.p.rho_ddsigma_alpha_rho(t0), self.p.q_axis) / Ag
        int_limit = np.where(self.p.times_t0 == t0)[0][0] + 1
        aexp_temp = [self.p.a(t, g, self.T) * int1 * np.exp(-2 * (t0 - t)) for t in self.p.times_t0]
        term2 = 2 * (self.p.epsilon ** 2) * np.trapz(aexp_temp[:int_limit], self.p.times_t0[:int_limit], axis=0)

        sq_temp = (self.p.dsigma(t0) ** 2) * self.p.rho(t0)
        term3 = ((self.p.epsilon * self.p.a(t0, g, self.T) / Ag) ** 2) * np.trapz(sq_temp, self.p.q_axis)

        return term1 + term2 + term3

    def cross_correlation(self, t0, g):
        Ag = self.p.A_fun(self.T, g)
        return self.p.epsilon * self.p.a(t0, g, self.T) * self.p.script_k(t0) / Ag

    def position_variance_g(self, t0, g):
        Ag = self.p.A_fun(self.T, g)

        term1 = self.p.var_t0(t0) - (self.p.mean_t0(t0) ** 2)
        term2 = 2 * (self.p.epsilon ** 2) * g * self.p.a(t0, g, self.T) * self.p.script_k(t0) / Ag

        coeff3 = -2 * self.p.script_k(t0) * (self.p.epsilon ** 2) * (1 + g) / Ag
        int_limit = np.where(self.p.times_t0 == t0)[0][0] + 1
        a_temp = [self.p.a(t, g, self.T) for t in self.p.times_t0]
        int1 = (t0 / self.T) * np.trapz(a_temp, self.p.times_t0, axis=0)
        int2 = np.trapz(a_temp[:int_limit], self.p.times_t0[:int_limit], axis=0)

        return term1 + term2 + coeff3 * (int1 - int2)

    def linear_position_cumulant(self, t0, g):
        A_minus_B = self.p.A_minus_B(self.T, g)

        term1 = (
            self.p.mean_t0(t0)
            + self.p.kappa(t0) * (self.p.epsilon ** 2) * g * self.p.a_minus_b2(t0, g, self.T) / A_minus_B
        )
        coeff2 = (self.p.epsilon ** 2) * (1 + g) * self.p.kappa(t0) / A_minus_B
        int_limit = np.where(self.p.times_t0 == t0)[0][0] + 1
        ab_temp = [self.p.a_minus_b2(t, g, self.T) for t in self.p.times_t0]
        int1 = (t0 / self.T) * np.trapz(ab_temp, self.p.times_t0, axis=0)
        int2 = np.trapz(ab_temp[:int_limit], self.p.times_t0[:int_limit], axis=0)

        return term1 - coeff2 * (int1 - int2)

class HalfBridgeCumulants:
    def __init__():
        self.p = HalfBridgePerturbation()
        self.T = self.p.Tf2/(self.p.epsilon**2)


def compute_and_save_predicted_cumulants(gs, bridgetype="Full", folder= None):
    """
    Computes cumulants based on perturbative predictions and saves to csv for plotting.
    """
    if folder is None: 
        folder = fetch_results_folder_from_cmd()
    
    if bridgetype == "Full":
        cumulants = FullBridgeCumulants()
    else:
        raise NotImplementedError

    cols = ["g", "t0", "pos_var", "mom_var", "mom_mean", "pos_mean", "xcorr"]
    rows = []

    for gi in gs:
        for t0 in cumulants.p.times_t0:
            rows.append([
                gi,
                t0,
                cumulants.position_variance_g(t0, gi),
                cumulants.momentum_variance(t0, gi),
                cumulants.mom_mean(t0, gi),
                cumulants.linear_position_cumulant(t0, gi),
                cumulants.cross_correlation(t0, gi),
            ])

    data = np.array(rows)
    header = ",".join(cols)
    np.savetxt(folder+f"/cumulants.csv", data, delimiter=",", header=header, comments="")


if __name__ == "__main__":
    folder = fetch_results_folder_from_cmd()
    gs = np.logspace(-1, -4, 4)
    compute_and_save_predicted_cumulants(gs, fileid="run1",folder=folder)