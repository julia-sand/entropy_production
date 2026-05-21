"""
This file stores all the functions for the calculation
of the distribution and drift, and the interpolation functions
used for the evolution of girsanov theorem and histogram plots

The CSV is loaded once via np.genfromtxt and columns are accessed by name.
Smoothing uses scipy.signal.savgol_filter throughout """

import numpy as np

from scipy.signal import savgol_filter
import scipy.interpolate as sci

from ep.utils.datafetch import open_array   # replaces open_df; see note below
from ep.utils.parser import fetch_results_folder_from_cmd
from ep.utils.misc import load_params_from_file


def _col(arr, t0, col):
    """Return values of *col* from structured array *arr* where arr['t0']==t0."""
    return arr[col][arr["t0"] == t0]


class Perturbation:
    def __init__(self):
        folder = fetch_results_folder_from_cmd()
        params = load_params_from_file(folder)
        
        try: 
            self.g = params["g"]
        except KeyError:
            self.g = None

        self.epsilon = params["epsilon"]
        self.Tf2 = params["Tf2"]
        
        self.peakloc = params["peak"]  # peak location (alpha)
        self.w2_dist = params["w2"]
 
        self.alpha = 0 
           
        self.window_length = 71   # savgol window length (must be odd)
        self.polyorder = 3        # savgol polynomial order

        # Load data as a NumPy structured array instead of a DataFrame
        self.data = open_array()
        self.times_t0 = np.unique(self.data["t0"])
        self.q_axis = np.unique(self.data["x"])
        self.dx = np.abs(self.q_axis[1] - self.q_axis[0])

    # ------------------------------------------------------------------
    # Internal helpers that mirror the old df-filter pattern
    # ------------------------------------------------------------------

    def _get(self, t0, col):
        """Fetch a column at a given t0 from the structured array."""
        return _col(self.data, t0, col)

    # ------------------------------------------------------------------
    # Filters / math utilities
    # ------------------------------------------------------------------

    def savgol_filter_(self, y, deriv=0):
        """Apply Savitzky-Golay filter using instance window_length, polyorder, and dx."""
        return savgol_filter(y, window_length=self.window_length,
                             polyorder=self.polyorder, deriv=deriv, delta=self.dx)
 
    def zchop(self, a, tol):
        """Clip values close to zero to prevent errors in logarithms."""
        a[np.abs(a) < tol] = 0.0
        return a

    # ------------------------------------------------------------------
    # Frequency / coefficient functions
    # ------------------------------------------------------------------

    def omega_fun(self, g):
        return np.sqrt((1 + g) / g)

    def B_fun(self, T, g):
        omega = self.omega_fun(g)
        return (
            -((1 + g) / T)
            * np.tanh(omega * T / 2)
            * (omega * np.tanh(T) - 2 * np.tanh(omega * T / 2))
            / (omega * np.tanh(omega * T / 2) - 2 * np.tanh(T))
        )

    def A_fun(self, T, g):
        omega = self.omega_fun(g)
        return (1 + g) * (
            1
            - (
                (((1 + g) / g) - 4)
                * (np.tanh(omega * T / 2) * np.tanh(T))
            )
            / (omega * T * (omega * np.tanh(omega * T / 2) - 2 * np.tanh(T)))
        )

    def A_minus_B(self, T, g):
        omega = self.omega_fun(g)
        return (1 + g) * (1 - (2 / (omega * T)) * np.tanh(omega * T / 2))

    # ------------------------------------------------------------------
    # Trajectory-space basis functions
    # ------------------------------------------------------------------

    def b(self, t0, g, T):
        omega = np.sqrt((1 + g) / g)
        denom1 = (
            np.cosh(T) * np.sinh(omega * T)
            - (2 * np.sinh(T) * np.cosh(omega * T) + 2 * np.sinh(T)) / omega
        )
        num3 = np.sinh(omega * (T - t0)) * np.exp(-T)
        num4 = np.sinh(omega * t0) * np.exp(T) - np.sinh(omega * T) * np.exp(2 * t0 - T)
        return (1 / denom1) * (num3 + num4)

    def a_minus_b2(self, t0, g, T):
        omega = np.sqrt((1 + g) / g)
        return 1 - (np.cosh(omega * t0) + np.cosh(omega * (t0 - T))) / (
            1 + np.cosh(T * omega)
        )

    def a(self, t0, g, T):
        return self.a_minus_b2(t0, g, T) + self.b(t0, g, T)

    def b_dot(self, t0, g, T):
        omega = np.sqrt((1 + g) / g)
        denom1 = (
            np.cosh(T) * np.sinh(omega * T)
            - (2 * np.sinh(T) * np.cosh(omega * T) + 2 * np.sinh(T)) / omega
        )
        num3 = -omega * np.cosh(omega * (T - t0)) * np.exp(-T)
        num4 = (
            omega * np.cosh(omega * t0) * np.exp(T)
            - 2 * np.sinh(omega * T) * np.exp(2 * t0 - T)
        )
        return (1 / denom1) * (num3 + num4)

    def a_dot(self, t0, g, T):
        omega = np.sqrt((1 + g) / g)
        term1 = -(np.sinh(omega * t0) + np.sinh(omega * (t0 - T))) / (
            1 + np.cosh(T * omega)
        )
        return self.b_dot(t0, g, T) + omega * term1

    # ------------------------------------------------------------------
    # Data-access methods
    # ------------------------------------------------------------------

    def rho(self, t0):
        """Get rho (overdamped PDF ptx) at time t0."""
        return self._get(t0, "ptx")

    def get_rhomask(self, t0, tol):
        """Return index of x-coords with probability mass greater than tol."""
        return np.where(self.zchop(self.rho(t0).copy(), tol) > 0)

    def dsigma(self, t0):
        """Return derivative of sigma."""
        return -self._get(t0, "dsigma")

    def distribution(self, t0):
        """Return underdamped distribution."""
        return self._get(t0, "UDpdf")

    def optimal_drift(self, t0):
        """Return underdamped drift."""
        return self._get(t0, "UDdrift")

    # ------------------------------------------------------------------
    # Moment / integral quantities
    # ------------------------------------------------------------------
 
    def kappa(self, t0):
        """Compute kappa at t0 (used in cumulants and other functions)."""
        integral = self.rho(t0) * self.dsigma(t0)
        return -np.trapz(integral, self.q_axis)
 
    def mean_t0(self, t0):
        rho_temp = self.rho(t0)
        return np.trapz(self.q_axis * rho_temp, self.q_axis)
 
    def var_t0(self, t0):
        rho_temp = self.rho(t0)
        return np.trapz((self.q_axis ** 2) * rho_temp, self.q_axis)
 
    def od_bound(self, T):
        """Compute overdamped bound."""
        return (1 / (1 + self.epsilon)) * (self.w2_dist / (T * (self.epsilon ** 2)))
 
    # ------------------------------------------------------------------
    # Derivative / gradient utilities
    # ------------------------------------------------------------------
   
    def dfun(self, vals):
        """First derivative with Savitzky-Golay filter."""
        return self.savgol_filter_(vals, deriv=1)

    def dlogrho(self, t0, tol=1e-10):
        """First derivative of log-rho via savgol_filter_(deriv=1)."""
        logrho = self._get(t0, "logptx")
        idx = self.get_rhomask(t0, tol)
        dlogout = np.zeros_like(logrho)
        dlogout[idx] = self.savgol_filter_(logrho[idx], deriv=1)
        return dlogout
 
    def drho(self, t0, tol=1e-10):
        """First derivative of rho via savgol_filter_(deriv=1)."""
        idx = self.get_rhomask(t0, tol)
        drho_vals = np.zeros_like(self.q_axis)
        drho_vals[idx] = self.savgol_filter_(self.rho(t0)[idx], deriv=1)
        return drho_vals
 
    # ------------------------------------------------------------------
    # Combined derivative expressions  (use self.alpha throughout)
    # ------------------------------------------------------------------
 
    def rho_dsigma_alpha_rho(self, t0,  tol=1e-10):
        result = (
            self.rho(t0) * self.dsigma(t0)
            + self.alpha * self.drho(t0,  tol)
        )
        result[np.isnan(result)] = 0
        return result
 
    def dsigma_alpha_rho(self, t0, tol=1e-10):
        result = (
            self.dsigma(t0)
            + self.alpha * self.dlogrho(t0, tol)
        )
        result[np.isnan(result)] = 0
        return result
 
    def rho_ddsigma_alpha_rho(self, t0, tol=1e-10):
        """Second derivatives of log-rho and sigma via savgol_filter_(deriv=2)."""
        idx = self.get_rhomask(t0, tol)
        ddlogrho = self.savgol_filter_(self._get(t0, "logptx")[idx], deriv=2)
        ddsigtemp = self.savgol_filter_(self.dsigma(t0)[idx], deriv=1)
        temp_vals_out = np.zeros_like(self.q_axis)
        temp_vals_out[idx] = self.alpha * ddlogrho + ddsigtemp
        return temp_vals_out * self.rho(t0)
 
    def script_k(self, t0, tol=1e-10):
        """Compute varsigma_dot / 2."""
        temp_vals = self.q_axis * self.rho_dsigma_alpha_rho(t0, tol)
        return (
            -np.trapz(temp_vals, self.q_axis)
            - self.kappa(t0) * self.mean_t0(t0)
        )
 
class FullBridgePerturbation(Perturbation):

    def __init__(self):
        __super__().__init__()

    # ------------------------------------------------------------------
    # f11 and its derivative coefficients
    # ------------------------------------------------------------------
 
    def coeff1_df11(self, g, t0, T):
        return -self.a(t0, g, T) / self.A_fun(T, g)
 
    def coeff2_df11(self, g, t0, T):
        Ag = self.A_fun(T, g)
        Bg = self.B_fun(T, g)
        return (Bg * self.a(t0, g, T) - Ag * self.b(t0, g, T)) / (
            Ag * self.A_minus_B(T, g)
        )
 
    def f11(self, t0, g, T, tol=1e-10):
        Ag = self.A_fun(T, g)
        Bg = self.B_fun(T, g)
        coeff1 = -self.a(t0, g, T) / Ag
        coeff2 = self.rho(t0) * (
            Bg * self.a(t0, g, T) - Ag * self.b(t0, g, T)
        ) / (Ag * self.A_minus_B(T, g))
        return (
            coeff1 * self.rho_dsigma_alpha_rho(t0, tol)
            + coeff2 * self.kappa(t0)
        )
 
    def calculate_df11(self, t0, g, T, tol=1e-10):
        drho_vals = self.drho(t0, tol)
        coeff1 = self.coeff1_df11(g, t0, T)
        num1 = (
            drho_vals * self.dsigma_alpha_rho(t0, tol)
            + self.rho_ddsigma_alpha_rho(t0, tol)
        )
        num1[np.isnan(num1)] = 0
        num1[np.isinf(num1)] = 0
        coeff2 = self.coeff2_df11(g, t0, T)
        return coeff1 * num1 + coeff2 * (self.kappa(t0) * drho_vals)
 
    # ------------------------------------------------------------------
    # f02 (second-order correction)
    # ------------------------------------------------------------------
 
    def f02_new(self, t0, g, T, tol=1e-10):
        t2_term1 = (
            self.drho(t0, tol)
            * self.dsigma_alpha_rho(t0, tol)
            + self.rho_ddsigma_alpha_rho(t0, tol)
        )
        t2_termrho = self.drho(t0, tol) * self.kappa(t0)
 
        term1 = -g * (
            self.coeff1_df11(g, t0, T) * t2_term1
            + self.coeff2_df11(g, t0, T) * t2_termrho
        )
 
        int_limit = np.where(self.times_t0 == t0)[0][0] + 1
        c1 = np.array([self.coeff1_df11(g, t, T) * t2_term1 for t in self.times_t0])
        c2 = np.array([self.coeff2_df11(g, t, T) * t2_termrho for t in self.times_t0])
 
        coeff1_int1 = (t0 / T) * np.trapz(c1, self.times_t0, axis=0)
        coeff2_int1 = (t0 / T) * np.trapz(c2, self.times_t0, axis=0)
        coeff1_int2 = np.trapz(c1[:int_limit], self.times_t0[:int_limit], axis=0)
        coeff2_int2 = np.trapz(c2[:int_limit], self.times_t0[:int_limit], axis=0)
 
        return term1 + (1 + g) * (
            coeff1_int1 + coeff2_int1 - (coeff1_int2 + coeff2_int2)
        )
 
    # ------------------------------------------------------------------
    # Optimal drift and distribution
    # ------------------------------------------------------------------
 
    def calculate_optimal_drift(self, t0, g, T, tol=1e-10):
        Ag = self.A_fun(T, g)
        Bg = self.B_fun(T, g)
        coeff1 = (self.a_dot(t0, g, T) + self.a(t0, g, T)) / Ag
        term1 = (self.alpha * coeff1 - 1) * self.dlogrho(t0, tol)
        term2 = coeff1 * self.dsigma(t0)
        coeff3 = self.kappa(t0) / (Ag * self.A_minus_B(T, g))
        term3 = (
            Bg * self.a_dot(t0, g, T) - Ag * self.b_dot(t0, g, T)
        ) + (Bg * self.a(t0, g, T) - Ag * self.b(t0, g, T))
        opt_drift = term1 + term2 - coeff3 * term3
        opt_drift[np.isnan(opt_drift)] = 0
        opt_drift[np.isinf(opt_drift)] = 0
        return -opt_drift
 
    def calculate_distribution(self, t0, g, T, tol=1e-10):
        return self.rho(t0) + (self.epsilon ** 2) * self.f02_new(t0, g, T, tol)
 
    # ------------------------------------------------------------------
    # Interpolation helpers
    # ------------------------------------------------------------------
 
    def dsigma_interp(self, t0, q, tol=1e-5):
        mask = self.get_rhomask(t0, tol)
        dsig_temp = self.dsigma(t0) - self.dlogrho(t0, tol)
        w_temp = self.rho(t0)
        interp_dsig = sci.splrep(self.q_axis[mask], dsig_temp[mask], w=w_temp[mask], k=3)
        return sci.splev(q, interp_dsig)
 
    def underdamped_drift_interp_function(self, t0, g, tol=1e-5):
        mask = self.get_rhomask(t0, tol)
        w_temp = self.distribution(t0)
        dsig_temp = self.optimal_drift(t0)
        return sci.splrep(self.q_axis[mask], dsig_temp[mask], w=w_temp[mask], k=5)
 
    def underdamped_drift_interp(self, t0, q, g, tol=1e-5):
        mask = self.get_rhomask(t0, tol)
        dsig_temp = self.optimal_drift(t0)[mask]
        return -np.interp(
            q,
            self.q_axis[mask],
            dsig_temp,
            left=np.abs(dsig_temp[0]),
            right=-np.abs(dsig_temp[-1]),
        )
 
    def d_underdamped_drift_interp(self, t0, q, g, tol=1e-5):
        return sci.splev(
            q, self.underdamped_drift_interp_function(t0, g, tol), der=1, ext=5
        )
