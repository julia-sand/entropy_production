"""
This file stores all the functions for the calculation of the distribution and drift, and the interpolation functions used for the evolution of
girsanov theorem
"""


#from scipy.ndimage import median_filter,generic_filter
import scipy.ndimage as sc
import scipy.interpolate as sci

from main import *  
from datafetch import *


def omega_fun(g):
  return np.sqrt((1+g)/g)

omega = omega_fun(g)

#constants
def B_fun(T,g):
  return -((1+g)/T)*np.tanh(omega*T/2)*(omega*np.tanh(T) - 2*np.tanh(omega*T/2))/(omega*np.tanh(omega*T/2) - 2*np.tanh(T))

def A_fun(T,g):
  return (1+g)*(1 - (((((1+g)/g) - 4)*(np.tanh(omega_fun(g)*T/2)*np.tanh(T)))/(omega_fun(g)*T*(omega_fun(g)*np.tanh(omega_fun(g)*T/2)-2*np.tanh(T)))))

A = A_fun(T,g)
B = B_fun(T,g)

#clips values close to zero to prevent errors in logarithms
def zchop(a,tol):
  """
  input: vector a
  output: vector with values close to zero clipped.
  """
  a[np.abs(a) < tol] = 0.0
  return a

#get qaxis
#def q_axis(t0):
  #t2 = round(t0*(epsilon**2),dps)
#  return df[df.t0 == t0].x.to_numpy()

#function to get rho at time t0.
def rho(t0):
  #t2 = round(t0*(epsilon**2),dps)
  #rho_temp = df[df.t0==t0].ptx.to_numpy()
  return df[df.t0==t0].ptx.to_numpy()#zchop(rho_temp,tol)


#get all xcoords where there is probability mass
def get_rhomask(t0,tol=tol):
  
  return np.where(zchop(rho(t0),tol)>0)

def od_bound(T,g):
  return (1/(1+g))*(w2_dist/(T*(epsilon**2)))


#derivative functions
def dsigma(t0 ):

  #t2 = round(t0*(epsilon**2),dps)
  sigtemp = df[df.t0==t0].dsigma.to_numpy()

  return -sigtemp
  #-generic_filter(sigtemp,sc.median,100,mode="nearest")


#used to compute cumulants and other functions
def kappa(t0): #mu dot 1
  integral = rho(t0) * dsigma(t0)
  return -np.trapz(integral,q_axis)



#function to get underdamped distribution
def distribution(t0 ):
  #t2 = round(t0*(epsilon**2),dps)
  dist = df[df.t0==t0].UDpdf.to_numpy()
  return dist

#function to get underdamped drift
def optimal_drift(t0 ):
  #t2 = round(t0*(epsilon**2),dps)
  drift_temp = df[df.t0 ==t0].UDdrift.to_numpy()
  return drift_temp

def A_minus_B(T,g):
  omega = omega_fun(g)

  return (1+g)*(1-((2/(omega*T))*(np.tanh(omega*T/2))))

def b(t0,g):
  omega = np.sqrt((1+g)/g)

  denom1 =np.cosh(T)*np.sinh(omega*T) -(2*np.sinh(T)*np.cosh(omega*T)+2*np.sinh(T))/omega

  num3 = np.sinh(omega*(T-t0))*np.exp(-T)
  num4 = np.sinh(omega*t0)*np.exp(T) - np.sinh(omega*T)*np.exp(2*t0 - T)
  return (1/denom1)*(num3+num4)

def a_minus_b2(t0,g):
  omega = np.sqrt((1+g)/g)

  return 1 -(np.cosh(omega*t0)+np.cosh(omega*(t0-T)))/(1+np.cosh(T*omega))

def a(t0,g):

  return a_minus_b2(t0,g) + b(t0,g)

def b_dot(t0,g):
  omega = np.sqrt((1+g)/g)

  denom1 =np.cosh(T)*np.sinh(omega*T) -(2*np.sinh(T)*np.cosh(omega*T)+2*np.sinh(T))/omega

  num3 = -omega*np.cosh(omega*(T-t0))*np.exp(-T)
  num4 = omega*np.cosh(omega*t0)*np.exp(T) - 2*np.sinh(omega*T)*np.exp(2*t0 - T)
  return (1/denom1)*(num3+num4)

def a_dot(t0,g):
  omega = np.sqrt((1+g)/g)

  term1 = -(np.sinh(omega*t0)+np.sinh(omega*(t0-T)))/(1+np.cosh(T*omega))

  return b_dot(t0,g) + omega*term1

#mean and variances
def mean_t0(t0):

  #t2 = round(t0*(epsilon**2),dps)
  #get q-axis
  q = df[df.t0 == t0].x.to_numpy()
  #get rho
  rho_temp = df[df.t0 == t0].ptx.to_numpy()

  return np.trapz(q*rho_temp,q)

def var_t0(t0):

  #t2 = round(t0*(epsilon**2),dps)
  #get q-axis
  #q = df[df.t0 == t0].x.to_numpy()
  #get rho
  rho_temp = df[df.t0 == t0].ptx.to_numpy()
  
  return np.trapz((q_axis**2)*rho_temp,q_axis)


def dfun(vals,qs):
  #finds numerical gradient using central differences and applies a small median filter to reduce outliers

  #dfun = np.gradient(vals,qs,edge_order=2)

  return np.gradient(vals,qs,edge_order=2)#generic_filter(dfun,sc.median,filter_delta,mode = "nearest")

def dlogrho(t0 ):

  #t2 = round(t0*(epsilon**2),dps)
  logrho = df[df.t0 ==t0].logptx.to_numpy()

  #get rid of nans
  #interpolate only on non-zero vals of rho
  idx = get_rhomask(t0)
  logrho_temp = logrho[idx]
  q_axis_temp = q_axis[idx]

  #differentiate and filter without edges
  dlogrho = np.gradient(logrho_temp,q_axis_temp,edge_order=2)
  #filter_dlogrho = dlogrho#generic_filter(dlogrho,sc.median,filter_delta,mode="nearest")

  #put back into right place
  dlogout = np.zeros_like(logrho)
  dlogout[idx] = dlogrho #filter_dlogrho

  return dlogout

def drho(t0):

  #interpolate only on non-zero vals of rho
  idx = get_rhomask(t0)
  rho_vals_temp = rho(t0)[idx]
  #q_axis_temp = q_axis[idx]

  #drho = np.gradient(rho_vals_temp,q_axis_temp,edge_order=2)

  #set values outside of range to zero to prevent extrapolation error
  drho_vals = np.zeros_like(q_axis)
  drho_vals[idx] = np.gradient(rho_vals_temp,q_axis[idx],edge_order=2)#drho #generic_filter(drho,sc.median,filter_delta,mode="constant")

  return drho_vals

def rho_dsigma_alpha_rho(t0):

  rho_dsigma_alpha_rho = rho(t0)*dsigma(t0) + alpha*drho(t0)

  #fill nan with zero
  rho_dsigma_alpha_rho[np.isnan(rho_dsigma_alpha_rho)] = 0

  return rho_dsigma_alpha_rho

def dsigma_alpha_rho(t0):

  dsigma_alpha_rho = dsigma(t0) + alpha*dlogrho(t0)

  #fill nan with zero
  dsigma_alpha_rho[np.isnan(dsigma_alpha_rho)] = 0

  return dsigma_alpha_rho

def rho_ddsigma_alpha_rho(t0):

  #get drho
  drhotemp = dlogrho(t0)

  #remove zeros first
  idx = get_rhomask(t0)
  #ddrhotemp = 
  #q_temp = 
  ddlogrho = np.gradient(drhotemp[idx],q_axis[idx],edge_order=2)

  ddlogrho = ddlogrho #generic_filter(ddlogrho,sc.mean,filter_delta,mode="nearest")

  #get ddsigma
  dsigtemp = dsigma(t0)
  #dsig_vals = dsigtemp[idx]
  ddsigtemp = np.gradient(dsigtemp[idx],q_axis[idx],edge_order=2)

  temp_out = alpha*ddlogrho + ddsigtemp#generic_filter(ddsigtemp,sc.mean,filter_delta,mode="nearest")

  temp_vals_out = np.zeros_like(q_axis)
  temp_vals_out[idx] = temp_out

  output_vals = temp_vals_out*rho(t0)
  return output_vals#generic_filter(output_vals,sc.mean,size=filter_delta,mode="constant")

def script_k(t0):#varsigma dot/2
  temp_vals = q_axis*rho_dsigma_alpha_rho(t0)
  return -np.trapz(temp_vals,q_axis) - kappa(t0)*mean_t0(t0)

##
def f11(t0,g):
  Ag = A_fun(T,g)
  Bg = A_fun(T,g)

  rho_vals = rho(t0)

  coeff1 = -a(t0,g)/Ag
  num1 = rho_dsigma_alpha_rho(t0)

  coeff2 = rho_vals *(Bg*a(t0,g) - Ag*b(t0,g))/(Ag*A_minus_B(T,g))

  return coeff1*num1 + coeff2*kappa(t0)

##
def calculate_df11(t0,g):
  Ag = A_fun(T,g)
  Bg = B_fun(T,g)

  drho_vals = drho(t0)

  coeff1 = -a(t0,g)/Ag
  num1 = (drho_vals*dsigma_alpha_rho(t0)) + rho_ddsigma_alpha_rho(t0)

  num1[np.isnan(num1)] = 0
  num1[np.isinf(num1)] = 0

  coeff2 = (Bg*a(t0,g) - Ag*b(t0,g))/(Ag*A_minus_B(T,g))

  df11_out = coeff1*num1 + coeff2*(kappa(t0)*drho_vals)# - (rho(t0)**2)*dsigma(t0))

  return df11_out


def coeff1_df11(g,t0,T):
  Ag = A_fun(T,g)
  coeff1 = -a(t0,g)/Ag
  return coeff1


def coeff2_df11(g,t0,T):
  Ag = A_fun(T,g)
  Bg = B_fun(T,g)
  coeff2 = (Bg*a(t0,g) - Ag*b(t0,g))/(Ag*A_minus_B(T,g))
  return coeff2

def f02_new(t0,g,T):

  #fetch the constants in t2
  t2_term1 = drho(t0)*dsigma_alpha_rho(t0) + rho_ddsigma_alpha_rho(t0)
  t2_termrho = drho(t0)*kappa(t0)

  term1 = -g*(coeff1_df11(g,t0,T)*t2_term1 + coeff2_df11(g,t0,T)*t2_termrho)

  int_limit = np.where(times_t0==t0)[0][0] + 1
  coeff1_df11_temp = [coeff1_df11(g,t,T)*t2_term1 for t in times_t0]
  coeff2_df11_temp = [coeff2_df11(g,t,T)*t2_termrho for t in times_t0]

  coeff1_int1 = (t0/T)*np.trapz(coeff1_df11_temp,times_t0,axis=0)
  coeff2_int1 = (t0/T)*np.trapz(coeff2_df11_temp,times_t0,axis=0)

  coeff1_int2 = np.trapz(coeff1_df11_temp[0:int_limit],times_t0[0:int_limit],axis=0)
  coeff2_int2 = np.trapz(coeff2_df11_temp[0:int_limit],times_t0[0:int_limit],axis=0)

  return term1 + (1+g)*(coeff1_int1 + coeff2_int1 - (coeff1_int2 + coeff2_int2))


def calculate_optimal_drift(t0,g): #-DU
  Ag = A_fun(T,g)
  Bg = B_fun(T,g)

  coeff1 = (a_dot(t0,g) + a(t0,g))/Ag

  term1 = (alpha*coeff1 - 1)*dlogrho(t0)
  term2 = coeff1*dsigma(t0)

  coeff3 = kappa(t0)/(Ag*A_minus_B(T,g))
  #kappa(t0)/(A*(A-B))
  term3 = (Bg*a_dot(t0,g) - Ag*b_dot(t0,g)) + (Bg*a(t0,g) - Ag*b(t0,g))
  opt_drift = term1 + term2 - coeff3*term3

  opt_drift[np.isnan(opt_drift)] = 0
  opt_drift[np.isinf(opt_drift)] = 0

  return -opt_drift


def calculate_distribution(t0,g):
  rvals = rho(t0)+ (epsilon**2)*f02_new(t0,g,T)

  #find normalising factor
  #norm_factor = np.trapz(np.abs(rvals),q_axis(t0))

  return rvals #/ norm_factor



#function to get underdamped distribution
def distribution(t0 ):
  #t2 = round(t0*(epsilon**2),dps)
  dist = df[df.t0==t0].UDpdf.to_numpy()
  return dist

#function to get underdamped drift
def optimal_drift(t0 ):
  #t2 = round(t0*(epsilon**2),dps)
  drift_temp = df[df.t0 ==t0].UDdrift.to_numpy()
  return drift_temp





#function for interpolated dsigma
def dsigma_interp(t0,q):
  t2 = round(epsilon**2*t0,dps)
  #q_temp = q_axis
  dsig_temp = dsigma(t0)
  w_temp = rho(t0)

  interp_dsig = sci.splrep(q_axis, dsig_temp,w = w_temp,k=3)
  return sci.splev(q,interp_dsig)

#function for interpolated DU
def underdamped_drift_interp_function(t0,g):

  #q_temp = q_axis

  mask = get_rhomask(t0)
  w_temp = distribution(t0)
  dsig_temp_underdamped = optimal_drift(t0)
  #dsigout = generic_filter(dsig_temp_underdamped,sc.mean,size=100)
  interp_dsig_underdamped = sci.splrep(q_axis[mask],dsig_temp_underdamped[mask],w=w_temp[mask],k=5)
  #sci.splrep(q_axis[(w_temp!=0).argmax():-(np.flip(w_temp)!=0).argmax()],
  #                                     dsig_temp_underdamped[(w_temp!=0).argmax():-(np.flip(w_temp)!=0).argmax()],
  #                                     w=w_temp[(w_temp!=0).argmax():-(np.flip(w_temp)!=0).argmax()], k=5)

  return interp_dsig_underdamped

def underdamped_drift_interp(t0,q,g):

  return -sci.splev(q,underdamped_drift_interp_function(t0,g),ext=3)


#function for derivative of interpolated DU
def d_underdamped_drift_interp(t0,q,g):

  return -sci.splev(q,underdamped_drift_interp_function(t0,g),der=1,ext=5)


