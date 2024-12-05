from main import *
import functions

gs = np.logspace(-1,-4,4)

##
def mom_mean(t0,g):

  #Ag = A_fun(T,g)
  #Bg = B_fun(T,g)
  #return epsilon*(a_minus_b2(t0))*kappa(t0)/(Ag-Bg)

  A_minus_B = (1+g)*(1-((2/(functions.omega_fun(g)*T))*(np.tanh(functions.omega_fun(g)*T/2))))
  return epsilon*(functions.a_minus_b2(t0,g))*functions.kappa(t0)/A_minus_B

#momentum variance - ONLY FOR EP.
def momentum_variance(t0,g):
  Ag = functions.A_fun(T,g)

  term1 = 1 - (((functions.kappa(t0)*epsilon/Ag)*functions.a(t0,g))**2)
  int1 = np.trapz(functions.rho_ddsigma_alpha_rho(t0),q_axis)/Ag
  int_limit = np.where(times_t0==t0)[0][0] + 1

  aexp_temp = [functions.a(t,g)*int1*np.exp(-2*(t0-t)) for t in times_t0]
  int2 = np.trapz(aexp_temp[0:int_limit],times_t0[0:int_limit],axis =0)

  term2 = 2*(epsilon**2)*int2

  sq_temp = ((functions.dsigma(t0))**2)*functions.rho(t0)

  term3 = (((epsilon)*functions.a(t0,g)/Ag)**2)*np.trapz(sq_temp,q_axis)

  return term1 + term2 + term3


#cross correlation
def cross_correlation(t0,g):
  Ag = functions.A_fun(T,g)
  return epsilon*functions.a(t0,g)*functions.script_k(t0)/Ag

#position process variance
def position_variance_g(t0,g):
  Ag = functions.A_fun(T,g)
  term1 = functions.var_t0(t0) - (functions.mean_t0(t0)**2)
  term2 = 2*(epsilon**2)*g*functions.a(t0,g)*functions.script_k(t0)/Ag

  coeff3 = -2*functions.script_k(t0)*(epsilon**2)*(1+g)/Ag

  int_limit = np.where(times_t0==t0)[0][0] + 1
  a_temp = [functions.a(t,g) for t in times_t0]

  int1 = (t0/T)*np.trapz(a_temp,times_t0,axis =0)
  int2 = np.trapz(a_temp[0:int_limit],times_t0[0:int_limit],axis =0)
  return term1 + term2 + coeff3*(int1 - int2)

#linear position cumulant aka position mean
def linear_position_cumulant(t0,g):
  Ag = functions.A_fun(T,g)
  A_minus_B = (1+g)*(1-((2/(functions.omega_fun(g)*T))*(np.tanh(functions.omega_fun(g)*T/2))))

  term1 = functions.mean_t0(t0) + functions.kappa(t0)*(epsilon**2)*g*(functions.a_minus_b2(t0,g))/A_minus_B
  coeff2 = (epsilon**2)*(1+g)*functions.kappa(t0)/A_minus_B
  int_limit = np.where(times_t0==t0)[0][0] + 1
  ab_temp = [functions.a_minus_b2(t,g) for t in times_t0]
  int1 = (t0/T)*np.trapz(ab_temp,times_t0,axis =0)
  int2 = np.trapz(ab_temp[0:int_limit],times_t0[0:int_limit],axis =0)

  return term1 - coeff2*(int1 - int2)


##compute cumulants at different g and save
df_ep_cumulants = pd.DataFrame(data =np.ones((len(gs)*len(times_t0),7)),columns = ["g","t0","pos_var","mom_var","mom_mean","pos_mean","xcorr"])

for g_ind in enumerate(gs):
  df_ep_cumulants.loc[(g_ind[0])*len(times_t0):(g_ind[0]+1)*len(times_t0),"g"] = g_ind[1]

for gi in gs:
  pos_var_temp = [position_variance_g(t0,gi) for t0 in times_t0]
  x_corr_temp = [cross_correlation(t0,gi) for t0 in times_t0]
  mom_var_temp = [momentum_variance(t0,gi) for t0 in times_t0]
  mom_mean_temp = [mom_mean(t0,gi) for t0 in times_t0]
  pos_mean_temp = [linear_position_cumulant(t0,gi) for t0 in times_t0]

  df_ep_cumulants.loc[df_ep_cumulants[df_ep_cumulants.g==gi].index,"t0"] = times_t0
  df_ep_cumulants.loc[df_ep_cumulants[df_ep_cumulants.g==gi].index,"pos_var"] = pos_var_temp
  df_ep_cumulants.loc[df_ep_cumulants[df_ep_cumulants.g==gi].index,"mom_var"] = mom_var_temp
  df_ep_cumulants.loc[df_ep_cumulants[df_ep_cumulants.g==gi].index,"mom_mean"] = mom_mean_temp
  df_ep_cumulants.loc[df_ep_cumulants[df_ep_cumulants.g==gi].index,"pos_mean"] = pos_mean_temp
  df_ep_cumulants.loc[df_ep_cumulants[df_ep_cumulants.g==gi].index,"xcorr"] = x_corr_temp

df_ep_cumulants.to_csv("cumulantscalculated.csv", index = False)
