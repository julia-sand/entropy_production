
import setup.functions as functions
from setup.main import *
from setup.datafetch import *

columns = ["g","Tf","Firstterm","EPcost","ODBound"]

####
#read csv if it exists, otherwise create a new dataframe. 
try:
    df_costs = pd.read_csv("ep_costs.csv")
except:
    df_costs = pd.DataFrame(columns = columns)



##These are the temporary functions until we need to use more T
def entropy_production_cost(T,g):
  #
  kappa_sq = functions.kappa(0)**2

  integral_t0 = np.ones(len(times_t0))
  for t0 in enumerate(times_t0):
      temp = (functions.dsigma(t0[1])**2)*functions.rho(t0[1])
      integral_t0[t0[0]] = np.trapz(temp,q_axis)

  ep_integral1 = np.trapz(integral_t0,t2_vec)

  integral_t0 *= 0
  for t0 in enumerate(times_t0):
      temp = (functions.dsigma(t0[1])**2)*functions.rho(t0[1])
      integral_t0[t0[0]] = np.trapz(temp,q_axis)- functions.kappa(t0[1])**2

  ep_integral2 = np.trapz(integral_t0,t2_vec)
  Ag = functions.A_fun(T,g)

  omg_sq = (1/g)+1
  coeff1 = 1/(1+g)

  coeff2 = (1-Ag+g)/((1*Ag)+(g*Ag))
  coeff3 = (T*epsilon**2)*(1 - functions.A_minus_B(T,g) + g )/((4*functions.A_minus_B(T,g)+4*g*functions.A_minus_B(T,g)))
  ep = (coeff1)*ep_integral1 + coeff2*ep_integral2 + coeff3*(kappa_sq)

  return ep

def first_term(T,g):

  kappa_sq = functions.kappa(0)**2

  integral_t0 = np.ones(len(times_t0))
  for t0 in enumerate(times_t0):
      temp = (functions.dsigma(t0[1])**2)*functions.rho(t0[1])
      integral_t0[t0[0]] = np.trapz(temp,q_axis)

  ep_integral1 = np.trapz(integral_t0,t2_vec)


  omg_sq = (1/g)+1
  coeff1 = 1/(1+g)

  ep = (coeff1)*ep_integral1
  return ep

##od bound

def od_bound(T,g):
  return (1/(1+g))*(w2_dist/(T*(epsilon**2)))


gs_temp = np.logspace(-9,-1,9)
first_term1 = [first_term(T,gs) for gs in gs_temp]
entropy_production_cost1 = [entropy_production_cost(T,gs) for gs in gs_temp]
overdamped_bound = [od_bound(T,gs) for gs in gs_temp]

data = [gs_temp,np.ones(len(gs_temp))*T,first_term1,entropy_production_cost1,overdamped_bound]

df_costs = pd.concat([df_costs,pd.DataFrame(dict(zip(columns, data)))])

#append to csv
df_costs.to_csv("ep_costs.csv", index=False)
