import scipy.interpolate as sci
from sklearn.neighbors import KernelDensity

import functions
from main import *
from datafetch import *

#get dataframe
#df = pd.read_csv("results.csv",index_col=0)

##add distributions as function to get sigma
for t0 in times_t0:
  t2 = round(t0*(epsilon**2),dps)
  new_vals = functions.calculate_distribution(t0,g)
  drift_vals = functions.calculate_optimal_drift(t0,g)
  df.loc[df[df.t==t2].index,"UDpdf"] = new_vals
  df.loc[df[df.t==t2].index,"UDdrift"] = drift_vals

#save the dataframe as a csv
df.to_csv("results_all.csv")
