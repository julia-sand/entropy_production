import scipy.interpolate as sci
from sklearn.neighbors import KernelDensity

import setup.functions as functions
from setup.main import *
from setup.datafetch import *

#add t0 col
#df["t0"] = np.round(df.t2/(epsilon**2) ,dps)

#fill infs with zeros
df.replace([np.inf,-np.inf,np.nan], 0, inplace=True)


#add two new columns

df["UDpdf"] = np.ones(len(df))
df["UDdrift"] = np.ones(len(df)) 

##add distributions as function to get sigma
for t0 in times_t0:
  print(t0)
  #t2 = round(t0*(epsilon**2),dps)
  new_vals = functions.calculate_distribution(t0,g)
  drift_vals = functions.calculate_optimal_drift(t0,g)
  df.loc[df[df.t0==t0].index,"UDpdf"] = new_vals
  df.loc[df[df.t0==t0].index,"UDdrift"] = drift_vals

#save the dataframe as a csv
temp = filename+".csv"
df.to_csv(temp, sep = " ", index=False)
