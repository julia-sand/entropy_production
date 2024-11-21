import pandas as pd
from main import *

df = pd.read_csv("results.csv", header = 0)

#add t0 col
df["t0"] = np.round(df["t"]/(epsilon**2),dps)

#fill infs with zeros
#df.replace([np.inf,-np.inf,np.nan], 0, inplace=True)
