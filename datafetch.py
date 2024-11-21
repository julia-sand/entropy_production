import pandas as pd
from main import *

df = pd.read_csv("results.csv",sep = " ", header = 0)

#add t0 column
df["t0"] = np.round((epsilon**2)*df.t ,dps)

#fill infs with zeros
#df.replace([np.inf,-np.inf,np.nan], 0, inplace=True)
