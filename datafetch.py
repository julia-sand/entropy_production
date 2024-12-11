import pandas as pd
from main import *


#df = pd.read_csv(filename+".csv", sep=" ", header = 0)
df = pd.read_csv("results_FLIP_TEMP.csv",sep=" ",index_col=0)

#df["t0"] = np.round(df.t2/(epsilon**2) ,dps)
