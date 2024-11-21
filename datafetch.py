import pandas as pd
from main import *

df = pd.read_csv("results.csv", header = 0)

#fill infs with zeros
#df.replace([np.inf,-np.inf,np.nan], 0, inplace=True)
