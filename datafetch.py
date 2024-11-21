import pandas as pd

df = pd.read_csv("results.csv",header = 1)

#fill infs with zeros
#df.replace([np.inf,-np.inf,np.nan], 0, inplace=True)
