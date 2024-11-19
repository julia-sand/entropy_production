import pandas as pd

df = pd.read_csv("results.csv",index_col=0)

#fill infs with zeros
#df.replace([np.inf,-np.inf,np.nan], 0, inplace=True)
