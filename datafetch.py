import pandas as pd
from main import *


df = pd.read_csv(filename+".csv", sep=" ", header = 0)

#round the t0 and t2 columns just in case it still isnt working
df = df.round({'t0': 3, 't2': 4})