import pandas as pd
from src.utils.params import *

try:
    df = pd.read_csv(filename+".csv", sep=" ", header = 0)

    #round the t0 and t2 columns just in case it still isnt working
    df = df.round({'t0': dps, 't2': dps})
except: 
    print("The requested results file could not be found. Please first solve the overdamped problem (sinkhorn.py) \n or check that you have entered the filename correctly.")