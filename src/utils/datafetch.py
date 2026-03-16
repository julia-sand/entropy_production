"""fetch the dataframe containing results"""
import pandas as pd

from src.utils.params import *

def open_df():
    try:
        df = pd.read_csv(filename+".csv", sep=" ", header = 0)

        #round the t0 and t2 columns
        df = df.round({'t0': dps, 't2': dps})
        return df
    except: 
        print("The requested results file could not be found. Please first solve the overdamped problem (sinkhorn.py) \n or check that you have entered the filename correctly.")
        raise BaseException

if __name__=="__main__":
    open_df()