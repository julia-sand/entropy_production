"""fetch the dataframe containing results"""
import pandas as pd

from ep.utils.parser import fetch_results_folder_from_cmd

def open_array(filename):
    folder = fetch_results_folder_from_cmd()+"/results.csv"
    
    try:
        df = pd.read_csv(filename+".csv", sep=" ", header = 0)

        #round the t0 and t2 columns
        dps = 5
        df = df.round({'t0': dps, 't2': dps})
        return df
    except: 
        print("The requested results file could not be found. Please first solve the overdamped problem (sinkhorn.py) \n or check that you have entered the filename correctly.")
        #solve_cell(n,filename)
        #raise BaseException

def open_array(filename):
      return np.genfromtxt(filename, delimiter=",", names=True, dtype=None, encoding="utf-8")

if __name__=="__main__":
    open_array(filename)
