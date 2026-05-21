"""fetch the dataframe containing results"""
import numpy as np

from ep.utils.parser import fetch_results_folder_from_cmd


def open_array():
    
    try: 

        filename = fetch_results_folder_from_cmd()+"/results_corrected.csv"
    
        return np.genfromtxt(filename, delimiter=" ", names=True, dtype=float, encoding="utf-8")

    except FileNotFoundError:
        filename = fetch_results_folder_from_cmd()+"/results.csv"

        return np.genfromtxt(filename, delimiter=" ", names=True, dtype=float, encoding="utf-8")

if __name__=="__main__":
    open_array(filename)
