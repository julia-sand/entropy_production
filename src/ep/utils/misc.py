import os
import datetime
import yaml

from ep.utils.parser import fetch_results_folder_from_cmd

def save_params(params,filename):
    # Create timestamped filename
    #timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    full_filename = f"{filename}.yaml"#f"{filename}{timestamp}.yaml"

    # Save dictionary as YAML
    with open(full_filename, "w") as file:
        yaml.dump(params, file, default_flow_style=False)

def append_to_dict(d, key, value):
    """
    Append a value to a dictionary entry.
    
    If the key does not exist, create it with a list.
    If the key exists and is not a list, convert it to a list.
    """
    
    if key not in d:
        d[key] = value
    else:
        if not isinstance(d[key], list):
            d[key] = [d[key]]
        d[key].append(value)

    return d

def load_params(paramfile):
    with open(paramfile, "r") as file:
        params = yaml.safe_load(file)

    return params

def make_results_dir():

    #check if there is a results folder to use 
    newpath = fetch_results_folder_from_cmd()

    if newpath is None:
        ##make a directory for the results in
        # the place the file is run
        dirname = os.getcwd()

        newpath = os.path.join(dirname,'results/'+f"{time.strftime("%Y%m%d-%H%M%S")}")
    
    os.makedirs(newpath, exist_ok = True)
    print("Output Directory created successfully.")  
    
    return newpath


def init_params_from_file():

    #check if there is a results folder to use 
    folder = fetch_results_folder_from_cmd()

    try:
        # Read YAML file
        with open(folder+'/results.yaml', 'r') as stream:
            data_loaded = yaml.safe_load(stream)

    except FileNotFoundError: #if no yaml file is provided, use the defaults
        print("No parameter file found in the specified folder. The plots will be created using the default values. See -h for values.")

    return data_loaded