import datetime
import yaml

def save_params(params,filename):
   # Create timestamped filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    full_filename = f"{filename}{timestamp}.yaml"

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