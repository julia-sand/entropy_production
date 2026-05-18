"""handles parameter loading from command line"""

import yaml 

import argparse

def fetch_results_folder_from_cmd():
    # Instantiate the parser
    parser6 = argparse.ArgumentParser()
    
    parser.add_argument('-filename','--filename', type=str, default="results",
                        help='name of csv for results')

    args6 = parser6.parse_known_args()

    return args6[0].filename

def cell_argparser():
    """Parse arguments for solving the cell problem"""
    
  # Instantiate the parser
    parser = argparse.ArgumentParser()

    parser.add_argument('-n','--n', type=int, default=1000,
                        help='how many x samples to use in the OT problem')
    
    parser.add_argument('-hstep','--hstep', type=int, default=0.05,
                        help='size of timestep')
    
    args = parser.parse_known_args()

    return vars(args[0])

def config_argparser():
    #general configs for the training process
    # Instantiate the parser
    parser = argparse.ArgumentParser()

    parser.add_argument('-bridgetype','--bridgetype', type=str, default="full",
                        help='bridge type')
    
  
    parser.add_argument('-g','--g', default=0.01,
                        help='coupling constant')
  
    parser.add_argument("--fileid", default="V1", type=str,
                        help="This can be used to add a version number at the end of filenames for outputs, eg csv, plot images")
  
    args = parser.parse_known_args()

    return vars(args[0])

def boundary_params():

    # Instantiate the parser
    parser = argparse.ArgumentParser()

    parser.add_argument('-peaklocation','--peaklocation', default=1,
                        help='location of initial peak')
  
    parser.add_argument('-denom','--denom', default=1,
                        help='denominator of the distributions, determines peak heights')
  
    args2 = parser.parse_known_args()

    return vars(args2[0])

if __name__=="__main__":
    

    params_test = config_argparser()

    print(params_test)
