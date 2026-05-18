"""
This file holds all the shared parameters
"""

import pandas as pd
import numpy as np
import numpy.random as npr
import datetime

import argparse,sys

#add the arguments to the parser
parser = argparse.ArgumentParser()

parser.add_argument("--epsilon", default=0.2, help="multiscale expansion parameter")
parser.add_argument("--Tf", default=2, help="final (underdamped) time")
parser.add_argument("--hstep", default=0.05, help="size of timemesh")
parser.add_argument("--g", default=0.01, help="momentum coupling constant")
parser.add_argument("--n", default=100000, help="number of points in optimal transport matching")
parser.add_argument("--mcsamples", default=10000, help="number of monte carlo trajectories in calculation of joint distribution")
parser.add_argument("--filename", default="resultsXL", help="filename for input and output file")
parser.add_argument("--pqsamples", default=51, help="number of samples to use for P and Q in the Girsanov calculation")
parser.add_argument("--peaklocation", default=1, help="location of initial peak")
parser.add_argument("--denom", default=1, help="denominator of the distributions")
parser.add_argument("--w2dist", default=1.1180988295435215, help="This is the wasserstein distance found from the cell problem. See the .txt data file for the value")
parser.add_argument("--fileid", default="V1", help="This can be used to add a version number at the end of filenames for outputs, eg csv, plot images")

args = parser.parse_args()

#get params
h0_step = float(args.hstep)
g = float(args.g)
n = int(args.n)
mc_samples = int(args.mcsamples)
filename = args.filename
p_samples = int(args.pqsamples)
q_samples = int(args.pqsamples)
denom = float(args.denom)
peak_center = float(args.peaklocation)
w2_dist = float(args.w2dist)
fileid = args.fileid

def make_t2_vec(h0_step):

  T = float(args.Tf)
  epsilon = float(args.epsilon)
  
  ### params
  Tf = (epsilon**2)*T  #final time for t2
  
  #decimal places for the time lookup.
  h_step = h0_step*(epsilon**2)
  dps =  int(np.ceil(-np.log10(h_step))+1)
  t_steps = int(T/h0_step) + 1 #number of timesteps
  times_t0 = np.round(np.linspace(0,T,t_steps,endpoint = True),dps)
  t2_vec = np.round(times_t0*(epsilon**2),dps)

  return t2_vec


alpha = 0

#xaxis for calculations
N = 50000
xmin = -3
xmax = 3
q_axis = np.linspace(xmin,xmax,N)

#tolerance to zero
tol = 1e-100

#size of smoothing filter
filter_delta = 500
