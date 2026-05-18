"""
This file defines the shared parameters and boundary conditions
"""
import datetime

import pandas as pd
import numpy as np

from ep.utils.parser import pert_argparser

#get params
p_samples = 10
q_samples = 10

def make_t2_vec(h2_step,epsilon,T):

  ### params
  Tf2 = (epsilon**2)*T  #final time for t2
  
  #decimal places for the time lookup.
  h_step = h2_step/(epsilon**2)
  dps =  int(np.ceil(-np.log10(h2_step))+1)
  t_steps = int(Tf2/h2_step) + 1 #number of timesteps
  #times_t0 = 
  t2_vec = np.round(np.linspace(0,Tf2,t_steps,endpoint = True),dps)

  return t2_vec

#tolerance to zero
tol = 1e-100

#size of smoothing filter
filter_delta = 500
