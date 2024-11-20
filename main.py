"""
This file holds all the shared parameters for the whole program."""


### import all the parameters here

import pandas as pd
import numpy as np
import numpy.random as npr
import datetime

import sys

#get params
epsilon = float(sys.argv[1])
T = float(sys.argv[2])
h_step = float(sys.argv[3]) #size of time mesh
g = float(sys.argv[4])
#n = int(sys.argv[5]) #number of samples for the optimal transport problem

### params
Tf = (epsilon**2)*T  #final time for t2

#decimal places for the time lookup.
dps =  int(np.ceil(-np.log10(h_step))+1)
t_steps = int(Tf/h_step) + 1 #number of timesteps
t2_vec = np.round(np.linspace(0,Tf,t_steps,endpoint = True),dps)
h0_step = np.round(h_step/(epsilon**2),4)

times_t0 = np.round(t2_vec/(epsilon**2),4)

alpha = 0

#tolerance to zero
tol = 1e-10

#size of smoothing filter
filter_delta = 20

#set up the boundary conditions
peak_center = 1
denom = 1
xmin = -5
xmax = 5

#q = np.linspace(xmin,xmax,n) #fixed axes of points
#qnorm = np.linspace(-15,15,50000) #used for computing the normalisation
#qchoice = np.linspace(xmin,xmax,n*100) #points to choose from for histograms


#exact boundary conditions
def p_initial_unnormalised(q):
  return np.exp(-(q-peak_center)**4/denom)
def p_final_unnormalised(q):
  return np.exp(-(((q**2 -peak_center**2)**2)/denom))

#compute normalisation constacomputents
pi_norm = np.abs(np.trapz(p_initial_unnormalised(np.linspace(-8,8,10000)),np.linspace(-8,8,10000)))
pf_norm = np.abs(np.trapz(p_final_unnormalised(np.linspace(-8,8,10000)),np.linspace(-8,8,10000)))

#normalised boundary conds
def p_initial(q):
  return p_initial_unnormalised(q)/pi_norm
def p_final(q):
  return p_final_unnormalised(q)/pf_norm


#underdamped boundary conditions
def ud_pinitial(p_samples,q_samples):
  """this is the initial distribution in the underdamped case
  p_samples = momenta
  q_samples = positions
  """

  return p_initial(q_samples)*np.exp(-(p_samples**2)/2)/np.sqrt(2*np.pi)

def ud_pfinal(p_samples,q_samples):
  """this is the final distribution in the underdamped case
  p_samples = momenta
  q_samples = positions
  """
  return p_final(q_samples)*np.exp(-(p_samples**2)/2)/np.sqrt(2*np.pi)
  #np.exp(-(((q_samples**2-1)**2)/4 + 0.5*p_samples**2))/(pf_norm*np.sqrt(2*np.pi))



