"""Solves an optimal transport problem
 corresponding to the overdamped cell problem
"""

import csv

import numpy as np
import ot
from sklearn.neighbors import KernelDensity

from ep.utils.params import make_t2_vec
from ep.utils.parser import cell_argparser,pert_argparser
from ep.utils.boundary import Boundary
from ep.utils.misc import *

def make_histograms(boundary,n):
  """Generate histograms of size n of initial and final distributions"""
  xs = np.random.choice(np.linspace(-10,10,n), size = n, p = boundary.p_initial(np.linspace(-10,10,n))/ sum(boundary.p_initial(np.linspace(-10,10,n))))
  xt = np.random.choice(np.linspace(-10,10,n), size = n, p = boundary.p_final(np.linspace(-10,10,n))/ sum(boundary.p_final(np.linspace(-10,10,n))))

  return xs,xt

def solve_sinkhorn(xs,xt):

  n = xs.shape[0]

  #solve OT problem using Python OT
  G0_data = ot.emd2_1d(xs.reshape((n, 1)), xt.reshape((n, 1)),log=True)

  #save the assignments and W2 distance
  #G0 = G0_data[1]["G"]
  w2_dist = G0_data[0]
  idx = np.argmax(G0_data[1]["G"],axis=1)

  return w2_dist, idx

def get_rho_lambda(i,idx,xs,xt,t2_vec):
  '''
  Compute lagrangian trajectories and burgers velocities for a mat
  args:
    i : int
      index of the initial coordinate
    idx : ndarray
      indices of the matched endpoints found using ot
    xs : ndarray
      sampled histogram of initial assigned distribution
    xt : ndarray
      sampled histogram of final assigned distribution

  returns:
    l_map : ndarray 
      approximation of the dynamic lagrangian map between the two distributions as a function of time
    dsigma_x : ndarray
      dsigma (burger's velocity) evaluated at time (index) and x (l_map)
  '''

  idx_j = idx[i]

  #get initial and final points
  xinit = xs[i]
  xfinal = xt[idx_j]

  Tf = t2_vec[-1]
 
  #make (discrete) lagrangian maps
  l_map = np.fromiter((((Tf - tcurr)/Tf)*xinit + (tcurr/Tf)*xfinal for tcurr in t2_vec), float)

  #get burgers velocity (dsigma)
  dsigma_x = np.ones_like(t2_vec)*(1/Tf)*(xfinal - xinit)

  return l_map,dsigma_x


def compute_results(idx,xs,xt,t2_vec):

  #preallocate array
  results = np.zeros((xs.shape[0],2,len(t2_vec)))

  for x in enumerate(xs):
    lmap,dsig = get_rho_lambda(x[0],idx,xs,xt,t2_vec)

    #save into numpy array
    results[x[0],0,:] = lmap.reshape((1,1,len(t2_vec)))
    results[x[0],1,:] = dsig.reshape((1,1,len(t2_vec)))
  
  return results
  
def save_results_to_csv(results,filename,t2_vec,times_t0):
  
  #xaxis for calculations
  N = 500
  xmin = -3
  xmax = 3
  q_axis = np.linspace(xmin,xmax,N)
 
  #add header 
  header=["t0","t2","x","dsigma","logptx","ptx"]
  
  with open(filename+".csv","w") as file: 
    writer = csv.writer(file,delimiter=" ", lineterminator="\n")
    writer.writerow(header)

  for t2 in enumerate(t2_vec):

    #select point clouds at each point
    xz = results[:,0,t2[0]]
    dsigmax = results[:,1,t2[0]]

    #sort by x
    xz_sort, sort_idx = np.unique(xz, return_index = True)

    #run kde on these points
    kde = KernelDensity(kernel='epanechnikov', bandwidth=0.2).fit(xz.reshape(-1, 1))

    #estimated pdf
    logrho_temp = kde.score_samples(q_axis.reshape(-1, 1))
    dens = np.exp(logrho_temp)

    #make new array with these
    data = np.column_stack((np.full(N,times_t0[t2[0]]),np.full(N,t2[1]), q_axis, 
                            np.interp(q_axis,xz_sort,dsigmax[sort_idx]),#interp_dsigma(x_axis), 
                            logrho_temp, dens))
    np.nan_to_num(data,copy=False,nan=0,posinf=0,neginf=0)

    #append to the csv
    with open(filename+".csv","a") as file:
      np.savetxt(file,data)
  
  return 

def solve_cell(filename):

  boundary = Boundary() #parse cmd line boundary conditions

  cell_args = cell_argparser()
  n = cell_args["n"]
  h2step = cell_args["hstep"]
  time_params = pert_argparser()

  params = {}
  append_to_dict(params,"n",n)
  append_to_dict(params,"hstep",h2step)
  append_to_dict(params,"peak",boundary.peak_center)
  append_to_dict(params,"height",boundary.denom)
  append_to_dict(params,"Tf2",time_params["Tf"])
  append_to_dict(params,"epsilon",time_params["epsilon"])

  t2_vec = make_t2_vec(h2step,time_params["epsilon"],time_params["Tf"])
  times_t0 = np.round(t2_vec/(time_params["epsilon"]**2),5)

  xs,xt = make_histograms(boundary,n)

  w2_dist, idx = solve_sinkhorn(xs,xt)
  append_to_dict(params,"w2",w2_dist)

  save_params(params,filename)

  results = compute_results(idx,xs,xt,t2_vec)
  save_results_to_csv(results,filename,t2_vec,times_t0)

if __name__=="__main__":
  from ep.utils.misc import make_results_dir
 
  folder = make_results_dir()


  solve_cell(folder+"/results")