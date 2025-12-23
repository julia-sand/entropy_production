"""Solves an optimal control problem corresponding to the overdamped 
cell problem
"""

import csv

import ot
from sklearn.neighbors import KernelDensity

from src.utils.params import *

def make_histograms(n):

  # Generate some large histograms from initial and final data
  xs = npr.choice(np.linspace(-10,10,n), size = n, p = p_initial(np.linspace(-10,10,n))/ sum(p_initial(np.linspace(-10,10,n))))
  xt = npr.choice(np.linspace(-10,10,n), size = n, p = p_final(np.linspace(-10,10,n))/ sum(p_final(np.linspace(-10,10,n))))

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

def get_rho_lambda(i,idx,xs,xt):
  '''
  Computes lagrangian trajectories and burgers velocities for a mat
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

  #make (discrete) lagrangian maps
  l_map = np.fromiter((((Tf - tcurr)/Tf)*xinit + (tcurr/Tf)*xfinal for tcurr in t2_vec), float)

  #get burgers velocity (dsigma)
  dsigma_x = np.ones(t_steps)*(1/Tf)*(xfinal - xinit)

  return l_map,dsigma_x


def compute_results(idx,xs,xt):

  #preallocate array
  results = np.zeros((n,2,t_steps))

  for x in enumerate(xs):
    lmap,dsig = get_rho_lambda(x[0],idx,xs,xt)

    #save into numpy array
    results[x[0],0,:] = lmap.reshape((1,1,t_steps))
    results[x[0],1,:] = dsig.reshape((1,1,t_steps))
  
  return results
  
def save_results_to_csv(results,filename):

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

    #interpolation of sigma
    #interp_dsigma = sci.interp1d(xz_sort,dsigmax[sort_idx], kind='cubic', bounds_error=False, fill_value=(dsigmax[0], dsigmax[-1]), assume_sorted=True)

    #make new array with these
    data = np.column_stack((np.full(N,times_t0[t2[0]]),np.full(N,t2[1]), q_axis, 
                            np.interp(q_axis,xz_sort,dsigmax[sort_idx]),#interp_dsigma(x_axis), 
                            logrho_temp, dens))
    np.nan_to_num(data,copy=False,nan=0,posinf=0,neginf=0)

    #append to the csv
    with open(filename+".csv","a") as file:
      np.savetxt(file,data)
  
  return 

def save_params(w2_dist,filename):

  #save parameters to txt file
  # open a file in write mode
  with open(filename+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.txt', 'w') as file:
      # write variables using repr() function
      file.write("epsilon = " + repr(epsilon) + '\n')
      file.write("T = " + repr(T) + '\n')
      file.write("g = " + repr(g) + '\n')
      file.write("h_step = " + repr(h_step) + '\n')
      file.write("w2_dist = " + repr(w2_dist) + '\n')

  return 

def solve_cell(n,filename):

  xs,xt = make_histograms(n)

  w2_dist, idx = solve_sinkhorn(xs,xt)

  save_params(w2_dist,filename)

  results = compute_results(idx,xs,xt)
  save_results_to_csv(results)

if __name__=="__main__":
  n = 1000
  filename = "test"
  solve_cell(n,filename)