import torch
import ot
import scipy.interpolate as sci
from sklearn.neighbors import KernelDensity
#from geomloss import SamplesLoss # See also ImagesLoss, VolumesLoss

#get the parameters
from main import *

n = int(sys.argv[5]) #number of samples for the optimal transport problem

# Create some large histograms from initial and final data
xs = npr.choice(np.linspace(xmin,xmax,n*100), size = (n,1), p = p_initial(np.linspace(xmin,xmax,n*100))/ sum(p_initial(np.linspace(xmin,xmax,n*100))))
xt = npr.choice(np.linspace(xmin,xmax,n*100), size = (n,1), p = p_final(np.linspace(xmin,xmax,n*100))/ sum(p_final(np.linspace(xmin,xmax,n*100))))


# loss matrix
M = ot.dist(xs, xt)
M /= M.max()


outs = ot.solve(M, reg=1, reg_type="entropy")

w2_dist = outs.value_linear
G0 = outs.plan

G0_out = np.zeros_like(G0)
G0_out[np.arange(len(G0)), G0.argmax(1)] = 1

xs = xs.flatten()
xt = xt.flatten()


#find lagrangian trajectories and burgers velocities
def get_rho_lambda(i,G0,xt):

  '''input:
  - i: index of the initial coordinate
  - t: time of evaluation

  returns:
  - l_map: approximation of the dynamic lagrangian map between the two distributions as a function of time
  - dsigma_x: dsigma (burger's velocity) evaluated at time (index) and x (l_map)
  '''

  idx_j = 0

  #get matching coordinate
  for j in range(0,n):
    if G0[i,j] > 0:
      idx_j = j
      break

  #get initial and final points
  xinit = xs[i]
  xfinal = xt[idx_j]

  #make (discrete) lagrangian maps
  l_map = np.ones(t_steps)
  for t in range(0,t_steps):
    tcurr = t2_vec[t]
    l_map[t] = ((Tf - tcurr)/Tf)*xinit + (tcurr/Tf)*xfinal

  #get burgers velocity (dsigma)
  dsigma_x = np.ones(t_steps)
  for x in enumerate(l_map):
    #evaluate sigma at x
    dsigma_x[x[0]] = (1/Tf)*(xfinal - xinit)

  return l_map,dsigma_x

results = np.zeros((n,2,len(t2_vec)))
for x in enumerate(xs):
  lmap,dsig = get_rho_lambda(x[0],G0_out,xt)

  #put into numpy array
  results[x[0],0,:] = lmap.reshape((1,1,9))
  results[x[0],1,:] = dsig.reshape((1,1,9))

#get a new equally spaced x for saving the results and integrating
N = 5000
x_axis = np.linspace(xmin,xmax,N)
df = pd.DataFrame()

for t2 in enumerate(t2_vec):
  xz = results[:,0,t2[0]]
  dsigmax = results[:,1,t2[0]]

  #sort by x
  xz_sort, sort_idx = np.unique(xz, return_index = True)

  #run kde on these points
  kde = KernelDensity(kernel='epanechnikov', bandwidth=0.15).fit(xz.reshape(-1, 1))

  #estimated pdf
  logrho_temp = kde.score_samples(x_axis.reshape(-1, 1))
  dens = np.exp(logrho_temp)

  #interpolation of sigma
  interp_dsigma = sci.interp1d(xz_sort,dsigmax[sort_idx], kind='cubic', bounds_error=False, fill_value=(dsigmax[0], dsigmax[-1]), assume_sorted=True)

  #make new df with these
  data= [t2[1]*np.ones(N), x_axis, interp_dsigma(x_axis), logrho_temp, dens]
  columns=["t","x","dsigma","logptx","ptx"]

  #append new
  df2append = pd.DataFrame(dict(zip(columns, data)))
  df = pd.concat([df,df2append])

#sort by
df.sort_values(["t","x"],inplace=True)

df = df.reset_index(drop=True)

df.replace([np.inf,-np.inf,np.nan], 0, inplace=True)

#save the dataframe as a csv
df.to_csv("results.csv",index=False)

#write out the parameters
filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# open a file in write mode
with open(filename+'.txt', 'w') as file:
    # write variables using repr() function
    file.write("epsilon = " + repr(epsilon) + '\n')
    file.write("T = " + repr(T) + '\n')
    file.write("g = " + repr(g) + '\n')
    file.write("h_step = " + repr(h_step) + '\n')
    file.write("w2_dist = " + repr(w2_dist) + '\n')

