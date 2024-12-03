"""
Compute the joint distribution using the perturbative drift and girsanov's theorem

Creates a plot similar to that in Fig 4 of https://arxiv.org/abs/2411.08518 for the entropy production case.
"""

import functions
import plots
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from main import *
from datafetch import *

#import sys
import csv

#mc_samples = int(sys.argv[5]) #number of sample trajectories

#set up the spacial coordinates
#p_samples = 1
#q_samples = 21

#initialise p,q axes
p_init = np.linspace(-8,8,p_samples)
q_init = np.linspace(-8,8,q_samples)

#make a mesh grid
P,Q = np.meshgrid(p_init,q_init)

#df_girspdf_ep = pd.DataFrame()

###BACKWARDS EVOLUTIONS

#reset plot index
plot_index = 5

#what times to plot
plot_times = np.array([2,1.5,1.0,0.5,0.25,0])#np.flip([0,1,2,3,4,5])/(5/T)

plot_titles = [f"$t = {plot_times[j]}$" for j in range(0,len(plot_times))]
plot_titles = np.flip(plot_titles)

#write header to file
header=["t","P","Q","ptx"]
with open("ep_girsanovjoint.csv","w") as file:
   writer = csv.writer(file,delimiter=" ", lineterminator="\n")
   writer.writerow(header)

for t in plot_times:
  print(t)

  start_index = np.where(times_t0==t)[0][0]

  print("startindex",start_index)
  #start the trajectories
  p_evo_UD_prev = np.broadcast_to(P,(mc_samples,p_samples,q_samples))
  q_evo_UD_prev = np.broadcast_to(Q,(mc_samples,p_samples,q_samples))

  #reset girsanov
  girsanov = np.zeros((mc_samples,p_samples,q_samples))

  #initialise drift

  for i in range(0,start_index):

    curr_time = times_t0[start_index-i]
    print("curr_time",curr_time)

    innovation = npr.randn(mc_samples,p_samples,q_samples)

    currdrift = functions.underdamped_drift_interp(curr_time,q_evo_UD_prev,g)

    #underdamped dynamics
    #backward (T-t0) evolution in t0
    q_evo_UD_prev = q_evo_UD_prev - epsilon*h0_step*(p_evo_UD_prev-g*epsilon*currdrift) - epsilon*np.sqrt(2*g*h0_step)*npr.randn(mc_samples,p_samples,q_samples)
    p_evo_UD_prev = p_evo_UD_prev + (epsilon*currdrift)*h0_step - np.sqrt(2*h0_step)*innovation

    #remove escaped particles
    #q_evo_UD_prev[q_evo_UD_prev>30] = np.nan
    #q_evo_UD_prev[q_evo_UD_prev<-30] = np.nan
    #p_evo_UD_prev[p_evo_UD_prev>30] = np.nan
    #p_evo_UD_prev[p_evo_UD_prev<-30] = np.nan
    #q_evo_UD_prev = q_evo_UD_new
    #p_evo_UD_prev = p_evo_UD_new


    #compute girsanov factor
    girsanov = girsanov + (np.sqrt(h0_step/2)*np.multiply(p_evo_UD_prev,innovation) + (h0_step/4)*np.square(p_evo_UD_prev))



  joint_out = np.nanmean(np.multiply(functions.ud_pinitial(p_evo_UD_prev,q_evo_UD_prev),np.exp(-girsanov)),axis=0)

  data = np.column_stack((t*np.ones(p_samples*q_samples),
                           P.flatten(),
                           Q.flatten(),
                           joint_out.flatten()))
  np.nan_to_num(data,copy=False,nan=0,posinf=0,neginf=0)

  #append to the csv
  with open("ep_girsanovjoint.csv","a") as file:
    np.savetxt(file,data)

  plot_index -= 1
  ####


df_girspdf_ep = pd.read_csv("ep_girsanovjoint.csv", sep=" ", header = 0)

#make the plot
vmax = np.max(df_girspdf_ep.ptx)

# Plotting the distributions -initialise the figure object & creates the gridspec
fig_joint_distributions_meshgrid = plt.figure(figsize=(15,10))
gs_joint_distributions = fig_joint_distributions_meshgrid.add_gridspec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1])

for k in enumerate(plot_times):
  plots.joint_distributions_scatter(fig_joint_distributions_meshgrid,gs_joint_distributions, 5-k[0],
                              df_girspdf_ep[df_girspdf_ep["t"] == k[1]].ptx.to_numpy(),
                              Q,P,
                              k[1],vmax)


#adjust spacing
fig_joint_distributions_meshgrid.subplots_adjust(
     wspace=0.85,# The width of the padding between subplots, as a fraction of the average Axes width.
    hspace=0.65# The height of the padding between subplots, as a fraction of the average Axes height.
)

plt.savefig("test.pdf",bbox_inches="tight")
