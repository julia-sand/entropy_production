"""
Compute the joint distribution using the perturbative drift and girsanov's theorem

save to csv
"""


import setup.functions as functions
from setup.main import *
from setup.datafetch import *

#import sys
import csv

#mc_samples = int(sys.argv[5]) #number of sample trajectories

#set up the spacial coordinates
#p_samples = 1
#q_samples = 21

#initialise p,q axes
p_init = np.linspace(-10,10,p_samples)
q_init = np.linspace(-3,3,q_samples)

#make a mesh grid
P,Q = np.meshgrid(p_init,q_init)

#df_girspdf_ep = pd.DataFrame()

###BACKWARDS EVOLUTIONS

#reset plot index
plot_index = 5

#what times to plot
plot_times = np.array([2,1.5,1.0,0.5,0.25,0])
#np.array([1.0,0.5,0.25,0])#np.array([2,1.5,1.0,0.5,0.25,0])#np.flip([0,1,2,3,4,5])/(5/T)

plot_titles = [f"t = {plot_times[j]}" for j in range(0,len(plot_times))]
plot_titles = np.flip(plot_titles)

#write header to file
header=["t","P","Q","ptx"]
filename_temp = "ep_girsanovjoint" +f"{fileid}"+".csv"
with open(filename_temp,"w") as file:
   writer = csv.writer(file, delimiter=" ", lineterminator="\n")
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

    currdrift = functions.underdamped_drift_interp(curr_time,q_evo_UD_prev,g,1e-5)

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
  with open(filename_temp,"a") as file:
    np.savetxt(file,data)

  plot_index -= 1
  ####

