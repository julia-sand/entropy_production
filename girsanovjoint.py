"""
Compute the joint distribution using the perturbative drift and girsanov's theorem

Creates a plot similar to that in Fig 4 of https://arxiv.org/abs/2411.08518 for the entropy production case.
"""
import functions
from main import *
from datafetch import *

#set up the time grid things
mc_samples = 300 #number of sample trajectories
p_samples = 40
q_samples = 40

#initialise p,q axes
p_init = np.linspace(-8,8,p_samples)
q_init = np.linspace(-8,8,q_samples)

#make a mesh grid
P,Q = np.meshgrid(p_init,q_init)


df_girspdf_ep = pd.DataFrame()

#get figure
#fig_joint_distributions_meshgrid = plt.figure(figsize=(15,10))

###BACKWARDS EVOLUTIONS

#reset plot index
plot_index = 5


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
  #currdrift = underdamped_drift_interp(t,q_evo_UD_prev)

  for i in range(0,start_index):

    curr_time = times_t0[start_index-i]
    print("curr_time",curr_time)

    innovation = npr.randn(mc_samples,p_samples,q_samples)

    currdrift = underdamped_drift_interp(curr_time,q_evo_UD_prev,g)

    #underdamped dynamics
    #backward (T-t0) evolution in t0
    q_evo_UD_prev = q_evo_UD_prev - epsilon*h0_step*(p_evo_UD_prev-g*epsilon*currdrift) - epsilon*np.sqrt(2*g*h0_step)*npr.randn(mc_samples,p_samples,q_samples)
    p_evo_UD_prev = p_evo_UD_prev + (epsilon*currdrift)*h0_step - np.sqrt(2*h0_step)*innovation

    #remove escaped particles
    q_evo_UD_prev[q_evo_UD_prev>30] = np.nan
    q_evo_UD_prev[q_evo_UD_prev<-30] = np.nan
    p_evo_UD_prev[p_evo_UD_prev>30] = np.nan
    p_evo_UD_prev[p_evo_UD_prev<-30] = np.nan
    #q_evo_UD_prev = q_evo_UD_new
    #p_evo_UD_prev = p_evo_UD_new


    #compute girsanov factor
    girsanov = girsanov + (np.sqrt(h0_step/2)*np.multiply(p_evo_UD_prev,innovation) + (h0_step/4)*np.square(p_evo_UD_prev))



  joint_out = np.nanmean(np.multiply(ud_pinitial(p_evo_UD_prev,q_evo_UD_prev),np.exp(-girsanov)),axis=0)

  #save the data
  data= [t*np.ones(p_samples*q_samples),
         P.flatten(),
         Q.flatten(),
         joint_out.flatten()]
  columns=["t","P","Q","ptx"]

  #append new
  df_temp = pd.DataFrame(dict(zip(columns, data)))

  df_girspdf_ep = pd.concat([df_girspdf_ep,df_temp])

  plot_index -= 1
  ####
