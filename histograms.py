
"""This file evolves the trajectories and plots sample histograms using the approximate underdamped drift
Note: the number of trajectories is ALSO called mc_samples, as in the Girsanov joint file. 
The default it 10000 samples. 
"""


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

from main import *
import functions
from plots import *

# Plotting the distributions
fig_distributions = plt.figure(figsize=(15,10))

nplots = 8
#print(int(round(nplots/2)))
#create plot grid
gs_distributions = fig_distributions.add_gridspec(2, int(round(nplots/2)), width_ratios=[1,1, 1, 1], height_ratios=[1, 1])

##SET UP: RUN BEFORE THE SIMULATIONS
#set up dataframe for cumulant results
df_ep_cumulants_exp = pd.DataFrame(columns = ["g","t0","pos_var","mom_var","mom_mean","pos_mean","xcorr","OD_mean","OD_var"])

##Evolve the underdamped dynamics using EM SCHEME
plot_index = 0

#evo_choice = np.linspace(-10,10,mc_samples) #starting points to choose from for histograms. Note. use a larger range for x
#sample_weights = p_initial(evo_choice)
#sample_weights /= sum(sample_weights)

x_evo = npr.choice(np.linspace(-10,10,mc_samples*10), size = mc_samples, p = p_initial(np.linspace(-10,10,mc_samples*10))/np.sum(p_initial(np.linspace(-10,10,mc_samples*10))))
q_evo_UD_prev = x_evo #npr.choice(evo_choice, size = mc_samples, p = sample_weights) #position from assigned initial dist.
p_evo_UD_prev = npr.randn(mc_samples) #momentum, independent standard Gaussian samples

#compute mean and variance of p and q using MLE at t=0
covmat = np.cov(q_evo_UD_prev,p_evo_UD_prev)

df_ep_cumulants_exp.loc[len(df_ep_cumulants_exp)] = [g, times_t0[0],
                                                     covmat[0,0],
                                                     covmat[1,1],
                                                     np.mean(p_evo_UD_prev),
                                                     np.mean(q_evo_UD_prev),
                                                     covmat[0,1],
                                                     np.nanmean(x_evo),
                                                     np.var(x_evo)]



#set up plots
#what times to plot


#get elements of the array
idx = np.round(np.linspace(0, t_steps - 1, nplots)).astype(int)

times_to_save = [0,0.5,0.75,1.0,1.25,1.5,1.75,2]
#times_t0[idx]
#0,0.25,0.5,0.75,1,,0.8,1]
times_to_save = np.round(times_to_save,4)


for i in range(0,len(times_t0)-1):
#enumerate(times_t0[0:-1]):
  print(i)

  #plot the selected times
  if (times_t0[i] in times_to_save):
    plot_distributions_ep(fig_distributions,gs_distributions,
                          plot_index,q_evo_UD_prev,x_evo,times_t0[i],nplots)
    plot_index += 1

  #overdamped
  #evolution in t2
  x_evo = x_evo - (h_step)*functions.dsigma_interp(times_t0[i],x_evo) + np.sqrt(2*g*h_step)*npr.randn(mc_samples)

  #underdamped
  #evolution in t0
  h0_step = h_step/(epsilon**2)
  q_evo_UD_prev = q_evo_UD_prev + epsilon*h0_step*(p_evo_UD_prev-g*epsilon*functions.underdamped_drift_interp(times_t0[i],q_evo_UD_prev,g,1e-5)) + epsilon*np.sqrt(2*g*h0_step)*npr.randn(mc_samples)
  p_evo_UD_prev = p_evo_UD_prev - (p_evo_UD_prev + epsilon*functions.underdamped_drift_interp(times_t0[i],q_evo_UD_prev,g,1e-5))*h0_step + np.sqrt(2*h0_step)*npr.randn(mc_samples)

  covmat = np.cov(q_evo_UD_prev,p_evo_UD_prev)

  #compute mean and variance of p and q using MLE
  df_ep_cumulants_exp.loc[len(df_ep_cumulants_exp)] = [g, times_t0[i+1],
                                                           covmat[0,0],
                                                            covmat[1,1],
                                                            np.mean(p_evo_UD_prev),
                                                            np.mean(q_evo_UD_prev),
                                                            covmat[0,1],
                                                            np.nanmean(x_evo),
                                                            np.var(x_evo)]


#add final time plot
plot_distributions_ep(fig_distributions,gs_distributions,plot_index,q_evo_UD_prev,x_evo,T,nplots);


#add legend
orange_line = mlines.Line2D([], [],color="orange",lw=lw)
blue_line = mlines.Line2D([], [],color=c1,lw=lw)
darkblue_line = mlines.Line2D([], [],color=c2,lw=lw)
legend = fig_distributions.legend(handles=[blue_line,darkblue_line,orange_line],
          labels = ["Underdamped (Evolved)","Underdamped (Predicted)","Overdamped (Evolved)"],
           #prop={"size":fontsizeticks},
          fontsize = fontsizetitles,
          frameon = False,
          handlelength = 1,
          ncols = 3,
          loc="upper center", bbox_to_anchor=(0.5, 0))


bbox = legend.get_window_extent(fig_distributions.canvas.get_renderer()).transformed(fig_distributions.transFigure.inverted())
fig_distributions.tight_layout(rect=(0, bbox.y1, 1, 1), h_pad=2, w_pad=2)




#save the cumulants
df_ep_cumulants_exp.to_csv("cumulants.csv", index=False)

#save the histogram
plt.savefig("histograms_test.pdf",bbox_inches="tight")

