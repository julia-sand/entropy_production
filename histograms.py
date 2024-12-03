
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
fig_distributions = plt.figure(figsize=(8,6))

#create plot grid
gs_distributions = fig_distributions.add_gridspec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1])

##SET UP: RUN BEFORE THE SIMULATIONS
#set up dataframe for cumulant results
df_ep_cumulants_exp = pd.DataFrame(columns = ["g","t0","pos_var","mom_var","mom_mean","pos_mean","xcorr","OD_mean","OD_var"])

##Evolve the underdamped dynamics using EM SCHEME
plot_index = 0

evo_choice = np.linspace(xmin,xmax,mc_samples*10) #starting points to choose from for histograms
sample_weights = p_initial(evo_choice)
sample_weights /= sum(sample_weights)

x_evo = npr.choice(evo_choice, size = mc_samples, p = sample_weights)
q_evo_UD_prev = npr.choice(evo_choice, size = mc_samples, p = sample_weights) #position from assigned initial dist.
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


for i in range(0,len(times_t0)-1):
#enumerate(times_t0[0:-1]):
  print(i)
  
  #plot the selected times
  if (times_t0[i]/T in times_to_save):
    plot_distributions_ep(fig_distributions,gs_distributions,
                          plot_index,q_evo_UD_prev,x_evo,times_t0[i])
    plot_index += 1

  #overdamped
  #evolution in t2
  #x_evo = x_evo - (h_step)*functions.dsigma_interp(times_t0[i],x_evo) + np.sqrt(2*g*h_step)*npr.randn(samples)

  #underdamped
  #evolution in t0
  h0_step = h_step/(epsilon**2)
  q_evo_UD_prev = q_evo_UD_prev + epsilon*h0_step*(p_evo_UD_prev-g*epsilon*functions.underdamped_drift_interp(times_t0[i],q_evo_UD_prev,g)) + epsilon*np.sqrt(2*g*h0_step)*npr.randn(mc_samples)
  p_evo_UD_prev = p_evo_UD_prev - (p_evo_UD_prev + epsilon*functions.underdamped_drift_interp(times_t0[i],q_evo_UD_prev,g))*h0_step + np.sqrt(2*h0_step)*npr.randn(mc_samples)


  #idx = get_rhomask(times_t0[i])
  #qmin = np.min(q_axis(times_t0[i]))
  #qmax = np.max(q_axis(times_t0[i]))
  #remove escaped particles
  #q_evo_UD_new[q_evo_UD_new>xmax] = np.nan
  #q_evo_UD_new[q_evo_UD_new<xmin] = np.nan

  #remove nans and infs
  #good_idx_q = np.argwhere(~np.isnan(q_evo_UD_new)) #np.where(q_evo_UD_mid != [np.nan,np.inf,-np.inf])
  #good_idx_p = np.argwhere(~np.isnan(p_evo_UD_new)) #np.where(p_evo_UD_new != [np.nan,np.inf,-np.inf])

  #idx_out = np.intersect1d(good_idx_q,good_idx_p)
  #go forward without nans and infs
  #q_evo_UD_prev = q_evo_UD_new[idx_out]
  #p_evo_UD_prev = p_evo_UD_new[idx_out]

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
plot_distributions_ep(fig_distributions,gs_distributions,plot_index,q_evo_UD_prev,x_evo,T);

#add legend
orange_line = mlines.Line2D([], [],color="orange",lw=lw)
blue_line = mlines.Line2D([], [],color=c1,lw=lw)
legend = fig_distributions.legend(handles=[blue_line,orange_line],
          labels = ["Underdamped","Overdamped"],
           #prop={"size":fontsizeticks},
          fontsize = fontsizetitles,
          frameon = False,
          handlelength = 1,
          ncols = 2,
          loc="upper center", bbox_to_anchor=(0.5, 0))


bbox = legend.get_window_extent(fig_distributions.canvas.get_renderer()).transformed(fig_distributions.transFigure.inverted())
fig_distributions.tight_layout(rect=(0, bbox.y1, 1, 1), h_pad=0.5, w_pad=0.5)




#save the cumulants
df_ep_cumulants_exp.to_csv("cumulants.csv", index=False)

#save the histogram
plt.savefig("histograms_test.pdf",bbox_inches="tight")

