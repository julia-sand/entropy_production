
"""
This file evolves the trajectories and plots sample histograms using the approximate underdamped drift

Note: the number of trajectories is ALSO called mc_samples, as in the Girsanov joint file. 

The default is 10000 samples. 
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

from src.utils.params import *
import src.utils.functions as functions
from src.utils.plots import *

def make_legend():
  """
  Makes a legend for the histogram plots
  returns: 
    legend:
      mpl artist object
  """ 
  
  #add legend
  orange_line = mlines.Line2D([], [],color="orange",lw=lw)
  blue_line = mlines.Line2D([], [],color=c1,lw=lw)
  darkblue_line = mlines.Line2D([], [],color=c2,lw=lw)
  legend = fig_distributions.legend(handles=[blue_line,darkblue_line,orange_line],
            labels = ["Underdamped (Evolved)","Underdamped (Predicted)","Overdamped (Evolved)"],
            frameon = False,
            handlelength = 1,
            ncols = 3,
            loc="upper center", bbox_to_anchor=(0.5, 0))

  return legend

def save_cumulant_step(df,i,g,q_evo_UD_prev,p_evo_UD_prev,x_evo):
    covmat = np.cov(q_evo_UD_prev,p_evo_UD_prev)

    #compute mean and variance of p and q using MLE
    df.loc[len(df)] = [g, times_t0[i+1],
                                                            covmat[0,0],
                                                              covmat[1,1],
                                                              np.nanmean(p_evo_UD_prev),
                                                              np.nanmean(q_evo_UD_prev),
                                                              covmat[0,1],
                                                              np.nanmean(x_evo),
                                                              np.nanvar(x_evo)]


def plot_histograms():
  """
  Simulates stochastic evolution of underdamped dynamics under predicted perturbative drft
  
  output:
    saves csv with simulated 
    saved plot of histograms of the distributions at final time
  """

  # Plotting the distributions
  fig_distributions = plt.figure(figsize=(15,10))

  #number of plots
  nplots = 8

  #create plot grid
  gs_distributions = fig_distributions.add_gridspec(2, int(round(nplots/2)), width_ratios=[1,1, 1, 1], height_ratios=[1, 1])

  #set up dataframe to save cumulants 
  df_ep_cumulants_exp = pd.DataFrame(columns = ["g","t0","pos_var","mom_var","mom_mean","pos_mean","xcorr","OD_mean","OD_var"])

  ##Evolve the underdamped dynamics using EM SCHEME
  plot_index = 0
  
  #initial position sampled from assigned initial dist.
  x_evo = npr.choice(np.linspace(-10,10,mc_samples*10), size = mc_samples, p = p_initial(np.linspace(-10,10,mc_samples*10))/np.sum(p_initial(np.linspace(-10,10,mc_samples*10))))
  q_evo_UD_prev = x_evo 
  p_evo_UD_prev = npr.randn(mc_samples) #momentum, independent standard Gaussian samples

  save_cumulant_step(df_ep_cumulants_exp,0,g,q_evo_UD_prev,p_evo_UD_prev,x_evo)
  
  #get elements of the array
  idx = np.round(np.linspace(0, t_steps - 1, nplots)).astype(int)

  times_to_save = [0,0.5,0.75,1.0,1.25,1.5,1.75,2]
  #times_t0[idx]
  #0,0.25,0.5,0.75,1,,0.8,1]
  times_to_save = np.round(times_to_save,4)

  for i in range(0,len(times_t0)-1):
  
    #plot the selected times
    if (times_t0[i] in times_to_save):
      plot_distributions_ep(fig_distributions,gs_distributions,
                            plot_index,q_evo_UD_prev,x_evo,times_t0[i],nplots)
      plot_index += 1

    #overdamped evolution step in t2
    x_evo = x_evo - (h_step)*functions.dsigma_interp(times_t0[i],x_evo,1e-5) + np.sqrt(2*h_step)*npr.randn(mc_samples)

    #underdamped step in t0
    #h0_step = h_step/(epsilon**2)
    q_evo_UD_prev = q_evo_UD_prev + epsilon*h0_step*(p_evo_UD_prev-g*epsilon*functions.underdamped_drift_interp(times_t0[i],q_evo_UD_prev,g,1e-5)) + epsilon*np.sqrt(2*g*h0_step)*npr.randn(mc_samples)
    p_evo_UD_prev = p_evo_UD_prev - (p_evo_UD_prev + epsilon*functions.underdamped_drift_interp(times_t0[i],q_evo_UD_prev,g,1e-5))*h0_step + np.sqrt(2*h0_step)*npr.randn(mc_samples)

    #remove escaped particles
    #q_evo_UD_prev[q_evo_UD_prev>30] = np.nan
    #q_evo_UD_prev[q_evo_UD_prev<-30] = np.nan
    #p_evo_UD_prev[p_evo_UD_prev>30] = np.nan
    #p_evo_UD_prev[p_evo_UD_prev<-30] = np.nan

    #compute and save cumulants to dataframe
    save_cumulant_step(df_ep_cumulants_exp,i,g,q_evo_UD_prev,p_evo_UD_prev,x_evo)

  #add final time plot
  plot_distributions_ep(fig_distributions,gs_distributions,plot_index,q_evo_UD_prev,x_evo,T,nplots)

  legend = make_legend()

  #adjust legend positioning
  bbox = legend.get_window_extent(fig_distributions.canvas.get_renderer()).transformed(fig_distributions.transFigure.inverted())
  fig_distributions.tight_layout(rect=(0, bbox.y1, 1, 1), h_pad=2, w_pad=2)

  #save simulated cumulants as csv 
  filename_temp = "cumulants" +f"{fileid}"+".csv"
  df_ep_cumulants_exp.to_csv(filename_temp, index=False)

  #save the histogram plots
  filename_temp = "histograms_test" +f"{fileid}"+".pdf"
  plt.savefig(filename_temp,bbox_inches="tight")


if __name__=="__main__":
  plot_histograms()