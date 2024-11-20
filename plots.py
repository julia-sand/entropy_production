"""
This file stores all the plotting functions

"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines


from main import *
import functions

#fontsizes
fontsize = 22
fontsizeticks = 16
fontsizetitles = 22

#linewidth
lw = 3

#colors
c2 = "#a6cee3" #lightblue
c3 = "#33a02c" #dark green
c1 = "#1f78b4" #darkblue
c4 = "#b2df8a" #light green

#set up plots
#panel labels:
label_titles = ["(a)","(b)","(c)","(d)","(e)","(f)"]


# Plotting the distributions -initialise the figure object & creates the gridspec
#fig_joint_distributions_meshgrid = plt.figure(figsize=(15,10))
#gs_joint_distributions = fig_joint_distributions_meshgrid.add_gridspec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1])


#axes formatting
def format_dist_axes(ax):

  #ax.patch.set_alpha(0)
  #ax.yaxis.tick_right()
  ax.tick_params(axis='x', labelsize=fontsizeticks)
  ax.tick_params(axis='y', labelsize=fontsizeticks)
  #ax.tick_params(labeltop='off', labelright='off')

  ax.set_ylim((-0.01,0.8))
  #ax.set_xlim((xlimmin,xlimmax))
  ax.spines['bottom'].set_zorder(1000)


#function for plotting
def joint_distributions_scatter(fig,gs,
                                plot_index,
                                joint_out,
                                Q,P,
                                time,vmax):

  """
  Function that plots the scatter plot of the monte carlo approximation of the joint distribution

  input:
  - fig: the figure for plotting
  - plot_index: location of plot
  - joint_out: joint distribution as a numpy array
  - Q: q-coordinate
  - P: p-coordinate
  - time
  - vmax: maximum value for the scatter plots
  """

  qmin = -8
  qmax = 8
  pmin = -8
  pmax = 8

  x_ind = int(np.floor(plot_index/3))
  y_ind = plot_index % 3

  plot_title_value = round(time/T,2)

  #get plot location
  ax = fig.add_subplot(gs[x_ind,y_ind])

  ax_pmarginal = ax.inset_axes([0, 1.05, 1, 0.6])
  ax_qmarginal = ax.inset_axes([1.1, 0, 0.6, 1])

  ax_pmarginal.text(-2.85,0.5,label_titles[plot_index],fontsize = fontsizetitles,zorder = 200)

  #compute joint distribution and set nans to zero
  joint_out[np.isnan(joint_out)] = 0

  pmarginal = np.nansum(joint_out.reshape((P.shape[1],P.shape[0])),axis=0)
  pnorm = np.trapz(pmarginal,P[0])
  #mom_mean_temp = df_ep_cumulants[((df_ep_cumulants["g"]==g) & (df_ep_cumulants["t0"]==time))].mom_mean.values
  #mom_var_temp = df_ep_cumulants[((df_ep_cumulants["g"]==g) & (df_ep_cumulants["t0"]==time))].mom_var.values
  #ax_pmarginal.plot(P[0], stats.norm.pdf(P[0], mom_mean_temp,np.sqrt(mom_var_temp)),color="orange",linestyle="dashed",lw=lw)
  ax_pmarginal.plot(P[0],pmarginal/pnorm
                    ,color=c1,lw=lw)


  qmarginal = np.nansum(joint_out.reshape((P.shape[1],P.shape[0])),axis=1)#[np.nansum(joint_out[i::samples]) for i in range(0,samples)]
  #qorder = np.argsort(q_init)
  qnorm = np.trapz(qmarginal,Q.T[0])

  ax_qmarginal.plot(functions.distribution(time),functions.q_axis(time),
                    color="orange",lw=lw)
  ax_qmarginal.plot(qmarginal/qnorm
                    ,Q.T[0],color=c1,lw=lw)

  format_dist_axes(ax_pmarginal)

  ax_pmarginal.set_xlim((pmin,pmax))
  ax_pmarginal.set_ylim((-0.05,0.7))

  ax_qmarginal.set_ylim((qmin,qmax))
  ax_qmarginal.set_xlim((-0.05,0.6))

  #plot scatter graphs
  zout = joint_out

  order = np.argsort(zout.flatten())
  pout = P.flatten().flatten()
  qout = Q.flatten().flatten()
  ax.scatter(pout[order],qout[order],c=zout.flatten()[order]
             ,cmap="Blues",norm="log",vmin=0.001,vmax=vmax)



  #ax.contour(np.meshgrid(functions.q_axis(time),functions.q_axis(time)),
  #           joint_distribution_perturbative(functions.q_axis(time),time))

  ###TICKS
  ax_pmarginal.yaxis.tick_right()

  ax_pmarginal.tick_params(
      axis='x',          # changes apply to the x-axis
      which='both',      # both major and minor ticks are affected
      bottom=False,      # ticks along the bottom edge are off
      top=False,         # ticks along the top edge are off
      labelbottom=False)
  ax_pmarginal.tick_params(
      axis='y',
      which='both',
      labelsize = fontsizeticks)
      #rotation = 70, pad=-5,
      #length =2 )
  ax_qmarginal.tick_params(
      axis='x',          # changes apply to the x-axis
      which='both',
      rotation=-70,
      labelsize = fontsizeticks, pad=-3,
      length =5,
      labelleft=False)
  ax_qmarginal.tick_params(
      axis='y',          # changes apply to the x-axis
      which='both',
      left=False,
      labelleft=False)

  ax.set_xlim((pmin,pmax))
  ax.set_ylim((qmin,qmax))

  ax.tick_params(axis='both', labelsize=fontsizeticks)

  #add labels to outside and remove ticks from inside plots
  ax_qmarginal.set_yticklabels([])
  ax_pmarginal.set_xticklabels([])

  if x_ind ==0:
    ax.set_xticklabels([])
    ax_qmarginal.set_xticklabels([])

  else:
    ax.set_xlabel(r"$p$",fontsize = fontsizetitles)
    ax_qmarginal.set_xlabel(r"$\rho_t(q)$",fontsize = fontsizetitles)

  if y_ind == 0:
    ax.set_ylabel(r"$q$",fontsize = fontsizetitles)
    ax_pmarginal.set_ylabel(r"$\rho_t(p)$",fontsize = fontsizetitles)
  else:
    ax.set_yticklabels([])
    #ax_pmarginal.set_yticklabels([])


  #add contour plots of the boundary conditions
  if plot_index ==0:
    #[X, Y] = np.meshgrid(q_axis(0), q_axis(0))

    # Creating 2-D grid of features
    Z = ud_pinitial(Q,P)

    # plots contour lines
    ax.contour(Q,P, Z,zorder =10000)
    ax.set_title(f"$t = 0$", loc = "center", fontsize=fontsizetitles)

  if plot_index == 5:

    # Creating 2-D grid of features
    #[X, Y] = np.meshgrid(q_axis(0), q_axis(0))
    Z = ud_pfinal(Q,P)

    # plots contour lines
    ax.contour(Q,P, Z,zorder =10000)
    ax.set_title(f"$t = t_f$", loc = "center", fontsize=fontsizetitles)

  if (0 <plot_index < 5):
    ax.set_title(f"$t = {plot_title_value}\ t_f$", loc = "center", fontsize=fontsizetitles)

  plt.setp(ax_qmarginal.xaxis.get_majorticklabels(), ha="left")
  ax.set_aspect('equal', adjustable='box')

