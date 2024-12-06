"""
This file stores all the plotting functions

"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.neighbors import KernelDensity
import scipy.stats as stats

import string
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

#dist title location
disttitlex = -2.8
disttitley = 0.52

# Plotting the graphs)
#COLORS
#c2 = "#a6cee3" #lightblue
#c3 = "orange" #"#33a02c" #dark green
#c1 = "#1f78b4" #darkblue
#c4 = "#b2df8a" #light green

def format_axes(ax,fontsize):

  ax.set_xlim((0,T))
  ax.set_xlabel(r"$\mathrm{t}$",fontsize = fontsizetitles)

  #ax.set_xticklabels(labels = [0,1,2,3,4,5],fontsize=10)
  ax.tick_params(axis='y', labelsize=fontsizeticks)
  ax.tick_params(axis='x', labelsize=fontsizeticks)
  return ax

def format_drift(ax):

  #ax.yaxis.tick_right()
  ax.tick_params(axis='x', labelsize=fontsizeticks)
  ax.tick_params(axis='y', labelsize=fontsizeticks)

  #ax.set_ylim((-30,30))
  ax.set_xlim((-3,3))
  ax.set_xlabel(r"$\mathrm{q}$",fontsize= fontsize,labelpad=7)

##plot set-up
def format_dist_axes(ax):

  #ax.patch.set_alpha(0)
  #ax.yaxis.tick_right()
  ax.tick_params(axis='x', labelsize=fontsizeticks)
  ax.tick_params(axis='y', labelsize=fontsizeticks)
  #ax.tick_params(labeltop='off', labelright='off')

  ax.set_ylim((-0.01,0.6))
  ax.set_xlim((-3,3))
  ax.spines['bottom'].set_zorder(1000)

  #ax.set_xticks([])
  #ax.set_xticklabels([])
  #ax.set_xlabel(r"$\mathrm{q}$",fontsize= fontsize)

##plot set-up
def format_scatter_axes(ax):

  ax.set_xlim((-4,3))
  ax.set_ylim((-3,3))

  ax.tick_params(axis='both', labelsize=fontsizeticks)

#axes formatting
def format_dist_axes(ax):

  #ax.patch.set_alpha(0)
  #ax.yaxis.tick_right()
  ax.tick_params(axis='x', labelsize=fontsizeticks)
  ax.tick_params(axis='y', labelsize=fontsizeticks)
  #ax.tick_params(labeltop='off', labelright='off')

  ax.set_ylim((-0.01,0.8))
  ax.spines['bottom'].set_zorder(1000)

#####-----DRIFT AND DISTRIBUTION PLOTS-----#####

# Plotting the graphs)
#COLORS
c2 = "#a6cee3" #lightblue
c3 = "orange" #"#33a02c" #dark green
c1 = "#1f78b4" #darkblue
c4 = "#b2df8a" #light green




xtitle = 0.07
ytitle = 0.88



def cleaner(arr,t0):

  #masks nans and infs and returns a pair for plotting
  #copy q axis
  #plotq = np.copy(q_axis)

  masknan = functions.get_rhomask(t0,1e-3)

  #this function removes nan's and the end points which come from the truncation of the gradients
  #arr = arr[np.min(masknan)+500:np.max(masknan)-300]
  #plotq = q_axis[np.min(masknan)+500:np.max(masknan)-300]
  return q_axis[masknan] , arr[masknan]


def plot_pair(tcurr,title,labels,gs,locy):

  from scipy.ndimage import median_filter,generic_filter
  import scipy.ndimage as sc  # t0 distribution
  plt.subplot(gs[0,locy])
  plt.title(title, loc = "center", fontsize=fontsizetitles)

  plt.plot(q_axis,functions.rho(tcurr),color=c3,lw=4)
  plt.plot(q_axis,functions.distribution(tcurr),color=c1,lw=3)
  #plt.plot(q_axis,generic_filter(functions.distribution(tcurr),sc.median,size=150,mode="constant"),color=c1,lw=3, label =r"$\mathrm{t} = 0$",zorder = 10000)


  ax = plt.gca()
  format_dist_axes(ax)
  ax.text(s = labels[0],fontsize = fontsizetitles,x = disttitlex, y =disttitley,zorder = 1000)

  ax.set_xticklabels([])
  ax.tick_params(axis='y', labelsize=fontsizeticks)

  ax0 = plt.subplot(gs[1, locy])
  plot_data = cleaner(functions.optimal_drift(tcurr),tcurr)

  #this just removes the areas of low statistics rho < tol
  qseries =  plot_data[0]
  yseries = plot_data[1]#generic_filter(plot_data[1],sc.median,size=10,mode="nearest")
  sigma_data = cleaner(functions.dsigma(tcurr),tcurr)
  sigmaseries = -sigma_data[1]#-functions.dsigma(tcurr)

  #ax0 = plt.gca()
  format_drift(ax0)
  ax0.text(s = labels[1],fontsize = fontsizetitles,x = 0.05, y =-0.25, zorder = 1000,transform=ax.transAxes)

  series1a, = ax0.plot(qseries,yseries,color = c1,lw=lw,label = r"Underdamped",zorder = 100)
  series1b, = ax0.plot(qseries,sigmaseries,color = c3,lw=lw,label = r"Overdamped")

  ax0.set_ylim((-45,30))
  #if gs == gs0:
  #  #ax.set_ylim((-250,0))
  #  ax0.set_ylim((-260,0))
  #else:
  #  ax0.set_ylim((-20,260))

  if locy ==0:
    ax.set_ylabel(r'$\mathrm{f}_{\mathrm{t}}(\mathrm{q})$',fontsize = fontsizetitles,labelpad= 7)
    ax0.set_ylabel(r'$-\partial U_{\mathrm{t}}(\mathrm{q})$',fontsize = fontsizetitles,labelpad= -5)
    if tcurr == 0: #gs == gs0:
      #for edges
      ax0.set_ylim((-270,0))
      ax.fill_between(q_axis,p_initial(q_axis),color = c2)
  #else:
  #   ax0.set_ylim((-45,30))
  #  ax0.set_yticklabels([])
  #  ax.set_yticklabels([])

  if tcurr == T: #locy ==-1 and gs == gs1:
    ax.fill_between(q_axis,p_final(q_axis),color = c2)
    ax0.set_ylim((-40,300))





##########-------HISTOGRAMS PLOTS-------##################

def plot_distributions_ep(fig,gs,plot_index,underdamped_data,overdamped_data,tcurr):

  """
  Function that plots the histogram, underdamped and overdamped distributions in a lil square

  input:
  -plot_index: where to put the plot
  -histogram data
  -current t0
  """


  x_ind = int(np.floor(plot_index/3))
  y_ind = plot_index % 3
  #get plot location
  ax = fig.add_subplot(gs[x_ind,y_ind])
  ax.set_title(f"t = {tcurr}", loc = "center", fontsize=fontsizetitles)
  ax.text(-2.4,0.5,"("+string.ascii_lowercase[plot_index]+")",fontsize = fontsizetitles)


  #plot the histograms
  ax.hist(underdamped_data, range=(-3,3), color = c1,bins = 60,density = True,alpha=0.6)

  #fit kde of the samples
  kde = KernelDensity(kernel='epanechnikov', bandwidth=0.20).fit(underdamped_data.reshape(-1, 1))

  #estimated pdf
  kde_estimate = np.exp(kde.score_samples(q_axis.reshape(-1, 1)))

  ax.plot(q_axis,functions.rho(tcurr),color="orange",lw=lw)
  ax.plot(q_axis,functions.distribution(tcurr),color="midnightblue",lw=lw,  label =r"$T=2$")
  ax.plot(q_axis,kde_estimate,color=c1,lw=lw)

  #format the axes
  format_dist_axes(ax)
  #overwrite axes lims
  ax.set_ylim((-0.01,0.6))
  ax.set_xlim((-2.5,2.5))

  ax.tick_params(axis='y', labelsize=fontsizeticks)
  if x_ind !=0:
    ax.set_xlabel(r"$q$",fontsize = fontsizetitles)

  if y_ind == 0:
    ax.set_ylabel(r"$\rho_t(q)$",fontsize = fontsizetitles)


##########--------------##################
#GIRSANOV JOINT DISTRIBUTION PLOT

def joint_distributions_scatter(fig,gs,
                                plot_index,
                                joint_out,
                                Q,P,
                                time,vmax):

  """
  Function that plots the scatter plot of the monte carlo approximation of the joint distribution

  input:
  - fig: the figure for plotting
  - gs: the gridspec, should be attached to the fig
  - plot_index: location of plot
  - joint_out: joint distribution as a numpy array
  - Q: q-coordinate
  - P: p-coordinate
  - time
  - vmax: maximum value for the scatter plots
  """

  ##try to get the cumulants
  try:
    df_ep_cumulants = pd.read_csv("cumulants.csv",header=0)
    cumulants_exist = True

  except:
    print("Cumulants file not given. Check the filename.\n")
    print("The plots will not have estimated marginals for the momentum.\n")
    cumulants_exist = False


  qmin = np.min(Q)
  qmax = np.max(Q)
  pmin = np.min(P)
  pmax = np.max(P)

  x_ind = int(np.floor(plot_index/3))
  y_ind = plot_index % 3

  plot_title_value = round(time/T,2)

  #get plot location
  ax = fig.add_subplot(gs[x_ind,y_ind])

  ax.set_xlim((pmin,pmax))
  ax.set_ylim((qmin,qmax))

  ax_pmarginal = ax.inset_axes([0, 1.05, 1, 0.6])
  ax_qmarginal = ax.inset_axes([-0.65, 0, 0.6, 1])

  ax_pmarginal.text(-9,0.4,"("+string.ascii_lowercase[plot_index]+")",fontsize = fontsizetitles,zorder = 200)

  #compute joint distribution and set nans to zero
  joint_out[np.isnan(joint_out)] = 0

  pmarginal = np.nansum(joint_out.reshape((P.shape[1],P.shape[0])),axis=0)
  pnorm = np.trapz(pmarginal,P[0])

  if cumulants_exist:
    mom_mean_temp = df_ep_cumulants[((df_ep_cumulants["g"]==g) & (df_ep_cumulants["t0"]==time))].mom_mean.values
    mom_var_temp = df_ep_cumulants[((df_ep_cumulants["g"]==g) & (df_ep_cumulants["t0"]==time))].mom_var.values
    ax_pmarginal.plot(P[0], stats.norm.pdf(P[0], mom_mean_temp,np.sqrt(mom_var_temp)),color="green",linestyle="dashed",lw=lw)


  ax_pmarginal.plot(P[0],pmarginal/pnorm
                    ,color=c1,lw=lw)


  qmarginal = np.nansum(joint_out.reshape((P.shape[1],P.shape[0])),axis=1)#[np.nansum(joint_out[i::samples]) for i in range(0,samples)]
  #qorder = np.argsort(q_init)
  qnorm = np.trapz(qmarginal,Q.T[0])

  ax_qmarginal.plot(functions.distribution(time),q_axis,
                    color="orange",lw=lw)
  ax_qmarginal.plot(qmarginal/qnorm
                    ,Q.T[0],color=c1,lw=lw)

  format_dist_axes(ax_pmarginal)

  ax_pmarginal.set_xlim((pmin,pmax))
  ax_pmarginal.set_ylim((-0.05,0.55))

  ax_qmarginal.set_ylim((qmin,qmax))
  ax_qmarginal.set_xlim((-0.05,0.6))

  #plot scatter graphs
  #zout = joint_out

  order = np.argsort(joint_out.flatten())
  pout = P.flatten().flatten()
  qout = Q.flatten().flatten()
  ax.scatter(pout[order],qout[order],c=joint_out.flatten()[order]
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
      labelsize = fontsizeticks, pad=-3,rotation = 45,
      length =5,
      labelleft=False)
  ax_qmarginal.tick_params(
      axis='y',          # changes apply to the x-axis
      which='both',
      left=False,
      labelleft=False)
  ax.tick_params(axis="x", which="both",rotation = 45)

  #plt.setp(ax_qmarginal.get_xticklabels(),
  #    rotation=90, va="top", rotation_mode="anchor")
  ax.tick_params(axis='both', labelsize=fontsizeticks)

  #add labels to outside and remove ticks from inside plots
  ax_qmarginal.set_yticklabels([])
  ax_pmarginal.set_xticklabels([])
  ax.yaxis.set_label_position("right")
  ax.yaxis.tick_right()
  ax_qmarginal.xaxis.set_label_position("top")
  ax_qmarginal.set_xlabel(r"$\rho_t(q)$",fontsize = fontsizetitles,labelpad = 7)

  if x_ind !=0:
    ax.set_xlabel(r"$p$",fontsize = fontsizetitles)

  ax_pmarginal.set_ylabel(r"$\rho_t(p)$",fontsize = fontsizetitles)
  
  if y_ind == 2:
    #make label and add text
    ax.set_ylabel(r"$q$",fontsize = fontsizetitles)
    ax.yaxis.set_label_position("right")
    
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

  plt.setp(ax_qmarginal.xaxis.get_majorticklabels(), ha="right")
  ax_qmarginal.invert_xaxis()
  #ax.set_aspect('equal', adjustable='box')
  plt.setp(ax.xaxis.get_majorticklabels(), ha="left")


