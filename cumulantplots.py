from main import *
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#formatting options
from plots import *

import functions

df_ep_cumulants = pd.read_csv("cumulants.csv",header=0)
cumulants_perturbative = pd.read_csv("cumulantscalculated.csv",header=0)
cumulant_plot_times = df_ep_cumulants.t0.unique()
cumulant_plot_times.sort()

# Plotting the cumulants
plt.figure(figsize=(15, 8))#, constrained_layout=True)

# Create a 1x5 grid with different widths for the bottom row
gs_cumulants = gridspec.GridSpec(2, 6, width_ratios=[1, 1, 1, 1,1,1], height_ratios=[1, 1])
                              #hspace = 0.3,wspace = 2.3)
                              #(1, 2, width_ratios=[6, 4], figure=fig)

# position mean
plt.subplot(gs_cumulants[0, 0:2])
plt.plot(times_t0,  cumulants_perturbative[cumulants_perturbative.g==g].pos_mean,lw=lw,color=c1)
plt.plot(times_t0, [functions.mean_t0(t0) for t0 in times_t0],lw=lw,color="orange")
plt.plot(cumulant_plot_times,  df_ep_cumulants[df_ep_cumulants.g==g].pos_mean,lw=lw,color=c3)
plt.title('(a)',fontweight = "bold",fontsize = fontsizetitles,pad = titlepad,x = titlex, y =titley,zorder = 1000000)
ax = format_axes(plt.gca(),fontsize)
#ax.set_ylim((-0.05,1.2))
ax.set_ylabel('Position Mean',fontsize = fontsizetitles)

# position variance
plt.subplot(gs_cumulants[0, 2:4])
plt.plot(times_t0,  cumulants_perturbative[cumulants_perturbative.g==g].pos_var,lw=lw,color=c1)
plt.plot(times_t0, [functions.var_t0(t0) for t0 in times_t0],lw=lw,color="orange")
plt.plot(cumulant_plot_times, df_ep_cumulants[df_ep_cumulants.g==g].pos_var,lw=lw,color=c3)
plt.title('(b)',fontweight = "bold",fontsize = fontsizetitles,pad = titlepad,x = titlex, y =titley,zorder = 1000000)
ax = format_axes(plt.gca(),fontsize)
ax.set_ylabel('Position Variance',fontsize = fontsizetitles)

# cross corr
plt.subplot(gs_cumulants[0, 4:])
plt.plot(times_t0,  cumulants_perturbative[cumulants_perturbative.g==g].xcorr,lw=lw,color=c1)
plt.plot(cumulant_plot_times, df_ep_cumulants[df_ep_cumulants.g==g].xcorr,lw=lw,color=c3)
plt.title('(c)',fontweight = "bold",fontsize = fontsizetitles,pad = titlepad,x = titlex, y =titley,zorder = 1000000)
ax = format_axes(plt.gca(),fontsize)
ax.set_ylabel('Cross Correlation',fontsize = fontsizetitles)


# momentum mean
plt.subplot(gs_cumulants[1,:3])
plt.plot(times_t0,  cumulants_perturbative[cumulants_perturbative.g==g].mom_mean,lw=lw,color=c1)
plt.plot(cumulant_plot_times, df_ep_cumulants[df_ep_cumulants.g==g].mom_mean,lw=lw,color=c3)
#plt.plot(times_t0, [mom_mean_alt(t0) for t0 in times_t0],lw=lw)
plt.title('(d)',fontweight = "bold",fontsize = fontsizetitles,pad = titlepad,x = titlex*(2/3), y =titley,zorder = 1000000)
ax = format_axes(plt.gca(),fontsize)
ax.set_ylabel('Momentum Mean',fontsize = fontsizetitles)

# momentum variance
mom_var = plt.subplot(gs_cumulants[1, 3:])
plt.plot(times_t0,  cumulants_perturbative[cumulants_perturbative.g==g].mom_var,lw=lw,color=c1)
plt.plot(cumulant_plot_times,  df_ep_cumulants[df_ep_cumulants.g==g].mom_var,lw=lw,color=c3)
plt.title('(e)',fontweight = "bold",fontsize = fontsizetitles,pad = titlepad,x = titlex*(2/3), y =titley,zorder = 1000000)
ax = format_axes(plt.gca(),fontsize)
ax.set_ylabel('Momentum Variance',fontsize = fontsizetitles)

plt.tight_layout()


#get legend
orange_line = mlines.Line2D([], [],color="orange",lw=lw)
blue_line = mlines.Line2D([], [],color=c1,lw=lw)
green_line = mlines.Line2D([], [],color=c3,lw=lw,linestyle="dashed")

legend = fig_joint_distributions_meshgrid.legend(handles=[blue_line,orange_line,green_line],
          labels = ["Overdamped","Underdamped (Perturbative)","Underdamped (From Evolution)"],
          fontsize = fontsizeticks,
          frameon = False,
          handlelength = 1,
          ncols = 3,
          bbox_to_anchor=(0.75,0.2))

#move the legend
legend.set_bbox_to_anchor(bbox=(0.95,0.2))

plt.savefig("ep_cumulants.pdf",bbox_inches="tight")
plt.close()
