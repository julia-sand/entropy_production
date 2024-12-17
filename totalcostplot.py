from plots import *
from main import *
import pandas as pd

#gridspec
gs_costs = gridspec.GridSpec(3, 1, width_ratios=[1], height_ratios=[1.3, 1, 1])


df_ep_cost_all = pd.read_csv("ep_costs.csv")

#format axes
def format_log_axes(ax):
  ax.set_xscale('log')
  ax.invert_xaxis()
  #ax.set_ylim((4.6,5.3))
  ax.set_xlim((0.11,(8e-7)))
  ax.tick_params(axis='y', labelsize=fontsizeticks)
  #ax.tick_params(axis='x', labelsize=fontsizeticks)
  #ax.set_ylabel(r"$\mathcal{E}$",fontsize = fontsizetitles)

fig1 = plt.figure(figsize = (15,6))

plt.subplot(gs_costs[0])
plt.plot(df_ep_cost_all[df_ep_cost_all.Tf==2].g,df_ep_cost_all[df_ep_cost_all.Tf==2].EPcost,lw=lw,label=r"EP Cost")#,yerr=0.04,ecolor="black",capsize=4,elinewidth=1,label=r"EP cost $\pm 0.04$")
plt.plot(df_ep_cost_all[df_ep_cost_all.Tf==2].g,df_ep_cost_all[df_ep_cost_all.Tf==2].Firstterm,lw=lw,label=r"Overdamped Bound")#,yerr=0.04,ecolor="black",capsize=4,elinewidth=1,label=r"EP cost $\pm 0.04$")
plt.plot(df_ep_cost_all[df_ep_cost_all.Tf==2].g,(1+df_ep_cost_all[df_ep_cost_all.Tf==2].g)*df_ep_cost_all[df_ep_cost_all.Tf==2].ODBound,linestyle="dashed",lw=lw,label=r"$\mathcal{W}_2$-distance")#,yerr=0.04,ecolor="black",capsize=4,elinewidth=1,label=r"EP cost $\pm 0.04$")
ax = plt.gca()
format_log_axes(ax)
ax.set_xticklabels([]) #remove x axis lables
#ax.set_ylabel(r"$\mathcal{E}$",fontsize = fontsizetitles)
ax.minorticks_off()
ax2 = ax.twinx()
ax2.set_ylabel(r"$\mathrm{t}_f=2$",rotation=270,fontsize = fontsizetitles,labelpad = 30)
ax2.set_yticks([])

plt.subplot(gs_costs[1])
plt.plot(df_ep_cost_all[df_ep_cost_all.Tf==5].g,df_ep_cost_all[df_ep_cost_all.Tf==5].EPcost,label=r"EP Cost",lw=lw)#,yerr=0.04,ecolor="black",capsize=4,elinewidth=1,label=r"EP cost $\pm 0.04$")
plt.plot(df_ep_cost_all[df_ep_cost_all.Tf==5].g,df_ep_cost_all[df_ep_cost_all.Tf==5].Firstterm,label=r"Overdamped Bound",lw=lw)#,yerr=0.04,ecolor="black",capsize=4,elinewidth=1,label=r"EP cost $\pm 0.04$")
plt.plot(df_ep_cost_all[df_ep_cost_all.Tf==5].g,(1+df_ep_cost_all[df_ep_cost_all.Tf==5].g)*df_ep_cost_all[df_ep_cost_all.Tf==5].ODBound,linestyle="dashed",label=r"$\mathcal{W}_2$-distance",lw=lw)#,yerr=0.04,ecolor="black",capsize=4,elinewidth=1,label=r"EP cost $\pm 0.04$")
format_log_axes(plt.gca())
ax = plt.gca()
ax.set_ylim((4.8,5.8))
ax.set_xticklabels([]) #remove x axis lables
ax.set_ylabel(r"Mean Entropy Production",fontsize = fontsizetitles,labelpad =35)
#ax.set_ylabel(r"$\mathcal{E}$", fontsize=fontsizetitles)
ax.minorticks_off()
ax2 = ax.twinx()
ax2.set_ylabel(r"$\mathrm{t}_f=5$",rotation=270, fontsize=fontsizetitles,labelpad = 30)
ax2.set_yticks([])

#####
plt.subplot(gs_costs[2])
plt.plot(df_ep_cost_all[df_ep_cost_all.Tf==50].g,df_ep_cost_all[df_ep_cost_all.Tf==50].EPcost,label=r"Average Entropy Production",lw=lw)#,yerr=0.04,ecolor="black",capsize=4,elinewidth=1,label=r"EP cost $\pm 0.04$")
plt.plot(df_ep_cost_all[df_ep_cost_all.Tf==50].g,df_ep_cost_all[df_ep_cost_all.Tf==50].Firstterm,label=r"Overdamped Bound",lw=lw)#,yerr=0.04,ecolor="black",capsize=4,elinewidth=1,label=r"EP cost $\pm 0.04$")
plt.plot(df_ep_cost_all[df_ep_cost_all.Tf==50].g,(1+df_ep_cost_all[df_ep_cost_all.Tf==50].g)*df_ep_cost_all[df_ep_cost_all.Tf==50].ODBound,linestyle="dashed",lw=lw,label=r"Squared $\mathcal{W}_2$-distance")#,yerr=0.04,ecolor="black",capsize=4,elinewidth=1,label=r"EP cost $\pm 0.04$")
ax = plt.gca()
format_log_axes(ax)
ax.set_xlabel(r"$g$",fontsize = fontsizetitles)
ax.set_ylim((0.5,0.585))
#ax.set_ylabel(r"$\mathcal{E}$",fontsize = fontsizetitles)
ax.tick_params(axis='x', labelsize=fontsizeticks)
ax.minorticks_off()
ax2 = ax.twinx()
ax2.set_ylabel(r"$\mathrm{t}_f=50$",rotation=270, fontsize=fontsizetitles,labelpad = 30)
ax2.set_yticks([])

#plt.subplots_adjust(left  = 0.2 , # the left side of the subplots of the figure
#                    right = 0.9  ,  # the right side of the subplots of the figure
#                    bottom = 0.12,   # the bottom of the subplots of the figure
#                    top = 0.95,      # the top of the subplots of the figure
#                    wspace = 0.2,   # the amount of width reserved for blank space between subplots
#                    hspace = 0.05)   # the amount of height reserved for white space between subplots


plt.tight_layout()
#get legend
orange_line = mlines.Line2D([], [],color="orange",lw=lw)
blue_line = mlines.Line2D([], [],color="royalblue",lw=lw)
green_line = mlines.Line2D([], [],color="green",lw=lw,linestyle="dashed")

legend = fig1.legend(handles=[blue_line,orange_line,green_line],
          labels=["Total Entropy Production","Overdamped Bound",r"$\mathcal{W}_2$-distance"],
          fontsize = fontsizetitles,
          frameon = False,
          handlelength = 1,
          ncols = 3)
          #bbox_to_anchor=(0.75,0.2))

#move the legend
legend.set_bbox_to_anchor(bbox=(0.95,0.1))

plt.savefig("ep_cost_all2.png",bbox_inches="tight")
plt.savefig("ep_cost_all.pdf",bbox_inches="tight")
plt.close()