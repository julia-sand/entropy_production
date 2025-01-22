import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable

from setup.main import *
from setup.datafetch import *
from setup.plots import *

#what times to plot
#plot_times = np.array([2,1.5,1.0,0.5,0.25,0])#np.flip([0,1,2,3,4,5])/(5/T)

#plot_titles = [f"$t = {plot_times[j]}$" for j in range(0,len(plot_times))]
#plot_titles = np.flip(plot_titles)


filename_temp = "ep_girsanovjoint" +f"{fileid}"+".csv"

df_girspdf_ep = pd.read_csv(filename_temp, sep=" ", header = 0)

#get plot times from csv
plot_times = df_girspdf_ep.t.unique()
plot_titles = [f"$t = {plot_times[j]}$" for j in range(0,len(plot_times))]
#plot_titles = np.flip(plot_titles)

p_init = df_girspdf_ep[df_girspdf_ep["t"] == 0].P.unique()
q_init = df_girspdf_ep[df_girspdf_ep["t"] == 0].Q.unique()

P,Q = np.meshgrid(p_init,q_init)

#make the plot
vmax = np.max(df_girspdf_ep.ptx)

# Plotting the distributions -initialise the figure object & creates the gridspec
fig_joint_distributions_meshgrid = plt.figure(figsize=(15,10))
gs_joint_distributions = fig_joint_distributions_meshgrid.add_gridspec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1])

for k in enumerate(plot_times):
  print(k)
  joint_distributions_scatter(fig_joint_distributions_meshgrid,gs_joint_distributions, 5-k[0],
                                    df_girspdf_ep[df_girspdf_ep["t"] == k[1]].ptx.to_numpy(),
                                    Q,P,
                                    k[1],vmax)



#adjust spacing
fig_joint_distributions_meshgrid.subplots_adjust(wspace=1.1,# The width of the padding between subplots, as a fraction of the average Axes width.
                                                hspace=1.1# The height of the padding between subplots, as a fraction of the average Axes height.
                                                )

#add the legend
sm =  ScalarMappable(norm="log",cmap=plt.get_cmap("Blues"))
sm.set_array([])
sm.set_clim(vmin=0.001, vmax=vmax)
sm2 =  ScalarMappable(norm="log",cmap=plt.get_cmap("viridis"))
sm2.set_array([])
sm2.set_clim(vmin=0.001, vmax=vmax)


cbar_ax = fig_joint_distributions_meshgrid.add_axes([0.2, -0.1, 0.55, 0.05])
cb = fig_joint_distributions_meshgrid.colorbar(sm,
                                          cax=cbar_ax,
                                          orientation="horizontal")

cbar_ax2 = fig_joint_distributions_meshgrid.add_axes([0.2, -0.1, 0.55, 0.05])
cb2 = fig_joint_distributions_meshgrid.colorbar(sm2,
                                          cax=cbar_ax2,
                                          orientation="horizontal")

cb.ax.tick_params(labelsize=fontsizeticks)
cb.ax.set_xlabel("Joint Distribution",fontsize=fontsizetitles)
cb2.ax.tick_params(labelsize=fontsizeticks)
cb2.ax.set_xlabel("Boundary Conditions",fontsize=fontsizetitles)

#get legend
blue_line = mlines.Line2D([], [],color=c1,lw=lw)
orange_line = mlines.Line2D([], [],color="orange",lw=lw)
green_line = mlines.Line2D([], [],color="green",lw=lw,linestyle="dashed")

legend = fig_joint_distributions_meshgrid.legend(handles=[blue_line,orange_line,green_line],
          labels = ["Monte Carlo w. Girsanov","Perturbative Prediction","Gaussian Prediction"],
          fontsize = fontsizetitles,
          frameon = False,
          handlelength = 1,
          ncols = 3,
          bbox_to_anchor=(0.75,0.2))

#move the legend
legend.set_bbox_to_anchor(bbox=(0.95,0.2))


fileout = "ep_jointpdf_" +f"{fileid}"+".pdf"

plt.savefig(fileout,bbox_inches="tight")
