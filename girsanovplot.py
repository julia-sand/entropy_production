import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from main import *
from datafetch import *
import plots

#what times to plot
plot_times = np.array([2,1.5,1.0,0.5,0.25,0])#np.flip([0,1,2,3,4,5])/(5/T)

plot_titles = [f"$t = {plot_times[j]}$" for j in range(0,len(plot_times))]
plot_titles = np.flip(plot_titles)



df_girspdf_ep = pd.read_csv("ep_girsanovjoint.csv", sep=" ", header = 0)
p_init = df_girspdf_ep[df_girspdf_ep["t"] == 0].P.unique()
q_init = df_girspdf_ep[df_girspdf_ep["t"] == 0].Q.unique()

P,Q = np.meshgrid(p_init,q_init)

#make the plot
vmax = np.max(df_girspdf_ep.ptx)

# Plotting the distributions -initialise the figure object & creates the gridspec
fig_joint_distributions_meshgrid = plt.figure(figsize=(15,10))
gs_joint_distributions = fig_joint_distributions_meshgrid.add_gridspec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1])

for k in enumerate(plot_times):
  plots.joint_distributions_scatter(fig_joint_distributions_meshgrid,gs_joint_distributions, 5-k[0],
                              df_girspdf_ep[df_girspdf_ep["t"] == k[1]].ptx.to_numpy(),
                              Q,P,
                              k[1],vmax)


#adjust spacing
fig_joint_distributions_meshgrid.subplots_adjust(
     wspace=0.85,# The width of the padding between subplots, as a fraction of the average Axes width.
    hspace=0.65# The height of the padding between subplots, as a fraction of the average Axes height.
)

plt.savefig("test.pdf",bbox_inches="tight")
