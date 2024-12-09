from main import *
import functions
from plots import *

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#from matplotlib import cm
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import string


#set up the gridspec

fig = plt.figure(figsize = (15,24)) #figsize = (width,height)

gs = gridspec.GridSpec(2, 1, height_ratios=[1,1],hspace=0.2)
gs0 = gridspec.GridSpecFromSubplotSpec(2, 5, height_ratios=[1,2], subplot_spec=gs[0], hspace=0.1, wspace=0.3)
gs1 = gridspec.GridSpecFromSubplotSpec(2, 5, height_ratios=[1,2], subplot_spec=gs[1], hspace=0.1, wspace=0.3)


#get elements of the array
idx = np.round(np.linspace(0, t_steps - 1, 10)).astype(int)

for i in enumerate(times_t0[idx]):
   if (i[0] ==0):
      title_panel = r"t = 0"
   elif (i[0] == 19):
      title_panel = r"$t=t_f$"
   else:
      title_panel = r"$t =$"+f"{i[0]}"+r"$/\ 9\ t_f$"	
   #gs = gs0 if (i[0]<5) else gs1
   plot_pair(i[1],title_panel,["("+string.ascii_lowercase[(i[0]//10)+i[0]]+")","("+string.ascii_lowercase[(i[0]//10)+i[0]+10]+")"],
                   (gs0 if (i[0]<5) else gs1),i[0]%5)


#make legend
l0 = mlines.Line2D([], [], color=c1, lw = lw)
l1 = mlines.Line2D([], [], color=c3, lw = lw)
p0 = mpatches.Patch(color = c2, alpha=1)


plt.legend([l0,l1,p0],["Underdamped","Overdamped","Assigned Boundary Conditions"],
              prop = {"size": fontsizetitles},
              ncol = 3,
              bbox_to_anchor=(0.7, -0.15),
              frameon = False)

#plt.savefig("ep_land_drift_v1.eps")
plt.savefig("ep_land_drift_v1.pdf")
#plt.savefig("ep_land_drift_v39.png")
plt.close()
