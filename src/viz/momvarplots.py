import pandas as pd
import matplotlib.pyplot as plt
from src.utils.plots import *

##momentum mean and var only

df_ep_cumulants = pd.read_csv("cumulantscalculated.csv")#pd.read_csv("ep_cumulants_g2.csv",index_col=0)

#round the g col
df_ep_cumulants = df_ep_cumulants.round({'g': 10})

times_t0 = df_ep_cumulants[df_ep_cumulants.g==0.1].t0.unique()
lw = 3


c1 = "#7fcdbb"
c2 = "#41b6c4"
c3 = "#2c7fb8"
c4 = "#253494"
#

fig = plt.figure(figsize = (15,7))

ax1 = plt.subplot(121)
#ax1 = ax.inset_axes([0.95,1.06,3.5,0.08], transform=ax.transData)
#ax1 = plt.gca()
ax1.plot(times_t0,df_ep_cumulants[df_ep_cumulants.g==0.1].mom_mean,label=r"$g=10^{-1}$",lw=lw,c = c1)
ax1.plot(times_t0,df_ep_cumulants[df_ep_cumulants.g==0.01].mom_mean,label=r"$g=10^{-2}$",lw=lw,c = c2)
ax1.plot(times_t0,df_ep_cumulants[df_ep_cumulants.g==0.001].mom_mean,label=r"$g=10^{-3}$",lw=lw,c = c3)
ax1.plot(times_t0,df_ep_cumulants[df_ep_cumulants.g==0.0001].mom_mean,label=r"$g=10^{-4}$",lw=lw,c = c4)
format_axes(ax1,"Momentum Mean")


ax1.set_xlim(-0.05,2.05)
#ax.set_ylim(0.97,1.155)
ax1.text(0.02,0.92,"(a)",  transform=ax1.transAxes)


plt.subplot(122)
#format axes
plt.plot(times_t0,df_ep_cumulants[df_ep_cumulants.g==0.1].mom_var,lw=lw,c = c1)
plt.plot(times_t0,df_ep_cumulants[df_ep_cumulants.g==0.01].mom_var,lw=lw,c = c2)
plt.plot(times_t0,df_ep_cumulants[df_ep_cumulants.g==0.001].mom_var,lw=lw,c = c3)
plt.plot(times_t0,df_ep_cumulants[df_ep_cumulants.g==0.0001].mom_var,lw=lw,c = c4)
ax = plt.gca()
plt.text(0.02,0.92,"(b)", transform=ax.transAxes)


format_axes(ax,"Momentum Variance")
ax.set_xlim(-0.05,2.05)



fig.legend(bbox_to_anchor=(0.85, 0.05),
            frameon=False, ncol=4,handlelength=1)


plt.tight_layout()
plt.savefig("ep_land_mom_cumulants.pdf",bbox_inches = "tight")
plt.savefig("ep_land_mom_cumulants.png",bbox_inches = "tight")
#plt.savefig("ep_land_mom_cumulants.pdf")
#plt.savefig("ep_land_mom_cumulants.eps")

plt.close()