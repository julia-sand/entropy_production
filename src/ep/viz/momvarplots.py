"""Script to produce Fig.3., momentum mean and variance at different values of
g, coupling constant """

import pandas as pd
import matplotlib.pyplot as plt
from src.utils.plots import *

if __name__=="__main__":

    ##momentum mean and var only
    update_mpl()

    df_ep_cumulants = pd.read_csv("cumulantscalculated.csv")#pd.read_csv("ep_cumulants_g2.csv",index_col=0)

    #round the g col
    df_ep_cumulants = df_ep_cumulants.round({'g': 10})

    times_t0 = df_ep_cumulants[df_ep_cumulants.g==0.1].t0.unique()
    
    fig = plt.figure(figsize = (15,7))

    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    
    for ax, title, textlabel in zip([ax1,ax2],["Momentum Mean","Momentum Variance"],["(a)","(b)"]):
        ax.set_prop_cycle(color=["#7fcdbb", "#41b6c4", "#2c7fb8", "#253494"])
        format_axes(ax,title)

        ax.set_xlim(-0.05,2.05)
        ax.text(0.02,0.92,textlabel,transform=ax.transAxes)

    for gi,label in zip([0.1,0.01,0.001,0.0001],[r"$g=10^{-1}$",r"$g=10^{-2}$",r"$g=10^{-3}$",r"$g=10^{-4}$"]):

        ax2.plot(times_t0,df_ep_cumulants[df_ep_cumulants.g==gi].mom_var,lw=3)
        
        ax1.plot(times_t0,df_ep_cumulants[df_ep_cumulants.g==gi].mom_mean,label=label,lw=3)
    
    fig.legend(bbox_to_anchor=(0.85, 0.05),
                frameon=False, ncol=4,handlelength=1)

    plt.tight_layout()
    plt.savefig("ep_land_mom_cumulants.pdf",bbox_inches = "tight")
    plt.savefig("ep_land_mom_cumulants.png",bbox_inches = "tight")
    #plt.savefig("ep_land_mom_cumulants.pdf")
    #plt.savefig("ep_land_mom_cumulants.eps")

    plt.close()