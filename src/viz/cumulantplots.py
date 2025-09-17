from main import *
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

#formatting options
from src.utils.plots import *
import src.utils.functions as functions

def plot_cumulants(simulated_cumulants_csv,cumulants_perturbative_csv):
    
    df_ep_cumulants = pd.read_csv(simulated_cumulants_csv,header=0)
    cumulants_perturbative = pd.read_csv(cumulants_perturbative_csv,header=0)
    cumulant_plot_times = df_ep_cumulants.t0.unique()
    cumulant_plot_times.sort()
    times_t0 = cumulants_perturbative.t0.unique()
    times_t0.sort()

    titlepad = 5
    titlex = 0.07
    titley = 0.88

    # Plotting the cumulants
    fig1 = plt.figure(figsize=(15, 8))#, constrained_layout=True)

    # Create a 1x5 grid with different widths for the bottom row
    gs_cumulants = gridspec.GridSpec(2, 6, width_ratios=[1, 1, 1, 1,1,1], height_ratios=[1, 1])
                                #hspace = 0.3,wspace = 2.3)
                                #(1, 2, width_ratios=[6, 4], figure=fig)

    odmeans = np.fromiter((functions.mean_t0(t0) for t0 in times_t0),float)



    # position mean
    plt.subplot(gs_cumulants[0, 0:2])
    plt.plot(times_t0,  cumulants_perturbative[cumulants_perturbative.g==g].pos_mean,color=c1)
    plt.plot(times_t0,odmeans,lw=lw,color="orange")
    plt.plot(cumulant_plot_times,  df_ep_cumulants[df_ep_cumulants.g==g].pos_mean,color=c2)
    plt.title('(a)',pad = titlepad,x = titlex, y =titley,zorder = 1000000)
    ax = format_axes(plt.gca(),'Position Mean')
    ax.set_ylim((-0.05,1.2))
    ax.set_xlim((-0.01,2.01))

    # position variance
    plt.subplot(gs_cumulants[0, 2:4])
    plt.plot(times_t0,  cumulants_perturbative[cumulants_perturbative.g==g].pos_var,lw=lw,color=c1)
    plt.plot(times_t0, [functions.var_t0(t0) for t0 in times_t0]-(odmeans**2),lw=lw,color="orange")
    plt.plot(cumulant_plot_times, df_ep_cumulants[df_ep_cumulants.g==g].pos_var,lw=lw,color=c2)
    plt.title('(b)',fontsize = fontsizetitles,pad = titlepad,x = titlex, y =titley,zorder = 1000000)
    ax = format_axes(plt.gca(),'Position Variance')
    ax.set_xlim((-0.01,2.01))

    # cross corr
    plt.subplot(gs_cumulants[0, 4:])
    plt.plot(times_t0,  cumulants_perturbative[cumulants_perturbative.g==g].xcorr,lw=lw,color=c1)
    plt.plot(cumulant_plot_times, df_ep_cumulants[df_ep_cumulants.g==g].xcorr,lw=lw,color=c2)
    plt.title('(c)',fontsize = fontsizetitles,pad = titlepad,x = titlex, y =titley,zorder = 1000000)
    ax = format_axes(plt.gca(),'Cross Correlation')
    ax.set_xlim((-0.01,2.01))

    # momentum mean
    plt.subplot(gs_cumulants[1,:3])
    plt.plot(times_t0,  cumulants_perturbative[cumulants_perturbative.g==g].mom_mean,lw=lw,color=c1)
    plt.plot(cumulant_plot_times, df_ep_cumulants[df_ep_cumulants.g==g].mom_mean,lw=lw,color=c2)
    #plt.plot(times_t0, [mom_mean_alt(t0) for t0 in times_t0],lw=lw)
    plt.title('(d)',fontsize = fontsizetitles,pad = titlepad,x = titlex*(2/3), y =titley,zorder = 1000000)
    ax = format_axes(plt.gca(),'Momentum Mean')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.set_xlim((-0.01,2.01))

    # momentum variance
    mom_var = plt.subplot(gs_cumulants[1, 3:])
    plt.plot(times_t0,  cumulants_perturbative[cumulants_perturbative.g==g].mom_var,lw=lw,color=c1)
    plt.plot(cumulant_plot_times,  df_ep_cumulants[df_ep_cumulants.g==g].mom_var,lw=lw,color=c2)
    plt.title('(e)',fontsize = fontsizetitles,pad = titlepad,x = titlex*(2/3), y =titley,zorder = 1000000)
    ax = format_axes(plt.gca(),'Momentum Variance')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.set_xlim((-0.01,2.01))


    plt.tight_layout()

    #get legend
    orange_line = mlines.Line2D([], [],color="orange",lw=lw)
    blue_line = mlines.Line2D([], [],color=c1,lw=lw)
    green_line = mlines.Line2D([], [],color=c2,lw=lw)

    legend = fig1.legend(handles=[orange_line,blue_line,green_line],
            labels = ["Overdamped","Underdamped (Predicted)","Underdamped (Evolved)"],
            fontsize = fontsizetitles,
            frameon = False,
            handlelength = 1,
            ncols = 3)

    #move the legend
    legend.set_bbox_to_anchor(bbox=(0.95,0.07))

    plt.savefig("ep_cumulants_V6.pdf",bbox_inches="tight")
    plt.savefig("ep_cumulants_V6.png",bbox_inches="tight")
    plt.close()

if __name__=="__main__":
    simulated_cumulants_csv = "cumulants_V6.csv"
    cumulants_perturbative_csv = "cumulantscalculated.csv"
    plot_cumulants(simulated_cumulants_csv,cumulants_perturbative_csv)