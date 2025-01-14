#from main import *
#import functions
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import string


from setup.plots import *

#format axes
def format_axes(ax):
  #ax.set_xscale('log')
  #ax.invert_xaxis()
  #ax.set_ylim((4.6,5.3))
  #ax.set_xlim((0.11,(8e-7)))
  ax.tick_params(axis='y', labelsize=fontsizeticks)
  ax.tick_params(axis='x', labelsize=fontsizeticks)
  ax.set_xlim((-0.01,2.01))
  #ax.set_ylabel(r"$\mathcal{E}$",fontsize = fontsizetitles)

f1 = "results_SYM.csv"
f2 = "results.csv"
f3 = "results_FLIP_TEMP.csv"

lw = 3

fig1 = plt.figure(figsize = (15,16))
gs1 = fig1.add_gridspec(3, 2
            ,width_ratios=[1,1],height_ratios=[1,1,1])


for peaks in enumerate([f1,f2,f3]):
  
  plot_idx = peaks[0]

  df_temp = pd.read_csv(peaks[1], sep=" ", header = 0)
  #round the t0 and t2 columns just in case it still isnt working
  df_temp = df_temp.round({'t0': dps, 't2': dps})
  times_t0_temp = df_temp.t0.unique()
  times_t0_temp.sort()

  q_axis_temp = df_temp.x.unique()
  q_axis_temp.sort()

  #function to get underdamped distribution
  def distribution_temp(t0 ):
    #t2 = round(t0*(epsilon**2),dps)
    dist = df_temp[df_temp.t0==t0].UDpdf.to_numpy()
    return dist


  #function to get underdamped distribution
  def rho_temp(t0):
    #t2 = round(t0*(epsilon**2),dps)
    dist = df_temp[df_temp.t0==t0].ptx.to_numpy()
    return dist

  if plot_idx ==1:
    times_t0_temp = times_t0_temp[::20]

  height_a = np.zeros(len(times_t0_temp))
  height_b = np.zeros(len(times_t0_temp))
  ODheight_a = np.zeros(len(times_t0_temp))
  ODheight_b = np.zeros(len(times_t0_temp))

  loc_a = np.zeros(len(times_t0_temp))
  loc_b = np.zeros(len(times_t0_temp))
  ODloc_a = np.zeros(len(times_t0_temp))
  ODloc_b = np.zeros(len(times_t0_temp))

  for t in enumerate(times_t0_temp):
    series = distribution_temp(t[1])
    rho_vals = rho_temp(t[1])
    peaks,props  = find_peaks(series,prominence=(0.01, None))
    peaks_rho,props_rho  = find_peaks(rho_vals,prominence=(0.01, None))

    prominences = props['prominences']
    prominences_rho = props_rho['prominences']


    ind1 = peaks[np.argmax(prominences)]
    ind1_rho = peaks_rho[np.argmax(prominences_rho)]
    try:
      ind2 = peaks[np.argsort(prominences)[-2]]
    except:
      ind2 = ind1
    try:
      ind2_rho = peaks_rho[np.argsort(prominences_rho)[-2]]
    except:
      ind2_rho = ind1_rho


    height_a[t[0]] = series[ind1]
    height_b[t[0]] = series[ind2]
    ODheight_a[t[0]] = rho_vals[ind1_rho]
    ODheight_b[t[0]] = rho_vals[ind2_rho]
    loc_a[t[0]] = q_axis_temp[ind1]
    loc_b[t[0]] = q_axis_temp[ind2]
    ODloc_a[t[0]] = q_axis_temp[ind1_rho]
    ODloc_b[t[0]] = q_axis_temp[ind2_rho]

  ax = fig1.add_subplot(gs1[plot_idx,0])#(gs1[peaks[0],0])#
  #plt.title("Height of peaks")
  letter_idx = int(2*plot_idx)
  ax.text(0.03,0.9,"("+string.ascii_lowercase[letter_idx]+")",
      fontsize = fontsizetitles,
       transform=ax.transAxes)

  ax.plot(times_t0_temp,height_a,lw=lw,color="royalblue",label = "Underdamped")
  ax.plot(times_t0_temp,height_b,lw=lw,color="royalblue")
  ax.plot(times_t0_temp,ODheight_a,lw=lw,color = "orange",label = "Overdamped")
  ax.plot(times_t0_temp,ODheight_b,lw=lw,color = "orange")
  #plt.ylim((0.25,0.45))

  #plots.format_axes(ax,fontsizeticks)
  ax.set_ylabel("Peak Height",fontsize=fontsizetitles,labelpad = 20)
  format_axes(ax)
  ax.set_ylim((0.48,0.58))

  ax2 = fig1.add_subplot(gs1[plot_idx,1])#(gs1[peaks[0],1])#
  #ax2.title("Location of peaks")
  ax2.scatter(times_t0_temp,loc_a,lw=lw,color="royalblue",label = "Underdamped")
  ax2.scatter(times_t0_temp,loc_b,lw=lw,color="royalblue")
  ax2.scatter(times_t0_temp,ODloc_a,lw=lw,color = "orange",label = "Overdamped")
  ax2.scatter(times_t0_temp,ODloc_b,lw=lw,color = "orange")
  format_axes(ax2)
  ax2.set_ylim((-1.8,1.8))
  ax2.set_ylabel("Peak Location",fontsize=fontsizetitles)
  ax2.text(0.03,0.9,"("+string.ascii_lowercase[letter_idx+1]+")"
        ,fontsize = fontsizetitles,transform=ax2.transAxes)
  ax3 = ax2.twinx()
  ax3.set_yticks([])
  peak_center = round(loc_a[0])
  ax3.set_ylabel(f"Center = {peak_center}",rotation=270, fontsize=fontsizetitles,labelpad = 30)

  #if plot_idx != 2:
  #  ax.set_xticklabels([])
  #  ax2.set_xticklabels([])
  #  ax.tick_params('both', length=10, which='both')
  #  ax2.tick_params('both', length=10, which='both')

#get legend
orange_line = mlines.Line2D([], [],color="orange",lw=lw)
blue_line = mlines.Line2D([], [],color="royalblue",lw=lw)


legend = fig1.legend(handles=[blue_line,orange_line],
          labels=["Underdamped","Overdamped"],
          fontsize = fontsizetitles,
          frameon = False,
          handlelength = 1,
          ncols = 3)


#move the legend
legend.set_bbox_to_anchor(bbox=(0.65,0.07))

plt.subplots_adjust(wspace=0.25)

#plt.tight_layout()

plt.savefig("peakheights.png",bbox_inches="tight")
plt.close()