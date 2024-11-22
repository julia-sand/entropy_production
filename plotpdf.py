import matplotlib.pyplot as plt


from main import *
import functions

##PLOT SETUP
#fontsizes
fontsize = 22
fontsizeticks = 16
fontsizetitles = 22

#xlims
xlimmax = 3
xlimmin = -3

ti = np.min(times_t0)
tf = np.max(times_t0)

tlim = (ti,tf)

titlepad = 5
titlex = 0.07
titley = 0.88

# Plotting the graphs
#COLORS
c2 = "#a6cee3" #lightblue
c3 = "#33a02c" #dark green
c1 = "#1f78b4" #darkblue
c4 = "#b2df8a" #light green

#linewidth
lw = 3

#axes formatting
def format_dist_axes(ax):

  #ax.patch.set_alpha(0)
  #ax.yaxis.tick_right()
  ax.tick_params(axis='x', labelsize=fontsizeticks)
  ax.tick_params(axis='y', labelsize=fontsizeticks)
  #ax.tick_params(labeltop='off', labelright='off')

  ax.set_ylim((-0.01,0.8))
  ax.set_xlim((xlimmin,xlimmax))
  ax.spines['bottom'].set_zorder(1000)

def format_axes(ax,fontsize):

  ax.set_xlim(tlim)
  ax.set_xlabel(r"$\mathrm{t}$",fontsize = fontsizetitles)

  #ax.set_xticklabels(labels = [0,1,2,3,4,5],fontsize=10)
  ax.tick_params(axis='y', labelsize=fontsizeticks)
  ax.tick_params(axis='x', labelsize=fontsizeticks)
  return ax

#format axes
def format_log_axes(ax):
  ax.set_xscale('log')
  ax.invert_xaxis()
  #ax.set_ylim((4.6,5.3))
  ax.set_xlim((0.13,(8e-7)))
  ax.tick_params(axis='y', labelsize=fontsizeticks)
  #ax.tick_params(axis='x', labelsize=fontsizeticks)
  #ax.set_ylabel(r"$\mathcal{E}$",fontsize = fontsizetitles)

def plot_pdf_nucleation(tcurr,title,labels,loc):
  '''This functions plots a distribution and labels it

  input:
  -tcurr: the time in terms of t0 to plot
  -title: title of the graph (if using)
  -label: label of the panel
  -loc: location as a matplotlib subplot code (or gridspec)

  output:
  -plot in location given by loc with the overdamped(in orange)
  and underdamped (in blue) of the marginal density of the position
  '''

  # t0 distribution
  plt.subplot(loc)
  plt.title(title, loc = "center", fontsize=fontsizetitles)

  plt.plot(functions.q_axis(tcurr),functions.distribution(tcurr),color=c1,lw=lw,  label =r"$T=5$",zorder = 10000)
  plt.plot(functions.q_axis(tcurr),functions.rho(tcurr),color="orange",lw=lw)

  ax = plt.gca()
  format_dist_axes(ax)

  #ax.set_xticklabels([])
  ax.tick_params(axis='y', labelsize=fontsizeticks)

plt.figure(figsize=(8,6.5))
plot_pdf_nucleation(0,"$\mathrm{t} = 0$",None,231)
plt.fill_between(functions.q_axis(0),functions.p_initial(functions.q_axis(0)),alpha = 0.5)
plt.gca().set_xticklabels([])
plt.gca().set_ylabel(r"$\rho_t(q)$",fontsize=fontsizetitles)
plot_pdf_nucleation(0.2*T,"$\mathrm{t} = 0.2\,\mathrm{t}_f$",None,232)
plt.gca().set_xticklabels([])
plt.gca().set_yticklabels([])
plot_pdf_nucleation(0.5*T,"$\mathrm{t} = 0.5\,\mathrm{t}_f$",None,233)
plt.gca().set_xticklabels([])
plt.gca().set_yticklabels([])
plot_pdf_nucleation(0.6*T,"$\mathrm{t} = 0.6\,\mathrm{t}_f$",None,234)
plt.gca().set_ylabel(r"$\rho_t(q)$",fontsize=fontsizetitles)
plt.gca().set_xlabel(r"$q$",fontsize=fontsizetitles)
plot_pdf_nucleation(0.8*T,"$\mathrm{t} = 0.8\,\mathrm{t}_f$",None,235)
plt.gca().set_yticklabels([])
plt.gca().set_xlabel(r"$q$",fontsize=fontsizetitles)
plot_pdf_nucleation(T,"$\mathrm{t} = \mathrm{t}_f$",None,236)
plt.fill_between(functions.q_axis(T),functions.p_final(functions.q_axis(T)),alpha = 0.5)
plt.gca().set_xlabel(r"$q$",fontsize=fontsizetitles)
plt.legend(labels = ["Underdamped","Overdamped"],
           prop={"size":fontsizeticks},
           frameon = False,
           loc = "upper center",
           handlelength = 1)
plt.gca().set_yticklabels([])

plt.tight_layout()
#plt.show()
plt.savefig("ep_distributions_T2.pdf")
