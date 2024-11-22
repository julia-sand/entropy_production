from main import *
import functions
import matplotlib.pylot as plt
import matplotlib.gridspec as gridspec
#from matplotlib import cm
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.lines as mlines

##make the plots

titlepad = 5
titlex = 0.15
titley = 0.85
titlezorder = 1000
titles = ["(a)","(b)","(c)","(d)","(e)","(f)","(g)","(h)","(i)","(j)","(k)","(l)","(m)","(n)","(o)","(p)","(q)"]
time_titles = ["$\mathrm{t} = 0$","$\mathrm{t} = 0.25\mathrm{t}_f$",
               "$\mathrm{t} = 0.4\mathrm{t}_f$","$\mathrm{t} = 0.5\mathrm{t}_f$",
               "$\mathrm{t} = 0.6\mathrm{t}_f$","$\mathrm{t} = 0.7\mathrm{t}_f$",
               "$\mathrm{t} = 0.8\mathrm{t}_f$","$\mathrm{t} = 0.95\mathrm{t}_f$",
               "$\mathrm{t} = \mathrm{t}_f$"]

#dist title location
disttitlex = -2.8
disttitley = 0.52

suptitlex = 0.07
suptitley = 1.1

dashlinezorder = 10

#alpha for filled shapes
shadingalpha = 1

# Plotting the graphs)
#COLORS
c2 = "#a6cee3" #lightblue
c3 = "orange" #"#33a02c" #dark green
c1 = "#1f78b4" #darkblue
c4 = "#b2df8a" #light green


#set ylim for plot
ymin = -30
ymax = 20

#xlims
xlimmax = 3
xlimmin = -3

lw = 3

#fontsizes
xmax = 5
fontsize = 22
fontsizeticks = 18
fontsizetitles = 22

#Cumulant plots parameters
#df = pd.read_csv("kl_gauss_T2_v8.csv")

fontsize = 22
fontsizeticks = 18
fontsizetitles = 22

titlepad = 10
xtitle = 0.07
ytitle = 0.88
titlezorder = 1000


lw = 3

# Data for the graphs
titlepad = 5
def format_axes(ax,fontsize):

  ax.set_xlim((0,T))
  ax.set_xlabel(r"$\mathrm{t}$",fontsize = fontsizetitles)

  #ax.set_xticklabels(labels = [0,1,2,3,4,5],fontsize=10)
  ax.tick_params(axis='y', labelsize=fontsizeticks)
  ax.tick_params(axis='x', labelsize=fontsizeticks)
  return ax

def format_drift(ax):

  #ax.yaxis.tick_right()
  ax.tick_params(axis='x', labelsize=fontsizeticks)
  ax.tick_params(axis='y', labelsize=fontsizeticks)

  ax.set_ylim((-30,30))
  ax.set_xlim((xlimmin,xlimmax))
  ax.set_xlabel(r"$\mathrm{q}$",fontsize= fontsize,labelpad=7)

##plot set-up
def format_dist_axes(ax):

  #ax.patch.set_alpha(0)
  #ax.yaxis.tick_right()
  ax.tick_params(axis='x', labelsize=fontsizeticks)
  ax.tick_params(axis='y', labelsize=fontsizeticks)
  #ax.tick_params(labeltop='off', labelright='off')

  #ax.set_ylim((-0.01,0.65))
  ax.set_xlim((xlimmin,xlimmax))
  ax.spines['bottom'].set_zorder(1000)

  #ax.set_xticks([])
  #ax.set_xticklabels([])
  #ax.set_xlabel(r"$\mathrm{q}$",fontsize= fontsize)



##plot set-up
def format_scatter_axes(ax):

  ax.set_xlim((-4,3))
  ax.set_ylim((-3,3))

  ax.tick_params(axis='both', labelsize=fontsizeticks)

def cleaner(arr,t0):

  #masks nans and infs and returns a pair for plotting
  #copy q axis
  plotq = np.copy(q_axis)

  masknan = functions.get_rhomask(t0)

  #this function removes nan's and the end points which come from the truncation of the gradients
  #arr = arr[np.min(masknan)+20:np.max(masknan)-20]
  #plotq = plotq[np.min(masknan)+20:np.max(masknan)-20]

  return plotq, arr
  #generic_filter(arr,sc.median,1,mode="constant")


#plot params

#fontsizes
xmax = 5
fontsize = 28
fontsizeticks = 22
fontsizetitles = 28

#set ylim for plot
ymin = -30
ymax = 30

def plot_pair(tcurr,title,labels,gs,locy):
  # t0 distribution
  plt.subplot(gs[0,locy])
  plt.title(title, loc = "center", fontsize=fontsizetitles)

  plt.plot(functions.q_axis(tcurr),functions.rho(tcurr),color=c3,lw=4)
  plt.plot(functions.q_axis(tcurr),functions.distribution(tcurr),color=c1,lw=3, label =r"$\mathrm{t} = 0$",zorder = 10000)

  ax = plt.gca()
  format_dist_axes(ax)
  ax.text(s = labels[0],fontsize = fontsizetitles,x = disttitlex, y =disttitley,zorder = titlezorder)

  ax.set_xticklabels([])
  ax.tick_params(axis='y', labelsize=fontsizeticks)

  drift0 = plt.subplot(gs[1, locy])
  #plot_data = functions.optimal_drift(tcurr)#cleaner(optimal_drift(tcurr),tcurr)

  #this just removes the areas of low statistics rho < tol
  qseries = functions.q_axis(tcurr)
  yseries = functions.optimal_drift(tcurr)
  #sigma_data = #cleaner(dsigma(tcurr),tcurr)
  sigmaseries = -functions.dsigma(tcurr)

  ax0 = plt.gca()
  format_drift(ax0)
  ax0.text(s = labels[1],fontsize = fontsizetitles,x = disttitlex, y =25, zorder = titlezorder)

  series1a, = ax0.plot(qseries,yseries,color = c1,lw=lw,label = r"Underdamped",alpha=0.5,zorder = 100)
  series1b, = ax0.plot(qseries,sigmaseries,color = c3,lw=lw,label = r"Overdamped",alpha=0.5)
  #series1c, = ax0.plot(qseries,generic_filter(yseries,sc.mean,100,mode="nearest"),color = c1,lw=lw,label = r"Underdamped",zorder = 10000)
  #series1d, = ax0.plot(qseries,generic_filter(sigmaseries,sc.mean,100,mode="nearest"),color = c3,lw=lw,label = r"Underdamped",zorder = 10000)

  if locy ==0:
    ax.set_ylabel(r'$\mathrm{f}_{\mathrm{t}}(\mathrm{q})$',fontsize = fontsizetitles,labelpad= 7)
    ax0.set_ylabel(r'$-\partial U_{\mathrm{t}}(\mathrm{q})$',fontsize = fontsizetitles,labelpad= -5)
    if gs == gs0:
      #for edges
      ax.fill_between(functions.q_axis(tcurr),p_initial(functions.q_axis(tcurr)),color = c2,alpha = shadingalpha)
  else:
    ax0.set_yticklabels([])
    ax.set_yticklabels([])

  if locy ==-1 and gs == gs1:
    ax.fill_between(functions.q_axis(tcurr),p_final(functions.q_axis(tcurr)) = c2,alpha = shadingalpha)

#set up the gridspec

fig = plt.figure(figsize = (18,24)) #figsize = (width,height)

gs = gridspec.GridSpec(2, 1, height_ratios=[1,1],hspace=0.2)
gs0 = gridspec.GridSpecFromSubplotSpec(2, 5, height_ratios=[1,2], subplot_spec=gs[0], hspace=0.1, wspace=0.05)
gs1 = gridspec.GridSpecFromSubplotSpec(2, 5, height_ratios=[1,2], subplot_spec=gs[1], hspace=0.1, wspace=0.05)

plot_pair(0,"$\mathrm{t} = 0$",["(a)","(f)"],gs0,0)
plot_pair(0.25*T,"$\mathrm{t} = 0.25\ \mathrm{t}_f$",["(b)","(g)"],gs0,1)
plot_pair(0.5*T,"$\mathrm{t} = 0.5\ \mathrm{t}_f$",["(c)","(h)"],gs0,2)
plot_pair(0.6*T,"$\mathrm{t} = 0.6\ \mathrm{t}_f$",["(d)","(i)"],gs0,3)
plot_pair(0.7*T,"$\mathrm{t} = 0.7\ \mathrm{t}_f$",["(e)","(j)"],gs0,-1)

plot_pair(0.8*T,"$\mathrm{t} = 0.8\ \mathrm{t}_f$",["(k)","(p)"],gs1,0)
plot_pair(0.85*T,"$\mathrm{t} = 0.85\ \mathrm{t}_f$",["(l)","(q)"],gs1,1)
plot_pair(0.9*T,"$\mathrm{t} = 0.9\ \mathrm{t}_f$",["(m)","(r)"],gs1,2)
plot_pair(0.95*T,"$\mathrm{t} = 0.95\ \mathrm{t}_f$",["(n)","(s)"],gs1,3)
plot_pair(T,"$\mathrm{t} = \mathrm{t}_f$",["(o)","(t)"],gs1,-1)

#make legend
l0 = mlines.Line2D([], [], color=c1, lw = lw)
l1 = mlines.Line2D([], [], color=c3, lw = lw)
p0 = mpatches.Patch(color = c2, alpha=shadingalpha)

plt.legend([l0,l1,p0],["Underdamped","Overdamped","Assigned Boundary Conditions"],
              prop = {"size": fontsizeticks },
              ncol = 3,
              bbox_to_anchor=(0.4, -0.1),
              frameon = False)


#plt.savefig("ep_land_drift_v1.eps")
plt.savefig("ep_land_drift_v1.pdf")
#plt.savefig("ep_land_drift_v39.png")
plt.close()