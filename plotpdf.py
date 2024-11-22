import matplotlib.pyplot as plt


from main import *
import functions


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

  plt.plot(q_axis(tcurr),distribution(tcurr),color=c1,lw=lw,  label =r"$T=5$",zorder = 10000)
  plt.plot(q_axis(tcurr),rho(tcurr),color="orange",lw=lw)

  ax = plt.gca()
  format_dist_axes(ax)

  #ax.set_xticklabels([])
  ax.tick_params(axis='y', labelsize=fontsizeticks)

plt.figure(figsize=(8,6.5))
plot_pdf_nucleation(0,"$\mathrm{t} = 0$",None,231)
plt.fill_between(q_axis(0),p_initial(q_axis(0)),alpha = 0.5)
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
plt.fill_between(q_axis(T),p_final(q_axis(T)),alpha = 0.5)
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
