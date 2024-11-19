import torch
from geomloss import SamplesLoss # See also ImagesLoss, VolumesLoss

#get the parameters
from main import *

# Create some large histograms from initial and final data
xs = npr.choice(np.linspace(xmin,xmax,n*100), size = (n,1), p = p_initial(np.linspace(xmin,xmax,n*100))/ sum(p_initial(np.linspace(xmin,xmax,n*100))))
xt = npr.choice(np.linspace(xmin,xmax,n*100), size = (n,1), p = p_final(np.linspace(xmin,xmax,n*100))/ sum(p_final(np.linspace(xmin,xmax,n*100))))

# Define a Sinkhorn (~Wasserstein) loss between sampled measures
S_e = SamplesLoss(loss="sinkhorn", p=2, blur=.05)

Sxy = S_e(torch.from_numpy(xs),torch.from_numpy(xt)) # By default, use constant weights = 1/number of samples
print(np.mean(Sxy.numpy()))
