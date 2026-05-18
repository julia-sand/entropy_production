"""controls boundary data"""
import numpy as np 

from ep.utils.parser import boundary_params

class Boundary():
  def __init__(self,):
    """loads initial and final boundary conditions"""
  
    self.params = boundary_params()
    
    #set up the boundary conditions
    self.peak_center = self.params["peaklocation"]
    self.denom = self.params["denom"]
    
    self.pi_norm = self.compute_norm(self.p_initial_unnormalised)
    self.pf_norm = self.compute_norm(self.p_final_unnormalised)
    
  #exact boundary conditions
  def p_initial_unnormalised(self,q):
    return np.exp(-(q-self.peak_center)**4/self.denom)
  
  def p_final_unnormalised(self,q):
    return np.exp(-(((q**2 -self.peak_center**2)**2)/self.denom))
  
  #compute normalisation constants
  def compute_norm(self,pfun):
    return np.abs(np.trapz(pfun(np.linspace(-8,8,10000)),np.linspace(-8,8,10000)))
    
  #normalised boundary conds
  def p_initial(self,q):
    return self.p_initial_unnormalised(q)/self.pi_norm
  def p_final(self,q):
    return self.p_final_unnormalised(q)/self.pf_norm
  
  #underdamped boundary conditions
  def ud_pinitial(self,p_samples,q_samples):
    """this is the initial distribution in the underdamped case
    p_samples = momenta
    q_samples = positions
    """
  
    return self.p_initial(q_samples)*np.exp(-(p_samples**2)/2)/np.sqrt(2*np.pi)
  
  def ud_pfinal(self,p_samples,q_samples):
    """this is the final distribution in the underdamped case
    p_samples = momenta
    q_samples = positions
    """
    return self.p_final(q_samples)*np.exp(-(p_samples**2)/2)/np.sqrt(2*np.pi)
