""" 
Functions to initialize and run the dynamics for EMRE

TODO: Implement support for an external time grid
"""

import numpy as np
from tqdm import tqdm

def initialize_dynamics(self,initial_values,startTime,endTime,delta_t,alpha=1.,volume=1.,measureResponse=False, ext_time_grid=None, crossCorrelator_tau=False):
    """
    Initializes the Effective Mesocopic Rate equations (System size expansion) or the LNA with also correction to the means. An additional input argument is volume, which as defined in the volume scaling.
    
    Parameters
    ----------
    - initial_values : array of floats
                Initial average values of the different species
    - startTime: float
                Start time
    - endTime  : float
                End Times
    - delta_t  : float
                Time step
    - ext_time_grid : array of floats (default = None)
                When an external (non-uniform) time grid needs to specified. NOT PROPERLY IMPLEMENTED!                
    - alpha   : float
                constant multiplying all k_3    
    - volume  : float
                volume as defined for the volume scaling.

    """
    
    if ext_time_grid is None:
        
        time_grid     = np.arange(startTime,endTime,delta_t)
        self.timeGrid = time_grid
        self.t        = 0.
        self.delta_t  = delta_t

    else:
        self.timeGrid = ext_time_grid
        self.t        = ext_time_grid[0]
        

    self.EMRE     = True
    self.y        = np.zeros([self.num_species,len(time_grid)])
    self.y[:,0]   = initial_values
    self.i        = 0
    self.alpha    = alpha
    self.volume   = volume
    
    #These are the correction to the MAK means!
    self.eps      = np.zeros([self.num_species,len(time_grid)])
    
    #These are the cross sepcies variances around the MAK means!
    self.lna_var  = np.zeros([self.num_species,self.num_species,len(time_grid)]) 

    # These are the response functions or the normalized two point correlation function to get by LNA (they can be defined for cross species)
    self.measureResponse = measureResponse
    
    self.crossCorrelator_tau = crossCorrelator_tau

    if self.measureResponse:
        self.resp = np.zeros([self.num_species,self.num_species,len(time_grid),len(time_grid)])
        for sp in range(self.num_species):
            self.resp[sp,sp,0,0] = 1.
            
    if self.crossCorrelator_tau:
        self.corr = np.zeros([self.num_species,self.num_species,len(time_grid),len(time_grid)])
        for i in range(self.num_species):
            if self.num_species > 1:
                self.corr[i,i,0,0] = initial_values[i]
            else:
                self.corr[i,i,0,0] = initial_values

    for i in range(self.num_species):
        if self.num_species > 1:
            self.lna_var[i,i,0] = initial_values[i]
        else:
            self.lna_var[i,i,0] = initial_values

    self.stchm_mat = np.zeros([self.num_species,self.num_reactions]) # The stochiometric matrix of the system

    j = 0
    for i in range(self.num_species):
        if j%2 == 0:
            self.stchm_mat[i,j] = 1 # The k1 spontaneous reaction
        j += 1
        if j%2 != 0:
            self.stchm_mat[i,j] = -1 # The k2 destruction reaction 
        j += 1

    for k in range(self.num_int):
        for i in range(self.num_species):
            self.stchm_mat[i,j+k] = self.s_i[k][i] - self.r_i[k][i]

    self.flux_mat = np.zeros([self.num_reactions,self.num_reactions]) # A diagonal matrix which stores the flux (without sign of every reactions)

    self.J_mat    = np.zeros([self.num_species,self.num_species]) # The J matrix required for calculating corrections 
        
        
def runDynamics(self):

    if self.k3 is float:
        self.k3 = np.array(k3)
    
    with tqdm(total=len(self.timeGrid)-2) as pbar:            
        while(self.i < len(self.timeGrid)-1):
            
            dydt       = np.zeros(self.num_species)
            depsdt     = np.zeros(self.num_species)
            dlna_vardt = np.zeros([self.num_species,self.num_species])                
            self.J_mat = np.zeros([self.num_species,self.num_species])
            delta_vec  = np.zeros(self.num_species)
            
            if self.measureResponse:
                dRdt       = np.zeros([self.num_species,self.num_species,self.i+1])
                
            if self.crossCorrelator_tau:
                dNdt       = np.zeros([self.num_species,self.num_species,self.i+1])
            
            #The following defines the MAK equations
            dydt   = self.k1 - self.k2*self.y[:,self.i]
            for j in range(self.num_int):
                x = self.k3[j]*self.alpha
                for k in range(self.num_species):
                    x *= self.y[k,self.i]**(self.r_i[j,k]) 
                #dydt[self.r_i[j].astype(bool)] -= x
                #dydt[self.s_i[j].astype(bool)] += x
                for k in range(self.num_species):
                    dydt[k] += (self.s_i[j,k] - self.r_i[j,k])*x
            
            #The flux matrix:
            j = 0
            for k in range(self.num_species):
                self.flux_mat[j,j] = self.k1[k]
                j+=1
                self.flux_mat[j,j] = self.k2[k]*self.y[k,self.i]
                j+=1
            
            for k in range(self.num_int):
                self.flux_mat[j,j] = self.alpha*self.k3[k]
                for m in range(self.num_species):
                    self.flux_mat[j,j] *= self.y[m,self.i]**(self.r_i[k,m]) 
                j+=1
                
            #Calculate the J matrix and delta_vec
            for i in range(self.num_species):                   
                
                for w in range(self.num_species):
                    for k in range(self.num_int):
                        
                        if self.r_i[k,w] > 0:
                            self.J_mat[i,w] += (self.s_i[k][i]-self.r_i[k][i])*self.r_i[k,w]* self.flux_mat[k+2*self.num_species,k+2*self.num_species]/self.y[w,self.i]                                    
                            
                        if self.r_i[k,w] > 1:
                            delta_vec[i] += self.J_mat[i,w]*self.lna_var[w,w,self.i]*(self.r_i[k,w]-1)/self.y[w,self.i] -self.J_mat[i,w]*(self.r_i[k,w]-1)
                            
                        for z in range(self.num_species):
                            if z!= w:
                                if self.r_i[k,z] > 0:
                                    delta_vec[i] += self.J_mat[i,w]*self.lna_var[w,z,self.i]*self.r_i[k,z]/self.y[z,self.i]
                                    
                #Remember to update the following diagonal enteries from the k2/destruction reaction only at the end becasue self.J_mat diagonal entries are used to construct the delta_vec, but the k2 reactions don't contribute to it!
                self.J_mat[i,i] += -self.k2[i]
                                                                                
            #Epsolion update eqns (remember alpha)
            depsdt = np.matmul(self.J_mat,self.eps[:,self.i]) + self.volume**(-0.5)*0.5*delta_vec
            
            #Variation update eqn
            dlna_vardt = np.matmul(self.J_mat,self.lna_var[:,:,self.i]) + np.matmul(self.lna_var[:,:,self.i],self.J_mat.T) + np.matmul(self.stchm_mat,np.matmul(self.flux_mat,self.stchm_mat.T))
            

            #Actual updates:
            self.y[:,self.i+1]         = self.y[:,self.i]         + self.delta_t*dydt
            self.eps[:,self.i+1]       = self.eps[:,self.i]       + self.delta_t*depsdt  
            self.lna_var[:,:,self.i+1] = self.lna_var[:,:,self.i] + self.delta_t*dlna_vardt
            
            if self.measureResponse:
                
                #Update eqn for the response functions:
                for t2 in range(self.i+1):
                    dRdt[:,:,t2]  = np.matmul(self.J_mat,self.resp[:,:,self.i,t2])
                
                for t2 in range(self.i+1):
                    self.resp[:,:,self.i+1,t2] = self.resp[:,:,self.i,t2] + self.delta_t*dRdt[:,:,t2]
                
                for sp in range(self.num_species):                    
                    self.resp[sp,sp,self.i+1,self.i+1] = 1.
                    
            if self.crossCorrelator_tau:
                
                for t2 in range(self.i+1):
                    dNdt[:,:,t2]  = np.matmul(self.J_mat,self.corr[:,:,self.i,t2]) #+ np.matmul(self.corr[:,:,self.i,t2],self.J_mat.T) + np.matmul(self.stchm_mat,np.matmul(self.flux_mat,self.stchm_mat.T))
                
                for t2 in range(self.i+1):
                    self.corr[:,:,self.i+1,t2] = self.corr[:,:,self.i,t2] + self.delta_t*dNdt[:,:,t2]
                    
                #for sp in range(self.num_species):                    
                self.corr[:,:,self.i+1,self.i+1] = self.lna_var[:,:,self.i+1]
            
            self.t += self.delta_t
            self.i += 1
                        
            pbar.update(1)
