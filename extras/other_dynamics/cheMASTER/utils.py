"""
Internal utility functions for cheMASTER
"""


import numpy as np

from .master_operator import *

def calculate_mean(self,p):

    """
    Calculates the mean number for any given p for all species.
    """

    x = np.zeros(self.num_species)

    for j in range(self.num_species):
        x[j] = np.sum(np.array(self.master_stateSpace)[:,j]*np.abs(p),axis=0)/np.sum(np.abs(p))

    return x

def calculate_secondMoment(self,p):

    """
    Calculates the second moment of the number distribution for p.
    """

    x = np.zeros(self.num_species)

    for j in range(self.num_species):
        x[j] = np.sum(np.array(self.master_stateSpace)[:,j]**2*np.abs(p),axis=0)/np.sum(np.abs(p))

    return x

def calculate_crossCorrelator(self,p,crossC):

    """
    Calculates the number cross correclation between two species crossC[0] and crossC[1] for given p at a given time.
    """

    return np.sum(np.array(self.master_stateSpace)[:,crossC[0]]*np.array(self.master_stateSpace)[:,crossC[1]]*np.abs(p),axis=0)/np.sum(np.abs(p))        

def normalize_response(self,measureResponse,measureResponse_par):
    
    k1_true     = np.copy(self.k1)
    
    if measureResponse == 'step':
        for k in range(len(self.timeGrid)):
            l = k
            while l < len(self.timeGrid):
                self.Response[:,l,k] = (self.Response[:,l,k] - self.y[:,l])/(k1_true*(measureResponse_par-1))
                l += 1
                
    elif measureResponse == 'finite':
        for k in range(len(self.timeGrid)):
            l = k
            while l < len(self.timeGrid):
                self.Response[:,l,k] = (self.Response[:,l,k] - self.y[:,l])/(k1_true*(measureResponse_par[0]-1))/(measureResponse_par[1]*self.delta_t)
                l += 1
                
    elif measureResponse == 'finite-cross':
        for k in range(len(self.timeGrid)):
            l = k
            while l < len(self.timeGrid):
                for j1 in range(self.num_species):
                    for j2 in range(self.num_species):
                        self.Response[j1,j2,l,k] = (self.Response[j1,j2,l,k] - self.y[j1,l])/(k1_true[j2]*(measureResponse_par[0]-1))/(measureResponse_par[1]*self.delta_t)
                l += 1
            

def perturb_measure_response(self,p,measureResponse,measureResponse_par,max_num):
    
    k1_true     = np.copy(self.k1)
    master_true = np.copy(self.master)
    
    if measureResponse != 'finite-cross':
    
        if measureResponse == 'finite':
            self.k1     = self.k1*measureResponse_par[0]
            masterOperator(self,max_num)
            tau = self.i
            q   = np.copy(p)
            
        else:
            self.k1     = self.k1*measureResponse_par
            masterOperator(self,max_num)
            tau = self.i
            q   = np.copy(p)
        
    
    if measureResponse == 'impulse':
        
        q   = np.matmul((self.master-master_true)/(k1_true*(measureResponse_par-1)),q)

        while tau < len(self.timeGrid):
            self.Response[:,tau,self.i] = calculate_mean(self,q)
            q += self.delta_t*np.matmul(master_true,q)
            tau += 1
        self.k1 = k1_true
        masterOperator(self,max_num)

    elif measureResponse == 'step':

        while tau < len(self.timeGrid):
            self.Response[:,tau,self.i] = calculate_mean(self,q)
            q += self.delta_t*np.matmul(self.master,q)
            tau += 1
        
    elif measureResponse == 'finite':

        while tau-self.i < measureResponse_par[1] and tau < len(self.timeGrid):
            #propogate the perturbed solution only for the given time measureResponse_finite[1]
            
            self.Response[:,tau,self.i] = calculate_mean(self,q)
            q += self.delta_t*np.matmul(self.master,q)
            tau += 1

        while tau < len(self.timeGrid):
            #propagate the original solution
            
            self.Response[:,tau,self.i] = calculate_mean(self,q)
            q += self.delta_t*np.matmul(master_true,q)
            tau += 1
        
    elif measureResponse == 'finite-cross':

        for j1 in range(self.num_species):

            temp = np.ones(self.num_species)
            temp[j1] = measureResponse_par[0]
            self.k1 = k1_true*temp
            masterOperator(self,max_num)
            # j1 is the species who's creation rate has been perturbed

            tau = self.i
            q = np.copy(p)

            while tau-self.i < measureResponse_par[1] and tau < len(self.timeGrid):
                #propogate the perturbed solution only for the given time measureResponse_finite[1]
                self.Response[:,j1,tau,self.i] = calculate_mean(self,q)
                q += self.delta_t*np.matmul(self.master,q)
                tau += 1

            while tau < len(self.timeGrid):
                #propagate the original solution
                self.Response[:,j1,tau,self.i] = calculate_mean(self,q)
                q += self.delta_t*np.matmul(master_true,q)
                tau += 1
            
    self.k1 = k1_true
    masterOperator(self,max_num)
    

def calculate_selfCorrelator_tau(self,p,tau):

    # TODO

    """
    Calculates the self Correlator as a function of lag time \tau in time Grid index values. Takes the full probability matrix p of the state_space_size*time_grid_length
    """

    #x = np.zeros([self.num_species,len(self.timeGrid)-tau])

    #for j in range(self.num_species):
    #    for time in range(np.shape(x)[1]):
    #        x[j,time] = np.sum(np.array(self.master_stateSpace)[:,j]**2*np.abs(p[:,time])*np.abs(p[:,time+tau]), axis=0)/np.sum(np.abs(p[:,time])*np.abs(p[:,time+tau]))

    return x