""" 
Functions to initialize and run the dynamics for cheMASTER

TODO: Implement support for an external time grid
"""

import numpy as np
import scipy as sc
from scipy.integrate import odeint
import itertools as it
import math
from tqdm import tqdm
from scipy.stats import poisson

from .utils import *
from .master_operator import *

def initialize_dynamics(self,initial_values,startTime,endTime,delta_t,ext_time_grid = None):

    """
    Initializes the dynamics. Arguments self explanatory.

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

    """
    
    if ext_time_grid is None:
        
        time_grid     = np.arange(startTime,endTime,delta_t)
        self.timeGrid = time_grid
        self.t        = 0.
        self.delta_t  = delta_t

    else:
        self.timeGrid = ext_time_grid
        self.t        = ext_time_grid[0]
    
    self.y        = np.zeros([self.num_species,len(self.timeGrid)])
    self.y[:,0]   = initial_values
    self.i        = 0

    return 0


def runDynamics(self, max_num=10, method='Euler', variance=False, crossCorrelator=None, selfCorrelator_tau=None, 
    crossCorrelator_tau=None, measureResponse = None, measureResponse_par = None ,initialization='poisson', initialization_par=None,
    return_probability=False):

    """
    Run the dynamics of the probability distribution dP/dt = MP where M is the master operator, and outputs the mean for each species. 
    
    Parameters
    ----------
    - max_num : int or array(int)
                The maximum number of particles or the maximum size for the support of the probability distribution. If int, the same value is used for all species and specific values for each species can be passed as an array otherwise.
                
    - method : 'Euler', 'RK2', 'RK4' or 'Eigenvalue'
                The method used to solve the equations. Some further measurement methods are method specific
                
    - variance: T or None
                If to measure variance or not
                
    - crossCorrelator: T or None
                If to measure the cross variance between the species
                
    - selfCorrelator_tau : T, 'connected', or None
                Measures the two time quantity <n_t1 n_t2> or the connected one
                
    - crossCorrelator_tau : T, 'connected', or None
                Measures the two time quantity <n_t1 n_t2> between the two species, disconencted or the connected one. The crossCorrelator_tau is slow, but includes all selfCorrelator_tau
                
    - measureResponse : 'impulse', 'step' , 'finite' , 'finite-cross' or None
                To measure the response function R(t1,t2) from one of these different perturbing methods
                'impulse' -- Measures the Response function by creating perturbation in the creation rate by changing it to k1*measureResponse_par, and then going to the original k1 
                'step' --  Measures the Response function by creating perturbation in the creation rate by changing it to k1*measureResponse_par.
                'finite' -- Measures the Response function by creating perturbation in the creation rate for a time measureResponse_finite[1] by changing it to k1*measureResponse_par[0] and then setting it back to the original value.
                'finite-cross' -- # Measures the Response function by creating perturbation in the creation rate for a time measureResponse_finite[1] by changing it to k1*measureResponse_par[0] and then setting it back to the original value.
                
    - measureResponse_par : float, array(float) or None
                Extra parameter for measuring response
                
    - initialization: 'poisson', 'fixed', 'uniform' (uniform is not yet implemented)
                The initial distribution shape - poisson, fixed - a delta function
                
    - initialization_par : any
                Extra paramters to define the initial distribution
    - return_probability : True or False
                Whether to return the full probability distribution or not
    """

    # We define some helper functions here which will help us later.
    def wrapper(t,y):
        return dpdt(t,y,self.master)

    def dpdt(t,y,master):
        return np.matmul(master,y)

    try:
        len(max_num)
    except:
        max_num = np.zeros(self.num_species) + max_num

    try:
        self.master
    except:
        masterOperator(self,max_num)

    if variance:
        self.variance = np.zeros([self.num_species,len(self.timeGrid)])

    if crossCorrelator is not None:
        self.crossC = np.zeros(len(self.timeGrid))
        
    if measureResponse is not None:
        
        if measureResponse != 'finite-cross':
            self.Response = np.zeros([self.num_species,len(self.timeGrid),len(self.timeGrid)])
            
        else:
            self.Response = np.zeros([self.num_species,self.num_species,len(self.timeGrid),len(self.timeGrid)])

    init_dist = stateSpace_initialDistribution(self,self.master_maxNum,initialization=initialization, initialization_par=initialization_par)

    p = init_dist[:]

    if selfCorrelator_tau is not None:
        # We need to store this value for all solution times and for all lag times

        correlator_tau = np.zeros([self.num_species,len(self.timeGrid),len(self.timeGrid)])

    if crossCorrelator_tau is not None:
        # We need to store this value for all solution times and for all lag times and for pairs of species

        correlator_tau = np.zeros([self.num_species,self.num_species,len(self.timeGrid),len(self.timeGrid)])

    if method == 'Euler':

        with tqdm(total=len(self.timeGrid)) as pbar:
            self.y[:,self.i] = calculate_mean(self,p)
            if selfCorrelator_tau is not None:
                p_full[:,self.i] = p               
            pbar.update(1)
            
            if measureResponse is not None:            
                perturb_measure_response(self,p,measureResponse,measureResponse_par,max_num)

            while self.i < len(self.timeGrid)-1:
                p += self.delta_t*np.matmul(self.master,p)
                self.y[:,self.i+1] = calculate_mean(self,p)
                if variance:
                    self.variance[:,self.i+1] = calculate_secondMoment(self,p) - self.y[:,self.i+1]**2

                if crossCorrelator is not None:
                    self.crossC[self.i+1] = calculate_crossCorrelator(self,p,crossCorrelator) - self.y[crossCorrelator[0],self.i+1]*self.y[crossCorrelator[1],self.i+1]
                
                self.i += 1
                self.t += self.delta_t
                
                if measureResponse is not None:                
                    perturb_measure_response(self,p,measureResponse,measureResponse_par,max_num)
                
                pbar.update(1)
        
        if measureResponse is not None:
            normalize_response(self,measureResponse,measureResponse_par)
        
    elif method == 'RK2':

        sol = sc.integrate.solve_ivp(wrapper,(self.timeGrid[0],self.timeGrid[-1]),init_dist,method='RK23',t_eval=self.timeGrid,dense_output=False)

        with tqdm(total=len(self.timeGrid)) as pbar:
            while self.i < len(self.timeGrid):
                self.y[:,self.i] = calculate_mean(self,sol.y[:,self.i])
                
                if variance:
                    self.variance[:,self.i] = calculate_secondMoment(self,sol.y[:,self.i]) - self.y[:,self.i]**2
                
                if crossCorrelator is not None:
                    self.crossC[self.i] = calculate_crossCorrelator(self,sol.y[:,self.i],crossCorrelator) - self.y[crossCorrelator[0],self.i]*self.y[crossCorrelator[1],self.i]

                self.i += 1
                self.t += self.delta_t

                pbar.update(1)
                
        if measureResponse is not None:
            print('Method and measurement are incompatible')

    elif method == 'RK4':

        sol = sc.integrate.solve_ivp(wrapper,(self.timeGrid[0],self.timeGrid[-1]),init_dist,method='RK45',t_eval=self.timeGrid,dense_output=False)

        with tqdm(total=len(self.timeGrid)) as pbar:
            while self.i < len(self.timeGrid):
                self.y[:,self.i] = calculate_mean(self,sol.y[:,self.i])
                
                if variance:
                    self.variance[:,self.i] = calculate_secondMoment(self,sol.y[:,self.i]) - self.y[:,self.i]**2
                    
                if crossCorrelator is not None:
                    self.crossC[self.i] = calculate_crossCorrelator(self,sol.y[:,self.i],crossCorrelator) - self.y[crossCorrelator[0],self.i]*self.y[crossCorrelator[1],self.i]

                self.i += 1
                self.t += self.delta_t

                pbar.update(1)

        if return_probability is not False:
            self.probability = sol.y
                
        if measureResponse is not None:
            print('Method and measurement are incompatible')

    elif method == 'Eigenvalue':

        evalue,evRight = np.linalg.eig(self.master)
        evLeft         = np.linalg.inv(evRight)
        ini_proj       = np.matmul(evLeft,init_dist)
        self.evLeft    = evLeft
        self.evRight   = evRight

        with tqdm(total=len(self.timeGrid)) as pbar:
            
            # Note that this is function is different here -- diagonalization at every step! TODO -- seems to be wrong         
            if measureResponse == 'impulse':
                k1_true     = np.copy(self.k1)
                master_true = np.copy(self.master)
                self.k1     = self.k1*measureResponse_impluse
                masterOperator(self,max_num)

                evalue_new,evRight_new = np.linalg.eig((self.master-master_true)/(k1_true*(measureResponse_impluse-1)))
                tau = self.i
                q   = np.copy(p)
                q   = np.matmul((self.master-master_true)/(k1_true*(measureResponse_impluse-1)),q)

                while tau < len(self.timeGrid):
                    self.Response[:,tau,self.i] = calculate_mean(self,q)
                    q += self.delta_t*np.matmul(master_true,q)
                    tau += 1
                self.k1 = k1_true
                masterOperator(self,max_num)
                
            elif measureResponse is not None:
                print('Method and measurement are incompatible')

            while self.i < len(self.timeGrid):

                # TODO: Implement less matrix multiplication here, will be faster
                p = np.real(np.matmul(evRight, np.matmul( np.identity(len(evalue))* np.exp(evalue*self.timeGrid[self.i]), ini_proj)))

                self.y[:,self.i] = calculate_mean(self,p)

                if selfCorrelator_tau is not None:
                    Q = []

                    for j in range(self.num_species):
                        Q.append(np.array(self.master_stateSpace)[:,j]*p)

                    tau = 0

                    while tau+self.i<len(self.timeGrid):
                        
                        if selfCorrelator_tau == 'connected':
                            y_ = calculate_mean(self,np.real(np.matmul(evRight, np.matmul( np.identity(len(evalue))* np.exp(evalue*self.timeGrid[self.i+tau]), ini_proj)))) 

                        for j in range(self.num_species):

                            correlator_tau[j,tau,self.i] = np.sum(np.array(self.master_stateSpace)[:,j]*Q[j],axis=0)

                            if selfCorrelator_tau == 'connected':

                                correlator_tau[j,tau,self.i] -= self.y[j,self.i]*y_[j]

                            Q[j] = np.real(np.matmul(evRight,np.matmul(np.identity(len(evalue))* np.exp(evalue*self.delta_t),np.matmul(evLeft,Q[j]))))

                        tau += 1

                if crossCorrelator_tau is not None:
                    Q = []

                    for j in range(self.num_species):
                        Q.append(np.array(self.master_stateSpace)[:,j]*p)

                    tau = 0

                    while tau+self.i<len(self.timeGrid):
                        if crossCorrelator_tau == 'connected':
                            y_ = calculate_mean(self,np.real(np.matmul(evRight, np.matmul( np.identity(len(evalue))* np.exp(evalue*self.timeGrid[self.i+tau]), ini_proj)))) 

                        for j in range(self.num_species):
                            for j1 in range(self.num_species):

                                correlator_tau[j,j1,tau,self.i] = np.sum(np.array(self.master_stateSpace)[:,j]*Q[j1],axis=0)
                                if crossCorrelator_tau == 'connected':

                                    correlator_tau[j,j1,tau,self.i] -= self.y[j1,self.i]*y_[j]
                                    # species j is at time tau+self.i and species j1 at time self.i

                        for j in range(self.num_species):
                            Q[j] = np.real(np.matmul(evRight,np.matmul(np.identity(len(evalue))*np.exp(evalue*self.delta_t),np.matmul(evLeft,Q[j]))))

                        tau += 1

                if variance:
                    self.variance[:,self.i] = calculate_secondMoment(self,p) - self.y[:,self.i]**2

                if crossCorrelator is not None:
                    self.crossC[self.i] = calculate_crossCorrelator(self,p,crossCorrelator) - self.y[crossCorrelator[0],self.i]*self.y[crossCorrelator[1],self.i]

                self.i += 1
                self.t += self.delta_t

                pbar.update(1)

            # TODO: Implement the operator that counts the mean number of molecules

    if selfCorrelator_tau is not None:
        self.correlator_tau = correlator_tau

    if crossCorrelator_tau is not None:
        self.correlator_tau = correlator_tau
        