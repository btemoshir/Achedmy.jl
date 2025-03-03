import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22,'font.family':'arial','text.usetex':False})
from scipy.integrate import odeint
import sys
import os
import itertools as it
import math
from functools import reduce
from operator import sub
#from numba.experimental import jitclass
from tqdm import tqdm
#from numba import int32, float64
from scipy.stats import poisson

#spec = [
#    ('time_grid', float32[:]), ('num_int', int32), ('r_i', int32) , ('s_i', int32)         
#    ('num_species', int32), , ('k1', float32[:]), ('k1', float64), ('k2', float64[:]), ('k3', int32[:,:])
#]

def doubleFactorialOdd(n):
    """Takes an array n of integer enteries and returns its double factorial only 
    if the number is Odd, returns 0 otherwise, if the number is 0, it returns 0."""
    
    try:
        len(n)
    except:
        n = np.array([n])
    
    x = np.zeros(len(n))
    
    for i in range(len(n)):
        if n[i] < 0:
            #x[i] = 0
            if n[i]%2 != 0:
                x[i] = 1
            else:
                x[i] = 0
        else:
            if n[i]%2 == 0:
                x[i] = 0
            else:
                x[i] = reduce(int.__mul__, range(int(n[i]), 0, -2))
        
    return x

def HalfonlyEven(n):
    """Takes an array n of integer enteries and returns half the value if its even and 0 otherwise"""
    
    try:
        len(n)
    except:
        n = np.array([n])
    
    x = np.zeros(len(n))
    
    for i in range(len(n)):
        if n[i]%2 == 0:
            x[i] = n[i]/2
        else:
            x[i] = 0
    
    return x

def unitVector(n):
    """Checks if the given vector n is a simple basis vector that is one non-zero value and the rest zeros.
    Returns 1, the position of the non-zero value and its value if its true and zero,zero otherwise."""
    
    n = np.array(n)
    
    if len(n[n==0])!= len(n)-1:
        return 0,0,0
    else:
        return 1,np.where(n!=0)[0][0],n[np.where(n!=0)][0]
    
    
def block_tri_lower_inverse_old(mat):
    """
    Calculates the inverse of a lower triangular matrix where each elemet itself is a block matrix. Used for mixed SBR series!
    mat should have the first two dimensions coressponding to the block of n, the last two to the time. Assumes there are 1's on the diagonal!
    
    """
    
    len_rows = np.shape(mat)[2]
    len_n    = np.shape(mat)[0]
    inv      = np.zeros(np.shape(mat))
    
    #Do the diagonals first:
    for diag in range(len_rows):
        inv[:,:,diag,diag] = np.linalg.inv(mat[:,:,diag,diag]) 
        #inv[:,:,diag,diag] = mat[:,:,diag,diag]
        
    for row in range(1,len_rows):
        for col in range(row):
            temp = np.zeros(np.shape(mat)[:2])
            for col2 in range(col,row):
                temp += np.matmul(mat[:,:,row,col2],inv[:,:,col2,col])                  
            
            inv[:,:,row,col] = -np.matmul(inv[:,:,row,row],temp)  
   
    return inv

def block_tri_lower_inverse(mat):
    """
    Calculates the inverse of a lower triangular matrix where each elemet itself is a block matrix. Used for mixed SBR series!
    mat should have the first two dimensions coressponding to the block of n, the last two to the time. Assumes there are 1's on the diagonal!
    
    """
    
    len_rows = np.shape(mat)[2]
    len_n    = np.shape(mat)[0]
    inv      = np.zeros(np.shape(mat))
    inv[:,:,0,0] = np.linalg.inv(mat[:,:,0,0]) 
    
    #optimize before or use greedy optimization for further speed up!
    
    for row in range(1,len_rows):
        D_inv  = np.linalg.inv(mat[:,:,row,row]) #Can get rid of this in principle!
        temp_n = np.einsum('ijl,jmlo->imo',mat[:,:,row,:row+1],inv[:,:,:row+1,:row+1],optimize='greedy')
        inv[:,:,row,:row+1] = np.einsum('ij,jmk->imk',D_inv,(block_identity(len_n,row+1)[:,:,row]-temp_n[:,:,:row+1]), optimize='greedy')
        
        #print(D_inv)
    return inv
    
def block_mat_mul(mat1,mat2):
    """
    Calculates the matrix product between two compatible matrices etc which have the block structure! The first two indices corresspond to the list index.
    """
    
    len_rows = np.shape(mat1)[2]
    mul      = np.zeros(np.shape(mat1))
    
    for i in range(len_rows):
        for j in range(len_rows):
            for k in range(len_rows):
                mul[:,:,i,j] += np.matmul(mat1[:,:,i,k],mat2[:,:,k,j])
    
    mul = np.einsum('ijkl,jnlo->inko',mat1,mat2,optimize='optimal')
    
    return mul

def block_lower_shift(mat):
    
    len_rows = np.shape(mat)[2]
    #len_n    = np.shape(mat)[0]
    #shifted  = np.zeros(np.shape(mat))
    L        = np.diag(np.ones(len_rows-1),k=-1)
    
    #for i in range(len_n):
        #for j in range(len_n):
            #shifted[i,j] = np.matmul(mat[i,j],L)
    
    shifted = np.einsum('ijkl,lm->ijkm',mat,L,optimize='optimal')
    
    return shifted
    

def block_mat_mix_mul(mat1,mat2):
    """
    Calculates the matrix product between a matrix which has the block structure with another matrix which has a structure with only one n but two time indices! The first two indices corresspond to the list index.
    """
    
    #len_rows = np.shape(mat1)[2]
    #mul      = np.zeros([np.shape(mat1)[0],np.shape(mat1)[2],np.shape(mat1)[3]])
    
    #for i in range(len_rows):
        #for j in range(len_rows):
            #for k in range(len_rows):
                #mul[:,i,j] += np.matmul(mat1[:,:,i,k],mat2[:,k,j])
    
    mul = np.einsum('ijkl,jlm->ikm',mat1,mat2,optimize='optimal')
    
    return mul
    
def block_mat_vec_mul(mat,vec):
    """
    Calculates the matrix product between a compatible matrix and a vector etc which have the block structure! The first two indices corresspond to the list index.
    """
    
    len_rows = np.shape(mat)[2]
    mul      = np.zeros([np.shape(mat)[0],np.shape(mat)[2]])
    
    for i in range(len_rows):
        for k in range(len_rows):
            mul[:,i] += np.matmul(mat[:,:,i,k],vec[:,k])
            
    return mul

def block_vec_mat_mul(vec,mat):
    """
    Calculates the matrix product between a compatible vector and a matrix which have the block structure! The first two indices corresspond to the list index.
    """
    
    len_rows = np.shape(mat)[2]
    mul      = np.zeros([np.shape(mat)[0],np.shape(mat)[2]])
    
    for i in range(len_rows):
        for k in range(len_rows):
            mul[:,i] += np.matmul(vec[:,k],mat[:,:,k,i])
            
    return mul
    
    
def block_identity(dim_list,dim_time):
    """
    Creates the identity for the block matrix system
    """
    
    identity = np.zeros([dim_list,dim_list,dim_time,dim_time])
    
    for i in range(dim_time):
        for j in range(dim_list):
            identity[j,j,i,i] = 1.
        
    return identity
    
#@jitclass(spec)

class plefka_system:
    
    """ 
    Class with methods on it to set up a Plefka system and run dynamics on it with linear Order Parameters or Quadratic Order parameters upto second order in the expansion parameter: alpha.
    
    Initialization inputs:
    
    - num_int: Number of interacting reactions over the baseline
    - num_species: Total number of reaction species in the system
    - rxn_par: A list of length 3 of arrays with the first array as all the creation rates of the baseline, the second as the destruction rate of the baseline and the third array with the rates of the interaction reactions (the length of first two must be equal to num_sepcies and of the third must be equal to num_int)
    - r_i: A list of list with length = num_int. Each sublist has length = num_species. This list defines the stochiometric coefficients of all the species being destroyed (reactant) as a result of the interaction reactions in order of the rates defined in rxn_par[2]
    - s_i: Same as r_i but for the species being created (product) in the interaction reactions.
    
    Author: Moshir Harsh
    moshir.harsh@theorie.physik.uni-goettingen.de
    
    """
    
    def __init__(self,num_int,num_species,rxn_par,r_i,s_i,alpha=1.):
        
        # Assign the numbers to the initial value
        self.num_species   = num_species
        self.num_int       = num_int
        self.r_i           = r_i
        self.s_i           = s_i
        self.num_reactions = 2*num_species + num_int
        self.alpha         = alpha
        
        
        # Assign reaction parameters
        self.k1 = rxn_par[0]
        self.k2 = rxn_par[1]
        self.k3 = rxn_par[2]
    
    def gillespie_timegrid(self,initial_values,max_timesteps):
        
        """Outputs the results of the simulation at the time_grid"""
        
        self.y        = np.zeros([self.num_species,len(self.timeGrid)])
        self.y[:,0]   = np.random.poisson(initial_values)
        self.i        = 0
        self.t        = 0.
    
        # These variables have scope only to this function
        
        y             = np.zeros([self.num_species,max_timesteps])
        y[:,0]        = self.y[:,0]
        rate          = np.zeros(self.num_reactions)
        t             = 0
        i             = 0
        
        while(self.i < len(self.timeGrid)-1):

            while(t < self.timeGrid[self.i+1]):

                # Calculate the reaction probability vector:

                rate[:self.num_species] = self.k1
                rate[self.num_species:2*self.num_species] = self.k2*y[:,i]

                for m in range(self.num_int):
                    rate[2*self.num_species+m] = self.k3[m]
                    for k in range(self.num_species):
                        #rate[(2*self.num_species)+m] *= y[k,i]**(self.r_i[m,k])
                        for p in range(int(self.r_i[m,k])):
                            if y[k,i] > p:
                                rate[(2*self.num_species)+m] *= (y[k,i]-p)
                            else:
                                rate[(2*self.num_species)+m] *= 0.


                rate_total = np.sum(rate)

                #Sample the time to the next reaction from an exponential distribution with mean = 1/rate_total 
                dt = np.random.exponential(scale = 1./rate_total)

                #Increase time
                t += dt
                #time_reaction[i+1] = t

                #Choose which reaction will occur based on the probability
                reaction_occur = np.random.choice(self.num_reactions, p = rate/rate_total) 

                #Update y i.e the state space
                y[:,i+1] = y[:,i]

                if reaction_occur < self.num_species:    
                    y[reaction_occur,i+1] = y[reaction_occur,i] + 1 #Creation reaction

                elif self.num_species-1 < reaction_occur and reaction_occur < 2*self.num_species:
                    y[reaction_occur-self.num_species,i+1] = max(y[reaction_occur-self.num_species,i] - 1,0) #Destruction reaction

                else:
                    # Do the reaction only when the number of molecules > 1 of all types reacting
                    if (y[(self.r_i[reaction_occur-2*self.num_species].astype(bool)),i]-1).all():
                        y[:,i+1] = y[:,i] - self.r_i[reaction_occur-2*self.num_species,:] +\
                        self.s_i[reaction_occur-2*self.num_species,:]

                i += 1

            #Update x only if the time crosses a certain grid point:
            self.y[:,self.i+1] = y[:,i]

            #Increase step
            self.i += 1
            self.t += self.delta_t
    
    def gillespie(self,initial_values,max_timesteps,endTime,alpha=1.,initialization='poisson'):
        
        """Outputs the results of the simulation"""
        
        # These variables have scope only to this function
        
        y             = np.zeros([self.num_species,max_timesteps])
        rate          = np.zeros(self.num_reactions)
        t             = 0.
        i             = 0
        time_rxn      = np.zeros(max_timesteps)
        
        if initialization is 'poisson':
            y[:,0]        = np.random.poisson(initial_values)
        elif initialization is 'fixed':
            y[:,0]        = initial_values

        while(i < max_timesteps-1 and t < endTime):

            # Calculate the reaction probability vector:
            rate[:self.num_species] = self.k1
            rate[self.num_species:2*self.num_species] = self.k2*y[:,i]

            for m in range(self.num_int):
                rate[2*self.num_species+m] = alpha*self.k3[m]
                for k in range(self.num_species):
                    #rate[(2*self.num_species)+m] *= y[k,i]**(self.r_i[m,k])
                    for p in range(int(self.r_i[m,k])):
                        if y[k,i] > p:
                            rate[(2*self.num_species)+m] *= (y[k,i]-p)
                        else:
                            rate[(2*self.num_species)+m] *= 0.
                            # This already puts the rate of this reaction to zero if the number of molecules to react are not enough.
            #print(rate)
            rate_total = np.sum(rate)
            
            if rate_total > 0:
            
                #Sample the time to the next reaction from an exponential distribution with mean = 1/rate_total 
                dt = np.random.exponential(scale = 1./rate_total)

                #Choose which reaction will occur based on the probability
                reaction_occur = np.random.choice(self.num_reactions, p = rate/rate_total) 

                #Update y i.e the state space
                y[:,i+1] = y[:,i]

                if reaction_occur < self.num_species:    
                    y[reaction_occur,i+1] += 1 #Creation reaction

                elif self.num_species-1 < reaction_occur and reaction_occur < 2*self.num_species:
                    #if y[reaction_occur-self.num_species,i] > 1:
                    y[reaction_occur-self.num_species,i+1] = max(y[reaction_occur-self.num_species,i]-1,0) #Destruction reaction

                else:
                    #if (y[(self.r_i[reaction_occur-2*self.num_species].astype(bool)),i]).all():
                    y[:,i+1] += - self.r_i[reaction_occur-2*self.num_species,:] + self.s_i[reaction_occur - 2*self.num_species,:]

                #Increase time
                t            += dt
                time_rxn[i+1] = t
                i            += 1
                
            else:
                t = endTime
                temp_idx = 1
                if temp_idx <max_timesteps:
                    y[:,i+temp_idx] = y[:,i]
                    temp_idx += 1
                time_rxn[i+1:] = endTime
        
        return [time_rxn,y]
        
    def gillespie_transform_timeGrid(self,gill,timeGrid,delta_t):
        
        """
        Now we put this on a time grid self.t and the concentrations in self.y
        
        """
        z       = np.zeros([self.num_species,len(timeGrid)])
        z[:,0]  = gill[1][:,0]
        i       = 1
        t       = 1

        while i < len(timeGrid):
            while gill[0][t] < timeGrid[i]:
                t += 1
            z[:,i] = gill[1][:,t-1]
            i += 1
            
        return z
   
    def gillespie_avg(self,num_repeats,initial_values,startTime,endTime,delta_t,max_timesteps,alpha=1., initialization='poisson'):
              
        time_grid       = np.arange(startTime,endTime,delta_t)
        self.timeGrid   = time_grid
        gill            = np.zeros([num_repeats,self.num_species,len(time_grid)])
        self.delta_t    = delta_t
        self.y          = np.zeros([self.num_species,len(self.timeGrid)])
        self.i          = 0
        self.t          = 0.

        for i in tqdm(range(num_repeats)):
            gill[i] = self.gillespie_transform_timeGrid(self.gillespie(initial_values,max_timesteps,endTime,alpha,initialization), self.timeGrid,self.delta_t)
            #print("repeat " + str(i))
        
        self.y          = np.mean(gill,axis=0,keepdims=False)
        self.gill_stdev = np.std(gill,axis=0,keepdims=False)/np.sqrt(num_repeats)
        #self.gill_stdev = np.std(gill,axis=0,keepdims=False)/np.sqrt(num_repeats)
        self.t          = self.timeGrid[-1]
        self.i          = len(self.timeGrid)
        
    def initializeMAK(self,initial_values,startTime,endTime,delta_t,alpha=1.):
        
        """
        Initializes the Mass action kinetics dynamics for the reaction. Arguments names are self explanatory.
        
        """
        
        self.MAK      = True
        time_grid     = np.arange(startTime,endTime,delta_t)
        self.y        = np.zeros([self.num_species,len(time_grid)])
        self.timeGrid = time_grid
        self.y[:,0]   = initial_values
        self.i        = 0
        self.t        = 0.
        self.delta_t  = delta_t
        self.alpha    = alpha
        self.EMRE     = False
        
    def initializeEMRE(self,initial_values,startTime,endTime,delta_t,alpha=1.,volume=1.,measureResponse=False,crossCorrelator_tau=False):
        """
        Initializes the Effective Mesocopic Rate equations (System size expansion) or the LNA with also correction to the means. An additional input argument is volume, which as defined in the volumen scaling.
        """
        
        self.EMRE     = True
        time_grid     = np.arange(startTime,endTime,delta_t)
        self.y        = np.zeros([self.num_species,len(time_grid)])
        self.timeGrid = time_grid
        self.y[:,0]   = initial_values
        self.i        = 0
        self.t        = 0.
        self.delta_t  = delta_t
        self.alpha    = alpha
        self.volume   = volume
        
        self.eps      = np.zeros([self.num_species,len(time_grid)]) #These are the correction to the MAK means!
        self.lna_var  = np.zeros([self.num_species,self.num_species,len(time_grid)]) #These are the cross sepcies variances around the MAK means!
        
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
            
            #for sp in range(self.num_species):
            #    self.resp[sp,sp,0,0] = 1.
        
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
        
        
        
    def initializePlefka(self,orderParameter,alphaOrder,alpha,C=True,fieldMethod=None,crossResponse=False,R_regularize=False, Theta_regularize=False,RidgeRegularizer=0., GaussianRegularizer=0.,initialization='Poisson'):
        
        """ 
        Initializes the Plefka system by defining the parameters required for Plefka.
        
        Inputs:
        
        - orderParameter: 'linear' or 'quad'
        - alphaOrder: 1 or 2, defines the expansion order
        - alpha: 0 < alpha < 1. Defines the value for the expansion parameter
        - C: Whether we constraint the correlation function in the Plefka free energy or not. By default it is True.
        - fieldMethod: The method of implementing the Plefka first and/or second order fields.
                      Options - 'None': Simple method which calculates c_mn by summing over all reactions first. 
                                'split': This splits the c_mn over different reactions and calculates pairwise products over the different reactions.
                                'split_ABC': Specific for the case for just 3 species with only A+B makes C interacting reaction. This method implements analytically derived explicit equations for the fields for this case.
                                
        - R_regularize: Whether to regularize the response function or not. ONLY useful for Quad OP at alpha^2. The response functions have a divergence when the second order \hat{R} becomes larger than the first order \hat{R}. This takes care of that.
                        Options - 'False': No regularization
                                - 'exp_smooth': Smoothly go to just using the first order field if the second order field becomes larger in magnitude with alpha as the scale of the exponential. DOES NOT INTRODUCE another TIMESCALE!
                                - 'max_1': Uses a regularizing function of the form: -alpha*R1/(1-0.5*alpha*R2/R1) if R1 != 0
                                - 'max_2': Uses a regularizing function of the form: -alpha*R1/sqrt(1-alpha*R2/R1) if R1 != 0
                                - 'max_3': Uses a regularizing function of the form: -alpha*R1*exp(0.5*alpha*R2/R1) if R1 != 0
                                
        - Theta_regularize: The same as R_regularize but for the Theta fields with the same possible options!
        
        - RidgeRegularizer: A float. The coefficient, lambda of the Ridge Regularizer. By deafult = 0. 
        
        - initialization: 'Poisson' or 'Fixed'
       
        """
        
        self.alpha          = alpha
        self.orderParameter = orderParameter
        self.alphaOrder     = alphaOrder
        self.mnList()
        self.exclusion_list = [list(np.zeros(self.num_species,dtype=int))]
        self.exclusion_list_bubble = [list(np.zeros(self.num_species,dtype=int))]
        self.C              = C
        self.fieldMethod    = fieldMethod
        self.EMRE           = False
        self.crossResponse  = crossResponse      
        
        self.R_regularize       = R_regularize
        self.Theta_regularize   = Theta_regularize
        self.RidgeRegularizer   = RidgeRegularizer
        self.GaussianRegularizer = GaussianRegularizer
        self.initialization     = initialization
        
        if self.orderParameter is 'linear':
            if self.alphaOrder == 1:
                t_array = []
            
            elif self.alphaOrder == 2:
                for k in range(self.num_species):
                    t_array    = list(np.zeros(self.num_species,dtype=int))
                    t_array[k] = 1
                    self.exclusion_list.append(t_array)
                    self.exclusion_list_bubble.append(t_array)
            
        elif self.orderParameter is 'quad':
            if self.alphaOrder == 1:
                t_array = []
            elif self.alphaOrder == 2:
                for k in range(self.num_species):
                    t_array1    = list(np.zeros(self.num_species,dtype=int))
                    t_array2    = list(np.zeros(self.num_species,dtype=int))
                    t_array1[k] = 1
                    self.exclusion_list.append(t_array1)
                    self.exclusion_list_bubble.append(t_array1)
                    t_array2[k] = 2
                    #self.exclusion_list.append(t_array2)  # Review this, should this term be included or not?? -- Should not be there!     
                    
    
    def initialize_dynamics(self,initial_values,startTime,endTime,delta_t,plefka=True):
        
        """
        Initializes any type of dynamics. Arguments self explanatory.
        
        """
        
        time_grid     = np.arange(startTime,endTime,delta_t)
        self.y        = np.zeros([self.num_species,len(time_grid)])
        self.timeGrid = time_grid
        self.y[:,0]   = initial_values
        self.i        = 0
        self.t        = 0.
        self.delta_t  = delta_t
        
        #if self.initialization == 'Fixed':
            #self.k1 += self.y[:,0]
            #print(self.k1)
        
        if plefka:
            # Assign response and correlation functions
            if not self.crossResponse:
                self.resp     = np.zeros([self.num_species,len(time_grid),len(time_grid)])
                try:
                    self.C
                    if self.C:
                        self.corr     = np.zeros([self.num_species,len(time_grid),len(time_grid)])
                except:
                    pass

                for i in range(self.num_species):
                    self.resp[i] = np.identity(len(time_grid))

                if self.orderParameter == 'linear':

                    self.hatTheta1 = np.zeros([self.num_species,len(time_grid)])

                    if self.alphaOrder == 2:
                        self.hatTheta2 = np.zeros([self.num_species,len(time_grid)])

                elif self.orderParameter == 'quad':

                    self.hatTheta1 = np.zeros([self.num_species,len(time_grid)])
                    self.hatR1     = np.zeros([self.num_species,len(time_grid),len(time_grid)])
                    self.hatB1     = np.zeros([self.num_species,len(time_grid),len(time_grid)])


                    if self.alphaOrder == 2:
                        self.hatR2     = np.zeros([self.num_species,len(time_grid),len(time_grid)])
                        self.hatB2     = np.zeros([self.num_species,len(time_grid),len(time_grid)])
                        self.hatTheta2 = np.zeros([self.num_species,len(time_grid)])
                        
            else:
                self.resp     = np.zeros([self.num_species,self.num_species,len(time_grid),len(time_grid)])
                try:
                    self.C
                    if self.C:
                        self.corr     = np.zeros([self.num_species,self.num_species,len(time_grid),len(time_grid)])
                except:
                    pass

                for i in range(self.num_species):
                    self.resp[i,i] = np.identity(len(time_grid))

                if self.orderParameter == 'linear':

                    self.hatTheta1 = np.zeros([self.num_species,len(time_grid)])

                    if self.alphaOrder == 2:
                        self.hatTheta2 = np.zeros([self.num_species,len(time_grid)])

                elif self.orderParameter == 'quad':

                    self.hatTheta1 = np.zeros([self.num_species,len(time_grid)])
                    self.hatR1     = np.zeros([self.num_species,self.num_species,len(time_grid),len(time_grid)])
                    self.hatB1     = np.zeros([self.num_species,self.num_species,len(time_grid),len(time_grid)])


                    if self.alphaOrder == 2:
                        self.hatR2     = np.zeros([self.num_species,self.num_species,len(time_grid),len(time_grid)])
                        self.hatB2     = np.zeros([self.num_species,self.num_species,len(time_grid),len(time_grid)])
                        self.hatTheta2 = np.zeros([self.num_species,len(time_grid)])
        
        
    def resetDynamics(self):
        
        del self.y, self.timeGrid, self.i, self.t, self.delta_t, self.resp, self.corr, self.hatR1, self.hatB1
        del self.hatR2, self.hatB2, self.hatTheta1, self.hatTheta2

    
    def mnList(self):
        
        """
        Creates the list of m and n vectors over which we need to sum over to calculate the fields. The FULL list must be used if one does not use the split fieldMethod since that generates these vectors for all reactions. 
        
        """
        
        self.m_list     = []
        self.n_list     = []
        self.m_listFULL = []
        self.n_listFULL = []
        
        for i in range(self.num_int):
            x = list(it.product(*list(np.arange(int(max(self.s_i[i][j],self.r_i[i][j])+1)) for j in range(self.num_species))))
            y = list(it.product(*list(np.arange(int(self.r_i[i][j]+1)) for j in range(self.num_species))))
        
            for j in range(len(x)):
                x[j] = list(x[j])
            for j in range(len(y)):
                y[j] = list(y[j])
            
            self.m_list.append(x)
            self.n_list.append(y)
        
        max1 = np.zeros(self.num_species,dtype=int)
        max2 = np.zeros(self.num_species,dtype=int)
        
        for j in range(self.num_species):
            max1[j] = int(max(list(self.s_i[i][j] for i in range(self.num_int))+ list(self.r_i[i][j] for i in range(self.num_int)))+1)
            max2[j] = int(max(list(self.r_i[i][j] for i in range(self.num_int)))+1)
        
        a = list(it.product(*list(np.arange(max1[j]) for j in range(self.num_species))))
        b = list(it.product(*list(np.arange(max2[j]) for j in range(self.num_species))))
        
        for j in range(len(a)):
            a[j] = list(a[j])
        for j in range(len(b)):
            b[j] = list(b[j])
        
        self.m_listFULL.append(a)
        self.n_listFULL.append(b)
        
        del a,b,x,y,max1,max2
        
    def c_mn(self,int_rxn_index,m,n,i):
        """
        Creates the c_mn for each individual beta reaction (including the \mu factors)
        """
        
        if i > 0:
            if any((n-self.r_i[int_rxn_index])>0):
                c_mnBeta = 0.

            else:
                c_mnBeta = self.k3[int_rxn_index]*(np.prod(sc.special.binom(self.s_i[int_rxn_index],m) *np.power(1,(self.s_i[int_rxn_index]-m))) - np.prod(sc.special.binom(self.r_i[int_rxn_index],m) *np.power(1,(self.r_i[int_rxn_index]-m) ))) *np.prod(sc.special.binom(self.r_i[int_rxn_index],n) *np.power(self.y[:,i-1],(self.r_i[int_rxn_index]-n)))
            
            return c_mnBeta
        
        else:
            return 0.
        
    def c_mn_no_mu(self,int_rxn_index,m,n):
        """
        Creates the c_mn for each individual beta reaction (without the \mu factors)
        """
        
        if any((n-self.r_i[int_rxn_index])>0):
            c_mnBeta = 0.

        else:
            c_mnBeta = self.k3[int_rxn_index]*(np.prod(sc.special.binom(self.s_i[int_rxn_index],m) *np.power(1,(self.s_i[int_rxn_index]-m))) - np.prod(sc.special.binom(self.r_i[int_rxn_index],m) *np.power(1,(self.r_i[int_rxn_index]-m) ))) *np.prod(sc.special.binom(self.r_i[int_rxn_index],n))
            
        return c_mnBeta
    
    def create_c_mn_dict(self):
        """
        Creates a dictionary for the c_mn for each individual beta reaction (without the \mu factors) where the output is a dictionary and can be looked up by [int(rxn_index),tuple(m),tuple(n)]
        """
        
        c_mn_dict = {}
        self.c_mn_dict = {}
        
        for int_rxn_index in range(self.num_int):
            for m in self.m_list[int_rxn_index]:
                for n in self.n_list[int_rxn_index]:
                    c_mn_dict[(int_rxn_index,tuple(m),tuple(n))] = self.c_mn_no_mu(int_rxn_index,m,n)

        self.c_mn_dict = c_mn_dict
    
    def c_mnStar(self,int_rxn_index,m,n,i):
  
        m1,m2,m3 = unitVector(m) 
        n1,n2,n3 = unitVector(n)
        
        
        c_mnStarBeta = self.c_mn(int_rxn_index,m,n,i)
        
        changed = False
        if i < len(self.timeGrid):
            if m1 == 1:
                if m3 == 1:
                    if n == list(np.zeros(len(n),dtype=int)):
                        c_mnStarBeta += self.hatTheta1[m2,i]
                        changed       = True
                    elif n1 == 1 and n2 == m2:
                        c_mnStarBeta += self.delta_t*self.hatR1[m2,i,i-1]
                        changed = True
                elif m3 == 2:
                    if n == list(np.zeros(len(n),dtype=int)):
                        c_mnStarBeta += self.delta_t*self.hatB1[m2,i,i]
                        changed = True
            
        return c_mnStarBeta
    
    def c_mnFULL(self,m,n,i):
        
        a = 0.
        for k in range(self.num_int):
            a += self.c_mn(k,m,n,i)
        return a
    
    def c_mnStarFULL(self,m,n,i):
        
        a = 0.
        for k in range(self.num_int):
            a += self.c_mnStar(k,m,n,i)
        return a
        
                        
    def massActionKinetics(self):
        
        if self.k3 is float:
            self.k3 = np.array(k3)

        dydt = np.zeros(self.num_species)
        dydt = self.k1 - self.k2*self.y[:,self.i] 

        for j in range(self.num_int):
            x = self.k3[j]*self.alpha
            for k in range(self.num_species):
                x *= self.y[k,self.i]**(self.r_i[j,k]) 
            #dydt[self.r_i[j].astype(bool)] -= x
            #dydt[self.s_i[j].astype(bool)] += x
            for k in range(self.num_species):
                dydt[k] += (self.s_i[j,k] - self.r_i[j,k])*x

        return dydt
    
    def runDynamics_EMRE(self):
        
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
                
                
                
    
    def massActionKineticsEqn(self,y):
        
        if self.k3 is float:
            self.k3 = np.array(k3)

        dydt = np.zeros(self.num_species)
        dydt = self.k1 - self.k2*y 

        for j in range(self.num_int):
            x = self.k3[j]*self.alpha
            for k in range(self.num_species):
                x *= y[k]**(self.r_i[j,k]) 
            #dydt[self.r_i[j].astype(bool)] -= x
            #dydt[self.s_i[j].astype(bool)] += x
            for k in range(self.num_species):
                dydt[k] += (self.s_i[j,k] - self.r_i[j,k])*x

        return dydt
    
    def massActionKineticsEqn_forParameters(self,par):
        
        # Don't keep all parameters variable, just a subset, for the others still use self.k_
        
        y  = self.steadyState
        k1 = par
        
        if self.k3 is float:
            self.k3 = np.array(self.k3)

        dydt = np.zeros(self.num_species)
        dydt = k1 - self.k2*y 

        for j in range(self.num_int):
            x = self.k3[j]*self.alpha
            for k in range(self.num_species):
                x *= y[k]**(self.r_i[j,k])
            #dydt[self.r_i[j].astype(bool)] -= x
            #dydt[self.s_i[j].astype(bool)] += x
            for k in range(self.num_species):
                dydt[k] += (self.s_i[j,k] - self.r_i[j,k])*x

        return dydt
        
    def massActionKineticsEqn_allBaseline(self,par,penalty,par_ini):
        
        y  = self.steadyState
        k1 = par[0:self.num_species]
        k2 = par[self.num_species:]
        
        if self.k3 is float:
            self.k3 = np.array(self.k3)

        dydt = np.zeros(self.num_species)
        dydt = k1 - k2*y 

        for j in range(self.num_int):
            x = self.k3[j]*self.alpha
            for k in range(self.num_species):
                x *= y[k]**(self.r_i[j,k])
            #dydt[self.r_i[j].astype(bool)] -= x
            #dydt[self.s_i[j].astype(bool)] += x
            for k in range(self.num_species):
                dydt[k] += (self.s_i[j,k] - self.r_i[j,k])*x

        return 0.5*np.sum((dydt)**2) + 0.5*penalty*np.sum((par-par_ini)**2)
    
    
    def findSteadyState(self,startGuess):
        
        """
        Find the Mass Action kinetics steady state by solving those equations in steady state.
        
        """
        
        roots = sc.optimize.root(self.massActionKineticsEqn,startGuess)
        self.steadyState = roots.x
        
        
    def par_constantSS(self,par_new,par_old_set):
        
        """
        Outputs the parameter that will give the same value of MAK steady state if we were to change an other parameter of the system. 
        
        """
                
        self.findSteadyState(self.y[:,0])
        self.k3 = par_new
        k1      = par_old_set[0]
        
        new_par = sc.optimize.root(self.massActionKineticsEqn_forParameters,k1)
        
        self.k3 = par_old_set[2]
        
        return new_par
    
    def par_constantSS_k1direct(self,par_new,par_old_set):
        
        """
        Outputs the parameter, k_1 that will give the same value of MAK steady state if we were to change another parameter of the system. k_1 can be obtained directly actually, no need for using scipy.optimize.root
        
        """
                
        self.findSteadyState(self.y[:,0])
        ss      = self.steadyState
        self.k3 = par_new
        
        if self.k3 is float:
            self.k3 = np.array(self.k3)

        dydt = np.zeros(self.num_species)
        dydt = -self.k2*ss 
        
        for j in range(self.num_int):
            x = self.k3[j]*self.alpha
            for k in range(self.num_species):
                x *= ss[k]**(self.r_i[j,k])
            dydt[self.r_i[j].astype(bool)] -= x
            dydt[self.s_i[j].astype(bool)] += x
        
        #print(dydt)
        new_par = -dydt
        
        self.k3 = par_old_set[2]
        
        return new_par
    
    def par_constantSS_allBaseline(self,par_new,par_old_set,penalty):
        
        """
        Outputs the parameter that will give the same value of MAK steady state if we were to change an other parameter of the system. This changes all the creation and the destruction rates of the system. The system is usually underdetermined - more uknowns than equations, so we find the solution with the smallest modulus using a non-linear bounded minimization algorithm.
        
        """
                
        self.findSteadyState(self.y[:,0])
        self.k3 = par_new
        start_guess = np.zeros(2*self.num_species)
        
        start_guess[0:self.num_species] = par_old_set[0]
        start_guess[self.num_species:] = par_old_set[1]
        
        b = ((0,None),)
        bd = ()
        
        for i in range(2*self.num_species):
            bd += b
        
        new_par = sc.optimize.minimize(self.massActionKineticsEqn_allBaseline, start_guess ,args=(penalty,start_guess),bounds=bd)
        
        self.k3 = par_old_set[2]
        
        return new_par
        
        
        
    def yDerivative(self):
        
        """
        Calculates the yDerivative to update the numbers of the species or y or \mu values.
        
        """
        
        dydt    = np.zeros(self.num_species)
        
        if self.alphaOrder == 1:
            for j in range(self.num_species):
                dydt[j] = self.k1[j] - self.k2[j]*self.y[j,self.i] - self.alpha*self.hatTheta1[j,self.i+1]

        elif self.alphaOrder == 2:
            for j in range(self.num_species):
                
                if not self.Theta_regularize:           
                    dydt[j] = self.k1[j] - self.k2[j]*self.y[j,self.i] - self.alpha*self.hatTheta1[j,self.i+1] \
                          - 0.5*self.alpha**2*self.hatTheta2[j,self.i+1]
                    
                elif self.Theta_regularize == 'exp_smooth':
                    if self.hatTheta1[j,self.i+1] == 0:
                        dydt[j] = self.k1[j] - self.k2[j]*self.y[j,self.i]
                    else:
                        dydt[j] = self.k1[j] - self.k2[j]*self.y[j,self.i] - self.alpha*self.hatTheta1[j,self.i+1] \
                          - 0.5*self.alpha**2*self.hatTheta2[j,self.i+1]* np.exp(0.5*self.alpha*self.hatTheta2[j,self.i+1]/self.hatTheta1[j,self.i+1])
                        
                elif self.Theta_regularize == 'max_1':
                    
                    if self.hatTheta1[j,self.i+1] == 0:
                        dydt[j] = self.k1[j] - self.k2[j]*self.y[j,self.i]
                    else:
                        dydt[j] = dydt[j] = self.k1[j] - self.k2[j]*self.y[j,self.i] - self.alpha*self.hatTheta1[j,self.i+1]/(1-0.5*self.alpha*self.hatTheta2[j,self.i+1]/self.hatTheta1[j,self.i+1])
                    
                elif self.Theta_regularize == 'max_2':
                    
                    if self.hatTheta1[j,self.i+1] == 0:
                        dydt[j] = self.k1[j] - self.k2[j]*self.y[j,self.i]
                    else:
                        dydt[j] = self.k1[j] - self.k2[j]*self.y[j,self.i] - self.alpha*self.hatTheta1[j,self.i+1]/np.sqrt(1-self.alpha*self.hatTheta2[j,self.i+1]/self.hatTheta1[j,self.i+1])
                    
                elif self.Theta_regularize == 'max_3':
                    
                    if self.hatTheta1[j,self.i+1] == 0:
                        dydt[j] = self.k1[j] - self.k2[j]*self.y[j,self.i]
                    else:
                        dydt[j] = self.k1[j] - self.k2[j]*self.y[j,self.i] - self.alpha*self.hatTheta1[j,self.i+1]*np.exp(0.5*self.alpha*self.hatTheta2[j,self.i+1]/self.hatTheta1[j,self.i+1])
                    

        return dydt
    
    def chiFunction(self,m,m_,n,n_,t,t_,species):
        
        def sum_k(k,C):
            return sc.special.factorial(min(n_-m,n-m_))*np.power(C[t-1,t_-1],k)\
            *np.power(C[t_-1,t_-1],HalfonlyEven(n_-m-k))*np.power(C[t-1,t-1],HalfonlyEven(n-m_-k))\
            *doubleFactorialOdd(n_-m-k-1)*doubleFactorialOdd(n-m_-k-1)/sc.special.factorial(min(n_-m,n-m_)-k)
        
        if m >= 0 and m_>=0 and n_ >= m and n >= m_:
            q = 0. 
            for k in range(0,int(min(n_-m,n-m_))+1):
                q += sum_k(k,self.corr[species])
            if m != 0 and m_ != 0:
                q *= np.power(self.resp[species,t_-1,t],m)*np.power(self.resp[species,t-1,t_],m_)
            elif m == 0 and m_ != 0:
                q *= np.power(self.resp[species,t-1,t_],m_)
            elif m != 0 and m_ == 0:
                q *= np.power(self.resp[species,t_-1,t],m)
            
            q *= sc.special.factorial(n_)*sc.special.factorial(n)/(sc.special.factorial(n_-m)*sc.special.factorial(n-m_))
            
            return q
            
        else:
            return 0.

    
    def chiFunctionProduct(self,m,m_,n,n_,t,t_,excluded=None):
        
        p = 1.
        if excluded is None:
            for k in range(self.num_species):
                p *= self.chiFunction(m[k],m_[k],n[k],n_[k],t,t_,k)
        else:
            for k in range(self.num_species):
                if k is not excluded:
                    p *= self.chiFunction(m[k],m_[k],n[k],n_[k],t,t_,k)      
        return p
    
    def EulerStep(self):
        
        """
        Euler step integration for the numbers or the y for MAK and Plefka.
        
        """
        
        try:
            self.MAK
        except AttributeError:
            self.MAK = False
            
        if self.MAK:
            dydt           = self.massActionKinetics()
        else:
            dydt           = self.yDerivative()
        
        self.y[:,self.i+1] = self.y[:,self.i] + self.delta_t*dydt
        #TODO: this condition formally
        for j in range(self.num_species):
            if self.y[j,self.i+1] < 0:
                self.y[j,self.i+1] = 0
            
        self.t            += self.delta_t
        self.i            += 1
        
    
    def updateResponses(self):
        
        """
        Updates the response and if C is True, also the correlation fucntion.
        
        """
        
        if self.i < len(self.timeGrid)-1:
            
            for j in range(self.num_species):

                if self.orderParameter == 'linear':
                    
                    if not self.crossResponse:
                        self.resp[j,self.i+1,self.i+1] = 1.                            
                    else:
                        self.resp[j,j,self.i+1,self.i+1] = 1.
                    
                    for k in np.arange(self.i,-1,-1):
                        if not self.crossResponse:
                        
                            self.resp[j,self.i+1,k] = self.resp[j,self.i,k] - self.delta_t*self.k2[j]*self.resp[j,self.i,k]
                            
                        else:                            
                            for j2 in range(self.num_species):                                    
                                self.resp[j,j2,self.i+1,k] = self.resp[j,j2,self.i,k] - self.delta_t*self.k2[j]*self.resp[j,j2,self.i,k]
                    
                    #This is only useful if I we start with non trivial initial correlator values!
                    #if self.C:
                    #    for k in range(self.i+2):
                            
                    #        self.corr[j,self.i+1,k] = self.corr[j,self.i,k] - self.delta_t*self.k2[j]*self.corr[j,self.i,k]
                    #        self.corr[j,k,self.i+1] = self.corr[j,self.i+1,k]
                        

                elif self.orderParameter == 'quad':

                    if self.alphaOrder == 1:
                        if not self.crossResponse:
                            self.resp[j,self.i+1,self.i+1] = 1.                            
                        else:
                            self.resp[j,j,self.i+1,self.i+1] = 1.

                        for k in np.arange(self.i,-1,-1):

                            #self.resp[j,self.i+1,k] = self.resp[j,self.i,k] - self.delta_t*self.k2[j]*self.resp[j,self.i,k] - self.alpha*self.delta_t**2*np.sum((self.hatR1[j,self.i+1,1:]*self.resp[j,:-1,k]),axis = 0)
                            
                            if not self.crossResponse:
                            
                                self.resp[j,self.i+1,k] = self.resp[j,self.i,k] - self.delta_t*self.k2[j]*self.resp[j,self.i,k] - self.alpha*self.delta_t**2*np.sum((self.hatR1[j,self.i+1,:-1]*self.resp[j,:-1,k]),axis = 0)
                                
                            else:
                                
                                for j2 in range(self.num_species):
                                    
                                    R_sum = 0
                                    
                                    for j_sum in range(self.num_species):
                                        
                                        R_sum += self.alpha*self.delta_t**2*np.sum((self.hatR1[j,j_sum,self.i+1,:-1] *self.resp[j_sum,j2,:-1,k]),axis = 0)
                                                                            
                                    self.resp[j,j2,self.i+1,k] = self.resp[j,j2,self.i,k] - self.delta_t*self.k2[j]*self.resp[j,j2,self.i,k] - R_sum

                        #if self.C:
                        #    for k in range(self.i+2):

                        #        self.corr[j,self.i+1,k] = self.corr[j,self.i,k] - self.delta_t*self.k2[j]*self.corr[j,self.i,k] - self.alpha*self.delta_t**2*np.sum((self.hatR1[j,self.i+1,:]*self.corr[j,k,:])[0:self.i+1],axis = 0) - self.alpha*self.delta_t**2*np.sum((self.hatB1[j,self.i+1,:]*self.resp[j,k,:])[0:k+1],axis = 0)

                         #       self.corr[j,k,self.i+1] = self.corr[j,self.i+1,k]

                    elif self.alphaOrder == 2:
                        if not self.crossResponse:
                            self.resp[j,self.i+1,self.i+1] = 1.                            
                        else:
                            self.resp[j,j,self.i+1,self.i+1] = 1.

                        for k in np.arange(self.i,-1,-1):
                            
                            if not self.crossResponse:
                            
                                R1 = self.delta_t**2*np.sum((self.hatR1[j,self.i+1,1:]*self.resp[j,:-1,k]),axis = 0)
                                R2 = self.delta_t**2*np.sum((self.hatR2[j,self.i+1,1:]*self.resp[j,:-1,k]),axis = 0)

                                if not self.R_regularize or type(self.R_regularize) is float:

                                    self.resp[j,self.i+1,k] = self.resp[j,self.i,k] - self.delta_t*self.k2[j]*self.resp[j,self.i,k] - self.alpha*R1 - 0.5*self.alpha**2*R2

                                elif self.R_regularize == 'exp_smooth':

                                    if R1 == 0:
                                        self.resp[j,self.i+1,k] = self.resp[j,self.i,k] - self.delta_t*self.k2[j]*self.resp[j,self.i,k]

                                    #Do this for all R2 or only when the derivative becomes negative?

                                    else:
                                        self.resp[j,self.i+1,k] = self.resp[j,self.i,k] - self.delta_t*self.k2[j]*self.resp[j,self.i,k] -self.alpha*R1 - 0.5*self.alpha**2*R2*np.exp(0.5*self.alpha*R2/R1)

                                elif self.R_regularize == 'max_1':

                                    if R1 == 0:
                                        self.resp[j,self.i+1,k] = self.resp[j,self.i,k] - self.delta_t*self.k2[j]*self.resp[j,self.i,k]

                                    else:
                                        #TODO: Take Abs Value, formalize this
                                        self.resp[j,self.i+1,k] = self.resp[j,self.i,k] - self.delta_t*self.k2[j]*self.resp[j,self.i,k] -abs(self.alpha*R1/(1-0.5*self.alpha*R2/R1))

                                elif self.R_regularize == 'max_2':

                                    if R1 == 0:
                                        self.resp[j,self.i+1,k] = self.resp[j,self.i,k] - self.delta_t*self.k2[j]*self.resp[j,self.i,k]

                                    else:
                                        self.resp[j,self.i+1,k] = self.resp[j,self.i,k] - self.delta_t*self.k2[j]*self.resp[j,self.i,k] -self.alpha*R1/np.sqrt(1-self.alpha*R2/R1)

                                elif self.R_regularize == 'max_3':

                                    if R1 == 0:
                                        self.resp[j,self.i+1,k] = self.resp[j,self.i,k] - self.delta_t*self.k2[j]*self.resp[j,self.i,k]

                                    else:
                                        self.resp[j,self.i+1,k] = self.resp[j,self.i,k] - self.delta_t*self.k2[j]*self.resp[j,self.i,k] -self.alpha*R1*np.exp(0.5*self.alpha*R2/R1)


                                #TODO: Check and formalize this!
                                #if self.resp[j,self.i+1,k] < 0:
                                #    self.resp[j,self.i+1,k] = 0

                                #if self.resp[j,self.i+1,k] > 1:
                                #    self.resp[j,self.i+1,k] = 1
                                    
                            else:

                                for j2 in range(self.num_species):
                                    
                                    R1 = 0.
                                    R2 = 0.
                                    for j_sum in range(self.num_species):
                                    
                                        R1 += self.delta_t**2*np.sum((self.hatR1[j,j_sum,self.i+1,1:]* self.resp[j_sum,j2,:-1,k]),axis = 0)
                                        R2 += self.delta_t**2*np.sum((self.hatR2[j,j_sum,self.i+1,1:]* self.resp[j_sum,j2,:-1,k]),axis = 0)
                                        
                                    self.resp[j,j2,self.i+1,k] = self.resp[j,j2,self.i,k] - self.delta_t*self.k2[j]*self.resp[j,j2,self.i,k] - self.alpha*R1 - 0.5*self.alpha**2*R2
                                    
                                #if self.resp[j,j2,self.i+1,k] < 0:
                                #    self.resp[j,j2,self.i+1,k] = 0

                                #if self.resp[j,j2,self.i+1,k] > 1:
                                #    self.resp[j,j2,self.i+1,k] = 1
                                    
                            if False:
                            #for k in range(self.i+2):
                                
                                #if not self.R_regularize:
                                
                                
                                #self.corr[j,self.i+1,k] = self.corr[j,self.i,k] - self.delta_t*self.k2[j]*self.corr[j,self.i,k] -self.alpha*self.delta_t**2*np.sum((self.hatR1[j,self.i+1,:]*self.corr[j,k,:])[0:self.i+2],axis = 0) -0.5*self.alpha**2*self.delta_t**2*np.sum((self.hatR2[j,self.i+1,:]*self.corr[j,k,:])[0:self.i+2],axis = 0) - self.alpha*self.delta_t**2*np.sum((self.hatB1[j,self.i+1,:]*self.resp[j,k,:])[0:k+1],axis = 0) - 0.5*self.alpha**2*self.delta_t**2*np.sum((self.hatB2[j,self.i+1,:]*self.resp[j,k,:])[0:k+1],axis = 0)
                                
                                
                                #This below is the last stable version
                                
                                #self.corr[j,self.i+1,k] = self.corr[j,self.i,k] - self.delta_t*self.k2[j]*self.corr[j,self.i,k] -self.alpha*self.delta_t**2*np.sum((self.hatR1[j,self.i+1,:]*self.corr[j,k,:])[0:self.i+1],axis = 0) -0.5*self.alpha**2*self.delta_t**2*np.sum((self.hatR2[j,self.i+1,:]*self.corr[j,k,:])[0:self.i+1],axis = 0) - self.alpha*self.delta_t**2*np.sum((self.hatB1[j,self.i+1,:]*self.resp[j,k,:])[0:k+1],axis = 0) - 0.5*self.alpha**2*self.delta_t**2*np.sum((self.hatB2[j,self.i+1,:]*self.resp[j,k,:])[0:k+1],axis = 0)

                                #self.corr[j,self.i+1,k] = self.corr[j,self.i,k] - self.delta_t*self.k2[j]*self.corr[j,self.i,k] - self.alpha*self.delta_t**2*np.sum((self.hatR1[j,self.i+1,:]*self.corr[j,k,:])[0:self.i+1],axis = 0) - 0.5*self.alpha**2*self.delta_t**2*np.sum((self.hatR2[j,self.i+1,:]*self.corr[j,k,:])[0:self.i+1],axis = 0) - self.alpha*self.delta_t**2*np.sum((self.hatB1[j,self.i+1,:]*self.resp[j,k,:])[0:k+1],axis = 0) - 0.5*self.alpha**2*self.delta_t**2*np.sum((self.hatB2[j,self.i+1,:]*self.resp[j,k,:])[0:k+1],axis = 0)

                                
                                if self.R_regularize == 'BLAH_BLAH': #TODO
                                    
                                    C1 = self.delta_t**2*np.sum((self.hatR1[j,self.i+1,:]*self.corr[j,k,:])[0:self.i+1],axis = 0)                                  
                                    C2 = self.delta_t**2*np.sum((self.hatR2[j,self.i+1,:]*self.corr[j,k,:])[0:self.i+1],axis = 0)
                                    B1 = self.delta_t**2*np.sum((self.hatB1[j,self.i+1,:]*self.resp[j,k,:])[0:k+1],axis = 0)
                                    B2 = self.delta_t**2*np.sum((self.hatB2[j,self.i+1,:]*self.resp[j,k,:])[0:k+1],axis = 0) 
                                    
                                    der = -self.delta_t*self.k2[j]*self.corr[j,self.i,k]
                                    
                                    if C1 != 0:
                                        der += -self.alpha*C1 - 0.5*self.alpha**2*C2*np.exp(0.5*self.alpha*C2/C1)
                                    
                                    if B1 != 0:
                                        der += -self.alpha*B1 - 0.5*self.alpha**2*B2*np.exp(0.5*self.alpha*B2/B1)
                                    
                                    #elif self.i>0 and der-0.5*self.alpha**2*B2 > 0 or self.corr[j,self.i,k] - self.corr[j,self.i-1,k] < 0: 
                                    #elif self.i==0:
                                        
                                        #der += -0.5*self.alpha**2*B2
                                    
                                    elif self.i>0 and self.corr[j,self.i,k]-self.corr[j,self.i-1,k] >= 0 and B2>0: 
                                        # Allow B2 even if B1 is zero but based on the condition that the correction should be of the correct sign.
                                        der += -0.5*self.alpha**2*B2
                                    
                                    #print(der)
                                    self.corr[j,self.i+1,k] = self.corr[j,self.i,k] + der
                                        
                                self.corr[j,k,self.i+1] = self.corr[j,self.i+1,k]
        
        #if self.C:
        #    if self.crossResponse:
        #        for j in range(self.num_species):
        #            for j2 in range(self.num_species):
        #                for j_sum1 in range(self.num_species):
        #                    for j_sum2 in range(self.num_species):
        #                        self.corr[j,j2,:self.i+2,:self.i+2] += np.matmul(self.resp[j,j_sum1,:self.i+2,:self.i+2],np.matmul((-0.5*self.alpha**2*self.hatB2[j_sum1,j_sum2,:self.i+2,:self.i+2]-self.alpha*self.hatB1[j_sum1,j_sum2,:self.i+2,:self.i+2]), self.resp[j2,j_sum2,:self.i+2,:self.i+2].T))*self.delta_t**2

        
        #CORR UPDATE!
        #At the last time step we use C = R \hat_B R^T
        
        if self.i == len(self.timeGrid)-2:
            if self.C:
                if self.orderParameter == 'quad':
                    for j in range(self.num_species):
                        if not self.crossResponse:
                            self.corr[j] = np.matmul(self.resp[j],np.matmul((-0.5*self.alpha**2*self.hatB2[j]-self.alpha*self.hatB1[j]), self.resp[j].T))*self.delta_t**2

                        else:
                            for j2 in range(self.num_species):
                                for j_sum1 in range(self.num_species):
                                    for j_sum2 in range(self.num_species):
                                        self.corr[j,j2,:,:] += np.matmul(self.resp[j,j_sum1],np.matmul((-0.5*self.alpha**2*self.hatB2[j_sum1,j_sum2]-self.alpha*self.hatB1[j_sum1,j_sum2]), self.resp[j2,j_sum2].T))*self.delta_t**2
                                        
                                #Note the order of the species index in the transposed response!!!

                                
                                
                            
                                
    def updateFields(self):
        
        """
        Updating the field values for fieldMethod = None and C = True.
        
        """
        
        if self.orderParameter == 'linear':
            
            for k in range(self.num_species):
                x    = 0
                m    = np.zeros(self.num_species)
                m[k] = 1.
                
                for j in range(self.num_int):
                    x += -self.c_mn(j,m,np.zeros(self.num_species),self.i+1)
                
                if self.i < len(self.timeGrid)-1:
                    
                    self.hatTheta1[k,self.i+1] = x
                
            if self.alphaOrder == 2:                
                for k in range(self.num_species):
                    x    = 0
                    m    = np.zeros(self.num_species)
                    m[k] = 1.

                    for n in self.n_listFULL[0]:
                        for time in np.arange(1,self.i+1):
                            if n not in self.exclusion_list:
                                x += self.c_mnFULL(m,n,self.i+1)*self.c_mnFULL(n,np.zeros(self.num_species),time)*np.prod((sc.special.factorial(np.array(n)))*np.power(self.resp[:,self.i,time],np.array(n)))
                    
                    x *= -2*self.delta_t
                    if self.i < len(self.timeGrid)-1:
                        self.hatTheta2[k,self.i+1] = x
                    del x
            
        elif self.orderParameter == 'quad':
            
            for k in range(self.num_species):
                x    = 0.
                y    = 0.
                z    = 0.
                m    = np.zeros(self.num_species)
                m[k] = 1.
                
                for j in range(self.num_int):
                    for n in self.n_list[j]:
                        x += self.c_mn(j,m,n,self.i+1)*np.prod(doubleFactorialOdd(np.array(n)-1) *np.power(self.corr[:,self.i,self.i],HalfonlyEven(np.array(n))))
                        
                        y += self.c_mn(j,m,n,self.i+1)*n[k]*doubleFactorialOdd(np.array([n[k]])-2)*np.power(self.corr[k,self.i,self.i], HalfonlyEven(np.array([n[k]])-1))* np.prod((doubleFactorialOdd(np.array(n)-1)*np.power(self.corr[:,self.i,self.i],HalfonlyEven(np.array(n))))[np.arange(self.num_species)!=k])

                        z += self.c_mn(j,2*m,n,self.i+1)*doubleFactorialOdd(np.array([n[k]])-1)*np.power(self.corr[k,self.i,self.i],HalfonlyEven(np.array([n[k]]))) *np.prod((doubleFactorialOdd(np.array(n)-1)*np.power(self.corr[:,self.i,self.i],HalfonlyEven(np.array(n))))[np.arange(self.num_species)!=k])
                
                if self.i < len(self.timeGrid)-1:
                    self.hatTheta1[k,self.i+1]      = -x
                
                if self.i > 0:
                    self.hatR1[k,self.i+1,self.i]   = -y/self.delta_t
                    self.hatB1[k,self.i+1,self.i+1] = -z/self.delta_t
                
                del x,y,z
                
            if self.alphaOrder == 2:
                
                for k in range(self.num_species):
                    
                    x,y,z = 0.,0.,0.
                    m     = np.zeros(self.num_species)
                    m[k]  = 1.
                    mZero = np.zeros(self.num_species)
    
                    for t in range(0,self.i+2):
                    
                        for n in self.n_listFULL[0]:
                            for n_ in self.n_listFULL[0]:
                                
                                for m_ in self.m_listFULL[0]:
                                    if m_ not in [mZero.tolist()]:
                                        
                                        if t > 0:
                                            x += self.c_mnStarFULL(m_,n_,t)*self.c_mnFULL(m,n,self.i+1) *self.chiFunctionProduct(mZero,m_,n,n_,self.i+1,t)

                                            self.hatR2[k,self.i+1,t-1] += -2*n_[k]*self.c_mnStarFULL(m,n,self.i+1)*self.c_mnStarFULL(m_,n_,t)* self.chiFunctionProduct(mZero,m_,n,n_-m,self.i+1,t)

                                            y += n[k]*self.c_mnStarFULL(m,n,self.i+1) *self.c_mnStarFULL(m_,n_,t)*self.chiFunctionProduct(mZero,m_,n-m,n_,self.i+1,t)
                                        
                                        if t < self.i+1:
                                            
                                            self.hatB2[k,self.i+1,t+1] += -2*m_[k]*self.c_mnStarFULL(m,n,self.i+1)*self.c_mnStarFULL(m_,n_,t+1)*self.chiFunctionProduct(mZero,m_-m,n,n_,self.i+1,t+1) -2*m_[k]*self.c_mnStarFULL(m,n,t+1)*self.c_mnStarFULL(m_,n_,self.i+1)*self.chiFunctionProduct(mZero,m_-m,n,n_,t+1,self.i+1)                                        
                                            z += self.c_mnStarFULL(2*m,n,self.i+1) *self.c_mnStarFULL(m_,n_,t+1)*self.chiFunctionProduct(mZero,m_,n,n_,self.i+1,t+1)
                                            
                                            self.hatB2[k,t+1,self.i+1] = self.hatB2[k,self.i+1,t+1]
                                            
                                ## Check this
                                
                                #if n[k] > 0 and n[k]%2 == 0:
                                    
                                    #x -= 0.5*n[k]*doubleFactorialOdd(n[k]-1)*self.c_mnFULL(m,n,self.i)*np.power(self.corr[k,self.i-1,self.i-1],-1 + HalfonlyEven(np.array(n[k])))*(self.c_mnStarFULL(m,n_,t)*self.chiFunction(0,1,2,n_[k],self.i,t,k) + self.c_mnStarFULL(2*m,n_,t)*self.chiFunction(0,2,2,n_[k],self.i,t,k))*np.prod((doubleFactorialOdd(np.array(n)-1)*np.power(self.corr[:,self.i-1,self.i-1],HalfonlyEven(np.array(n))))[np.arange(self.num_species)!=k])*self.chiFunctionProduct(mZero,mZero,mZero,n_,self.i,t,excluded=k)
                        
                        #if t < self.i:
                        
                    
                    if self.i < len(self.timeGrid)-1:
                        self.hatTheta2[k,self.i+1]   = -2*self.delta_t*x
                    
                    self.hatR2[k,self.i+1,self.i]   += -2*y
                    self.hatB2[k,self.i+1,self.i+1] += -2*z
                    
                    del x,y,z,m,mZero

    
    def updateFields_beta(self):
        
        """
        Updating the field values for fieldMethod = 'split' and C = True.
        
        """

        if self.orderParameter == 'linear':
            
            for k in range(self.num_species):
                x    = 0
                m    = np.zeros(self.num_species)
                m[k] = 1.
                
                for j in range(self.num_int):
                    x += -self.c_mn(j,m,np.zeros(self.num_species),self.i+1)
                
                if self.i < len(self.timeGrid)-1:
                    self.hatTheta1[k,self.i+1] = x
                
            if self.alphaOrder == 2:                
                for k in range(self.num_species):
                    x    = 0
                    m    = np.zeros(self.num_species)
                    m[k] = 1.
                    
                    for j in range(self.num_int):
                        for l in range(self.num_int):
                            for n in self.n_listFULL[0]:
                                if n not in self.exclusion_list:
                                    
                                    for time in np.arange(1,self.i+1):
                                            x += self.c_mn(j,m,n,self.i+1)*self.c_mn(l,n,np.zeros(self.num_species),time)*\
                                         np.prod((sc.special.factorial(np.array(n)))*np.power(self.resp[:,self.i,time],np.array(n)))

                    x *= -2*self.delta_t
                    
                    if self.i < len(self.timeGrid)-1:
                        self.hatTheta2[k,self.i+1] = x
                    
                    del x
            
        elif self.orderParameter == 'quad':
            
            for k in range(self.num_species):
                x    = 0.
                y    = 0.
                z    = 0.
                m    = np.zeros(self.num_species)
                m[k] = 1.
                
                for j in range(self.num_int):
                    for n in self.n_listFULL[0]:
                        
                        x += self.c_mn(j,m,n,self.i+1)*np.prod(doubleFactorialOdd(np.array(n)-1) *np.power(self.corr[:,self.i,self.i],HalfonlyEven(np.array(n))))
                        
                        y += self.c_mn(j,m,n,self.i+1)*n[k]*doubleFactorialOdd(np.array([n[k]])-2)*np.power(self.corr[k,self.i,self.i],HalfonlyEven(np.array([n[k]])-1))* np.prod((doubleFactorialOdd(np.array(n)-1)*np.power(self.corr[:,self.i,self.i],HalfonlyEven(np.array(n))))[np.arange(self.num_species)!=k])       

                        z += self.c_mn(j,2*m,n,self.i+1)*doubleFactorialOdd(np.array([n[k]])-1)*np.power(self.corr[k,self.i,self.i],HalfonlyEven(np.array([n[k]])))* np.prod((doubleFactorialOdd(np.array(n)-1)*np.power(self.corr[:,self.i,self.i],HalfonlyEven(np.array(n))))[np.arange(self.num_species)!=k])
                
                if self.i < len(self.timeGrid)-1:
                    self.hatTheta1[k,self.i+1]      = -x
                    
                if self.i > 0 :
                    self.hatR1[k,self.i+1,self.i]   = -y/self.delta_t
                    self.hatB1[k,self.i+1,self.i+1] = -z/self.delta_t
                
                del x,y,z
                
            if self.alphaOrder == 2:
                
                for k in range(self.num_species):
                    
                    x,y,z = 0.,0.,0.
                    m     = np.zeros(self.num_species)
                    m[k]  = 1.
                    mZero = np.zeros(self.num_species)
                    
                    for j in range(self.num_int):
                        for l in range(self.num_int):

                            for n in self.n_listFULL[0]:
                                for n_ in self.n_listFULL[0]:
                            
                                    for t in range(0,self.i+2):
                                    
                                        for m_ in self.m_listFULL[0]:
                                            if m_ not in [mZero.tolist()]:
                                                
                                                if t>0:

                                                    x += self.c_mnStar(l,m_,n_,t)*self.c_mn(j,m,n,self.i+1)*\
                                                self.chiFunctionProduct(mZero,m_,n,n_,self.i+1,t)

                                                    self.hatR2[k,self.i+1,t-1] += -2*n_[k]*self.c_mnStar(j,m,n,self.i+1)*self.c_mnStar(l,m_,n_,t)*self.chiFunctionProduct(mZero,m_,n,n_-m,self.i+1,t) 

                                                    y += n[k]*self.c_mnStar(j,m,n,self.i+1)*self.c_mnStar(l,m_,n_,t)*self.chiFunctionProduct(mZero,m_,n-m,n_,self.i+1,t)

                                                if t < self.i+1:
                                                    
                                                    self.hatB2[k,self.i+1,t+1] += -2*m_[k]*self.c_mnStar(j,m,n,self.i+1)*self.c_mnStar(l,m_,n_,t+1)*self.chiFunctionProduct(mZero,m_-m,n,n_,self.i+1,t+1) + -2*m_[k]*self.c_mnStar(j,m,n,t+1)*self.c_mnStar(l,m_,n_,self.i+1)*self.chiFunctionProduct(mZero,m_-m,n,n_,t+1,self.i+1)

                                                    z += self.c_mnStar(j,2*m,n,self.i+1)*self.c_mnStar(l,m_,n_,t+1)*self.chiFunctionProduct(mZero,m_,n,n_,self.i+1,t+1)
                                                    
                                                    self.hatB2[k,t+1,self.i+1] = self.hatB2[k,self.i+1,t+1]
                                        #TODO
                                        
                                        #if n[k] > 0 and n[k]%2 == 0: 
                                            #x -= 0.5*n[k]*doubleFactorialOdd(n[k]-1)*self.c_mn(j,m,n,self.i+1)*np.power(self.corr[k,self.i,self.i],-1 + HalfonlyEven(np.array(n[k])))*(self.c_mnStar(l,m,n_,t)*self.chiFunction(0,1,2,n_[k],self.i+1,t,k) + self.c_mnStar(l,2*m,n_,t)*self.chiFunction(0,2,2,n_[k],self.i+1,t,k)) *np.prod((doubleFactorialOdd(np.array(n)-1)*np.power(self.corr[:,self.i,self.i],HalfonlyEven(np.array(n))))[np.arange(self.num_species)!=k])*self.chiFunctionProduct(mZero,mZero,mZero,n_,self.i+1,t,excluded=k)
                                        
                    
                    
                    if self.i < len(self.timeGrid)-1:
                        self.hatTheta2[k,self.i+1]   = -2*self.delta_t*x
                    
                    self.hatR2[k,self.i+1,self.i]   += -2*y
                    self.hatB2[k,self.i+1,self.i+1] += -2*z
                    
                    del x,y,z,m,mZero
                    
                    
    def updateFields_no_C(self):

            """
            Updating the field values for fieldMethod = None and C = False.

            """

            if self.orderParameter == 'linear':

                for k in range(self.num_species):
                    x    = 0
                    m    = np.zeros(self.num_species)
                    m[k] = 1.

                    for j in range(self.num_int):
                        x += -self.c_mn(j,m,np.zeros(self.num_species),self.i+1)

                    if self.i < len(self.timeGrid)-1:
                        self.hatTheta1[k,self.i+1] = x

                if self.alphaOrder == 2:                
                    for k in range(self.num_species):
                        x    = 0
                        m    = np.zeros(self.num_species)
                        m[k] = 1.
                        
                        for n in self.n_listFULL[0]:
                            if n not in self.exclusion_list:
                                for time in np.arange(1,self.i+1):
                                    x += self.c_mnFULL(m,n,self.i+1)*self.c_mnFULL(n,np.zeros(self.num_species),time)*np.prod((sc.special.factorial(np.array(n)))*np.power(self.resp[:,self.i,time],np.array(n)))

                        if self.i < len(self.timeGrid)-1:
                            self.hatTheta2[k,self.i+1] = -2*self.delta_t*x
                        
                        del x

            elif self.orderParameter == 'quad':

                for k in range(self.num_species):
                    x    = 0.
                    y    = 0.
                    z    = 0.
                    m    = np.zeros(self.num_species)
                    m[k] = 1.

                    for j in range(self.num_int):
                        x += self.c_mn(j,m,np.zeros(self.num_species),self.i+1)
                        y += self.c_mn(j,m,m,self.i+1)

                    if self.i < len(self.timeGrid)-1:
                        self.hatTheta1[k,self.i+1]      = -x

                    if self.i > 0:
                        self.hatR1[k,self.i+1,self.i]   = -y/self.delta_t

                    del x,y,z

                if self.alphaOrder == 2:

                    for k in range(self.num_species):
                        x,y,z = 0.,0.,0.
                        m     = np.zeros(self.num_species)
                        m[k]  = 1.
                        mZero = np.zeros(self.num_species)

                        for n in self.n_listFULL[0]:
                            
                            if n not in self.exclusion_list:
                                for t in range(0,self.i+2):

                                    x += self.c_mnFULL(m,n,self.i+1)*self.c_mnFULL(n,mZero,t)*np.prod((sc.special.factorial(np.array(n)))*np.power(self.resp[:,self.i,t],np.array(n)))

                            if n not in [mZero.astype(int).tolist(),m.astype(int).tolist()]:
                                for t in range(1,self.i+2):

                                    self.hatR2[k,self.i+1,t-1] += -2*self.c_mnFULL(m,n,self.i+1)*self.c_mnFULL(n,m,t)*np.prod((sc.special.factorial(np.array(n)))*np.power(self.resp[:,self.i,t],np.array(n)))

                        if self.i < len(self.timeGrid)-1:
                            self.hatTheta2[k,self.i+1]   = -2*self.delta_t*x

                        del x,y,z,m,mZero
                    
            
    def updateFields_no_C_old(self):
        
        """
        Updating the field values for fieldMethod = None and C = False. This is the slightly old version, might be slower! With extra hat(B) calculations on top!
        
        """

        if self.orderParameter == 'linear':
            
            for k in range(self.num_species):
                x    = 0
                m    = np.zeros(self.num_species)
                m[k] = 1.
                
                for j in range(self.num_int):
                    x += -self.c_mn(j,m,np.zeros(self.num_species),self.i+1)
                    
                if self.i < len(self.timeGrid)-1:
                    self.hatTheta1[k,self.i+1] = x
                
            if self.alphaOrder == 2:                
                for k in range(self.num_species):
                    x    = 0
                    m    = np.zeros(self.num_species)
                    m[k] = 1.

                    for n in self.n_listFULL[0]:
                        for time in np.arange(1,self.i+1):
                            if n not in self.exclusion_list:
                                x += self.c_mnFULL(m,n,self.i+1)*self.c_mnFULL(n,np.zeros(self.num_species),time)*\
                                    np.prod((sc.special.factorial(np.array(n)))*np.power(self.resp[:,self.i,time],np.array(n)))
                                
                    x *= -2*self.delta_t
                    if self.i < len(self.timeGrid)-1:
                        self.hatTheta2[k,self.i+1] = x
                    
                    del x
            
        elif self.orderParameter == 'quad':
            
            for k in range(self.num_species):
                x    = 0.
                y    = 0.
                z    = 0.
                m    = np.zeros(self.num_species)
                m[k] = 1.
                
                for j in range(self.num_int):
                    x += self.c_mn(j,m,np.zeros(self.num_species),self.i+1)
                    y += self.c_mn(j,m,m,self.i+1)
                    #z += self.c_mn(j,2*m,np.zeros(self.num_species),self.i+1)
                    
                if self.i < len(self.timeGrid)-1:
                    self.hatTheta1[k,self.i+1]      = -x
                
                if self.i > 0:
                    self.hatR1[k,self.i+1,self.i]   = -y/self.delta_t
                    #self.hatB1[k,self.i+1,self.i+1] = -z/self.delta_t
                
                del x,y,z
                
            if self.alphaOrder == 2:
                
                for k in range(self.num_species):
                    x,y,z = 0.,0.,0.
                    m     = np.zeros(self.num_species)
                    m[k]  = 1.
                    mZero = np.zeros(self.num_species)
                    
                    #for t in range(1,self.i+2):
                        #self.hatR2[k,self.i+1,t-1] += self.RidgeRegularizer*self.resp[k,self.i,t]
                        
                    for n in self.n_listFULL[0]:
                        
                        if n not in self.exclusion_list:
                            for t in range(0,self.i+2):
                                
                                x += self.c_mnFULL(m,n,self.i+1)*self.c_mnFULL(n,mZero,t)*np.prod((sc.special.factorial(np.array(n)))*np.power(self.resp[:,self.i,t],np.array(n)))
                                
                        if n not in [mZero.astype(int).tolist(),m.astype(int).tolist()]:
                            for t in range(1,self.i+2):

                                self.hatR2[k,self.i+1,t-1] += -2*self.c_mnFULL(m,n,self.i+1)*self.c_mnFULL(n,m,t)*np.prod((sc.special.factorial(np.array(n)))*np.power(self.resp[:,self.i,t],np.array(n)))
                                    #if n not in [(3*m.astype(int)).tolist()] and n[k]>0:

                                        #y += n[k]*self.c_mnFULL(m,n,self.i+1)*self.c_mnFULL(n-m,mZero,t) *sc.special.factorial(n[k]-1)*np.power(self.resp[k,self.i,t],n[k]-1)* np.prod((sc.special.factorial(np.array(n))*np.power(self.resp[:,self.i,t],np.array(n)))[np.arange(self.num_species)!=k])
                                        
                        #if n not in [mZero.astype(int).tolist(),m.astype(int).tolist()]:
                            #for t in range(0,self.i+1):

                                #if (n+m).astype(int).tolist() in self.m_listFULL[0]:
                                
                                    #self.hatB2[k,self.i+1,t+1] += -2*(n[k]+1)*self.c_mnFULL(m,n,self.i+1)* self.c_mnFULL(n+m,mZero,t+1) *np.prod((sc.special.factorial(np.array(n)))*np.power(self.resp[:,self.i,t+1],np.array(n)))
                                    
                                    #z += self.c_mnFULL(2*m,n,self.i+1)*self.c_mnFULL(n,mZero,t+1)*np.prod((sc.special.factorial(np.array(n)))*np.power(self.resp[:,self.i,t+1],np.array(n)))

                                    #self.hatB2[k,t+1,self.i+1] = self.hatB2[k,self.i+1,t+1]
                    
                    ## Apply a Gaussian cut-off w.r.t the time difference on the hatR2 field:
                    
                    #for t in range(1,self.i+2):
                        #self.hatR2[k,self.i+1,t-1] = self.hatR2[k,self.i+1,t-1]*np.exp(-0.5*(self.i-t+1)**2*self.GaussianRegularizer**2)
                                                
                    if self.i < len(self.timeGrid)-1:
                        self.hatTheta2[k,self.i+1]   = -2*self.delta_t*x
                    
                    #self.hatR2[k,self.i+1,self.i]   += -2*y
                    #self.hatB2[k,self.i+1,self.i+1] += -2*z
                    
                    del x,y,z,m,mZero
                    
                    
    def updateFields_no_C_bubble_sum_shifted(self):
        """
        Updating the field values for fieldMethod = bubble_sum and C = False.
        
        """

        if self.orderParameter == 'linear':
            
            for k in range(self.num_species):
                x    = 0
                m    = np.zeros(self.num_species)
                m[k] = 1.
                
                for j in range(self.num_int):
                    x += -self.c_mn(j,m,np.zeros(self.num_species),self.i+1)
                    
                if self.i < len(self.timeGrid)-1:
                    self.hatTheta1[k,self.i+1] = x
                
            if self.alphaOrder == 2:                
                for k in range(self.num_species):
                    x    = 0
                    m    = np.zeros(self.num_species)
                    m[k] = 1.
                    mZero= np.zeros(self.num_species)
                    
                    cNN     = np.zeros(self.i+1)
                    M_v     = np.zeros(self.i+1)
                    chi_mat = np.zeros([self.i+1,self.i+1])
                    Y_mat   = np.zeros([self.i+1,self.i+1])
                                        
                    for n in self.n_listFULL[0]:
                        if n not in self.exclusion_list_bubble:
                            
                            for t in range(self.i+1):
                                cNN[t] = self.c_mnFULL(n,n,t)
                                M_v[t] = self.c_mnFULL(n,mZero,t)
                                for t2 in range(self.i+1):
                                    Y_mat[t,t2] = np.prod((sc.special.factorial(np.array(n)))*np.power(self.resp[:,t,t2], np.array(n)))
                                    
                            L = np.diag(np.ones(self.i),k=-1)
                            for t in range(self.i+1):
                                chi_mat[t,:] = cNN[:]*Y_mat[t,:]

                            temp = np.matmul(np.matmul(sc.linalg.solve_triangular(a=np.identity(self.i+1)-self.alpha*self.delta_t*np.matmul(chi_mat,L), b=np.identity(self.i+1), lower=True,overwrite_b=True),Y_mat),M_v)
                            
                            x += self.c_mnFULL(m,n,self.i+1)*temp[self.i]
                    
                    if self.i < len(self.timeGrid)-1:
                        self.hatTheta2[k,self.i+1] = -2*self.delta_t*x
                    
                    del x
            
        elif self.orderParameter == 'quad':
            
            for k in range(self.num_species):
                x    = 0.
                y    = 0.
                z    = 0.
                m    = np.zeros(self.num_species)
                m[k] = 1.
                
                for j in range(self.num_int):
                    x += self.c_mn(j,m,np.zeros(self.num_species),self.i+1)
                    y += self.c_mn(j,m,m,self.i+1)
                    
                if self.i < len(self.timeGrid)-1:
                    self.hatTheta1[k,self.i+1]      = -x
                
                if self.i > 0:
                    self.hatR1[k,self.i+1,self.i]   = -y/self.delta_t
                
                del x,y,z
                
            if self.alphaOrder == 2:
                
                for k in range(self.num_species):
                    x,y,z = 0.,0.,0.
                    m     = np.zeros(self.num_species)
                    m[k]  = 1.
                    mZero = np.zeros(self.num_species)
                    
                    cNN     = np.zeros(self.i+1)
                    M_v     = np.zeros(self.i+1)
                    chi_mat = np.zeros([self.i+1,self.i+1])
                    Y_mat   = np.zeros([self.i+1,self.i+1])
                    M_f     = np.zeros([self.i+1,self.i+1])
                                        
                    for n in self.n_listFULL[0]:
                        if n not in self.exclusion_list_bubble:
                            
                            for t in range(self.i+1):
                                cNN[t] = self.c_mnFULL(n,n,t)
                                M_v[t] = self.c_mnFULL(n,mZero,t)
                                for t2 in range(self.i+1):
                                    Y_mat[t,t2] = np.prod((sc.special.factorial(np.array(n)))*np.power(self.resp[:,t,t2], np.array(n)))
                                    
                            L = np.diag(np.ones(self.i),k=-1)
                            for t in range(self.i+1):
                                chi_mat[t,:] = cNN[:]*Y_mat[t,:]

                            temp = np.matmul(np.matmul(sc.linalg.solve_triangular(a=np.identity(self.i+1)-self.alpha*self.delta_t*np.matmul(chi_mat,L), b=np.identity(self.i+1), lower=True,overwrite_b=True),Y_mat),M_v)
                            
                            x += self.c_mnFULL(m,n,self.i+1)*temp[self.i] 
                                
                        #if n not in [mZero.astype(int).tolist(),m.astype(int).tolist()]:
                        if n not in self.exclusion_list_bubble:
                            for t in range(self.i+1):
                                M_f[t,t] = self.c_mnFULL(n,m,t)
                                
                            L = np.diag(np.ones(self.i),k=-1)
                            
                            temp = np.matmul(np.matmul(sc.linalg.solve_triangular(a=np.identity(self.i+1)-self.alpha*self.delta_t*np.matmul(chi_mat,L), b=np.identity(self.i+1), lower=True,overwrite_b=True),Y_mat),M_f)
                                                        
                            for t in range(1,self.i+1):
                                self.hatR2[k,self.i+1,t-1] += -2*self.c_mnFULL(m,n,self.i+1)*temp[self.i,t]
                                                    
                    if self.i < len(self.timeGrid)-1:
                        self.hatTheta2[k,self.i+1] = -2*self.delta_t*x
                    
                    del x,y,z,m,mZero
                    
                    
    def updateFields_no_C_bubble_sum_wo_inversion(self):
        """
        Updating the field values for fieldMethod = bubble_sum and C = False. New version, slightly faster. This does not invert the matrix, but uses block inversion relations! Needs to save an extra matrix "inv_mat" which we keep appending with the new inverted values. But this needs to done for each element n of the reaction list!
        
        """

        if self.orderParameter == 'linear':
            
            for k in range(self.num_species):
                x    = 0
                m    = np.zeros(self.num_species)
                m[k] = 1.
                
                for j in range(self.num_int):
                    x += -self.c_mn(j,m,np.zeros(self.num_species),self.i+1)
                    
                if self.i < len(self.timeGrid)-1:
                    self.hatTheta1[k,self.i+1] = x
                
            if self.alphaOrder == 2:
                
                try:
                    self.inv_mat
                except:
                    self.inv_mat = np.zeros(np.append(len(self.n_listFULL[0]),np.shape(self.resp)))
                    self.inv_mat[:,:,0,0] = 1.
                
                for k in range(self.num_species):
                    
                    x    = 0
                    m    = np.zeros(self.num_species)
                    m[k] = 1.
                    mZero= np.zeros(self.num_species)
                    inv_index_n = 0
                    
                    cNN     = np.zeros(self.i+1)
                    M_v     = np.zeros(self.i+1)
                    chi_mat = np.zeros([self.i+1,self.i+1])
                    Y_mat   = np.zeros([self.i+1,self.i+1])
                    L       = np.diag(np.ones(self.i),k=-1)
                    
                    for n in self.n_listFULL[0]:
                        if n not in self.exclusion_list_bubble:
                            
                            for t in range(self.i+1):
                                cNN[t] = self.c_mnFULL(n,n,t)
                                M_v[t] = self.c_mnFULL(n,mZero,t)
                                for t2 in range(self.i+1):
                                    Y_mat[t,t2] = np.prod((sc.special.factorial(np.array(n)))*np.power(self.resp[:,t,t2], np.array(n)))                            
                            for t in range(self.i+1):
                                chi_mat[t] = cNN*Y_mat[t]
                            
                            self.inv_mat[inv_index_n,k,self.i+1,:self.i+1] = np.matmul(self.alpha*self.delta_t*chi_mat[self.i,:self.i+1], self.inv_mat[inv_index_n,k,:self.i+1,:self.i+1])
                            
                            self.inv_mat[inv_index_n,k,self.i+1,self.i+1]  = 1. 
                            
                            temp = np.matmul(self.inv_mat[inv_index_n,k,:self.i+1,:self.i+1],Y_mat)
                            
                            inv_index_n += 1
                            
                            x += self.c_mnFULL(m,n,self.i+1)*np.sum(temp[self.i]*M_v)
                    
                    if self.i < len(self.timeGrid)-1:
                        self.hatTheta2[k,self.i+1] = -2*self.delta_t*x
                    
                    del x
            
        elif self.orderParameter == 'quad':
            
            for k in range(self.num_species):
                x    = 0.
                y    = 0.
                z    = 0.
                m    = np.zeros(self.num_species)
                m[k] = 1.
                
                for j in range(self.num_int):
                    x += self.c_mn(j,m,np.zeros(self.num_species),self.i+1)
                    y += self.c_mn(j,m,m,self.i+1)
                    
                if self.i < len(self.timeGrid)-1:
                    self.hatTheta1[k,self.i+1]      = -x
                
                if self.i > 0:
                    self.hatR1[k,self.i+1,self.i+1]   = -y/self.delta_t
                    #self.hatR1[k,self.i+1,self.i]   = -y/self.delta_t
                
                del x,y,z
                
            if self.alphaOrder == 2:
                
                try:
                    self.inv_mat
                except:
                    self.inv_mat = np.zeros(np.append(len(self.n_listFULL[0]),np.shape(self.resp)))
                    self.inv_mat[:,:,0,0] = 1.
                
                for k in range(self.num_species):                                       
                    x,y,z = 0.,0.,0.
                    m     = np.zeros(self.num_species)
                    m[k]  = 1.
                    mZero = np.zeros(self.num_species)
                    inv_index_n = 0
                    
                    cNN     = np.zeros(self.i+1)
                    M_v     = np.zeros(self.i+1)
                    chi_mat = np.zeros([self.i+1,self.i+1])
                    Y_mat   = np.zeros([self.i+1,self.i+1])
                    M_f     = np.zeros(self.i+1)
                                        
                    for n in self.n_listFULL[0]:
                        if n not in self.exclusion_list_bubble:
                            
                            for t in range(self.i+1):
                                cNN[t] = self.c_mnFULL(n,n,t)
                                M_v[t] = self.c_mnFULL(n,mZero,t)
                                M_f[t] = self.c_mnFULL(n,m,t)
                                
                                for t2 in range(self.i+1):
                                    Y_mat[t,t2] = np.prod((sc.special.factorial(np.array(n)))*np.power(self.resp[:,t,t2], np.array(n)))                                  
                            for t in range(self.i+1):
                                chi_mat[t] = cNN*Y_mat[t]                                                            
                            
                            #The inverse of the new extended matrix is obatined by multiplying the new chi_matrix with the old inverse and appending a 1 at the last value!
                            if self.i==0: 
                                self.inv_mat[inv_index_n,k,self.i+1,:self.i] = np.matmul(self.alpha*self.delta_t*chi_mat[self.i,:self.i],self.inv_mat[inv_index_n,k,:self.i,:self.i])                           
                            else:
                                self.inv_mat[inv_index_n,k,self.i,:self.i] = np.matmul(self.alpha*self.delta_t*chi_mat[self.i,1:self.i+1], self.inv_mat[inv_index_n,k,:self.i,:self.i]) 
                                                        
                            self.inv_mat[inv_index_n,k,self.i+1,self.i+1]  = 1. 
                            
                            temp = np.matmul(self.inv_mat[inv_index_n,k,:self.i+1,:self.i+1],Y_mat)                           
                                                                                     
                            x += self.c_mnFULL(m,n,self.i+1)*np.sum(temp[self.i]*M_v) 
                                                        
                            self.hatR2[k,self.i+1,:self.i+1] = -2*self.c_mnFULL(m,n,self.i+1)*(temp[self.i]*M_f)
                            
                            inv_index_n += 1
                            
                            del temp
                        
                        elif n in self.exclusion_list_bubble:
                            if n not in [mZero.astype(int).tolist(),m.astype(int).tolist()]:
                                                                
                                for t in range(self.i+1):
                                    M_f[t] = self.c_mnFULL(n,m,t)
                                    cNN[t] = self.c_mnFULL(n,n,t)

                                    for t2 in range(self.i+1):
                                        Y_mat[t,t2] = np.prod((sc.special.factorial(np.array(n)))*np.power(self.resp[:,t,t2], np.array(n)))
                                for t in range(self.i+1):
                                    chi_mat[t] = cNN*Y_mat[t]
                                    
                                if self.i==0:
                                    self.inv_mat[inv_index_n,k,self.i+1,:self.i] = np.matmul(self.alpha*self.delta_t*chi_mat[self.i,:self.i],self.inv_mat[inv_index_n,k,:self.i,:self.i])                           
                                else:
                                    self.inv_mat[inv_index_n,k,self.i,:self.i] = np.matmul(self.alpha*self.delta_t*chi_mat[self.i,1:self.i+1], self.inv_mat[inv_index_n,k,:self.i,:self.i]) 
                            
                                self.inv_mat[inv_index_n,k,self.i+1,self.i+1]  = 1. 
                            
                                temp = np.matmul(self.inv_mat[inv_index_n,k,:self.i+1,:self.i+1],Y_mat)
                                
                                self.hatR2[k,self.i+1,:self.i+1] = -2*self.c_mnFULL(m,n,self.i+1)*(temp[self.i]*M_f)
                                
                                inv_index_n += 1
                                
                                del temp, R2_temp
                                                    
                    if self.i < len(self.timeGrid)-1:
                        self.hatTheta2[k,self.i+1] = -2*self.delta_t*x
                    
                    del x,y,z,m,mZero
                    
                    
    def updateFields_no_C_bubble_sum_faster(self):
        """
        Updating the field values for fieldMethod = bubble_sum and C = False. New version, slightly faster by implementing minimal matrix multiplication! Additionally uses the dtritri function from LAPAC to invert the matrix, which is supposed to be faster, but practically its not!
        
        """

        if self.orderParameter == 'linear':
            
            for k in range(self.num_species):
                x    = 0
                m    = np.zeros(self.num_species)
                m[k] = 1.
                
                for j in range(self.num_int):
                    x += -self.c_mn(j,m,np.zeros(self.num_species),self.i+1)
                    
                if self.i < len(self.timeGrid)-1:
                    self.hatTheta1[k,self.i+1] = x
                
            if self.alphaOrder == 2:                
                for k in range(self.num_species):
                    x    = 0
                    m    = np.zeros(self.num_species)
                    m[k] = 1.
                    mZero= np.zeros(self.num_species)
                    
                    cNN     = np.zeros(self.i+1)
                    M_v     = np.zeros(self.i+1)
                    chi_mat = np.zeros([self.i+1,self.i+1])
                    Y_mat   = np.zeros([self.i+1,self.i+1])
                    L       = np.diag(np.ones(self.i),k=-1)
                    
                    for n in self.n_listFULL[0]:
                        if n not in self.exclusion_list_bubble:
                            
                            for t in range(self.i+1):
                                cNN[t] = self.c_mnFULL(n,n,t)
                                M_v[t] = self.c_mnFULL(n,mZero,t)
                                for t2 in range(self.i+1):
                                    Y_mat[t,t2] = np.prod((sc.special.factorial(np.array(n)))*np.power(self.resp[:,t,t2], np.array(n)))                            
                            for t in range(self.i+1):
                                chi_mat[t] = cNN*Y_mat[t]
                                
                                
                            temp = np.matmul((sc.linalg.lapack.dtrtri(np.identity(self.i+1)-self.alpha*self.delta_t*np.matmul(chi_mat,L),lower=1,overwrite_c=1,unitdiag=1))[0],Y_mat)                                
                                                            
                            #temp = np.matmul(sc.linalg.solve_triangular(a=np.identity(self.i+1)-self.alpha*self.delta_t*np.matmul(chi_mat,L), b=np.identity(self.i+1), lower=True,overwrite_b=True,check_finite=False),Y_mat)
                                                        
                            x += self.c_mnFULL(m,n,self.i+1)*np.sum(temp[self.i]*M_v)
                    
                    if self.i < len(self.timeGrid)-1:
                        self.hatTheta2[k,self.i+1] = -2*self.delta_t*x
                    
                    del x
            
        elif self.orderParameter == 'quad':
            
            for k in range(self.num_species):
                x    = 0.
                y    = 0.
                z    = 0.
                m    = np.zeros(self.num_species)
                m[k] = 1.
                
                for j in range(self.num_int):
                    x += self.c_mn(j,m,np.zeros(self.num_species),self.i+1)
                    y += self.c_mn(j,m,m,self.i+1)
                    
                if self.i < len(self.timeGrid)-1:
                    self.hatTheta1[k,self.i+1]      = -x
                
                if self.i > 0:
                    self.hatR1[k,self.i+1,self.i+1]   = -y/self.delta_t
                    #self.hatR1[k,self.i+1,self.i]   = -y/self.delta_t
                
                del x,y,z
                
            if self.alphaOrder == 2:
                
                for k in range(self.num_species):
                    #x,y,z = 0.,0.,0.
                    x     = 0.
                    m     = np.zeros(self.num_species)
                    m[k]  = 1.
                    mZero = np.zeros(self.num_species)
                    
                    cNN     = np.zeros(self.i+1)
                    M_v     = np.zeros(self.i+1)
                    chi_mat = np.zeros([self.i+1,self.i+1])
                    Y_mat   = np.zeros([self.i+1,self.i+1])
                    M_f     = np.zeros(self.i+1)
                    L       = np.diag(np.ones(self.i),k=-1)
                                        
                    for n in self.n_listFULL[0]:
                        if n not in self.exclusion_list_bubble:
                            
                            for t in range(self.i+1):
                                cNN[t]   = self.c_mnFULL(n,n,t)
                                M_v[t]   = self.c_mnFULL(n,mZero,t)
                                M_f[t] = self.c_mnFULL(n,m,t)
                                
                                for t2 in range(self.i+1):
                                    Y_mat[t,t2] = np.prod((sc.special.factorial(np.array(n)))*np.power(self.resp[:,t,t2], np.array(n)))                                  
                            for t in range(self.i+1):
                                chi_mat[t] = cNN*Y_mat[t]
                            
                            temp = np.matmul((sc.linalg.lapack.dtrtri(np.identity(self.i+1)-self.alpha*self.delta_t*np.matmul(chi_mat,L),lower=1,overwrite_c=1,unitdiag=1))[0],Y_mat)   
                            
                            #temp = np.matmul(sc.linalg.solve_triangular(a=np.identity(self.i+1)-self.alpha*self.delta_t*np.matmul(chi_mat,L), b=np.identity(self.i+1), lower=True,overwrite_b=True,check_finite=False),Y_mat)
                            
                            x += self.c_mnFULL(m,n,self.i+1)*np.sum(temp[self.i]*M_v)
                                                            
                            self.hatR2[k,self.i+1,:self.i+1] = -2*self.c_mnFULL(m,n,self.i+1)*(temp[self.i]*M_f)
                                
                            del temp
                        
                        elif n in self.exclusion_list_bubble:
                            if n not in [mZero.astype(int).tolist(),m.astype(int).tolist()]:
                                
                                for t in range(self.i+1):
                                    M_f[t] = self.c_mnFULL(n,m,t)
                                    cNN[t] = self.c_mnFULL(n,n,t)

                                    for t2 in range(self.i+1):
                                        Y_mat[t,t2] = np.prod((sc.special.factorial(np.array(n)))*np.power(self.resp[:,t,t2], np.array(n)))
                                for t in range(self.i+1):
                                    chi_mat[t] = cNN*Y_mat[t]

                                temp = np.matmul((sc.linalg.lapack.dtrtri(np.identity(self.i+1)-self.alpha*self.delta_t*np.matmul(chi_mat,L),lower=1,overwrite_c=1,unitdiag=1))[0],Y_mat)
                                
                                #temp = np.matmul(sc.linalg.solve_triangular(a=np.identity(self.i+1)-self.alpha*self.delta_t*np.matmul(chi_mat,L), b=np.identity(self.i+1), lower=True,overwrite_b=True,check_finite=False),Y_mat)   
                                
                                self.hatR2[k,self.i+1,:self.i+1] = -2*self.c_mnFULL(m,n,self.i+1)*(temp[self.i]*M_f)

                                del temp
                                                    
                    if self.i < len(self.timeGrid)-1:
                        self.hatTheta2[k,self.i+1] = -2*self.delta_t*x
                    
                    del x,y,z,m,mZero
                    
    def updateFields_hatB_bubble_sum(self):
        """
        Updating the field values for fieldMethod = bubble_sum and C = False. Modification of the faster version. This additionally claculates the hat(B) fields, only as a function of the Response. This uses teh SBR series, but also had additional terms at the next order.
        
        """

        if self.orderParameter == 'linear':
            
            for k in range(self.num_species):
                x    = 0
                m    = np.zeros(self.num_species)
                m[k] = 1.
                
                for j in range(self.num_int):
                    x += -self.c_mn(j,m,np.zeros(self.num_species),self.i+1)
                    
                if self.i < len(self.timeGrid)-1:
                    self.hatTheta1[k,self.i+1] = x
                
            if self.alphaOrder == 2:                
                for k in range(self.num_species):
                    x    = 0
                    m    = np.zeros(self.num_species)
                    m[k] = 1.
                    mZero= np.zeros(self.num_species)
                    
                    cNN     = np.zeros(self.i+1)
                    M_v     = np.zeros(self.i+1)
                    chi_mat = np.zeros([self.i+1,self.i+1])
                    Y_mat   = np.zeros([self.i+1,self.i+1])
                    L       = np.diag(np.ones(self.i),k=-1)
                    
                    for n in self.n_listFULL[0]:
                        if n not in self.exclusion_list_bubble:
                            
                            for t in range(self.i+1):
                                cNN[t] = self.c_mnFULL(n,n,t)
                                M_v[t] = self.c_mnFULL(n,mZero,t)
                                for t2 in range(self.i+1):
                                    Y_mat[t,t2] = np.prod((sc.special.factorial(np.array(n)))*np.power(self.resp[:,t,t2], np.array(n)))                            
                            for t in range(self.i+1):
                                chi_mat[t] = cNN*Y_mat[t]

                            temp = np.matmul(np.matmul(sc.linalg.solve_triangular(a=np.identity(self.i+1)-self.alpha*self.delta_t*np.matmul(chi_mat,L), b=np.identity(self.i+1), lower=True,overwrite_b=True,check_finite=False),Y_mat),M_v)
                            
                            x += self.c_mnFULL(m,n,self.i+1)*temp[self.i]
                    
                    if self.i < len(self.timeGrid)-1:
                        self.hatTheta2[k,self.i+1] = -2*self.delta_t*x
                    
                    del x
            
        elif self.orderParameter == 'quad':
            
            for k in range(self.num_species):
                x    = 0.
                y    = 0.
                z    = 0.
                m    = np.zeros(self.num_species)
                m[k] = 1.
                
                for j in range(self.num_int):
                    x += self.c_mn(j,m,np.zeros(self.num_species),self.i+1)
                    y += self.c_mn(j,m,m,self.i+1)
                    z += 2*self.c_mn(j,2*m,np.zeros(self.num_species),self.i+1)
                    
                if self.i < len(self.timeGrid)-1:
                    self.hatTheta1[k,self.i+1]      = -x
                
                if self.i > 0:
                    self.hatR1[k,self.i+1,self.i+1]   = -y/self.delta_t
                    self.hatB1[k,self.i+1,self.i+1]   = -z/self.delta_t
                    #self.hatR1[k,self.i+1,self.i]   = -y/self.delta_t
                    
                del x,y,z
                
            if self.alphaOrder == 2:
                
                for k in range(self.num_species):
                    x,y,z = 0.,0.,0.
                    m     = np.zeros(self.num_species)
                    m[k]  = 1.
                    mZero = np.zeros(self.num_species)
                    
                    cNN     = np.zeros(self.i+1)
                    M_v     = np.zeros(self.i+1)
                    #B_v     = np.zeros(self.i+1)
                    chi_mat = np.zeros([self.i+1,self.i+1])
                    Y_mat   = np.zeros([self.i+1,self.i+1])
                    M_f     = np.zeros([self.i+1,self.i+1])
                    B_f     = np.zeros([self.i+1,self.i+1])
                    B_f2    = np.zeros([self.i+1,self.i+1])
                    L       = np.diag(np.ones(self.i),k=-1)
                    #L       = np.diag(np.ones(self.i+1),k=0)
                                        
                    for n in self.n_listFULL[0]:
                        if n not in self.exclusion_list_bubble:
                            
                            for t in range(self.i+1):
                                cNN[t]   = self.c_mnFULL(n,n,t)
                                M_v[t]   = self.c_mnFULL(n,mZero,t)
                                #B_v[t]   = self.c_mnFULL(n,mZero,t)
                                M_f[t,t] = self.c_mnFULL(n,m,t)
                                B_f[t,t] = self.c_mnFULL(m+n,mZero,t)
                                B_f2[t,t]= self.c_mnFULL(m+n,n,t)
                                
                                for t2 in range(self.i+1):
                                    Y_mat[t,t2] = np.prod((sc.special.factorial(np.array(n)))*np.power(self.resp[:,t,t2], np.array(n)))                                  
                            for t in range(self.i+1):
                                chi_mat[t] = cNN*Y_mat[t]

                            #temp = np.matmul(np.matmul(sc.linalg.solve_triangular(a=np.identity(self.i+1)-self.alpha*self.delta_t*np.matmul(chi_mat,L), b=np.identity(self.i+1), lower=True,overwrite_b=True),Y_mat),M_v)
                            
                            temp = np.matmul(sc.linalg.solve_triangular(a=np.identity(self.i+1)-self.alpha*self.delta_t*np.matmul(chi_mat,L), b=np.identity(self.i+1), lower=True,overwrite_b=True,check_finite=False),Y_mat)
                            
                            x += self.c_mnFULL(m,n,self.i+1)*(np.matmul(temp,M_v)[self.i]) 
                            
                            R2_temp = -2*self.c_mnFULL(m,n,self.i+1)*np.matmul(temp,M_f)
                            
                            z += self.c_mnFULL(2*m,n,self.i+1)*(np.matmul(temp,M_v)[self.i])                            
                            
                            #B2_temp = -2*(n[k]+1)*self.c_mnFULL(m,n,self.i+1)*np.matmul(temp,B_f) -2*(n[k]+1)*self.c_mnFULL(m,n,self.i+1)*np.array([[np.matmul(temp,B_f2)[i,j]*np.matmul(temp,M_v)[j] for i in range(self.i+1)] for j in range(self.i+1)])
                            
                            B2_temp = -2*(n[k]+1)*self.c_mnFULL(m,n,self.i+1)*np.matmul(temp,B_f)[self.i] -2*(n[k]+1)*self.c_mnFULL(m,n,self.i+1)*np.matmul(temp,B_f2)[self.i,:]*np.matmul(temp,M_v)[:]
                                                        
                            for t in range(self.i+1):
                                #self.hatR2[k,self.i+1,t] += -2*self.c_mnFULL(m,n,self.i+1)*(np.matmul(temp,M_f)[self.i,t])
                                self.hatR2[k,self.i+1,t] += R2_temp[self.i,t]
                                #self.hatB2[k,self.i+1,t] += -2*(n[k]+1)*self.c_mnFULL(m,n,self.i+1)*self.c_mnFULL(m+n,mZero,t)*Y_mat[self.i,t]
                                #self.hatB2[k,self.i+1,t] += B2_temp[self.i,t]
                                self.hatB2[k,self.i+1,t] += B2_temp[t]
                                
                            del temp
                        
                        elif n in self.exclusion_list_bubble:
                            if n not in [mZero.astype(int).tolist(),m.astype(int).tolist()]:
                                #print(k,n)
                                for t in range(self.i+1):
                                    M_f[t,t] = self.c_mnFULL(n,m,t)
                                    M_v[t]   = self.c_mnFULL(n,mZero,t)
                                    cNN[t]   = self.c_mnFULL(n,n,t)
                                    B_f[t,t] = self.c_mnFULL(m+n,mZero,t)
                                    B_f2[t,t]= self.c_mnFULL(m+n,n,t)

                                    for t2 in range(self.i+1):
                                        Y_mat[t,t2] = np.prod((sc.special.factorial(np.array(n)))*np.power(self.resp[:,t,t2], np.array(n)))
                                for t in range(self.i+1):
                                    chi_mat[t] = cNN*Y_mat[t]

                                temp = np.matmul(sc.linalg.solve_triangular(a=np.identity(self.i+1)-self.alpha*self.delta_t*np.matmul(chi_mat,L), b=np.identity(self.i+1), lower=True,overwrite_b=True,check_finite=False),Y_mat)
                                
                                R2_temp = -2*self.c_mnFULL(m,n,self.i+1)*np.matmul(temp,M_f)
                                
                                z += 2*self.c_mnFULL(2*m,n,self.i+1)*(np.matmul(temp,M_v)[self.i])
                                
                                #B2_temp = -2*(n[k]+1)*self.c_mnFULL(m,n,self.i+1)*np.matmul(temp,B_f) -2*(n[k]+1)*self.c_mnFULL(m,n,self.i+1)*np.array([[np.matmul(temp,B_f2)[i,j]*np.matmul(temp,M_v)[j] for i in range(self.i+1)] for j in range(self.i+1)])
                                
                                B2_temp = -2*(n[k]+1)*self.c_mnFULL(m,n,self.i+1)*np.matmul(temp,B_f)[self.i] -2*(n[k]+1)*self.c_mnFULL(m,n,self.i+1)*np.matmul(temp,B_f2)[self.i,:]*np.matmul(temp,M_v)[:]
                                
                                #Should add a shift in between the two!
                                
                                for t in range(self.i+1):
                                    self.hatR2[k,self.i+1,t] += R2_temp[self.i,t]                         
                                    self.hatB2[k,self.i+1,t] += B2_temp[t]
                                    
                                del temp
                                
                            elif n in [m.astype(int).tolist()]:
                                
                                for t in range(self.i+1):
                                    M_v[t]   = self.c_mnFULL(n,mZero,t)
                                    cNN[t]   = self.c_mnFULL(n,n,t)
                                    B_f[t,t] = self.c_mnFULL(m+n,mZero,t)
                                    B_f2[t,t]= self.c_mnFULL(m+n,n,t)

                                    for t2 in range(self.i+1):
                                        Y_mat[t,t2] = np.prod((sc.special.factorial(np.array(n)))*np.power(self.resp[:,t,t2], np.array(n)))
                                for t in range(self.i+1):
                                    chi_mat[t] = cNN*Y_mat[t]

                                #temp = np.matmul(sc.linalg.solve_triangular(a=np.identity(self.i+1)-self.alpha*self.delta_t*np.matmul(chi_mat,L), b=np.identity(self.i+1), lower=True,overwrite_b=True,check_finite=False),Y_mat)
                                
                                #z += self.c_mnFULL(2*m,n,self.i+1)*(np.matmul(temp,M_v)[self.i])
                                
                                #What about this diagonal term?
                                
                                #B2_temp = -2*(n[k]+1)*self.c_mnFULL(m,n,self.i+1)*np.matmul(temp,B_f)[self.i] -2*(n[k]+1)*self.c_mnFULL(m,n,self.i+1)*np.matmul(temp,B_f2)[self.i,:]*np.matmul(temp,M_v)[:]
                                
                                #Should add a shift in between the two!
                                
                                #for t in range(self.i+1):                       
                                    #self.hatB2[k,self.i+1,t] += B2_temp[t]
                                    
                                #del temp
                                                    
                    if self.i < len(self.timeGrid)-1:
                        self.hatTheta2[k,self.i+1] = -2*self.delta_t*x
                        self.hatB2[k,self.i+1,self.i+1] = -2*z
                                                
                    for t in range(self.i+1): 
                        
                        # This is temporary, just to handle the n = 1 which is exluded above!
                        
                        #if t < self.i:
                            #self.hatB2[k,self.i+1,t+1] += 2*(n[k]+1)*self.c_mnFULL(m,m,self.i+1)*self.c_mnFULL(m+m,mZero,t+1)*self.resp[:,self.i,t+1]
                        
                        self.hatB2[k,t,self.i+1] = self.hatB2[k,self.i+1,t]
                        
                    del x,y,z,m,mZero
                    
                    
                    
    def updateFields_hatB_bubble_sum_mixed_SBR(self):
        """
        Updating the field values for fieldMethod = bubble_sum and C = False. Modification of the faster version. This additionally claculates the hat(B) fields, only as a function of the Response. This uses the SBR series, but also had additional terms at the next order.
        Additionally, this calculates the cross corrections or the mixed corrections to the SBR series, which is useful for the multispecies case!
        
        """

        if self.orderParameter == 'linear':
            
            for k in range(self.num_species):
                x    = 0
                m    = np.zeros(self.num_species)
                m[k] = 1.
                
                for j in range(self.num_int):
                    x += -self.c_mn(j,m,np.zeros(self.num_species),self.i+1)
                    
                if self.i < len(self.timeGrid)-1:
                    self.hatTheta1[k,self.i+1] = x
                    
            if self.alphaOrder == 2:
                
                len_n_list = 0                
                #Determine the length of non-zero values for n list:
                for n in self.n_listFULL[0]:
                    if n not in self.exclusion_list_bubble:
                        len_n_list += 1
                
                cMN     = np.zeros([len_n_list,len_n_list,self.i+1])
                chi_mat = np.zeros([len_n_list,len_n_list,self.i+1,self.i+1])
                Y_mat   = np.zeros([len_n_list,self.i+1,self.i+1])
                M_v     = np.zeros([len_n_list,self.i+1,self.i+1])
                cNzero  = np.zeros([len_n_list,self.i+1,self.i+1])
                mZero   = np.zeros(self.num_species)
                
                ind1 = 0                
                for n in self.n_listFULL[0]:
                    if n not in self.exclusion_list_bubble:
                        
                        for t in range(self.i+1):                            
                            for t2 in range(self.i+1):                                
                                Y_mat[ind1,t,t2] = np.prod((sc.special.factorial(np.array(n)))*np.power(self.resp[:,t,t2], np.array(n)))
                            for t2 in range(self.i+1):
                                cNzero[ind1,t,t2] = self.c_mnFULL(n,mZero,t2)*Y_mat[ind1,t,t2]
                        
                        ind2 = 0
                        
                        for n2 in self.n_listFULL[0]:
                            if n2 not in self.exclusion_list_bubble:
                                for t in range(self.i+1):
                                    cMN[ind1,ind2,t] = self.c_mnFULL(n,n2,t)                                                                
                                for t in range(self.i+1):
                                    chi_mat[ind1,ind2,t] = cMN[ind1,ind2]*Y_mat[ind1,t]
                                
                                ind2 += 1
                                
                        ind1 += 1
                        
                temp  = block_tri_lower_inverse(block_identity(len_n_list,self.i+1)-self.alpha*self.delta_t*block_lower_shift(chi_mat))
                
                temp2 = np.sum(block_mat_mix_mul(temp,cNzero),axis=2)[:,self.i]
                
                for k in range(self.num_species):
                    
                    m    = np.zeros(self.num_species)
                    m[k] = 1.
                    
                    czeroN = np.zeros(len_n_list)
                    ind1 = 0
                    for n in self.n_listFULL[0]:
                        if n not in self.exclusion_list_bubble:
                            czeroN[ind1] = self.c_mnFULL(m,n,self.i+1)                        
                            ind1+=1
                    
                    if self.i < len(self.timeGrid)-1:
                        self.hatTheta2[k,self.i+1] = -2*self.delta_t*np.dot(czeroN,temp2)
                    
            
        elif self.orderParameter == 'quad':
            
            for k in range(self.num_species):
                x    = 0.
                y    = 0.
                z    = 0.
                m    = np.zeros(self.num_species)
                m[k] = 1.
                
                for j in range(self.num_int):
                    x += self.c_mn(j,m,np.zeros(self.num_species),self.i+1)
                    y += self.c_mn(j,m,m,self.i+1)
                    z += 2*self.c_mn(j,2*m,np.zeros(self.num_species),self.i+1)
                    
                if self.i < len(self.timeGrid)-1:
                    self.hatTheta1[k,self.i+1]      = -x
                
                if self.i > 0:
                    self.hatR1[k,self.i+1,self.i+1]   = -y/self.delta_t
                    self.hatB1[k,self.i+1,self.i+1]   = -z/self.delta_t
                
                del x,y,z
                
            if self.alphaOrder == 2:
                
                len_n_list = 0
                n_list     = []
                
                #Determine the length of non-zero values for n list:
                for n in self.n_listFULL[0]:
                    if n not in self.exclusion_list_bubble:
                        n_list.append(n)
                        len_n_list += 1
                
                cMN     = np.zeros([len_n_list,len_n_list,self.i+1])
                chi_mat = np.zeros([len_n_list,len_n_list,self.i+1,self.i+1])
                Y_mat   = np.zeros([len_n_list,self.i+1,self.i+1])
                cNzero  = np.zeros([len_n_list,self.i+1,self.i+1])
                mZero   = np.zeros(self.num_species)
                
                ind1 = 0                
                for n in self.n_listFULL[0]:
                    if n not in self.exclusion_list_bubble:
                        
                        for t in range(self.i+1):                            
                            for t2 in range(self.i+1):                                
                                Y_mat[ind1,t,t2] = np.prod((sc.special.factorial(np.array(n)))*np.power(self.resp[:,t,t2], np.array(n)))
                            for t2 in range(self.i+1):
                                cNzero[ind1,t,t2] = self.c_mnFULL(n,mZero,t2)*Y_mat[ind1,t,t2]
                        
                        ind2 = 0
                        
                        for n2 in self.n_listFULL[0]:
                            if n2 not in self.exclusion_list_bubble:
                                for t in range(self.i+1):
                                    cMN[ind1,ind2,t] = self.c_mnFULL(n,n2,t)                                                                
                                for t in range(self.i+1):
                                    chi_mat[ind1,ind2,t] = cMN[ind1,ind2]*Y_mat[ind1,t]
                                
                                ind2 += 1
                                
                        ind1 += 1
                        
                temp  = block_tri_lower_inverse(block_identity(len_n_list,self.i+1)-self.alpha*self.delta_t*block_lower_shift(chi_mat))
                
                tempZ = block_mat_mix_mul(temp,cNzero)
                
                temp2 = np.sum(tempZ,axis=2)[:,self.i]
                
                for k in range(self.num_species):
                    
                    m    = np.zeros(self.num_species)
                    m[k] = 1.
                    
                    czeroN  = np.zeros(len_n_list)
                    czeroN2 = np.zeros(len_n_list)
                    ind1 = 0
                    
                    for n in self.n_listFULL[0]:
                        if n not in self.exclusion_list_bubble:
                            czeroN[ind1]  = self.c_mnFULL(m,n,self.i+1)
                            czeroN2[ind1] = 2*self.c_mnFULL(2*m,n,self.i+1) 
                            ind1+=1
                    
                    if self.i < len(self.timeGrid)-1:
                        self.hatTheta2[k,self.i+1] = -2*self.delta_t*np.dot(czeroN,temp2)
                    
                    #z = np.dot(czeroN2,np.sum(tempZ,axis=2)[:,self.i])
                    
                    if self.i < len(self.timeGrid)-1:
                        self.hatB2[k,self.i+1,self.i+1] = -2*np.dot(czeroN2,temp2) #-2*z 
                
                #Have to do everything for each species, because the list depends on the species index!
                for k in range(self.num_species):
                    
                    z     = 0.
                    m     = np.zeros(self.num_species)
                    m[k]  = 1.
                    mZero = np.zeros(self.num_species)
                    
                    #Calculate the series with additional n vectors for hat R and hat B:                
                    len_n_list2 = 0
                    n_list2     = []

                    #Determine the length of non-zero values for n list:
                    for n in self.n_listFULL[0]:
                        if n not in self.exclusion_list_bubble:
                            n_list2.append(n)
                            len_n_list2 += 1

                        elif n in self.exclusion_list_bubble:
                            if n not in [mZero.astype(int).tolist(),m.astype(int).tolist()]:
                                         #(2*m).astype(int).tolist()]:
                                #Should this last factor be there? It will not affect the hatR, but only hat B bacause we have a binary system?
                                n_list2.append(n)
                                len_n_list2 += 1
                    
                    cMN     = np.zeros([len_n_list2,len_n_list2,self.i+1])
                    chi_mat = np.zeros([len_n_list2,len_n_list2,self.i+1,self.i+1])
                    Y_mat   = np.zeros([len_n_list2,self.i+1,self.i+1])
                    Y_mat_MN= np.zeros([len_n_list2,self.i+1,self.i+1])
                    cNzero  = np.zeros([len_n_list2,self.i+1,self.i+1])
                    mZero   = np.zeros(self.num_species)
                    czeroN  = np.zeros(len_n_list2)
                    #czeroN2 = np.zeros(len_n_list2)
                    #czeroN3 = np.zeros(len_n_list2)
                    cNone   = np.zeros([len_n_list2,self.i+1,self.i+1])
                    
                    cNplus1zero = np.zeros([len_n_list2,self.i+1,self.i+1])
                    cMplus1N    = np.zeros([len_n_list2,len_n_list2,self.i+1,self.i+1])

                    ind1 = 0                
                    for n in n_list2:
                        
                        czeroN[ind1]  = self.c_mnFULL(m,n,self.i+1)
                        #czeroN2[ind1] = 2*self.c_mnFULL(2*m,n,self.i+1) 
                        
                        for t in range(self.i+1):                            
                            for t2 in range(self.i+1):                                
                                Y_mat[ind1,t,t2] = np.prod((sc.special.factorial(np.array(n)))*np.power(self.resp[:,t,t2], np.array(n)))
                                #This keeps track of the correct combinatorial factor while dealing with C_N+1 terms
                                Y_mat_MN[ind1,t,t2] = np.prod((sc.special.factorial(np.array(m+n)))*np.power(self.resp[:,t,t2], np.array(n)))
                            for t2 in range(self.i+1):
                                cNzero[ind1,t,t2] = self.c_mnFULL(n,mZero,t2)*Y_mat[ind1,t,t2]
                                cNone[ind1,t,t2]  = self.c_mnFULL(n,m,t2)*Y_mat[ind1,t,t2]
                                
                                #cNplus1zero[ind1,t,t2] = (n[k]+1)*self.c_mnFULL(m+n,mZero,t2)*Y_mat[ind1,t,t2]
                                cNplus1zero[ind1,t,t2] = self.c_mnFULL(m+n,mZero,t2)*Y_mat_MN[ind1,t,t2]
                                
                                ind2 = 0
                                for n2 in n_list2:
                                    #cMplus1N[ind1,ind2,t,t2] = (n[k]+1)*self.c_mnFULL(m+n,n2,t2)*Y_mat[ind1,t,t2]
                                    cMplus1N[ind1,ind2,t,t2] = self.c_mnFULL(m+n,n2,t2)*Y_mat_MN[ind1,t,t2]
                                    ind2 += 1

                        ind2 = 0                        
                        for n2 in n_list2:                            
                            for t in range(self.i+1):
                                cMN[ind1,ind2,t] = self.c_mnFULL(n,n2,t)                                                                
                            for t in range(self.i+1):
                                chi_mat[ind1,ind2,t] = cMN[ind1,ind2]*Y_mat[ind1,t]

                            ind2 += 1

                        ind1 += 1
                        
                    temp  = block_tri_lower_inverse(block_identity(len_n_list2,self.i+1)-self.alpha*self.delta_t*block_lower_shift(chi_mat))
                    
                    #Implemented this
                    #temp2 = np.sum(block_mat_mix_mul(temp,cNzero),axis=2)[:,self.i]
                    
                    R2_temp = np.zeros([self.i+1,self.i+1])
                    B2_temp = np.zeros([self.i+1,self.i+1])
                    
                    temp2   = block_mat_mix_mul(temp,cNone)
                    temp3   = block_mat_mix_mul(temp,cNplus1zero)
                    #temp4   = block_mat_mix_mul(temp,cNzero)
                    temp5   = block_mat_mul(temp,cMplus1N)
                    
                    #The following makes the temp4 nlist2 indices 0 which are not in n_list
                    #for ind2 in range(len_n_list2):
                    #    if n_list2[ind2] not in n_list:
                    #        temp4[ind2,:,:] = 0
                    
                    for ind1 in range(len_n_list2):
                        R2_temp += -2*(czeroN[ind1]*temp2[ind1])                        
                        B2_temp += -2*(czeroN[ind1]*temp3[ind1])
                                                
                    for ind1 in range(len_n_list2):
                        for ind2 in range(len_n_list2):
                            for ind3 in range(len_n_list):
                                if n_list2[ind2] == n_list[ind3]:
                        #if n_list2[ind2] in n_list:
                        #Note that index2 in this case cannot take all values, because some are already included in tadpole summation of \mu -- it can only take he values for \mu list!

                                    for t1 in range(self.i+1):
                                        for t2 in range(self.i+1):
                                            #B2_temp[t1,t2] += -2*self.delta_t*czeroN[ind1]*temp5[ind1,ind2,t1,t2]*(np.sum(temp4,axis=2))[ind2,t2]
                                            B2_temp[t1,t2] += -2*self.delta_t*czeroN[ind1]*temp5[ind1,ind2,t1,t2]*(np.sum(tempZ,axis=2))[ind3,t2]
                                        
                                        ## TODO: Change this temp4 by temp which does not include R_B in there!
                    
                    #z = np.dot(czeroN2,np.sum(temp4,axis=2)[:,self.i])                    
                    #Change this z as well, it needs to be included without the R_B connectors list!!!
                    
                    for t in range(self.i+1):
                        self.hatR2[k,self.i+1,t] += R2_temp[self.i,t]
                        self.hatB2[k,self.i+1,t] += B2_temp[self.i,t]
                        
                    #if self.i < len(self.timeGrid)-1:
                    #    self.hatB2[k,self.i+1,self.i+1] = -2*z
                                                
                    for t in range(self.i+1):                         
                        self.hatB2[k,t,self.i+1] = self.hatB2[k,self.i+1,t]
                        
                    del z,m,mZero
                    
    def response_combinations(self,n1,n2,resp):
        
        #First do it assuming only two response connections -- thus the length limit!
        
        resp_comb = np.zeros(np.shape(resp)[2:])
        
        x1 = np.where(np.array(n1) != 0)[0]
        x2 = np.where(np.array(n2) != 0)[0]
        
        if len(x1) == 2 and len(x2) == 2:
        
            resp_comb = resp[x1[0],x2[0],:,:]*resp[x1[1],x2[1],:,:] + resp[x1[0],x2[1],:,:]*resp[x1[1],x2[0],:,:]
            
        elif len(x1) == 1 and len(x2) == 2:
            
            resp_comb = 2*resp[x1[0],x2[0],:,:]*resp[x1[0],x2[1],:,:]
        
        elif len(x1) == 2 and len(x2) == 1:
            
            resp_comb = 2*resp[x1[0],x2[0],:,:]*resp[x1[1],x2[0],:,:]
            
            #print(x1[0],x2[0],x1[1],x2[0])
            
        elif len(x1) == 1 and len(x2) == 1:
            
            resp_comb = 2*resp[x1[0],x2[0],:,:]*resp[x1[0],x2[0],:,:]
        
        #print(n1,n2,resp_comb)
        
        #for i1 in range(self.num_species):
        #    for i2 in range(self.num_species):
        #        resp_comb += resp[n1[i1],n2[i2],:,:]
            
        return resp_comb

    def updateFields_hatB_bubble_sum_mixed_SBR_crossResponse_C_loop(self):
        """
        DOES the cross responses additionally !!! Uses the equal time C in hat R and theta to update their values, does not do the explicit sum!
        
        Updating the field values for fieldMethod = bubble_sum and C = True. Modification of the faster version. This additionally claculates the hat(B) fields, only as a function of the Response. This uses the SBR series, but also had additional terms at the next order.
        Additionally, this calculates the cross corrections or the mixed corrections to the SBR series, which is useful for the multispecies case!
        
        """                  
            
        if self.orderParameter == 'quad':
            
            for k in range(self.num_species):
                x    = 0.
                m    = np.zeros(self.num_species)
                m[k] = 1.
                
                for j in range(self.num_int):
                    x += self.c_mn(j,m,np.zeros(self.num_species),self.i+1)

                    for j_p in range(self.num_species):
                        m_p    = np.zeros(self.num_species)
                        m_p[j_p] = 1.
                        
                        for j_q in range(j_p+1):
                            m_q    = np.zeros(self.num_species)
                            m_q[j_q] = 1.
                            
                            x += self.c_mn(j,m,m_p+m_q,self.i+1)*self.corr[j_p,j_q,self.i,self.i]
                    
                if self.i < len(self.timeGrid)-1:
                    self.hatTheta1[k,self.i+1]      = -x
                    
                for k2 in range(self.num_species):
                    y    = 0.
                    z    = 0.
                    m2    = np.zeros(self.num_species)
                    m2[k2] = 1.
                    
                    for j in range(self.num_int):
                        y += self.c_mn(j,m,m2,self.i+1)
                        z += np.prod(sc.special.factorial(m+m2))*self.c_mn(j,m+m2,np.zeros(self.num_species),self.i+1)

                        for j_p in range(self.num_species):
                            m_p    = np.zeros(self.num_species)
                            m_p[j_p] = 1.
                            
                            for j_q in range(j_p+1):
                            #for j_q in range(self.num_species):
                                m_q    = np.zeros(self.num_species)
                                m_q[j_q] = 1.

                                #The factorial product gives 2 when m = m2 and 1 otherwise
                                z += np.prod(sc.special.factorial(m+m2))*self.c_mn(j,m+m2,m_p+m_q,self.i+1)*self.corr[j_p,j_q,self.i,self.i]                    

                    #IMP: TAKE CARE OF THIS CONDITION -- Self.i must be geq 0!!
                    if self.i >= 0:
                        self.hatR1[k,k2,self.i+1,self.i+1]   = -y/self.delta_t
                        self.hatB1[k,k2,self.i+1,self.i+1]   = -z/self.delta_t
                    
                del x,y,z
                
            if self.alphaOrder == 2:
                
                len_n_list = 0
                n_list     = []
                
                #Determine the length of non-zero values for n list:
                #for n in self.n_listFULL[0]:
                #    if n not in self.exclusion_list_bubble:
                #        n_list.append(n)
                #        len_n_list += 1
                
                # I changed this to m_list but with the sum restriction so only two connections are present!
                for n in self.m_listFULL[0]:
                    if n not in self.exclusion_list_bubble:
                        if np.sum(np.array(n)) < 3:
                            n_list.append(n)
                            len_n_list += 1
                
                #print(n_list)
                
                cMN      = np.zeros([len_n_list,len_n_list,self.i+1,self.i+1])
                chi_mat  = np.zeros([len_n_list,len_n_list,self.i+1,self.i+1])
                Y_mat    = np.zeros([len_n_list,len_n_list,self.i+1,self.i+1])
                #cNzero   = np.zeros([len_n_list,self.i+1,self.i+1])
                #cNzeroTMP= np.zeros([len_n_list,self.i+1,self.i+1])
                mZero    = np.zeros(self.num_species)                
                
                ind1 = 0                
                for n in n_list:
                    ind2 = 0
                    for n2 in n_list:
                        #Do it only if the number of legs on both sides match -- CHECK whether to restrict n or n2
                        if np.sum(n) == np.sum(n2):
                            Y_mat[ind1,ind2,:,:] = self.response_combinations(n,n2,self.resp[:,:,:self.i+1,:self.i+1])
                        
                        for t in range(self.i+1):
                            cMN[ind1,ind2,t,t]   = self.c_mnFULL(n,n2,t)
                        
                        ind2 += 1
                    
                    #for t in range(self.i+1):
                        #cNzeroTMP[ind1,t,t] = self.c_mnFULL(n,mZero,t)
                    
                    ind1 += 1                    
                
                chi_mat = block_mat_mul(Y_mat,cMN)
                #cNzero  = block_mat_mix_mul(Y_mat,cNzeroTMP)
                
                temp  = block_tri_lower_inverse(block_identity(len_n_list,self.i+1)-self.alpha*self.delta_t*block_lower_shift(chi_mat))
                
                #temp2 = np.sum(block_mat_mix_mul(temp,cNzero),axis=2)[:,self.i]
                
                for k in range(self.num_species):
                    
                    m    = np.zeros(self.num_species)
                    m[k] = 1.
                    
                    czeroN = np.zeros(len_n_list)
                    ind1 = 0
                    
                    for n in n_list:
                        
                        czeroN[ind1] = self.c_mnFULL(m,n,self.i+1)

                        ind1+=1
                    
                    #if self.i < len(self.timeGrid)-1:
                    #    self.hatTheta2[k,self.i+1] = -2*self.delta_t*np.dot(czeroN,temp2)
                    
                    for k2 in range(self.num_species):
                        z      = 0.
                        m2     = np.zeros(self.num_species)
                        m2[k2] = 1.
                        
                        #czeroN2 = np.zeros(len_n_list)
                        cNone   = np.zeros([len_n_list,self.i+1,self.i+1])
                        
                        ind1 = 0                
                        for n in n_list:
                            
                            #czeroN2[ind1] = np.prod(sc.special.factorial(m+m2))*self.c_mnFULL(m+m2,n,self.i+1)
                            
                            for t in range(self.i+1):
                                cNone[ind1,t,t]  = self.c_mnFULL(n,m2,t)           
                            
                            ind1 += 1
                            
                        cNone   = block_mat_mix_mul(Y_mat,cNone)
                            
                        R2_temp = np.zeros([self.i+1,self.i+1])

                        temp3   = block_mat_mix_mul(temp,cNone)

                        for ind1 in range(len_n_list):
                            R2_temp += -2*(czeroN[ind1]*temp3[ind1])                        

                        #z = np.dot(czeroN2,temp2)

                        for t in range(self.i+1):
                            self.hatR2[k,k2,self.i+1,t] += R2_temp[self.i,t]

                        #if self.i < len(self.timeGrid)-1:
                        #    self.hatB2[k,k2,self.i+1,self.i+1] = -2*z
                        
                del z,m,mZero
                    
                    
    def updateFields_hatB_bubble_sum_mixed_SBR_crossResponse(self):
        """
        DOES the cross responses additionally !!!
        Updating the field values for fieldMethod = bubble_sum and C = False. Modification of the faster version. This additionally claculates the hat(B) fields, only as a function of the Response. This uses the SBR series, but also had additional terms at the next order.
        Additionally, this calculates the cross corrections or the mixed corrections to the SBR series, which is useful for the multispecies case!
        
        """

        if self.orderParameter == 'linear':
            
            for k in range(self.num_species):
                x    = 0
                m    = np.zeros(self.num_species)
                m[k] = 1.
                
                for j in range(self.num_int):
                    x += -self.c_mn(j,m,np.zeros(self.num_species),self.i+1)
                    
                if self.i < len(self.timeGrid)-1:
                    self.hatTheta1[k,self.i+1] = x
                    
            if self.alphaOrder == 2:
                
                len_n_list = 0
                n_list     = []
                
                # I changed this to m_list but with the sum restriction so only two connections are present!
                for n in self.m_listFULL[0]:
                    if n not in self.exclusion_list_bubble:
                        if np.sum(np.array(n)) < 3:
                            n_list.append(n)
                            len_n_list += 1
                
                cMN      = np.zeros([len_n_list,len_n_list,self.i+1,self.i+1])
                chi_mat  = np.zeros([len_n_list,len_n_list,self.i+1,self.i+1])
                Y_mat    = np.zeros([len_n_list,len_n_list,self.i+1,self.i+1])
                cNzero   = np.zeros([len_n_list,self.i+1,self.i+1])
                cNzeroTMP= np.zeros([len_n_list,self.i+1,self.i+1])
                mZero    = np.zeros(self.num_species)                
                
                ind1 = 0                
                for n in n_list:
                    ind2 = 0
                    for n2 in n_list:
                        #Do it only if the number of legs on both sides match -- CHECK whether to restrict n or n2
                        if np.sum(n) == np.sum(n2):
                            Y_mat[ind1,ind2,:,:] = self.response_combinations(n,n2,self.resp[:,:,:self.i+1,:self.i+1])
                        
                        for t in range(self.i+1):
                            cMN[ind1,ind2,t,t]   = self.c_mnFULL(n,n2,t)
                        
                        ind2 += 1
                    
                    for t in range(self.i+1):
                        cNzeroTMP[ind1,t,t] = self.c_mnFULL(n,mZero,t)
                    
                    ind1 += 1                    
                
                chi_mat = block_mat_mul(Y_mat,cMN)
                cNzero  = block_mat_mix_mul(Y_mat,cNzeroTMP)
                
                temp  = block_tri_lower_inverse(block_identity(len_n_list,self.i+1)-self.alpha*self.delta_t*block_lower_shift(chi_mat))
                
                temp2 = np.sum(block_mat_mix_mul(temp,cNzero),axis=2)[:,self.i]
                
                for k in range(self.num_species):
                    
                    m    = np.zeros(self.num_species)
                    m[k] = 1.
                    
                    czeroN = np.zeros(len_n_list)
                    ind1 = 0
                    
                    for n in n_list:
                        
                        czeroN[ind1] = self.c_mnFULL(m,n,self.i+1)

                        ind1+=1
                    
                    if self.i < len(self.timeGrid)-1:
                        self.hatTheta2[k,self.i+1] = -2*self.delta_t*np.dot(czeroN,temp2)
                    
            
        elif self.orderParameter == 'quad':
            
            for k in range(self.num_species):
                x    = 0.
                m    = np.zeros(self.num_species)
                m[k] = 1.
                
                for j in range(self.num_int):
                    x += self.c_mn(j,m,np.zeros(self.num_species),self.i+1)
                    
                if self.i < len(self.timeGrid)-1:
                    self.hatTheta1[k,self.i+1]      = -x
                    
                for k2 in range(self.num_species):
                    y    = 0.
                    z    = 0.
                    m2    = np.zeros(self.num_species)
                    m2[k2] = 1.
                    
                    for j in range(self.num_int):
                        y += self.c_mn(j,m,m2,self.i+1)
                        z += np.prod(sc.special.factorial(m+m2))*self.c_mn(j,m+m2,np.zeros(self.num_species),self.i+1)
                    #The factorial product gives 2 when m = m2 and 1 otherwise
                    
                    if self.i > 0:
                        self.hatR1[k,k2,self.i+1,self.i+1]   = -y/self.delta_t
                        self.hatB1[k,k2,self.i+1,self.i+1]   = -z/self.delta_t
                
                del x,y,z
                
            if self.alphaOrder == 2:
                
                len_n_list = 0
                n_list     = []
                
                #Determine the length of non-zero values for n list:
                #for n in self.n_listFULL[0]:
                #    if n not in self.exclusion_list_bubble:
                #        n_list.append(n)
                #        len_n_list += 1
                
                # I changed this to m_list but with the sum restriction so only two connections are present!
                for n in self.m_listFULL[0]:
                    if n not in self.exclusion_list_bubble:
                        if np.sum(np.array(n)) < 3:
                            n_list.append(n)
                            len_n_list += 1
                
                #print(n_list)
                
                cMN      = np.zeros([len_n_list,len_n_list,self.i+1,self.i+1])
                chi_mat  = np.zeros([len_n_list,len_n_list,self.i+1,self.i+1])
                Y_mat    = np.zeros([len_n_list,len_n_list,self.i+1,self.i+1])
                cNzero   = np.zeros([len_n_list,self.i+1,self.i+1])
                cNzeroTMP= np.zeros([len_n_list,self.i+1,self.i+1])
                mZero    = np.zeros(self.num_species)                
                
                ind1 = 0                
                for n in n_list:
                    ind2 = 0
                    for n2 in n_list:
                        #Do it only if the number of legs on both sides match -- CHECK whether to restrict n or n2
                        if np.sum(n) == np.sum(n2):
                            Y_mat[ind1,ind2,:,:] = self.response_combinations(n,n2,self.resp[:,:,:self.i+1,:self.i+1])
                        
                        for t in range(self.i+1):
                            cMN[ind1,ind2,t,t]   = self.c_mnFULL(n,n2,t)
                        
                        ind2 += 1
                    
                    for t in range(self.i+1):
                        cNzeroTMP[ind1,t,t] = self.c_mnFULL(n,mZero,t)
                    
                    ind1 += 1                    
                
                chi_mat = block_mat_mul(Y_mat,cMN)
                cNzero  = block_mat_mix_mul(Y_mat,cNzeroTMP)
                
                temp  = block_tri_lower_inverse(block_identity(len_n_list,self.i+1)-self.alpha*self.delta_t*block_lower_shift(chi_mat))
                
                temp2 = np.sum(block_mat_mix_mul(temp,cNzero),axis=2)[:,self.i]
                
                for k in range(self.num_species):
                    
                    m    = np.zeros(self.num_species)
                    m[k] = 1.
                    
                    czeroN = np.zeros(len_n_list)
                    ind1 = 0
                    
                    for n in n_list:
                        
                        czeroN[ind1] = self.c_mnFULL(m,n,self.i+1)

                        ind1+=1
                    
                    if self.i < len(self.timeGrid)-1:
                        self.hatTheta2[k,self.i+1] = -2*self.delta_t*np.dot(czeroN,temp2)
                    
                    for k2 in range(self.num_species):
                        z      = 0.
                        m2     = np.zeros(self.num_species)
                        m2[k2] = 1.
                        
                        czeroN2 = np.zeros(len_n_list)
                        cNone   = np.zeros([len_n_list,self.i+1,self.i+1])
                        
                        ind1 = 0                
                        for n in n_list:
                            
                            czeroN2[ind1] = np.prod(sc.special.factorial(m+m2))*self.c_mnFULL(m+m2,n,self.i+1)
                            
                            for t in range(self.i+1):
                                cNone[ind1,t,t]  = self.c_mnFULL(n,m2,t)           
                            
                            ind1 += 1
                            
                        cNone   = block_mat_mix_mul(Y_mat,cNone)
                            
                        R2_temp = np.zeros([self.i+1,self.i+1])

                        temp3   = block_mat_mix_mul(temp,cNone)

                        for ind1 in range(len_n_list):
                            R2_temp += -2*(czeroN[ind1]*temp3[ind1])                        

                        z = np.dot(czeroN2,temp2)

                        for t in range(self.i+1):
                            self.hatR2[k,k2,self.i+1,t] += R2_temp[self.i,t]

                        if self.i < len(self.timeGrid)-1:
                            self.hatB2[k,k2,self.i+1,self.i+1] = -2*z
                        
                del z,m,mZero
                
    def updateFields_hatB_bubble_sum_mixed_SBR_crossResponse_fixed_initialization(self):
        """
        FIXED or DELTA function initialization at t= 0!!!
        DOES the cross responses additionally !!!
        Updating the field values for fieldMethod = bubble_sum and C = False. Modification of the faster version. This additionally claculates the hat(B) fields, only as a function of the Response. This uses the SBR series, but also had additional terms at the next order.
        Additionally, this calculates the cross corrections or the mixed corrections to the SBR series, which is useful for the multispecies case!
        
        """

        if self.orderParameter == 'linear':
            
            for k in range(self.num_species):
                x    = 0
                m    = np.zeros(self.num_species)
                m[k] = 1.
                
                for j in range(self.num_int):
                    x += -self.c_mn(j,m,np.zeros(self.num_species),self.i+1)
                    
                if self.i < len(self.timeGrid)-1:
                    self.hatTheta1[k,self.i+1] = x
                    
            if self.alphaOrder == 2:
                
                len_n_list = 0
                n_list     = []
                
                # I changed this to m_list but with the sum restriction so only two connections are present!
                for n in self.m_listFULL[0]:
                    if n not in self.exclusion_list_bubble:
                        if np.sum(np.array(n)) < 3:
                            n_list.append(n)
                            len_n_list += 1
                
                cMN      = np.zeros([len_n_list,len_n_list,self.i+1,self.i+1])
                chi_mat  = np.zeros([len_n_list,len_n_list,self.i+1,self.i+1])
                Y_mat    = np.zeros([len_n_list,len_n_list,self.i+1,self.i+1])
                cNzero   = np.zeros([len_n_list,self.i+1,self.i+1])
                cNzero2  = np.zeros([len_n_list,self.i+1,self.i+1])
                cNzeroTMP= np.zeros([len_n_list,self.i+1,self.i+1])
                cNzeroTMP2= np.zeros([len_n_list,self.i+1,self.i+1])
                mZero    = np.zeros(self.num_species)                
                
                ind1 = 0                
                for n in n_list:
                    ind2 = 0
                    for n2 in n_list:
                        #Do it only if the number of legs on both sides match -- CHECK whether to restrict n or n2
                        if np.sum(n) == np.sum(n2):
                            Y_mat[ind1,ind2,:,:]   = self.response_combinations(n,n2,self.resp[:,:,:self.i+1,:self.i+1])
                            #Y_mat2[[ind1,ind2,:,:] = np.prod((sc.special.factorial(np.array(n)))*np.power(self.resp[:,:,t,t2], np.array(n)))
                        
                        for t in range(self.i+1):
                            cMN[ind1,ind2,t,t]   = self.c_mnFULL(n,n2,t)
                        
                        ind2 += 1
                    
                    for t in range(self.i+1):
                        cNzeroTMP[ind1,t,t]  = self.c_mnFULL(n,mZero,t)
                        cNzeroTMP2[ind1,t,t] = np.prod((sc.special.factorial(np.array(n)))*np.power(self.y[:,0], np.array(n)))
                        
                        #Think if this factorial n factor should be here or not!
                        
                    ind1 += 1                    
                
                chi_mat = block_mat_mul(Y_mat,cMN)
                cNzero  = block_mat_mix_mul(Y_mat,cNzeroTMP)
                cNzero2 = block_mat_mix_mul(Y_mat,cNzeroTMP2)
                
                temp  = block_tri_lower_inverse(block_identity(len_n_list,self.i+1)-self.alpha*self.delta_t*block_lower_shift(chi_mat))
                
                temp2 = np.sum(block_mat_mix_mul(temp,cNzero),axis=2)[:,self.i]                
                temp3 = block_mat_mix_mul(temp,cNzero2)[:,self.i,0]
                
                #come_back
                
                for k in range(self.num_species):
                    
                    m    = np.zeros(self.num_species)
                    m[k] = 1.
                    
                    czeroN = np.zeros(len_n_list)
                    ind1 = 0
                    
                    for n in n_list:
                        
                        czeroN[ind1] = self.c_mnFULL(m,n,self.i+1)

                        ind1+=1
                    
                    if self.i < len(self.timeGrid)-1:
                        self.hatTheta2[k,self.i+1] = -2*self.delta_t*np.dot(czeroN,temp2) + np.dot(czeroN,temp3)
                    
            
        elif self.orderParameter == 'quad':
            
            for k in range(self.num_species):
                x    = 0.
                m    = np.zeros(self.num_species)
                m[k] = 1.
                
                for j in range(self.num_int):
                    x += self.c_mn(j,m,np.zeros(self.num_species),self.i+1)
                    
                if self.i < len(self.timeGrid)-1:
                    self.hatTheta1[k,self.i+1]      = -x
                    
                for k2 in range(self.num_species):
                    y    = 0.
                    z    = 0.
                    m2    = np.zeros(self.num_species)
                    m2[k2] = 1.
                    
                    for j in range(self.num_int):
                        y += self.c_mn(j,m,m2,self.i+1)
                        z += np.prod(sc.special.factorial(m+m2))*self.c_mn(j,m+m2,np.zeros(self.num_species),self.i+1)
                    #The factorial product gives 2 when m = m2 and 1 otherwise
                    
                    if self.i > 0:
                        self.hatR1[k,k2,self.i+1,self.i+1]   = -y/self.delta_t
                        self.hatB1[k,k2,self.i+1,self.i+1]   = -z/self.delta_t
                
                del x,y,z
                
            if self.alphaOrder == 2:
                
                len_n_list = 0
                n_list     = []
                
                #Determine the length of non-zero values for n list:
                #for n in self.n_listFULL[0]:
                #    if n not in self.exclusion_list_bubble:
                #        n_list.append(n)
                #        len_n_list += 1
                
                # I changed this to m_list but with the sum restriction so only two connections are present!
                for n in self.m_listFULL[0]:
                    if n not in self.exclusion_list_bubble:
                        if np.sum(np.array(n)) < 3:
                            n_list.append(n)
                            len_n_list += 1
                
                #print(n_list)
                
                cMN      = np.zeros([len_n_list,len_n_list,self.i+1,self.i+1])
                chi_mat  = np.zeros([len_n_list,len_n_list,self.i+1,self.i+1])
                Y_mat    = np.zeros([len_n_list,len_n_list,self.i+1,self.i+1])
                cNzero   = np.zeros([len_n_list,self.i+1,self.i+1])
                #cNzero2  = np.zeros([len_n_list,self.i+1,self.i+1])
                cNzeroTMP= np.zeros([len_n_list,self.i+1,self.i+1])
                #cNzeroTMP2= np.zeros([len_n_list,self.i+1,self.i+1])
                mZero    = np.zeros(self.num_species)                
                
                ind1 = 0                
                for n in n_list:
                    ind2 = 0
                    for n2 in n_list:
                        #Do it only if the number of legs on both sides match -- CHECK whether to restrict n or n2
                        if np.sum(n) == np.sum(n2):
                            Y_mat[ind1,ind2,:,:] = self.response_combinations(n,n2,self.resp[:,:,:self.i+1,:self.i+1])
                        
                        for t in range(self.i+1):
                            cMN[ind1,ind2,t,t]   = self.c_mnFULL(n,n2,t)
                        
                        ind2 += 1
                    
                    for t in range(self.i+1):
                        cNzeroTMP[ind1,t,t] = self.c_mnFULL(n,mZero,t)
                        #cNzeroTMP2[ind1,t,t] = np.prod(sc.special.factorial(np.array(n)))*np.prod(self.y[:,0])
                        #cNzeroTMP2[ind1,t,t] = np.prod((sc.special.factorial(np.array(n)))*np.power(self.y[:,0], np.array(n)))

                    ind1 += 1                    
                
                chi_mat = block_mat_mul(Y_mat,cMN)
                cNzero  = block_mat_mix_mul(Y_mat,cNzeroTMP)
                #cNzero2 = block_mat_mix_mul(Y_mat,cNzeroTMP2)
                
                temp  = block_tri_lower_inverse(block_identity(len_n_list,self.i+1)-self.alpha*self.delta_t*block_lower_shift(chi_mat))
                
                temp2  = np.sum(block_mat_mix_mul(temp,cNzero),axis=2)[:,self.i]
                #temp2f = block_mat_mix_mul(temp,cNzero2)[:,self.i,0]
                
                for k in range(self.num_species):
                    
                    m    = np.zeros(self.num_species)
                    m[k] = 1.
                    
                    czeroN = np.zeros(len_n_list)
                    cNzero2  = np.zeros([len_n_list,self.i+1,self.i+1])
                    ind1 = 0
                    
                    for n in n_list:
                        
                        czeroN[ind1] = self.c_mnFULL(m,n,self.i+1)
                        cNzero2[ind1,:,:] = self.response_combinations(n,2*m,self.resp[:,:,:self.i+1,:self.i+1]) #self.resp[n,k,:self.i+1,:self.i+1]*self.y[k,0]
                
                        ind1+=1
                    
                    temp2f = (block_mat_mix_mul(temp,cNzero2)[:,self.i,0])    
                    
                    if self.i < len(self.timeGrid)-1:
                        self.hatTheta2[k,self.i+1] = -2*self.delta_t*np.dot(czeroN,temp2) + np.dot(czeroN,temp2f)*self.y[k,0] #+ 0.5*np.dot(czeroN,temp2f**2)*self.y[k,0]
                    
                    for k2 in range(self.num_species):
                        z      = 0.
                        m2     = np.zeros(self.num_species)
                        m2[k2] = 1.
                        
                        czeroN2 = np.zeros(len_n_list)
                        cNone   = np.zeros([len_n_list,self.i+1,self.i+1])
                        
                        ind1 = 0                
                        for n in n_list:
                            
                            czeroN2[ind1] = np.prod(sc.special.factorial(m+m2))*self.c_mnFULL(m+m2,n,self.i+1)
                            
                            for t in range(self.i+1):
                                cNone[ind1,t,t]  = self.c_mnFULL(n,m2,t)           
                            
                            ind1 += 1
                            
                        cNone   = block_mat_mix_mul(Y_mat,cNone)
                            
                        R2_temp = np.zeros([self.i+1,self.i+1])

                        temp3   = block_mat_mix_mul(temp,cNone)

                        for ind1 in range(len_n_list):
                            R2_temp += -2*(czeroN[ind1]*temp3[ind1])                        

                        z = np.dot(czeroN2,temp2)

                        for t in range(self.i+1):
                            self.hatR2[k,k2,self.i+1,t] += R2_temp[self.i,t]
                        
                        ## Note that hatB will also significantly change, and hatR as well to a small extent!
                        
                        if self.i < len(self.timeGrid)-1:
                            self.hatB2[k,k2,self.i+1,self.i+1] = -2*z
                        
                del z,m,mZero
                                        
                    
    def updateFields_no_C_bubble_sum(self):
        """
        Updating the field values for fieldMethod = bubble_sum and C = False. New version, slightly faster.
        
        """

        if self.orderParameter == 'linear':
            
            for k in range(self.num_species):
                x    = 0
                m    = np.zeros(self.num_species)
                m[k] = 1.
                
                for j in range(self.num_int):
                    x += -self.c_mn(j,m,np.zeros(self.num_species),self.i+1)
                    
                if self.i < len(self.timeGrid)-1:
                    self.hatTheta1[k,self.i+1] = x
                
            if self.alphaOrder == 2:                
                for k in range(self.num_species):
                    x    = 0
                    m    = np.zeros(self.num_species)
                    m[k] = 1.
                    mZero= np.zeros(self.num_species)
                    
                    cNN     = np.zeros(self.i+1)
                    M_v     = np.zeros(self.i+1)
                    chi_mat = np.zeros([self.i+1,self.i+1])
                    Y_mat   = np.zeros([self.i+1,self.i+1])
                    L       = np.diag(np.ones(self.i),k=-1)
                    
                    for n in self.n_listFULL[0]:
                        if n not in self.exclusion_list_bubble:
                            
                            for t in range(self.i+1):
                                cNN[t] = self.c_mnFULL(n,n,t)
                                M_v[t] = self.c_mnFULL(n,mZero,t)
                                for t2 in range(self.i+1):
                                    Y_mat[t,t2] = np.prod((sc.special.factorial(np.array(n)))*np.power(self.resp[:,t,t2], np.array(n)))                            
                            for t in range(self.i+1):
                                chi_mat[t] = cNN*Y_mat[t]

                            temp = np.matmul(np.matmul(sc.linalg.solve_triangular(a=np.identity(self.i+1)-self.alpha*self.delta_t*np.matmul(chi_mat,L), b=np.identity(self.i+1), lower=True,overwrite_b=True,check_finite=False,unit_diagonal=True),Y_mat),M_v)
                            
                            x += self.c_mnFULL(m,n,self.i+1)*temp[self.i]
                    
                    if self.i < len(self.timeGrid)-1:
                        self.hatTheta2[k,self.i+1] = -2*self.delta_t*x
                    
                    del x
            
        elif self.orderParameter == 'quad':
            
            for k in range(self.num_species):
                x    = 0.
                y    = 0.
                z    = 0.
                m    = np.zeros(self.num_species)
                m[k] = 1.
                
                for j in range(self.num_int):
                    x += self.c_mn(j,m,np.zeros(self.num_species),self.i+1)
                    y += self.c_mn(j,m,m,self.i+1)
                    
                if self.i < len(self.timeGrid)-1:
                    self.hatTheta1[k,self.i+1]      = -x
                
                if self.i > 0:
                    self.hatR1[k,self.i+1,self.i+1]   = -y/self.delta_t
                    #self.hatR1[k,self.i+1,self.i]   = -y/self.delta_t
                
                del x,y,z
                
            if self.alphaOrder == 2:
                
                for k in range(self.num_species):
                    x,y,z = 0.,0.,0.
                    m     = np.zeros(self.num_species)
                    m[k]  = 1.
                    mZero = np.zeros(self.num_species)
                    
                    cNN     = np.zeros(self.i+1)
                    M_v     = np.zeros(self.i+1)
                    chi_mat = np.zeros([self.i+1,self.i+1])
                    Y_mat   = np.zeros([self.i+1,self.i+1])
                    M_f     = np.zeros([self.i+1,self.i+1])
                    L       = np.diag(np.ones(self.i),k=-1)
                    #L       = np.diag(np.ones(self.i+1),k=0)
                                        
                    for n in self.n_listFULL[0]:
                        if n not in self.exclusion_list_bubble:
                            
                            for t in range(self.i+1):
                                cNN[t]   = self.c_mnFULL(n,n,t)
                                M_v[t]   = self.c_mnFULL(n,mZero,t)
                                M_f[t,t] = self.c_mnFULL(n,m,t)
                                
                                for t2 in range(self.i+1):
                                    Y_mat[t,t2] = np.prod((sc.special.factorial(np.array(n)))*np.power(self.resp[:,t,t2], np.array(n)))                                  
                            for t in range(self.i+1):
                                chi_mat[t] = cNN*Y_mat[t]

                            #temp = np.matmul(np.matmul(sc.linalg.solve_triangular(a=np.identity(self.i+1)-self.alpha*self.delta_t*np.matmul(chi_mat,L), b=np.identity(self.i+1), lower=True,overwrite_b=True),Y_mat),M_v)
                            
                            temp = np.matmul(sc.linalg.solve_triangular(a=np.identity(self.i+1)-self.alpha*self.delta_t*np.matmul(chi_mat,L), b=np.identity(self.i+1), lower=True,overwrite_b=True,check_finite=False,unit_diagonal=True),Y_mat)
                            
                            x += self.c_mnFULL(m,n,self.i+1)*(np.matmul(temp,M_v)[self.i]) 
                            
                            R2_temp = -2*self.c_mnFULL(m,n,self.i+1)*np.matmul(temp,M_f)
                            
                            for t in range(self.i+1):
                                #self.hatR2[k,self.i+1,t] += -2*self.c_mnFULL(m,n,self.i+1)*(np.matmul(temp,M_f)[self.i,t])
                                self.hatR2[k,self.i+1,t] += R2_temp[self.i,t]
                                
                            del temp
                        
                        elif n in self.exclusion_list_bubble:
                            if n not in [mZero.astype(int).tolist(),m.astype(int).tolist()]:
                                #print(k,n)
                                for t in range(self.i+1):
                                    M_f[t,t] = self.c_mnFULL(n,m,t)
                                    cNN[t] = self.c_mnFULL(n,n,t)

                                    for t2 in range(self.i+1):
                                        Y_mat[t,t2] = np.prod((sc.special.factorial(np.array(n)))*np.power(self.resp[:,t,t2], np.array(n)))
                                for t in range(self.i+1):
                                    chi_mat[t] = cNN*Y_mat[t]

                                temp = np.matmul(sc.linalg.solve_triangular(a=np.identity(self.i+1)-self.alpha*self.delta_t*np.matmul(chi_mat,L), b=np.identity(self.i+1), lower=True,overwrite_b=True,check_finite=False,unit_diagonal=True),Y_mat)
                                
                                R2_temp = -2*self.c_mnFULL(m,n,self.i+1)*np.matmul(temp,M_f)

                                for t in range(self.i+1):
                                    self.hatR2[k,self.i+1,t] += R2_temp[self.i,t]

                                del temp
                                                    
                    if self.i < len(self.timeGrid)-1:
                        self.hatTheta2[k,self.i+1] = -2*self.delta_t*x
                    
                    del x,y,z,m,mZero
                    
                    
    def updateFields_no_C_bubble_sum_old(self):
        """
        Updating the field values for fieldMethod = bubble_sum and C = False. Old slower version! But always use as a reference to compare newer methods!
        
        """

        if self.orderParameter == 'linear':
            
            for k in range(self.num_species):
                x    = 0
                m    = np.zeros(self.num_species)
                m[k] = 1.
                
                for j in range(self.num_int):
                    x += -self.c_mn(j,m,np.zeros(self.num_species),self.i+1)
                    
                if self.i < len(self.timeGrid)-1:
                    self.hatTheta1[k,self.i+1] = x
                
            if self.alphaOrder == 2:                
                for k in range(self.num_species):
                    x    = 0
                    m    = np.zeros(self.num_species)
                    m[k] = 1.
                    mZero= np.zeros(self.num_species)
                    
                    cNN     = np.zeros(self.i+1)
                    M_v     = np.zeros(self.i+1)
                    chi_mat = np.zeros([self.i+1,self.i+1])
                    Y_mat   = np.zeros([self.i+1,self.i+1])
                                        
                    for n in self.n_listFULL[0]:
                        if n not in self.exclusion_list_bubble:
                            
                            for t in range(self.i+1):
                                cNN[t] = self.c_mnFULL(n,n,t)
                                M_v[t] = self.c_mnFULL(n,mZero,t)
                                for t2 in range(self.i+1):
                                    Y_mat[t,t2] = np.prod((sc.special.factorial(np.array(n)))*np.power(self.resp[:,t,t2], np.array(n)))
                                    
                            L = np.diag(np.ones(self.i),k=-1)
                            for t in range(self.i+1):
                                chi_mat[t,:] = cNN[:]*Y_mat[t,:]

                            temp = np.matmul(np.matmul(sc.linalg.solve_triangular(a=np.identity(self.i+1)-self.alpha*self.delta_t*np.matmul(chi_mat,L), b=np.identity(self.i+1), lower=True,overwrite_b=True),Y_mat),M_v)
                            
                            x += self.c_mnFULL(m,n,self.i+1)*temp[self.i]
                    
                    if self.i < len(self.timeGrid)-1:
                        self.hatTheta2[k,self.i+1] = -2*self.delta_t*x
                    
                    del x
            
        elif self.orderParameter == 'quad':
            
            for k in range(self.num_species):
                x    = 0.
                y    = 0.
                z    = 0.
                m    = np.zeros(self.num_species)
                m[k] = 1.
                
                for j in range(self.num_int):
                    x += self.c_mn(j,m,np.zeros(self.num_species),self.i+1)
                    y += self.c_mn(j,m,m,self.i+1)
                    
                if self.i < len(self.timeGrid)-1:
                    self.hatTheta1[k,self.i+1]      = -x
                
                if self.i > 0:
                    self.hatR1[k,self.i+1,self.i+1]   = -y/self.delta_t
                
                del x,y,z
                
            if self.alphaOrder == 2:
                
                for k in range(self.num_species):
                    x,y,z = 0.,0.,0.
                    m     = np.zeros(self.num_species)
                    m[k]  = 1.
                    mZero = np.zeros(self.num_species)
                    
                    cNN     = np.zeros(self.i+1)
                    M_v     = np.zeros(self.i+1)
                    chi_mat = np.zeros([self.i+1,self.i+1])
                    Y_mat   = np.zeros([self.i+1,self.i+1])
                    M_f     = np.zeros([self.i+1,self.i+1])
                                        
                    for n in self.n_listFULL[0]:
                        if n not in self.exclusion_list_bubble:
                            
                            for t in range(self.i+1):
                                cNN[t] = self.c_mnFULL(n,n,t)
                                M_v[t] = self.c_mnFULL(n,mZero,t)
                                for t2 in range(self.i+1):
                                    Y_mat[t,t2] = np.prod((sc.special.factorial(np.array(n)))*np.power(self.resp[:,t,t2], np.array(n)))
                                    
                            L = np.diag(np.ones(self.i),k=-1)
                            for t in range(self.i+1):
                                chi_mat[t,:] = cNN[:]*Y_mat[t,:]

                            temp = np.matmul(np.matmul(sc.linalg.solve_triangular(a=np.identity(self.i+1)-self.alpha*self.delta_t*np.matmul(chi_mat,L), b=np.identity(self.i+1), lower=True,overwrite_b=True),Y_mat),M_v)
                            
                            x += self.c_mnFULL(m,n,self.i+1)*temp[self.i] 
                                
                        if n not in [mZero.astype(int).tolist(),m.astype(int).tolist()]:
                        #if n not in self.exclusion_list_bubble:
                            for t in range(self.i+1):
                                M_f[t,t] = self.c_mnFULL(n,m,t)
                                cNN[t] = self.c_mnFULL(n,n,t)
                                for t2 in range(self.i+1):
                                    Y_mat[t,t2] = np.prod((sc.special.factorial(np.array(n)))*np.power(self.resp[:,t,t2], np.array(n)))
                                
                            L = np.diag(np.ones(self.i),k=-1)
                            for t in range(self.i+1):
                                chi_mat[t,:] = cNN[:]*Y_mat[t,:]
                            
                            temp = np.matmul(np.matmul(sc.linalg.solve_triangular(a=np.identity(self.i+1)-self.alpha*self.delta_t*np.matmul(chi_mat,L), b=np.identity(self.i+1), lower=True,overwrite_b=True),Y_mat),M_f)
                                                        
                            for t in range(self.i+1):
                                self.hatR2[k,self.i+1,t] += -2*self.c_mnFULL(m,n,self.i+1)*temp[self.i,t]
                                                    
                    if self.i < len(self.timeGrid)-1:
                        self.hatTheta2[k,self.i+1] = -2*self.delta_t*x
                    
                    del x,y,z,m,mZero
    
    
    def updateFields_beta_no_C(self):
        
        """
        Updating the field values for fieldMethod = 'split' and C = False. New version, does not calculate c_mn at every time point, but only once at the beginning and then looks up the value!
        
        """
        try:
            self.c_mn_dict
        except:
            self.create_c_mn_dict()
                
        if self.orderParameter == 'linear':
            
            for k in range(self.num_species):
                x    = 0
                m    = np.zeros(self.num_species)
                m[k] = 1.
                
                for j in range(self.num_int):
                    x += -self.c_mn(j,m,np.zeros(self.num_species),self.i+1)
                
                if self.i < len(self.timeGrid)-1:
                    self.hatTheta1[k,self.i+1] = x
                
            if self.alphaOrder == 2:                
                for k in range(self.num_species):
                    x    = 0
                    y    = 0
                    m    = np.zeros(self.num_species,dtype=int)
                    m[k] = 1.
                    
                    for n in self.n_listFULL[0]:
                        if n not in self.exclusion_list:
                            for j in range(self.num_int):
                                for l in range(self.num_int):
                                    for time in np.arange(1,self.i+1):
                                        try:
                                            x += self.c_mn_dict[(j,tuple(m),tuple(n))]*self.c_mn_dict[(l,tuple(n),tuple(np.zeros(self.num_species)))]*np.prod(np.power(self.y[:,self.i],(self.r_i[j]-n))*np.power(self.y[:,time-1],(self.r_i[l]-np.zeros(self.num_species)))*(sc.special.factorial(np.array(n)))*np.power(self.resp[:,self.i,time],np.array(n)))
                                        except:
                                            x += 0                           
                                            
                    #for j in range(self.num_int):
                        #for l in range(self.num_int):
                            
                            #n_list_intersection = [elm for elm in self.n_list[j] if elm in self.n_list[l] if elm not in self.exclusion_list]
                            #for n in n_list_intersection:
                                #for time in np.arange(1,self.i+1):
                                    #x+= self.c_mn_dict[(j,tuple(m),tuple(n))]*self.c_mn_dict[(l,tuple(n),tuple(np.zeros(self.num_species)))]*np.prod(np.power(self.y[:,self.i],(self.r_i[j]-n))*np.power(self.y[:,time-1],(self.r_i[l]-np.zeros(self.num_species)))*(sc.special.factorial(np.array(n)))*np.power(self.resp[:,self.i,time],np.array(n)))
                                                        
                    if self.i < len(self.timeGrid)-1:
                        self.hatTheta2[k,self.i+1] = -2*self.delta_t*x
                    
                    del x
            
        elif self.orderParameter == 'quad':
            
            for k in range(self.num_species):
                x,y,z= 0.,0.,0.
                m    = np.zeros(self.num_species)
                m[k] = 1.
                
                for j in range(self.num_int):
                    x += self.c_mn(j,m,np.zeros(self.num_species),self.i+1)
                    y += self.c_mn(j,m,m,self.i+1)
                    #z += self.c_mn(j,2*m,np.zeros(self.num_species),self.i+1)
                
                if self.i < len(self.timeGrid)-1:
                    self.hatTheta1[k,self.i+1]      = -x
                
                if self.i > 0:
                    self.hatR1[k,self.i+1,self.i]   = -y/self.delta_t
                    #self.hatB1[k,self.i+1,self.i+1] = -z/self.delta_t

                del x,y
                
            if self.alphaOrder == 2:
                
                for k in range(self.num_species):
                    x,y,z = 0.,0.,0.
                    m     = np.zeros(self.num_species)
                    m[k]  = 1.
                    mZero = np.zeros(self.num_species)
                    
                    for n in self.n_listFULL[0]:
                        for j in range(self.num_int):
                            for l in range(self.num_int):
                                for time in np.arange(1,self.i+1):
                                    if n not in self.exclusion_list:
                                        #print(j,l,n)
                                        try:
                                            x += self.c_mn_dict[(j,tuple(m),tuple(n))]*self.c_mn_dict[(l,tuple(n),tuple(np.zeros(self.num_species)))]*np.prod(np.power(self.y[:,self.i],(self.r_i[j]-n))*np.power(self.y[:,time-1],(self.r_i[l]-np.zeros(self.num_species)))*(sc.special.factorial(np.array(n)))*np.power(self.resp[:,self.i,time],np.array(n)))
                                            #x += self.c_mn_dict[(j,tuple(m),tuple(n))]* self.c_mn_dict[(l,tuple(n),tuple(mZero))]*np.prod(np.power(self.y[:,self.i],(self.r_i[j]-n))*np.power(self.y[:,time-1],(self.r_i[l]-mZero))*(sc.special.factorial(np.array(n)))*np.power(self.resp[:,self.i,time],np.array(n)))
                                            #print(x)
                                        except:
                                            x += 0
                                            
                                    if n not in [mZero.astype(int).tolist(),m.astype(int).tolist()]:
                                        try:                                    
                                            self.hatR2[k,self.i+1,time-1] += -2*self.c_mn_dict[(j,tuple(m),tuple(n))]* self.c_mn_dict[(l,tuple(n),tuple(m))]*np.prod(np.power(self.y[:,self.i],(self.r_i[j]-n))*np.power(self.y[:,time-1],(self.r_i[l]-m))*(sc.special.factorial(np.array(n)))*np.power(self.resp[:,self.i,time],np.array(n)))
                                        except:
                                            self.hatR2[k,self.i+1,time-1] += 0
                    
                    #for j in range(self.num_int):
                        #for l in range(self.num_int):
                            
                            #n_list_intersection1 = [elm for elm in self.n_list[j] if elm in self.n_list[l] if elm not in self.exclusion_list]
                            #for n in n_list_intersection1:
                                #for time in np.arange(1,self.i+1):
                                    #x += self.c_mn_dict[(j,tuple(m),tuple(n))]*self.c_mn_dict[(l,tuple(n),tuple(mZero))]*np.prod(np.power(self.y[:,self.i],(self.r_i[j]-n))*np.power(self.y[:,time-1],(self.r_i[l]-mZero))*sc.special.factorial(np.array(n)))*np.power(self.resp[:,self.i,time],np.array(n))
                                    
                            #n_list_intersection2 = [elm for elm in self.n_list[j] if elm in self.n_list[l] if elm not in [mZero.astype(int).tolist(),m.astype(int).tolist()]]
                            
                            #for n in n_list_intersection2:
                                #for time in range(1,self.i+2):
                                    
                                    #self.hatR2[k,self.i+1,t-1] += -2*self.c_mn_dict[(j,tuple(m),tuple(n))]* self.c_mn_dict[(l,tuple(n),tuple(m))]*np.prod(np.power(self.y[:,self.i],(self.r_i[j]-n))*np.power(self.y[:,time],(self.r_i[l]-m))*(sc.special.factorial(np.array(n)))*np.power(self.resp[:,self.i,time],np.array(n)))
                                    
                                    #1#Write the expression
                            
                                    #for t in range(0,self.i+2):
                                        #x += self.c_mn(l,m,n,self.i+1)*self.c_mn(j,n,mZero,t) *np.prod((sc.special.factorial(np.array(n)))*np.power(self.resp[:,self.i,t],np.array(n)))                 
                                        
                                #if n not in [mZero.astype(int).tolist(),m.astype(int).tolist()]:
                                    #for t in range(1,self.i+2):
                                        
                                        #self.hatR2[k,self.i+1,t-1] += -2*self.c_mn(j,m,n,self.i+1)* self.c_mn(l,n,m,t)*np.prod((sc.special.factorial(np.array(n)))*np.power(self.resp[:,self.i,t],np.array(n)))

                    if self.i < len(self.timeGrid)-1:
                        self.hatTheta2[k,self.i+1]   = -2*self.delta_t*x

                    del x,y,z,m,mZero
                    
                    
    def updateFields_beta_no_C_old(self):
        
        """
        Updating the field values for fieldMethod = 'split' and C = False. Old version, may be slightly slower. With extra hat(B) calculations as well.
        
        """
        
        if self.orderParameter == 'linear':
            
            for k in range(self.num_species):
                x    = 0
                m    = np.zeros(self.num_species)
                m[k] = 1.
                
                for j in range(self.num_int):
                    x += -self.c_mn(j,m,np.zeros(self.num_species),self.i+1)
                
                if self.i < len(self.timeGrid)-1:
                    self.hatTheta1[k,self.i+1] = x
                
            if self.alphaOrder == 2:                
                for k in range(self.num_species):
                    x    = 0
                    m    = np.zeros(self.num_species)
                    m[k] = 1.
                    
                    for j in range(self.num_int):
                        for l in range(self.num_int):
                            
                            for n in self.n_listFULL[0]:
                                if n not in self.exclusion_list:
                                    
                                    for time in np.arange(1,self.i+1):
                                        x += self.c_mn(j,m,n,self.i+1)*self.c_mn(l,n,np.zeros(self.num_species),time)*np.prod((sc.special.factorial(np.array(n)))*np.power(self.resp[:,self.i,time],np.array(n)))
                                
                    x *= -2*self.delta_t
                    
                    if self.i < len(self.timeGrid)-1:
                        self.hatTheta2[k,self.i+1] = x
                    
                    del x
            
        elif self.orderParameter == 'quad':
            
            for k in range(self.num_species):
                x,y,z= 0.,0.,0.
                m    = np.zeros(self.num_species)
                m[k] = 1.
                
                for j in range(self.num_int):
                    x += self.c_mn(j,m,np.zeros(self.num_species),self.i+1)
                    y += self.c_mn(j,m,m,self.i+1)
                    z += self.c_mn(j,2*m,np.zeros(self.num_species),self.i+1)
                
                if self.i < len(self.timeGrid)-1:
                    self.hatTheta1[k,self.i+1]          = -x
                
                if self.i > 0:
                    self.hatR1[k,self.i+1,self.i]   = -y/self.delta_t
                    self.hatB1[k,self.i+1,self.i+1] = -z/self.delta_t

                del x,y,z
                
            if self.alphaOrder == 2:
                
                for k in range(self.num_species):
                    x,y,z = 0.,0.,0.
                    m     = np.zeros(self.num_species)
                    m[k]  = 1.
                    mZero = np.zeros(self.num_species)
                    
                    for j in range(self.num_int):
                        for l in range(self.num_int):
                            
                            for n in self.n_listFULL[0]: 
                            
                                if n not in self.exclusion_list:
                                    for t in range(0,self.i+2):
                                        
                                        x += self.c_mn(l,m,n,self.i+1)*self.c_mn(j,n,mZero,t) *np.prod((sc.special.factorial(np.array(n)))*np.power(self.resp[:,self.i,t],np.array(n)))                 
                                        
                                if n not in [mZero.astype(int).tolist(),m.astype(int).tolist()]:
                                    for t in range(1,self.i+2):
                                        
                                        self.hatR2[k,self.i+1,t-1] += -2*self.c_mn(j,m,n,self.i+1)* self.c_mn(l,n,m,t)*np.prod((sc.special.factorial(np.array(n)))*np.power(self.resp[:,self.i,t],np.array(n)))

                                        #if n not in [(3*m.astype(int)).tolist()]:
                                            
                                        #    y += n[k]*self.c_mn(j,m,n,self.i)*self.c_mn(l,n-m,mZero,t) *sc.special.factorial(n[k]-1)*np.power(self.resp[k,self.i-1,t],n[k]-1) *np.prod((sc.special.factorial(np.array(n))*np.power(self.resp[:,self.i-1,t],np.array(n)))[np.arange(self.num_species)!=k])

                                if n not in [mZero.astype(int).tolist(),m.astype(int).tolist()]:
                                    for t in range(0,self.i+1):
                                        
                                        if (n+m).astype(int).tolist() in self.m_listFULL[0]:
                                        
                                            self.hatB2[k,self.i+1,t+1] += -2*(n[k]+1)*self.c_mn(j,m,n,self.i+1)*self.c_mn(l,n+m,mZero,t+1)*np.prod((sc.special.factorial(np.array(n))) *np.power(self.resp[:,self.i,t+1],np.array(n))) 
                                            
                                            z += self.c_mn(j,2*m,n,self.i+1)*self.c_mn(l,n,mZero,t+1) *np.prod((sc.special.factorial(np.array(n)))*np.power(self.resp[:,self.i,t+1],np.array(n)))

                                            self.hatB2[k,t+1,self.i+1] = self.hatB2[k,self.i+1,t+1]

                    if self.i < len(self.timeGrid)-1:
                        self.hatTheta2[k,self.i+1]   = -2*self.delta_t*x
                    
                    self.hatR2[k,self.i+1,self.i]   += -2*y
                    self.hatB2[k,self.i+1,self.i+1] += -2*z

                    del x,y,z,m,mZero
        
        
    def ABCupdateFields(self):
        
        """
        Updating the field values for fieldMethod = 'split_ABC' with C = True.
        
        """
        
        if self.orderParameter == 'linear':
            
            if self.i < len(self.timeGrid)-1:
            
                self.hatTheta1[0,self.i+1] = self.k3[0]*self.y[0,self.i]*self.y[1,self.i]
                self.hatTheta1[1,self.i+1] = self.k3[0]*self.y[0,self.i]*self.y[1,self.i]
                self.hatTheta1[2,self.i+1] = -self.k3[0]*self.y[0,self.i]*self.y[1,self.i]

                x = 0
                if self.alphaOrder == 2:
                    for t in range(1,self.i+1):
                        x  += -2*self.delta_t*self.k3[0]**2*\
                        (self.y[0,t-1]*self.y[1,t-1]*self.resp[0,self.i,t]*self.resp[1,self.i,t])

                    self.hatTheta2[0,self.i+1] = x
                    self.hatTheta2[1,self.i+1] = x
                    self.hatTheta2[2,self.i+1] = -x
        
        # The following is the Plefka LinearOP Mod which should be accurate to O(\alpha^2):
        
        #if self.orderParameter == 'linear':
        #    
        #    if self.i < len(self.timeGrid)-1:
        #    
        #        self.hatTheta1[0,self.i+1] = self.k3[0]*self.y[0,self.i]*self.y[1,self.i]
        #        self.hatTheta1[1,self.i+1] = self.k3[0]*self.y[0,self.i]*self.y[1,self.i]
        #        self.hatTheta1[2,self.i+1] = -self.k3[0]*self.y[0,self.i]*self.y[1,self.i]

         #       if self.alphaOrder == 2:
         #           x,y,z = 0,0,0
         #
         #           for t in range(1,self.i+1):

          #              x  += -2*self.delta_t*self.k3[0]**2*(self.y[0,t-1]*self.y[1,t-1]*self.resp[0,self.i,t]*self.resp[1,self.i,t])

           #             y += -2*self.delta_t**2*self.k3[0]**2*self.y[1,t-1]*(self.y[0,0]*self.resp[0,t,0]**2 + np.sum(((self.k1[0] + self.k2[0]*self.y[0,:])*self.resp[0,t,:]**2)[0:t]))*self.resp[0,self.i,t]*self.resp[1,self.i,t]

            #            z += -2*self.delta_t**2*self.k3[0]**2*self.y[0,t-1]*(self.y[1,0]*self.resp[1,t,0]**2 + np.sum(((self.k1[1] + self.k2[1]*self.y[1,:])*self.resp[1,t,:]**2)[0:t]))*self.resp[0,self.i,t]*self.resp[1,self.i,t]


             #       self.hatTheta2[0,self.i+1] = -x + y + z 
             #       self.hatTheta2[1,self.i+1] = -x + y + z
             #       self.hatTheta2[2,self.i+1] = x - y - z
            
        elif self.orderParameter == 'quad':
            
            if self.i < len(self.timeGrid)-1:
            
                self.hatTheta1[0,self.i+1] = self.k3[0]*self.y[0,self.i]*self.y[1,self.i]
                self.hatTheta1[1,self.i+1] = self.k3[0]*self.y[0,self.i]*self.y[1,self.i]
                self.hatTheta1[2,self.i+1] = -self.k3[0]*self.y[0,self.i]*self.y[1,self.i]

                self.hatB1[:,:,:] = 0.

                self.hatR1[0,self.i+1,self.i] = self.k3[0]*self.y[1,self.i]/self.delta_t
                self.hatR1[1,self.i+1,self.i] = self.k3[0]*self.y[0,self.i]/self.delta_t
                self.hatR1[2,self.i+1,self.i] = 0.
                
            if self.alphaOrder == 2:
                
                RA = self.resp[0]
                RB = self.resp[1]
                CA = self.corr[0]
                CB = self.corr[1]
                
                if self.i < len(self.timeGrid)-1:
                
                    x = 0
                    for t in range(1,self.i+2):

                        x  += -2*self.delta_t*self.k3[0]**2*(self.y[0,t-1]*self.y[1,t-1]*RA[self.i,t]*RB[self.i,t] + self.y[0,t-1]*RA[self.i,t]*CB[self.i,t-1] + self.y[1,t-1]*RB[self.i,t]*CA[self.i,t-1])

                        self.hatR2[0,self.i+1,t-1] = -2*self.k3[0]**2*(self.y[0,self.i]*self.y[1,t-1]*RB[self.i,t] + self.y[1,t-1]*RA[self.i,t]*RB[self.i,t] + RA[self.i,t]*CB[self.i,t-1])

                        self.hatR2[1,self.i+1,t-1] = -2*self.k3[0]**2*(self.y[1,self.i]*self.y[0,t-1]*RA[self.i,t] + self.y[0,t-1]*RA[self.i,t]*RB[self.i,t] + RB[self.i,t]*CA[self.i,t-1])

                        self.hatR2[2,self.i+1,t-1] = 0.

                    for t in range(0,self.i+1):

                        self.hatB2[0,self.i+1,t+1] = -2*self.k3[0]**2*(2*self.y[0,self.i]*self.y[0,t]*CB[self.i,t] + 2*CA[self.i,t]*CB[self.i,t] + self.y[0,self.i]*self.y[0,t]*self.y[1,t]*RB[self.i,t+1] + self.y[1,t]*CA[self.i,t]*RB[self.i,t+1])

                        self.hatB2[1,self.i+1,t+1] = -2*self.k3[0]**2*(2*self.y[1,self.i]*self.y[1,t]*CA[self.i,t] + 2*CB[self.i,t]*CA[self.i,t] + self.y[1,self.i]*self.y[1,t]*self.y[0,t]*RA[self.i,t+1] + self.y[0,t]*CB[self.i,t]*RA[self.i,t+1])

                        self.hatB2[2,self.i+1,t+1] = -2*self.k3[0]**2*(2*self.y[1,self.i]*self.y[1,t]*CA[self.i,t] + 2*self.y[0,self.i]*self.y[0,t]*CB[self.i,t] + 2*CA[self.i,t]*CB[self.i,t])

                        self.hatB2[:,t+1,self.i+1] = self.hatB2[:,self.i+1,t+1]

                    self.hatTheta2[0,self.i+1] = x
                    self.hatTheta2[1,self.i+1] = x
                    self.hatTheta2[2,self.i+1] = -x
                    
    def MMPupdateFields(self):
        
        """
        Updating the field values for fieldMethod = 'split_MMP' with C = True.
        
        """
        
        if self.orderParameter == 'linear':
            
            if self.i < len(self.timeGrid)-1:

                self.hatTheta1[1,self.i+1] = -self.k3[0]*self.y[0,self.i]
            
        elif self.orderParameter == 'quad':
            
            if self.i < len(self.timeGrid)-1:
            
                self.hatTheta1[1,self.i+1] = -self.k3[0]*self.y[0,self.i]
                
            if self.alphaOrder == 2:
                
                RM = self.resp[0]
                RP = self.resp[1]
                CM = self.corr[0]
                CP = self.corr[1]
                
                if self.i < len(self.timeGrid)-1:
                    for t in range(0,self.i+1):
                        
                        self.hatB2[1,self.i+1,t+1] = -2*self.k3[0]**2*(RM[self.i,t+1]*self.y[0,t])
                        self.hatB2[:,t+1,self.i+1] = self.hatB2[:,self.i+1,t+1]
                
    
    def ABCupdateFields_no_C(self):
        
        """
        Updating the field values for fieldMethod = 'split_ABC' with C = False.
        
        """
        
        if self.orderParameter == 'linear':
            
            if self.i < len(self.timeGrid)-1:
            
                self.hatTheta1[0,self.i+1] = self.k3[0]*self.y[0,self.i]*self.y[1,self.i]
                self.hatTheta1[1,self.i+1] = self.k3[0]*self.y[0,self.i]*self.y[1,self.i]
                self.hatTheta1[2,self.i+1] = -self.k3[0]*self.y[0,self.i]*self.y[1,self.i]

                x = 0
                if self.alphaOrder == 2:
                    for t in range(1,self.i+1):
                        x  += -2*self.delta_t*self.k3[0]**2*\
                        (self.y[0,t-1]*self.y[1,t-1]*self.resp[0,self.i,t]*self.resp[1,self.i,t])

                    self.hatTheta2[0,self.i+1] = x
                    self.hatTheta2[1,self.i+1] = x
                    self.hatTheta2[2,self.i+1] = -x
            
        elif self.orderParameter == 'quad':
            
            if self.i < len(self.timeGrid)-1:
            
                self.hatTheta1[0,self.i+1] = self.k3[0]*self.y[0,self.i]*self.y[1,self.i]
                self.hatTheta1[1,self.i+1] = self.k3[0]*self.y[0,self.i]*self.y[1,self.i]
                self.hatTheta1[2,self.i+1] = -self.k3[0]*self.y[0,self.i]*self.y[1,self.i]

                self.hatB1[:,:,:] = 0.

                self.hatR1[0,self.i+1,self.i] = self.k3[0]*self.y[1,self.i]/self.delta_t
                self.hatR1[1,self.i+1,self.i] = self.k3[0]*self.y[0,self.i]/self.delta_t
                self.hatR1[2,self.i+1,self.i] = 0.

                if self.alphaOrder == 2:

                    RA = self.resp[0]
                    RB = self.resp[1]
                    
                    if self.i < len(self.timeGrid)-1:
                    
                        x = 0
                        for t in range(1,self.i+2):

                            x  += -2*self.delta_t*self.k3[0]**2*(self.y[0,t-1]*self.y[1,t-1]*RA[self.i,t]*RB[self.i,t])

                            self.hatR2[0,self.i+1,t-1] = -2*self.k3[0]**2*(self.y[0,self.i]*self.y[1,t-1]*RB[self.i,t] + self.y[1,t-1]*RA[self.i,t]*RB[self.i,t])

                            self.hatR2[1,self.i+1,t-1] = -2*self.k3[0]**2*(self.y[1,self.i]*self.y[0,t-1]*RA[self.i,t] + self.y[0,t-1]*RA[self.i,t]*RB[self.i,t])

                            self.hatR2[2,self.i+1,t-1] = 0.

                        for t in range(0,self.i+1):

                            self.hatB2[0,self.i+1,t+1] = -2*self.k3[0]**2*(self.y[0,self.i]*self.y[0,t]*self.y[1,t]*RB[self.i,t+1])

                            self.hatB2[1,self.i+1,t+1] = -2*self.k3[0]**2*(self.y[1,self.i]*self.y[1,t]*self.y[0,t]*RA[self.i,t+1])

                            self.hatB2[2,self.i+1,t+1] = 0

                            self.hatB2[:,t+1,self.i+1] = self.hatB2[:,self.i+1,t+1]
                            
                        for t in range(1,self.i+2):
                            self.hatR2[0,self.i+1,t-1] = self.hatR2[0,self.i+1,t-1]*np.exp(-0.5*(self.i-t+1)**2*self.GaussianRegularizer**2)
                            
                            self.hatR2[1,self.i+1,t-1] = self.hatR2[1,self.i+1,t-1]*np.exp(-0.5*(self.i-t+1)**2*self.GaussianRegularizer**2)

                        self.hatTheta2[0,self.i+1] = x
                        self.hatTheta2[1,self.i+1] = x
                        self.hatTheta2[2,self.i+1] = -x
                        
    def AA_AupdateFields_no_C(self):
        
        """
        Updating the field values for fieldMethod = 'split_AA_A' with C = False.
        
        """
        
        if self.orderParameter == 'linear':
            
            if self.i < len(self.timeGrid)-1:
                #Added a new term here to take care of one molecule number less
            
                self.hatTheta1[0,self.i+1] = self.k3[0]*(self.y[0,self.i]**2) #- self.k3[0]*self.y[0,self.i]

                x = 0
                if self.alphaOrder == 2:
                    for t in range(1,self.i+1):
                        x  += -4*self.delta_t*self.k3[0]**2*self.y[0,t-1]**2*self.resp[0,self.i,t]**2
                    self.hatTheta2[0,self.i+1] = x
            
        elif self.orderParameter == 'quad':
            
            if self.i < len(self.timeGrid)-1:
            
                self.hatTheta1[0,self.i+1] = self.k3[0]*self.y[0,self.i]**2
                self.hatB1[:,:,:] = 0.
                self.hatR1[0,self.i+1,self.i] = 2*self.k3[0]*self.y[0,self.i]/self.delta_t

                if self.alphaOrder == 2:
                    RA = self.resp[0]
                    
                    if self.i < len(self.timeGrid)-1:
                        x = 0
                        for t in range(1,self.i+2):
                            x  += -4*self.delta_t*self.k3[0]**2*(self.y[0,t-1]**2*RA[self.i,t]**2)
                            
                            # An attempt with the disconnected Corrleation function
                            #x  += -4*self.delta_t*self.k3[0]**2*(self.y[0,t-1]**2*RA[self.i,t]**2) - 4*self.delta_t*self.k3[0]**2*(RA[self.i,t]*self.y[0,self.i]*self.y[0,t-1]**2)
                            
                            self.hatR2[0,self.i+1,t-1] = -8*self.k3[0]**2*(self.y[0,t-1]*RA[self.i,t]**2)

                        for t in range(0,self.i+1):

                            #self.hatB2[0,self.i+1,t+1] = -2*self.k3[0]**2*(self.y[0,self.i]*self.y[0,t]*self.y[1,t]*RB[self.i,t+1])
                            self.hatB2[0,self.i+1,t+1] = 0
                            self.hatB2[:,t+1,self.i+1] = self.hatB2[:,self.i+1,t+1]
                            
                        for t in range(1,self.i+2):
                            self.hatR2[0,self.i+1,t-1] = self.hatR2[0,self.i+1,t-1]*np.exp(-0.5*(self.i-t+1)**2*self.GaussianRegularizer**2)

                        self.hatTheta2[0,self.i+1] = x
                        
                                                                     
    def AA_AupdateFields_no_C_all_corrections(self):
        
        """
        Updating the field values for fieldMethod = 'split_AA_A' with C = False and including all the corrections by summing the Geometric series. This is possible only for Quadratic Order Paramters.
        
        """
            
        if self.orderParameter == 'quad':
            
            if self.i < len(self.timeGrid)-1:
            
                self.hatTheta1[0,self.i+1] = self.k3[0]*self.y[0,self.i]**2
                self.hatB1[:,:,:] = 0.
                self.hatR1[0,self.i+1,self.i] = self.k3[0]*self.y[0,self.i]/self.delta_t

                if self.alphaOrder == 2:
                    RA = self.resp[0]
                    #Lower Shift_matrix
                    size=self.i+1
                    L = np.diag(np.ones(size-1),k=-1)
                    
                    if self.i < len(self.timeGrid)-1:
                        x = 0
                        
                        #Using scipy solve triangular system to calculate the inverse with Shift Matrix
                        
                        temp_mat = np.matmul(RA[:size,:size]**2,sc.linalg.solve_triangular(a=np.identity(size)+ 2*self.alpha*self.k3[0]*self.delta_t*np.matmul(L,RA[:size,:size]**2),b=np.identity(size),lower=True,overwrite_b=True))
                                                
                        #Using scipy solve triangular system to calculate the inverse
                        #temp_mat = np.matmul(RA[:self.i+1,:self.i+1]**2,sc.linalg.solve_triangular(a=np.identity(self.i+1)+ 2*self.alpha*self.k3[0]*self.delta_t*RA[:self.i+1,:self.i+1]**2,b=np.identity(self.i+1),lower=True,overwrite_b=True))             
                        
                        # Vanilla inversion                        
                        #temp_mat = np.matmul(RA[:self.i+1,:self.i+1]**2,np.linalg.inv(np.identity(self.i+1)+ 2*self.alpha*self.k3[0]*self.delta_t*np.matmul(L,RA[:size,:size]**2)))
                        
                        #for t in range(1,self.i+1): #Made a change here, self.i+2 to self.i + 1
                        #    x  += -4*self.delta_t*self.k3[0]**2*(self.y[0,t-1]**2*temp_mat[self.i,t])
                            
                        #    self.hatR2[0,self.i+1,t] = -8*self.k3[0]**2*(self.y[0,t-1]*temp_mat[self.i,t])
                        
                        for t in range(1,self.i+1): #Made a change here, self.i+2 to self.i + 1
                            x  += -4*self.delta_t*self.k3[0]**2*(self.y[0,t-1]**2*temp_mat[self.i,t])
                            
                            #for tt in range(1,self.i+1):
                            #    x += 8*self.alpha*self.delta_t**2*self.k3[0]**3*(RA[self.i,t]*self.y[0,t-1]**2*RA[self.i,tt]*self.y[0,tt-1]*RA[tt-1,t])
                            
                            self.hatR2[0,self.i+1,t-1] = -4*self.k3[0]**2*(self.y[0,t-1]*temp_mat[self.i,t])
                        
                        
                        self.hatTheta2[0,self.i+1] = x
                        
                        
    def AA_AupdateFields_no_C_power_corrections(self):
        
        """
        Updating the field values for fieldMethod = 'split_AA_A' with C = False and including all the corrections by summing the Geometric series. This is possible only for Quadratic Order Paramters.
        
        """
            
        if self.orderParameter == 'quad':
            
            if self.i < len(self.timeGrid)-1:
            
                self.hatTheta1[0,self.i+1] = self.k3[0]*self.y[0,self.i]**2
                self.hatB1[:,:,:] = 0.
                self.hatR1[0,self.i+1,self.i] = 2*self.k3[0]*self.y[0,self.i]/self.delta_t

                if self.alphaOrder == 2:
                    RA = self.resp[0]
                    
                    if self.i < len(self.timeGrid)-1 and self.i > 0:
                        x = 0
                        size=self.i+1
                        L = np.diag(np.ones(size-1),k=-1)
                        
                        #Using scipy solve triangular system to calculate the inverse
                        temp_mat = np.matmul(RA[:self.i+1,:self.i+1]**2, np.matmul(np.identity(self.i+1)-np.linalg.matrix_power(-2*self.alpha*self.k3[0]*self.delta_t*np.matmul(L,RA[:self.i+1,:self.i+1]**2), self.i-1) ,sc.linalg.solve_triangular(a=np.identity(self.i+1) + 2*self.alpha*self.k3[0]*self.delta_t* np.matmul(L,RA[:self.i+1,:self.i+1]**2),b=np.identity(self.i+1),lower=True,overwrite_b=True)))
                                                                       
                        for t in range(1,self.i+1): #Made a change here, self.i+2 to self.i + 1
                            x  += -4*self.delta_t*self.k3[0]**2*(self.y[0,t-1]**2*temp_mat[self.i,t])
                            
                            self.hatR2[0,self.i+1,t-1] = -8*self.k3[0]**2*(self.y[0,t-1]*temp_mat[self.i,t])

                        self.hatTheta2[0,self.i+1] = x
                                            
    def AA_AupdateFields_no_C_finite_power(self):
        
        """
        Updating the field values for fieldMethod = 'split_AA_A' with C = False and including all the corrections by summing the Geometric series. This is possible only for Quadratic Order Paramters.
        
        """
        
        if self.orderParameter == 'linear':
            
            if self.i < len(self.timeGrid)-1:
            
                self.hatTheta1[0,self.i+1] = self.k3[0]*self.y[0,self.i]**2
                
                if self.alphaOrder == 2:
                    RA = self.resp[0]
                    
                    if self.i < len(self.timeGrid)-1 and self.i > 0:
                        x = 0
                        size=self.i+1
                        L = np.diag(np.ones(size-1),k=-1)
                        
                        power = 3 
                        
                        #Using scipy solve triangular system to calculate the inverse
                        temp_mat = np.matmul(RA[:self.i+1,:self.i+1]**2, np.matmul(np.identity(self.i+1)-np.linalg.matrix_power(-2*self.alpha*self.k3[0]*self.delta_t*np.matmul(L,RA[:self.i+1,:self.i+1]**2), power) ,sc.linalg.solve_triangular(a=np.identity(self.i+1) + 2*self.alpha*self.k3[0]*self.delta_t* np.matmul(L,RA[:self.i+1,:self.i+1]**2),b=np.identity(self.i+1),lower=True,overwrite_b=True)))
                                                
                        for t in range(1,self.i+1): #Made a change here, self.i+2 to self.i + 1
                            x  += -4*self.delta_t*self.k3[0]**2*(self.y[0,t-1]**2*temp_mat[self.i,t])
                            
                        self.hatTheta2[0,self.i+1] = x
            
        if self.orderParameter == 'quad':
            
            if self.i < len(self.timeGrid)-1:
            
                self.hatTheta1[0,self.i+1] = self.k3[0]*self.y[0,self.i]**2
                self.hatB1[:,:,:] = 0.
                self.hatR1[0,self.i+1,self.i] = 2*self.k3[0]*self.y[0,self.i]/self.delta_t

                if self.alphaOrder == 2:
                    RA = self.resp[0]
                    
                    if self.i < len(self.timeGrid)-1 and self.i > 0:
                        x = 0
                        size=self.i+1
                        L = np.diag(np.ones(size-1),k=-1)
                        
                        power = 3 ## Put the power you want to raise it to here (the number of terms in the expansion that are kept)
                        
                        #Using scipy solve triangular system to calculate the inverse
                        temp_mat = np.matmul(RA[:self.i+1,:self.i+1]**2, np.matmul(np.identity(self.i+1)-np.linalg.matrix_power(-2*self.alpha*self.k3[0]*self.delta_t*np.matmul(L,RA[:self.i+1,:self.i+1]**2), power) ,sc.linalg.solve_triangular(a=np.identity(self.i+1) + 2*self.alpha*self.k3[0]*self.delta_t* np.matmul(L,RA[:self.i+1,:self.i+1]**2),b=np.identity(self.i+1),lower=True,overwrite_b=True)))
                                                
                        for t in range(1,self.i+1): #Made a change here, self.i+2 to self.i + 1
                            x  += -4*self.delta_t*self.k3[0]**2*(self.y[0,t-1]**2*temp_mat[self.i,t])
                            
                            self.hatR2[0,self.i+1,t-1] = -8*self.k3[0]**2*(self.y[0,t-1]*temp_mat[self.i,t])

                        self.hatTheta2[0,self.i+1] = x
                                            
                  
    def runDynamics(self,Print=None):
        
        """
        Runs the dynamics after things have been properly initialized. Could optionally Print 'time' or the current step 'i'.
        
        """
        
        try:
            self.MAK
        except AttributeError:
            self.MAK = False
        
        with tqdm(total=len(self.timeGrid)-2) as pbar:
            
            while(self.i < len(self.timeGrid)-1):
                
                if not self.MAK:
                    if self.C:
                        if self.fieldMethod == None:
                            self.updateFields()
                        elif self.fieldMethod == 'split':
                            self.updateFields_beta()
                        elif self.fieldMethod == 'split_ABC':
                            self.ABCupdateFields()
                        elif self.fieldMethod == 'split_MMP':
                            self.MMPupdateFields()
                        elif self.fieldMethod == 'bubble_sum':
                            self.updateFields_hatB_bubble_sum()
                        elif self.fieldMethod == 'bubble_sum_mixed_SBR':
                            if not self.crossResponse:
                                self.updateFields_hatB_bubble_sum_mixed_SBR()
                            else:
                                if self.initialization == 'Poisson':
                                    self.updateFields_hatB_bubble_sum_mixed_SBR_crossResponse()
                                elif self.initialization == 'Fixed':
                                    self.updateFields_hatB_bubble_sum_mixed_SBR_crossResponse_fixed_initialization()
                        elif self.fieldMethod == 'bubble_sum_mixed_SBR_C_loop':
                            self.updateFields_hatB_bubble_sum_mixed_SBR_crossResponse_C_loop()
                        
                    else:
                        if self.fieldMethod == None:
                            self.updateFields_no_C()
                        elif self.fieldMethod == 'split':
                            self.updateFields_beta_no_C()
                        elif self.fieldMethod == 'split_ABC':
                            self.ABCupdateFields_no_C()
                        elif self.fieldMethod == 'split_AA_A':
                            self.AA_AupdateFields_no_C()
                        elif self.fieldMethod == 'AA_A_finite_power':
                            self.AA_AupdateFields_no_C_finite_power()
                        elif self.fieldMethod == 'AA_A_all_corrections':
                            self.AA_AupdateFields_no_C_all_corrections()
                        elif self.fieldMethod == 'AA_A_all_phi_minus_one':
                            self.AA_AupdateFields_no_C_phi_minus_one()
                        elif self.fieldMethod == 'AA_A_power_finite':    
                            self.AA_AupdateFields_no_C_finite_power()
                        elif self.fieldMethod == 'bubble_sum':
                            self.updateFields_no_C_bubble_sum()
                        elif self.fieldMethod == 'bubble_sum_old':
                            self.updateFields_no_C_bubble_sum_old()
                        elif self.fieldMethod == 'bubble_sum_shifted':
                            self.updateFields_no_C_bubble_sum_shifted()
                        elif self.fieldMethod == 'bubble_sum_no_inv':
                            self.updateFields_no_C_bubble_sum_wo_inversion()
                        elif self.fieldMethod == 'bubble_sum_faster':
                            self.updateFields_no_C_bubble_sum_faster()
                        elif self.fieldMethod == 'bubble_sum_mixed_SBR':
                            self.updateFields_hatB_bubble_sum_mixed_SBR()
                            
                    self.updateResponses()

                self.EulerStep()

                if Print is 'time':
                    print('Time = ',np.around(self.t,2))
                if Print is 'i':
                    print('step_i = ',self.i)
                
                pbar.update(1)

                        
    def effective_alpha_ABC(self):
        
        alpha_eff = self.alpha + 0.5*self.alpha**2*self.hatTheta2[0,self.i]/(self.y[0,:]*self.y[1,:])/self.k3[0]
        
        self.alpha_eff = alpha_eff
        
    def masterOperator(self,max_num=10):
    
        """
        This function creates the master operator by restricting the state space to a maximum max_num number of particles. The size of the state space is then = (max_num of A)*(max_num of B)....
        """
        try:
            len(max_num)
        except:
            max_num = np.zeros(self.num_species) + max_num
        
        state_space = list(it.product(*list(np.arange(max_num[j]) for j in range(self.num_species))))
        
        for j in range(len(state_space)):
            state_space[j] = list(state_space[j])
        
        master = np.zeros([len(state_space),len(state_space)])
                
        for i in state_space:
            
            for j in range(self.num_species):
                    
                t     = i[:]
                t[j] += 1

                if t[j]-max_num[j] < 0:
                    master[state_space.index(i),state_space.index(t)] += self.k2[j]*state_space[state_space.index(t)][j]
                    
                t     = i[:]
                t[j] -= 1

                if t[j] >= 0:
                    master[state_space.index(i),state_space.index(t)] += self.k1[j]
                    
            for k in range(self.num_int):
                    
                if all([(state_space[0][m] - i[m] + self.s_i[k][m] - self.r_i[k][m]) <= 0 for m in range(len(i))] ) and all([(state_space[-1][m] - i[m] + self.s_i[k][m] - self.r_i[k][m]) >=0 for m in range(len(i))] ):
                    
                    x = self.alpha*self.k3[k]
                    for n in range(self.num_species):
                        for p in range(int(self.r_i[k][n])):
                            x *= (state_space[state_space.index([(i[m] - self.s_i[k][m] + self.r_i[k][m]) for m in range(len(i))])][n]-p)
                    
                    master[state_space.index(i),state_space.index([(i[m] - self.s_i[k][m] + self.r_i[k][m]) for m in range(len(i))])] += x
        
        np.fill_diagonal(master,-np.sum(master,axis=0))
        
        self.master = master
        self.master_stateSpace = state_space
        self.master_maxNum = max_num
    
    def SteadyState_masterOP(self,max_num=10):
        
        """
        This outputs the steady state from the probability distribution found by the top Eigenvector of the Master Operator.
        
        """
        try:
            self.master
        except:
            self.masterOperator(max_num)
            
        evalue,evector = np.linalg.eig(self.master)
        x = np.zeros(self.num_species)
        
        for j in range(self.num_species):
            x[j] = np.sum(np.array(self.master_stateSpace)[:,j]* np.abs(evector[:,np.argmax(evalue)]))/np.sum(np.abs(evector[:,np.argmax(evalue)]))
        
        #for i in range(len(self.master_stateSpace)):
        #    for j in range(self.num_species):
        #        x[j] += (self.master_stateSpace)[i][j]* np.abs(evector[i,np.argmax(evalue)])/np.sum(np.abs(evector[:,np.argmax(evalue)]))
        
        self.ss_masterOP = x
        
        return evalue,evector
    
    
    def runDynamics_masterOP(self,max_num=10,method='Euler',variance=False,crossCorrelator=None,selfCorrelator_tau=None, crossCorrelator_tau=None, measureResponse=None, measureResponse_full=None, measureResponse_finite=None, measureResponse_finite_cross=None,initialization='poisson',initialization_par=None,return_probability=False):
        
        """
        Run the dynamics of the probability distribution dP/dt = MP where M is the master operator, and outputs the mean for each species. Method is the integration method. Default is Euler first order, other options are RK2 and RK4. Initialization and initialization_par are for choosing other probability distributions and to pass extra parameters to it!
        """
        
        def calculate_mean(self,p):
            
            """
            Calculates the mean number for any given p for all species.
            """
            
            x = np.zeros(self.num_species)
            
            for j in range(self.num_species):
                x[j] = np.sum(np.array(self.master_stateSpace)[:,j]*np.abs(p),axis=0)/np.sum(np.abs(p))
            
            #for i in range(len(self.master_stateSpace)):
            #    for j in range(self.num_species):
            #        x[j] += (self.master_stateSpace)[i][j]*np.abs(p[i])/np.sum(np.abs(p))
            
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
            
            for i in range(len(self.master_stateSpace)):
                np.array(self.master_stateSpace)[:,crossC[0]]
            
            return np.sum(np.array(self.master_stateSpace)[:,crossC[0]]*np.array(self.master_stateSpace)[:,crossC[1]]*np.abs(p),axis=0)/np.sum(np.abs(p))        
        
        def mean_operator(self):
            #TODO
            
            # Calculate the operator which tells the mean.
            mean_operator
        
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
            self.masterOperator(max_num)
            
        if variance:
            self.variance = np.zeros([self.num_species,len(self.timeGrid)])
        
        if crossCorrelator is not None:
            self.crossC = np.zeros(len(self.timeGrid))
            
        if measureResponse is not None:
            # Measures the Response function by creating perturbation in the creation rate by changing it to k1*measureResponse, and then going to the original k1            
            self.Response = np.zeros([self.num_species,len(self.timeGrid),len(self.timeGrid)])
            
        if measureResponse_full is not None:
            # Measures the Response function by creating perturbation in the creation rate by changing it to k1*measureResponse.
            self.Response = np.zeros([self.num_species,len(self.timeGrid),len(self.timeGrid)])
        
        if measureResponse_finite is not None:
            # Measures the Response function by creating perturbation in the creation rate for a time measureResponse_finite[1] by changing it to k1*measureResponse and then setting it back to the original value.
            
            self.Response = np.zeros([self.num_species,len(self.timeGrid),len(self.timeGrid)])
            
        if measureResponse_finite_cross is not None:
            # Measures the Response function by creating perturbation in the creation rate for a time measureResponse_finite[1] by changing it to k1*measureResponse and then setting it back to the original value.
            
            self.Response = np.zeros([self.num_species,self.num_species,len(self.timeGrid),len(self.timeGrid)])            
        
        
        init_dist = self.stateSpace_initialDistribution(self.master_maxNum,initialization=initialization, initialization_par=initialization_par)
        p = init_dist[:]
        
        if selfCorrelator_tau is not None:
            
            # To calculate the self correlator as a function of time lag \tau, we need to store this value for all solution times and for all lag times
            
            correlator_tau = np.zeros([self.num_species,len(self.timeGrid),len(self.timeGrid)])
            
        if crossCorrelator_tau is not None:
            
            # To calculate the cross correlator as a function of time lag \tau, we need to store this value for all solution times and for all lag times and for pairs of species. Use only one of selfCorrelator_tau or crossCorrelator_tau
            
            correlator_tau = np.zeros([self.num_species,self.num_species,len(self.timeGrid),len(self.timeGrid)])
        
        if method == 'Euler':
            
            with tqdm(total=len(self.timeGrid)) as pbar:
                self.y[:,self.i] = calculate_mean(self,p)
                if selfCorrelator_tau is not None:
                    p_full[:,self.i] = p               
                pbar.update(1)
                
                if measureResponse is not None:
                    k1_true = np.copy(self.k1)
                    master_true = np.copy(self.master)
                    self.k1 = self.k1*measureResponse
                    self.masterOperator(max_num)

                    tau = self.i
                    q = np.copy(p)
                    q = np.matmul((self.master-master_true)/(k1_true*(measureResponse-1)),q)

                    while tau < len(self.timeGrid):
                        self.Response[:,tau,self.i] = calculate_mean(self,q)
                        #self.Response[:,tau,self.i] = calculate_mean(self,q)/(k1_true*(measureResponse-1))
                        q += self.delta_t*np.matmul(master_true,q)
                        #q += self.delta_t*np.matmul(self.master,q)
                        tau += 1
                    self.k1 = k1_true
                    self.masterOperator(max_num)
                    
                if measureResponse_full is not None:
                    k1_true = np.copy(self.k1)
                    master_true = np.copy(self.master)
                    self.k1 = self.k1*measureResponse_full
                    self.masterOperator(max_num)

                    tau = self.i
                    q = np.copy(p)
                    
                    while tau < len(self.timeGrid):
                        self.Response[:,tau,self.i] = calculate_mean(self,q)
                        q += self.delta_t*np.matmul(self.master,q)
                        tau += 1
                        
                    self.k1 = k1_true
                    self.masterOperator(max_num)
                    
                if measureResponse_finite is not None:
                    k1_true = np.copy(self.k1)
                    master_true = np.copy(self.master)
                    self.k1 = self.k1*measureResponse_finite[0]
                    self.masterOperator(max_num)

                    tau = self.i
                    q = np.copy(p)
                    
                    while tau-self.i < measureResponse_finite[1] and tau < len(self.timeGrid):
                        #propogate the perturbed solution only for the given time measureResponse_finite[1]
                        self.Response[:,tau,self.i] = calculate_mean(self,q)
                        q += self.delta_t*np.matmul(self.master,q)
                        tau += 1
                        
                    while tau < len(self.timeGrid):
                        #propagate the original solution
                        self.Response[:,tau,self.i] = calculate_mean(self,q)
                        q += self.delta_t*np.matmul(master_true,q)
                        tau += 1
                        
                    self.k1 = k1_true
                    self.masterOperator(max_num)
                    
                if measureResponse_finite_cross is not None:
                    k1_true = self.k1.copy()
                    master_true = np.copy(self.master)
                    
                    for j1 in range(self.num_species):
                        
                        temp = np.ones(self.num_species)
                        temp[j1] = measureResponse_finite_cross[0]
                        self.k1 = k1_true*temp
                        self.masterOperator(max_num)
                        # j1 is the species who's creation rate has been perturbed

                        tau = self.i
                        q = np.copy(p)

                        while tau-self.i < measureResponse_finite_cross[1] and tau < len(self.timeGrid):
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
                        self.masterOperator(max_num)
                
                while self.i < len(self.timeGrid)-1:
                    p += self.delta_t*np.matmul(self.master,p)
                    self.y[:,self.i+1] = calculate_mean(self,p)
                    #if selfCorrelator_tau is not None:
                    #    p_full[:,self.i+1] = p
                    if variance:
                        self.variance[:,self.i+1] = calculate_secondMoment(self,p) - self.y[:,self.i+1]**2
                    
                    if crossCorrelator is not None:
                        self.crossC[self.i+1] = calculate_crossCorrelator(self,p,crossCorrelator) - self.y[crossCorrelator[0],self.i+1]*self.y[crossCorrelator[1],self.i+1]
                        
                    if measureResponse_full is not None:
                        k1_true = np.copy(self.k1)
                        master_true = np.copy(self.master)
                        self.k1 = self.k1*measureResponse_full
                        self.masterOperator(max_num)

                        tau = self.i+1
                        q = np.copy(p)
                        while tau < len(self.timeGrid):
                            self.Response[:,tau,self.i+1] = calculate_mean(self,q)
                            q += self.delta_t*np.matmul(self.master,q)
                            tau += 1

                        self.k1 = k1_true
                        self.masterOperator(max_num)
                        
                    if measureResponse is not None:
                        k1_true = np.copy(self.k1)
                        master_true = np.copy(self.master)
                        self.k1 = self.k1*measureResponse
                        self.masterOperator(max_num)
                        
                        tau = self.i+1
                        q = np.copy(p)
                        q = np.matmul((self.master-master_true)/(k1_true*(measureResponse-1)),q)
                        while tau < len(self.timeGrid):
                            self.Response[:,tau,self.i+1] = calculate_mean(self,q)
                            #self.Response[:,tau,self.i+1] = calculate_mean(self,q)/(k1_true*(measureResponse-1))
                            q += self.delta_t*np.matmul(master_true,q)
                            #q += self.delta_t*np.matmul(self.master,q)
                            tau += 1
                        self.k1 = k1_true
                        self.masterOperator(max_num)
                        
                    if measureResponse_finite is not None:
                        k1_true = np.copy(self.k1)
                        master_true = np.copy(self.master)
                        self.k1 = self.k1*measureResponse_finite[0]
                        self.masterOperator(max_num)

                        tau = self.i+1
                        q = np.copy(p)

                        while tau-self.i-1 < measureResponse_finite[1] and tau < len(self.timeGrid):
                            #propogate the perturbed solution only for the given time measureResponse_finite[1]
                            self.Response[:,tau,self.i+1] = calculate_mean(self,q)
                            q += self.delta_t*np.matmul(self.master,q)
                            tau += 1

                        while tau < len(self.timeGrid):
                            #propagate the original solution
                            self.Response[:,tau,self.i+1] = calculate_mean(self,q)
                            q += self.delta_t*np.matmul(master_true,q)
                            tau += 1

                        self.k1 = k1_true
                        self.masterOperator(max_num)
                        
                    if measureResponse_finite_cross is not None:
                        k1_true = self.k1.copy()
                        master_true = np.copy(self.master)
                        
                        for j1 in range(self.num_species):
                            temp = np.ones(self.num_species)
                            temp[j1] = measureResponse_finite_cross[0]
                            self.k1 = k1_true*temp
                            
                            self.masterOperator(max_num)
                            
                            tau = self.i+1
                            q = np.copy(p)

                            while tau-self.i-1 < measureResponse_finite_cross[1] and tau < len(self.timeGrid):
                                #propogate the perturbed solution only for the given time measureResponse_finite[1]
                                self.Response[:,j1,tau,self.i+1] = calculate_mean(self,q)
                                q += self.delta_t*np.matmul(self.master,q)
                                tau += 1

                            while tau < len(self.timeGrid):
                                #propagate the original solution
                                self.Response[:,j1,tau,self.i+1] = calculate_mean(self,q)
                                q += self.delta_t*np.matmul(master_true,q)
                                tau += 1

                            self.k1 = k1_true
                            self.masterOperator(max_num)
                                                
                    self.i += 1
                    self.t += self.delta_t
                    pbar.update(1)
                    
            if measureResponse_full is not None:
                for k in range(len(self.timeGrid)):
                    l = k
                    while l < len(self.timeGrid):
                        self.Response[:,l,k] = (self.Response[:,l,k] - self.y[:,l])/(k1_true*(measureResponse_full-1))
                        l += 1
                        
            if measureResponse_finite is not None:
                for k in range(len(self.timeGrid)):
                    l = k
                    while l < len(self.timeGrid):
                        self.Response[:,l,k] = (self.Response[:,l,k] - self.y[:,l])/(k1_true*(measureResponse_finite[0]-1))/(measureResponse_finite[1]*self.delta_t)
                        l += 1
                        
            if measureResponse_finite_cross is not None:
                for k in range(len(self.timeGrid)):
                    l = k
                    while l < len(self.timeGrid):
                        for j1 in range(self.num_species):
                            for j2 in range(self.num_species):
                                self.Response[j1,j2,l,k] = (self.Response[j1,j2,l,k] - self.y[j1,l])/(k1_true[j2]*(measureResponse_finite_cross[0]-1))/(measureResponse_finite_cross[1]*self.delta_t)
                        l += 1
            
        if method == 'RK2':

            sol = sc.integrate.solve_ivp(wrapper,(self.timeGrid[0],self.timeGrid[-1]),init_dist,method='RK23',t_eval=self.timeGrid,dense_output=False)

            with tqdm(total=len(self.timeGrid)) as pbar:
                while self.i < len(self.timeGrid):
                    self.y[:,self.i] = calculate_mean(self,sol.y[:,self.i])
                    #if selfCorrelator_tau is not None:
                    #    p_full[:,self.i] = sol.y[:,self.i]
                    if variance:
                        self.variance[:,self.i] = calculate_secondMoment(self,sol.y[:,self.i]) - self.y[:,self.i]**2
                    if crossCorrelator is not None:
                        self.crossC[self.i] = calculate_crossCorrelator(self,sol.y[:,self.i],crossCorrelator) - self.y[crossCorrelator[0],self.i]*self.y[crossCorrelator[1],self.i]
                        
                    self.i += 1
                    self.t += self.delta_t

                    pbar.update(1)

        if method == 'RK4':

            sol = sc.integrate.solve_ivp(wrapper,(self.timeGrid[0],self.timeGrid[-1]),init_dist,method='RK45',t_eval=self.timeGrid,dense_output=False)

            with tqdm(total=len(self.timeGrid)) as pbar:
                while self.i < len(self.timeGrid):
                    self.y[:,self.i] = calculate_mean(self,sol.y[:,self.i])
                    #if selfCorrelator_tau is not None:
                    #    p_full[:,self.i] = sol.y[:,self.i]
                    if variance:
                        self.variance[:,self.i] = calculate_secondMoment(self,sol.y[:,self.i]) - self.y[:,self.i]**2
                    if crossCorrelator is not None:
                        self.crossC[self.i] = calculate_crossCorrelator(self,sol.y[:,self.i],crossCorrelator) - self.y[crossCorrelator[0],self.i]*self.y[crossCorrelator[1],self.i]
                    
                    self.i += 1
                    self.t += self.delta_t

                    pbar.update(1)
                    
            if return_probability is not False:
                self.probability = sol.y
                    
        if method == 'Eigenvalue':
            
            evalue,evRight = np.linalg.eig(self.master)
            #evalue,evLeft,evRight = sc.linalg.eig(self.master,left=True,right=True)
            
            #print(evalue)
            if any(evalue == 0):
                #evLeft         = self.fix_zero_eingevalues(evalue,evRight)
                evLeft         = np.linalg.pinv(evRight)
            else:            
                evLeft         = np.linalg.inv(evRight)
                
            ini_proj       = np.matmul(evLeft,init_dist)
            self.evLeft    = evLeft
            self.evRight   = evRight
            
            with tqdm(total=len(self.timeGrid)) as pbar:
                
                #print(np.real(np.matmul(evRight,ini_proj))-init_dist)
                
                if measureResponse is not None:
                    k1_true     = np.copy(self.k1)
                    master_true = np.copy(self.master)
                    self.k1     = self.k1*measureResponse
                    self.masterOperator(max_num)
                    
                    evalue_new,evRight_new = np.linalg.eig((self.master-master_true)/(k1_true*(measureResponse-1)))
                    #evalue_new,evLeft_new,evRight_new = sc.linalg.eig((self.master-master_true)/(k1_true*(measureResponse-1)),left=True,right=True)
                    #evLeft_new             = np.linalg.inv(evRight_new)
                    #ini_proj               = np.matmul(evLeft,init_dist)

                    tau = self.i
                    q   = np.copy(p)
                    q   = np.matmul((self.master-master_true)/(k1_true*(measureResponse-1)),q)

                    while tau < len(self.timeGrid):
                        self.Response[:,tau,self.i] = calculate_mean(self,q)
                        q += self.delta_t*np.matmul(master_true,q)
                        tau += 1
                    self.k1 = k1_true
                    self.masterOperator(max_num)
                
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


                                Q[j] = np.real(np.matmul(evRight,np.matmul(np.identity(len(evalue))*np.exp(evalue*self.delta_t),np.matmul(evLeft,Q[j]))))

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
        
        if selfCorrelator_tau is not None:
            self.correlator_tau = correlator_tau
            
        if crossCorrelator_tau is not None:
            self.correlator_tau = correlator_tau
            
            
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
    
    
    def fix_zero_eingevalues(self,evalue,evec):
        
        zero_ev_index       = np.argwhere(evalue==np.amax(evalue))[:,0]
        zero_evecs_right    = np.zeros([len(zero_ev_index),len(evec)])
        nonzero_evecs_right = np.zeros([len(evalue) - len(zero_ev_index), len(evalue) - len(zero_ev_index)])

        for i in range(len(zero_ev_index)):
            zero_evecs_right[i,:] = evec[:,zero_ev_index[i]]

        x = [i for i in range(len(evalue)) if i not in zero_ev_index]

        for i in range(len(nonzero_evecs_right)):
            for j in range(len(nonzero_evecs_right)):
                nonzero_evecs_right[j,i] = evec[x[j],x[i]]

        nonzero_evecs_left  = np.linalg.inv(nonzero_evecs_right)
        combined_evecs_left = np.zeros([len(evalue),len(evalue)])

        for i in range(len(nonzero_evecs_right)):
            for j in range(len(nonzero_evecs_right)):
                combined_evecs_left[x[j],x[i]] = nonzero_evecs_left[j,i]

        for i in zero_ev_index:
            for j in zero_ev_index:
                #print(evec[i,j])
                combined_evecs_left[j,i] = evec[i,j]
        
        return combined_evecs_left

    
    def stateSpace_initialDistribution(self,max_num=10,initialization='poisson',initialization_par=None):
        
        try:
            len(max_num)
        except:
            max_num = np.zeros(self.num_species) + max_num
        
        try:
            self.master_stateSpace
        except:
            self.masterOperator(self,max_num)
            
            
        if initialization == 'poisson':
            p = np.ones(len(self.master_stateSpace))
            j = 0
            
            for i in self.master_stateSpace:
                for k in range(self.num_species):
                    p[j] *= poisson.pmf(self.master_stateSpace[j][k],self.y[k,0])    
                j += 1
                
        elif initialization == 'fixed':
            if any(self.y[:,0]%1):
                print('error: Use integer valued initialization for all species')
            else:                
                p = np.zeros(len(self.master_stateSpace))
                p[self.master_stateSpace.index(self.y[:,0].astype(int).tolist())] = 1.
                
        elif initialization == 'uniform':
            #Do this
            p = p
            
        return p/np.sum(p)
    
    
                    
class plot(plefka_system):

    def __init__(self):
        super().__init__()

    # The following routines create plots

    def plotTraces(self,figsize=(20,5)):
        
        """Error bars only for gillespie"""
        
        fig,ax = plt.subplots(figsize=figsize)
        
        try:
            self.gill_stdev
            for i in range(self.num_species):
                ax.plot(self.timeGrid,self.y[i,:],linestyle='dashed',linewidth=6)
                ax.fill_between(self.timeGrid,self.y[i,:]-self.gill_stdev[i,:],self.y[i,:]+self.gill_stdev[i,:],alpha=0.4)
            ax.legend([chr(65 + int(i)) for i in range(self.num_species)])
            ax.set_xlabel('Time')
            ax.set_ylabel('Copy Numbers')
        
        except:
            for i in range(self.num_species):
                ax.plot(self.timeGrid,self.y[i,:],linestyle='dashed')
            ax.legend([chr(65 + int(i)) for i in range(self.num_species)])
            ax.set_xlabel('Time')
            ax.set_ylabel('Copy Numbers')
        
        return fig,ax
        
    def plotDeviation(self,reference,figsize=(20,5)):

        fig,ax = plt.subplots(figsize=figsize)
        
        for i in range(self.num_species):
            ax.plot(self.timeGrid,(self.y[i,:]-reference.y[i,:])/reference.y[i,:],linestyle='dashed')
            try:
                referece.gill_std
                ax.fill_between(self.timeGrid,(self.y[i,:]-reference.y[i,:])/reference.y[i,:] - self.y[i,:]*refence.gill_stdev[i,:]/(reference.y[i,:]**2),(self.y[i,:]-reference.y[i,:])/reference.y[i,:] + self.y[i,:]*reference.gill_stdev[i,:]/(reference.y[i,:]**2),alpha=0.5)
            except:
                pass
        ax.legend([chr(65 + int(i)) for i in range(self.num_species)])
        ax.set_xlabel('Time')
        ax.set_ylabel('Relative deviation')
        
        return fig,ax

    def plotTracesandDeviation(self,reference,figsize=(20,5),relative=True):

        fig,ax = plt.subplots(figsize=figsize,ncols=2,nrows=1)
        
        for i in range(self.num_species):
            ax[0].plot(self.timeGrid,self.y[i,:],linestyle='dashed')

            try:
                self.steadyState
            except AttributeError:
                self.findSteadyState(self.y[:,0])

            #ax[0].scatter(1.1*self.timeGrid[-1],self.steadyState[i],s=20,marker='o')

        ax[0].legend([chr(65 + int(i)) for i in range(self.num_species)])
        ax[0].set_xlabel('Time')
        ax[0].set_ylabel('Copy Numbers')

        for i in range(self.num_species):
            
            if relative:
                ax[1].plot(self.timeGrid,(self.y[i,:]-reference.y[i,:])/reference.y[i,:],linestyle='dashed')
                try:
                    referece.gill_std
                    ax[1].fill_between(self.timeGrid,(self.y[i,:]-reference.y[i,:])/reference.y[i,:] - np.abs(self.y[i,:]*reference.gill_stdev[i,:]/(reference.y[i,:]**2)),(self.y[i,:]-reference.y[i,:])/reference.y[i,:] + np.abs(self.y[i,:]*reference.gill_stdev[i,:]/(reference.y[i,:]**2)),alpha=0.5)
                except:
                    pass
            else:
                ax[1].plot(self.timeGrid,(self.y[i,:]-reference.y[i,:]),linestyle='dashed')
                try:
                    referece.gill_std
                    ax[1].fill_between(self.timeGrid,(self.y[i,:]- reference.y[i,:] - np.abs(reference.gill_stdev[i,:])),(self.y[i,:]-reference.y[i,:] + np.abs(reference.gill_stdev[i,:])),alpha=0.5)
                except:
                    pass
        ax[1].legend([chr(65 + int(i)) for i in range(self.num_species)])
        ax[1].set_xlabel('Time')
        ax[1].set_ylabel('Relative deviation')
        
        return fig,ax

    def plotResponses(self,figsize=(20,5),ncols=3):

        resp = np.copy(self.resp)
        vmin = np.min(resp) 
        vmax = np.max(resp)

        for i in range(self.num_species):    
            for j in range(len(self.timeGrid)):
                resp[i,j,:] = np.roll(self.resp[i,j,::-1],j+1)

        fig, axes = plt.subplots(figsize=figsize,ncols=ncols,nrows=int(np.ceil(float(self.num_species)/ncols)))
        i = 0
        for ax in axes.flat:
            if i < self.num_species:
                im = ax.imshow(resp[i],aspect='equal',extent=[self.timeGrid[0],self.timeGrid[-1],\
                              self.timeGrid[-1],self.timeGrid[0]],vmin=vmin,vmax=vmax)
                ax.set_xlabel(r'$\tau - \tau \prime$')
                ax.set_label(r'$\tau $')
                ax.set_title(r'$R(\tau,\tau \prime)$ for ' + chr(65 + i))
                i += 1

        fig.colorbar(im, ax=axes.ravel().tolist())
        plt.show()
        
        return fig,axes

    def plotCorrelation(self,figsize=(20,5),ncols=3):

        if self.C:
            corr = np.copy(self.corr)
            vmin = np.min(corr)
            vmax = np.max(corr)
            if vmax == 0 and vmin == 0:
                vmax = 1.

            #for i in range(self.num_species):
            #    for j in range(len(self.timeGrid)):
            #        corr[i,j,:] = self.corr[i,j,::-1]

            fig, axes = plt.subplots(figsize=figsize,ncols=ncols,nrows=int(np.ceil(float(self.num_species)/ncols)))
            i = 0
            for ax in axes.flat:
                if i < self.num_species:
                    im = ax.imshow(corr[i],aspect='equal',extent=[self.timeGrid[0],self.timeGrid[-1],\
                                  self.timeGrid[-1],self.timeGrid[0]],vmin=vmin,vmax=vmax)
                    ax.set_xlabel(r'$\tau \prime$')
                    ax.set_label(r'$\tau $')
                    ax.set_title(r'$C(\tau,\tau \prime)$ for ' + chr(65 + i))
                    i += 1

            fig.colorbar(im, ax=axes.ravel().tolist())
            plt.show()
            
        return fig,axes

    def plotHatR(self,figsize=(20,5),ncols=3):

        hatR1 = np.copy(self.hatR1)

        vmin = np.min(hatR1)
        vmax = np.max(hatR1)
        if vmax == 0 and vmin == 0:
            vmax = 1.

        for i in range(self.num_species):
            for j in range(len(self.timeGrid)):
                hatR1[i,j,:] = np.roll(self.hatR1[i,j,::-1],j+1)

        fig, axes = plt.subplots(figsize=figsize,ncols=ncols,nrows=int(np.ceil(float(self.num_species)/ncols)))
        i = 0
        for ax in axes.flat:
            if i < self.num_species:
                im = ax.imshow(hatR1[i],aspect='equal',extent=[self.timeGrid[0],self.timeGrid[-1],\
                              self.timeGrid[-1],self.timeGrid[0]],vmin=vmin,vmax=vmax)
                ax.set_xlabel(r'$\tau - \tau \prime$')
                ax.set_label(r'$\tau $')
                ax.set_title(r"$\hat{R}_1(\tau,\tau\prime)$ for " + chr(65 + i))
                i += 1

        fig.colorbar(im, ax=axes.ravel().tolist())
        plt.show()

        if self.alphaOrder == 2:

            hatR2 = np.copy(self.hatR2)
            vmin  = np.min(hatR2)
            vmax  = np.max(hatR2)

            if vmax == 0 and vmin == 0:
                vmax = 1.

            for i in range(self.num_species):

                for j in range(len(self.timeGrid)):
                    hatR2[i,j,:] = np.roll(self.hatR2[i,j,::-1],j+1)

            fig, axes = plt.subplots(figsize=figsize,ncols=ncols,nrows=int(np.ceil(float(self.num_species)/ncols)))
            i = 0
            for ax in axes.flat:
                if i < self.num_species:
                    im = ax.imshow(hatR2[i],aspect='equal',extent=[self.timeGrid[0],self.timeGrid[-1],\
                                  self.timeGrid[-1],self.timeGrid[0]],vmin=vmin,vmax=vmax)
                    ax.set_xlabel(r'$\tau - \tau \prime$')
                    ax.set_label(r'$\tau $')
                    ax.set_title(r"$\hat{R}_2(\tau,\tau\prime)$ for " + chr(65 + i))
                    i += 1

            fig.colorbar(im, ax=axes.ravel().tolist())
            plt.show()
            
        return fig,axes

    def plotHatB(self,figsize=(20,5),ncols=3):

        hatB1 = np.copy(self.hatB1)
        vmin  = np.min(hatB1)
        vmax  = np.max(hatB1)

        if vmax == 0 and vmin == 0:
            vmax = 1.

        #plt.figure(figsize=figsize)

        #for i in range(self.num_species):            
        #    for j in range(len(self.timeGrid)):
        #        hatB1[i,j,:] = self.hatB1[i,j,::-1]

        fig, axes = plt.subplots(figsize=figsize,ncols=ncols,nrows=int(np.ceil(float(self.num_species)/ncols)))
        i = 0
        for ax in axes.flat:
            if i < self.num_species:
                im = ax.imshow(hatB1[i],aspect='equal',extent=[self.timeGrid[0],self.timeGrid[-1],\
                              self.timeGrid[-1],self.timeGrid[0]],vmin=vmin,vmax=vmax)
                ax.set_xlabel(r'$\tau - \tau\prime$')
                ax.set_label(r'$\tau $')
                ax.set_title(r"$\hat{B}_1(\tau,\tau\prime)$ for " + chr(65 + i))
                i += 1

        fig.colorbar(im, ax=axes.ravel().tolist())
        plt.show()

        if self.alphaOrder == 2:

            hatB2 = np.copy(self.hatB2)
            vmin  = np.min(hatB2)
            vmax  = np.max(hatB2)
            if vmax == 0 and vmin == 0:
                vmax = 1.

            #for i in range(self.num_species):
            #    for j in range(len(self.timeGrid)):
            #        hatB2[i,j,:] = self.hatB2[i,j,::-1]

            fig, axes = plt.subplots(figsize=figsize,ncols=ncols,nrows=int(np.ceil(float(self.num_species)/ncols)))
            i = 0
            for ax in axes.flat:
                if i < self.num_species:
                    im = ax.imshow(hatB2[i],aspect='equal',extent=[self.timeGrid[0],self.timeGrid[-1],\
                                  self.timeGrid[-1],self.timeGrid[0]],vmin=vmin,vmax=vmax)
                    ax.set_xlabel(r'$\tau \prime$')
                    ax.set_label(r'$\tau $')
                    ax.set_title(r"$\hat{B}_2(\tau,\tau\prime)$ for " + chr(65 + i))
                    i += 1

            fig.colorbar(im, ax=axes.ravel().tolist())
            plt.show()
            
        return fig,axes

    def plotHatTheta(self,figsize=(20,5)):

        if self.alphaOrder == 1:

            fig,ax = plt.subplots(figsize=figsize)

            for i in range(self.num_species):
                ax.plot(self.timeGrid,self.hatTheta1[i])
                ax.set_xlabel(r'$\tau $')
                ax.set_ylabel(r'$\tilde{\theta}_1$')

            ax.legend([chr(65 + int(i)) for i in range(self.num_species)])

        if self.alphaOrder == 2:

            fig,ax = plt.subplots(figsize=figsize,ncols=2,nrows=1)

            #ax = fig.subplot(121)
            for i in range(self.num_species):
                ax[0].plot(self.timeGrid,self.hatTheta1[i])
                ax[0].set_xlabel(r'$\tau $')
                ax[0].set_ylabel(r'$\tilde{\theta}_1$')
            ax[0].legend([chr(65 + int(i)) for i in range(self.num_species)])

            #ax.subplot(122)
            for i in range(self.num_species):
                ax[1].plot(self.timeGrid,self.hatTheta2[i])
                ax[1].set_xlabel(r'$\tau $')
                ax[1].set_ylabel(r'$\tilde{\theta}_2$')
            ax[1].legend([chr(65 + int(i)) for i in range(self.num_species)])
            
        return fig,ax

    def plotTimeSlices(self,figsize=(20,5),time_indices=None,quantity='resp',ncols=3):

        if time_indices is None:
            spacing = int(len(self.timeGrid)/10)

        if quantity is 'resp':
            resp = np.copy(self.resp)
        elif quantity is 'hatR1':
            hatR1 = np.copy(self.hatR1)
        elif quantity is 'hatR2':
            hatR2 = np.copy(self.hatR2)

        for i in range(self.num_species):
            for j in range(len(self.timeGrid)):

                if quantity is 'resp':
                    resp[i,j,:] = np.roll(self.resp[i,j,::-1],j+1)
                elif quantity is 'hatR1':
                    hatR1[i,j,:] = np.roll(self.hatR1[i,j,::-1],j+1)
                elif quantity is 'hatR2':
                    hatR2[i,j,:] = np.roll(self.hatR2[i,j,::-1],j+1)

        fig, axes = plt.subplots(figsize=figsize,ncols=ncols,nrows=int(np.ceil(float(self.num_species)/ncols)))
        i = 0

        if time_indices is None:

            for ax in axes.flat:
                if i < self.num_species:

                    if quantity is 'resp':
                        im = ax.plot(self.timeGrid,resp[i,::spacing].T,'--')
                        ax.set_xlabel(r'$\tau - \tau \prime$')
                        ax.set_ylabel(r'$R(\tau,\tau \prime)$ for ' + chr(65 + i))

                    elif quantity is 'hatR1':
                        im = ax.plot(self.timeGrid,hatR1[i,::spacing,:].T,'--')
                        ax.set_xlabel(r'$\tau - \tau \prime$')
                        ax.set_ylabel(r"$\hat{R}_1(\tau,\tau\prime)$ for " + chr(65 + i))

                    elif quantity is 'hatR2':
                        im = ax.plot(self.timeGrid,hatR2[i,::spacing,:].T,'--')
                        ax.set_xlabel(r'$\tau - \tau \prime$')
                        ax.set_ylabel(r"$\hat{R}_2(\tau,\tau\prime)$ for " + chr(65 + i))

                    elif quantity is 'corr':
                        im = ax.plot(self.timeGrid,self.corr[i,::spacing,:].T,'--')
                        ax.set_xlabel(r'$\tau \prime$')
                        ax.set_ylabel(r'$C(\tau,\tau \prime)$ for ' + chr(65 + i))

                    elif quantity is 'hatB1':
                        im = ax.plot(self.timeGrid,self.hatB1[i,::spacing,:].T,'--')
                        ax.set_xlabel(r'$\tau \prime$')
                        ax.set_ylabel(r'$\hat{B}_1(\tau,\tau \prime)$ for ' + chr(65 + i))

                    elif quantity is 'hatB2':
                        im = ax.plot(self.timeGrid,self.hatB2[i,::spacing,:].T,'--')
                        ax.set_xlabel(r'$\tau \prime$')
                        ax.set_ylabel(r'$\hat{B}_2(\tau,\tau \prime)$ for ' + chr(65 + i))

                    #plt.legend(np.around(self.timeGrid[::spacing],2),loc='upper right', bbox_to_anchor=(1.4, 1),title=r'$\tau $')
                    plt.legend(np.around(self.timeGrid[::spacing],2),bbox_to_anchor=(0.8, -0.2), ncol=len(self.timeGrid)/10, title=r'$\tau $',fontsize=18)
                    i += 1

            plt.show()

        else:
            for ax in axes.flat:
                if i < self.num_species:

                    if quantity is 'resp':
                        im = ax.plot(self.timeGrid,resp[i,time_indices,:].T,'--')
                        ax.set_xlabel(r'$\tau - \tau \prime$')
                        ax.set_ylabel(r'$R(\tau,\tau \prime)$ for ' + chr(65 + i))

                    elif quantity is 'hatR1':
                        im = ax.plot(self.timeGrid,hatR1[i,time_indices,:].T,'--')
                        ax.set_xlabel(r'$\tau - \tau \prime$')
                        ax.set_ylabel(r"$\hat{R}_1(\tau,\tau\prime)$ for " + chr(65 + i))

                    elif quantity is 'hatR2':
                        im = ax.plot(self.timeGrid,hatR2[i,time_indices,:].T,'--')
                        ax.set_xlabel(r'$\tau - \tau \prime$')
                        ax.set_ylabel(r"$\hat{R}_2(\tau,\tau\prime)$ for " + chr(65 + i))

                    elif quantity is 'corr':
                        im = ax.plot(self.timeGrid,self.corr[i,time_indices,:].T,'--')
                        ax.set_xlabel(r'$\tau \prime$')
                        ax.set_ylabel(r'$C(\tau,\tau \prime)$ for ' + chr(65 + i))

                    elif quantity is 'hatB1':
                        im = ax.plot(self.timeGrid,self.hatB1[i,time_indices,:].T,'--')
                        ax.set_xlabel(r'$\tau \prime$')
                        ax.set_ylabel(r'$\hat{B}_1(\tau,\tau \prime)$ for ' + chr(65 + i))

                    elif quantity is 'hatB2':
                        im = ax.plot(self.timeGrid,self.hatB2[i,time_indices,:].T,'--')
                        ax.set_xlabel(r'$\tau \prime$')
                        ax.set_ylabel(r'$\hat{B}_2(\tau,\tau \prime)$ for ' + chr(65 + i))

                    #plt.legend(np.around(self.timeGrid[time_indices],2),loc='upper right', bbox_to_anchor=(1.4, 1),title=r'$\tau $')
                    plt.legend(np.around(self.timeGrid[time_indices],2),loc='lower left', ncol=len(time_indices), title=r'$\tau $')
                    
                    i += 1

            plt.show()
            
        return fig,axes
            
            
class analysis(plefka_system):
    
    def __init__(self):
        super().__init__()
    
    def mean_deviation(self,reference,average_over_species=False,relative=True):
        
        """
        Outputs the mean (the mean being over length of time grid) relative deviation of the time trajectory of self from reference. Outputs an array with each entry being for each species if average over species is False.
        
        """
        if average_over_species:
            out = np.zeros(2)
        else:
            out = np.zeros([2,self.num_species])
        
        try:
            reference.y
            if relative:
                if self.EMRE:
                    mean = np.mean(np.abs((self.y+self.eps-reference.y)/reference.y),axis=1,keepdims=False)
                else:
                    mean = np.mean(np.abs((self.y-reference.y)/reference.y),axis=1,keepdims=False)
                #mean = np.mean(np.abs((self.y-reference.y)),axis=1,keepdims=False)/reference.y[:,-1]
                try:
                    reference.gill_stdev
                    std  = np.mean(np.abs((self.y*reference.gill_stdev)/(reference.y**2)),axis=1,keepdims=False)
                except:
                    pass
            else:
                mean = np.mean(np.abs(self.y-reference.y),axis=1,keepdims=False)
                try:
                    reference.gill_stdev
                    std  = np.mean(np.abs(reference.gill_stdev),axis=1,keepdims=False)
                except:
                    pass
        except:
            if relative:
                mean = np.mean(np.abs((self.y-reference[0])/reference[0]),axis=1,keepdims=False)
                try:
                    reference[1]
                    std  = np.mean(np.abs((self.y*reference[1])/(reference[0]**2)),axis=1,keepdims=False)
                except:
                    pass
            else:
                mean = np.mean(np.abs(self.y-reference[0]),axis=1,keepdims=False)
                try:
                    reference[1]
                    std  = np.mean(np.abs(reference[1]),axis=1,keepdims=False)
                except:
                    pass
        
        if average_over_species:
            out[0] = np.mean(mean,keepdims=False)
            try:
                std
                out[1] = np.mean(std,keepdims=False)
                return out
            except:
                return out[0]
        else:
            out[0] = mean
            try:
                std
                out[1] = std
                return out
            except:
                return out[0]
    
    def meanSquared_deviation(self,reference,average_over_species=False,relative=True):
        
        """
        Outputs the mean (the mean being over length of time grid) relative deviation of the time trajectory of self from reference. Outputs an array with each entry being for each species if average over species is False.
        
        """
        if average_over_species:
            out = np.zeros(2)
        else:
            out = np.zeros([2,self.num_species])
        
        try:
            reference.y
            if relative:
                mean = np.mean((self.y-reference.y)**2/reference.y**2,axis=1,keepdims=False)
                try:
                    reference.gill_stdev
                    std  = np.mean(np.abs(2*(self.y-reference.y)*(self.y*reference.gill_stdev)/(reference.y**3)),axis=1,keepdims=False)
                except:
                    pass
            else:
                mean = np.mean((self.y-reference.y)**2,axis=1,keepdims=False)
                try:
                    reference.gill_stdev
                    std  = np.mean(np.abs(2*(self.y-reference.y)*reference.gill_stdev),axis=1,keepdims=False)
                except:
                    pass
        except:
            if relative:
                mean = np.mean(((self.y-reference[0])**2/reference[0]**2),axis=1,keepdims=False)
                try:
                    reference[1]
                    std  = np.mean(np.abs(2*(self.y-reference[0])*(self.y*reference[1])/(reference[0]**3)),axis=1,keepdims=False)
                except:
                    pass
            else:
                mean = np.mean((self.y-reference[0])**2,axis=1,keepdims=False)
                try:
                    reference[1]
                    std  = np.mean(np.abs(2*(self.y-reference[0])*reference[1]),axis=1,keepdims=False)
                except:
                    pass
        
        if average_over_species:
            out[0] = np.mean(mean,keepdims=False)
            try:
                std
                out[1] = np.mean(std,keepdims=False)
                return out
            except:
                out[0]
        else:
            out[0] = mean
            try:
                std
                out[1] = std
                return out
            except:
                return out[0]
            
    
    def steadyState_deviation(self,reference,average_over_species=False,relative=True):
    
        """
        Outputs the mean (the mean being over different last state of Gillespie repeats) relative deviation of the last time value of self from that of reference. Outputs an array with each entry being for each species if average over species is False.
        
        """
        if average_over_species:
            out = np.zeros(2)
        else:
            out = np.zeros([2,self.num_species])
        
        try:
            reference.y
            if relative:
                mean = (np.abs((self.y-reference.y)/reference.y))[:,-1]
                std  = (np.abs((self.y*reference.gill_stdev)/(reference.y**2)))[:,-1]
            else:
                mean = (np.abs(self.y-reference.y))[:,-1]
                std  = (np.abs(reference.gill_stdev))[:,-1]
        except:
            if relative:
                mean = (np.abs((self.y-reference[0])/reference[0]))[:,-1]
                std  = (np.abs((self.y*reference[1])/(reference[0]**2)))[:,-1]
            else:
                mean = (np.abs(self.y-reference[0]))[:,-1]
                std  = (np.abs(reference[1]))[:,-1]
        
        if average_over_species:
            out[0] = np.mean(mean,keepdims=False)
            out[1] = np.mean(std,keepdims=False)
        else:
            out[0] = mean
            out[1] = std
            
        return out
    

    
    def SteadyState_deviation_masterOP(self,average_over_species=False,relative=True):
        
        if average_over_species:
            out = np.zeros(1)
        else:
            out = np.zeros([self.num_species])
        
        if relative:
            mean = np.abs((self.y[:,-1]-self.ss_masterOP)/self.ss_masterOP)
        else:
            mean = np.abs(self.y[:,-1]-self.ss_masterOP)
        
        if average_over_species:
            out = np.mean(mean,keepdims=False)
        else:
            out = mean
            
        return out
        
        
    def loop_alpha(self,alpha_range):
        
        """
        Calculates the Plefka approximations over a range of alpha.
        
        """
        
        
        
        
    def alpha_dependence():
        print("hello World")
        


