"""
Event based solutions for the CME solution, also called the Gillespie algorithm

"""

import numpy as np
from tqdm import tqdm

def gillespie_avg(self,num_repeats,initial_values,startTime,endTime,delta_t,max_timesteps,alpha=1., initialization='poisson'):
    """
    Averages the stochastic time traces over "num_repeats" number of runs.    
    
    """

    time_grid       = np.arange(startTime,endTime,delta_t)
    self.timeGrid   = time_grid
    gill            = np.zeros([num_repeats,self.num_species,len(time_grid)])
    self.delta_t    = delta_t
    self.y          = np.zeros([self.num_species,len(self.timeGrid)])
    self.i          = 0
    self.t          = 0.

    for i in tqdm(range(num_repeats)):
        gill[i] = gillespie_transform_timeGrid(self,gillespie(self,initial_values,max_timesteps,endTime,alpha,initialization), self.timeGrid,self.delta_t)
        #print("repeat " + str(i))

    self.y          = np.mean(gill,axis=0,keepdims=False)
    self.gill_stdev = np.std(gill,axis=0,keepdims=False)/np.sqrt(num_repeats)
    #self.gill_stdev = np.std(gill,axis=0,keepdims=False)/np.sqrt(num_repeats)
    self.t          = self.timeGrid[-1]
    self.i          = len(self.timeGrid)


def gillespie(self,initial_values,max_timesteps,endTime,alpha=1.,initialization='poisson'):

    """Outputs the results of the gillespie simulation over the variable event times
    
    """

    # These variables have scope only to this function

    y             = np.zeros([self.num_species,max_timesteps])
    rate          = np.zeros(self.num_reactions)
    t             = 0.
    i             = 0
    time_rxn      = np.zeros(max_timesteps)

    if initialization == 'poisson':
        y[:,0]        = np.random.poisson(initial_values)
    elif initialization == 'fixed':
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
