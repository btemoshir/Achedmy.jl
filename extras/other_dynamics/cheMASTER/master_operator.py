"""
Functions to create the state space, the initial distribution over the state space, to construct the master operator on this space and find the steady state of the system.

TODO: Separate the state_space function from the master Operator
"""


import numpy as np
from scipy.stats import poisson
import itertools as it


def masterOperator(self,max_num=10):
    
    """
    This function creates the master operator by restricting the state space to a maximum number of particles. The size of the state space is then = (max_num of A)*(max_num of B)....

    Parameters
    ----------
    - self : chemical_reaction_class
    - max_num : int or array of int
                The maximum number of particles or the maximum size for the support of the probability distribution. If int, the same value is used for all species and specific values for each species can be passed as an array otherwise.        

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

                x = self.k3[k]
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

    Paramters
    ---------
    - max_num : int or array of int
                The maximum number of particles or the maximum size for the support of the probability distribution. If int, the same value is used for all species and specific values for each species can be passed as an array otherwise.

    Returns
    --------

    - evalue : the eigenvalues of the master oeprator

    - evector : the eigenvectors corressponding to those eigenvalues

    """

    try:
        self.master
    except:
        masterOperator(self,max_num)

    evalue,evector = np.linalg.eig(self.master)
    x = np.zeros(self.num_species)

    for j in range(self.num_species):
        x[j] = np.sum(np.array(self.master_stateSpace)[:,j]* np.abs(evector[:,np.argmax(evalue)]))/np.sum(np.abs(evector[:,np.argmax(evalue)]))

    self.ss_masterOP = x

    return evalue,evector


def stateSpace_initialDistribution(self,max_num=10,initialization='poisson',initialization_par=None):
    """
    Write docstring
    
    """

    try:
        len(max_num)
    except:
        max_num = np.zeros(self.num_species) + max_num

    try:
        self.master_stateSpace
    except:
        masterOperator(self,max_num)


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
        #Do this TODO
        p = p

    return p/np.sum(p)