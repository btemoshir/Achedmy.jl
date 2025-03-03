import numpy as np
import scipy as sc
from scipy.integrate import odeint
import sys
import os
import itertools as it
import math
from tqdm import tqdm
from scipy.stats import poisson


class chemical_system_class:
    
    """ 
    Class with methods on it to set up the chemical system and run dynamics on it. 
    
    Parameters
    -----------    
    - num_int : int 
                Number of interacting reactions over the baseline
    - num_species : int
                Total number of reaction species in the system
    - rxn_par : list of floats
                A list of length 3 of arrays with the first array as all the creation rates of the baseline, the second as the destruction rate of the baseline and the third array with the rates of the interaction reactions (the length of first two must be equal to num_sepcies and of the third must be equal to num_int)                
    - r_i : list of int
                A list of list with length = num_int. Each sublist has length = num_species. This list defines the stochiometric coefficients of all the species being destroyed (reactant) as a result of the interaction reactions in order of the rates defined in rxn_par[2]
    - s_i : list of int
            Same as r_i but for the species being created (product) in the interaction reactions.
    
    """
    
    def __init__(self,num_int,num_species,rxn_par,r_i,s_i):
        
        # Assign the numbers to the initial value
        self.num_species   = num_species
        self.num_int       = num_int
        self.r_i           = r_i
        self.s_i           = s_i
        self.num_reactions = 2*num_species + num_int        
        
        # Assign reaction parameters
        self.k1 = rxn_par[0]
        self.k2 = rxn_par[1]
        self.k3 = rxn_par[2]
    

    
    
    