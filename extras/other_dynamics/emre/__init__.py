"""
    Dynamics for the solution of the Effective Mesoscopic Rate Equations (EMRE) for chemical reactions.
    
    This by definition goes beyond the Linear Noise Approximation (LNA).
    
    The reactions can be defined by arbitary reactions and starting from arbitrary values. 
    
    Two time quantities like number correlators or response to perturbations can also be measured.
    

    Author: Moshir Harsh
    btemoshir@gmail.com
    
    #TODO: Add proper support for an external time grid.
         : Add spatially segregated reactions.

"""


from .chemical_system import chemical_system_class

from .dynamics import (initialize_dynamics,runDynamics)

