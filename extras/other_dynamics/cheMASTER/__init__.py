"""
    cheMASTER for the solution of the Chemical Master Equation (CME) for chemical reactions.
    
    This is useful for very precise numerical solution of the CME when the number of molecules involved are small. This works by bounding the state space from above by 'max_num' of molecules.
    
    The reactions can be defined by arbitary reactions and starting from arbitrary values (perferrably small). 
    
    Two time quantities like number correlators or response to different kinds of perturbations can also be measured.
    

    Author: Moshir Harsh
    btemoshir@gmail.com
    
    #TODO: Add proper support for an external time grid.
         : Add spatially segregated reactions.
         : Add support for the Gillespie algorithm.

"""


from .chemical_system import chemical_system_class

from .master_operator import (masterOperator,SteadyState_masterOP,stateSpace_initialDistribution)

from .dynamics import (initialize_dynamics,runDynamics)

from .gillespie import (gillespie,gillespie_avg)
