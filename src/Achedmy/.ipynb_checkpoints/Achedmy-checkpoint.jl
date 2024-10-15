module Achedmy
"""
Achedmy -- Adaptive CHEmical Dynamics using MemorY

This module implements the memory corrections (SBR - self-consistent bubble resummation) for the dynamics of Chemical Reaction Networks (CRNs), calculating the one point and two point quantities accurately.

The CRNs are defined using catalyst.jl and the module uses KB.jl to solve the resulting two time equations using adaptive time step given some tolerance values.

Author: Moshir Harsh
email : btemoshir@gmail.com

TODO: Implement Correlation function calculation!
TODO: Add support for passing rates and initial values defined outside of Catalyst
"""

#using Pkg
#Pkg.activate()

#import Pkg
#Pkg.add(path="/home/harsh/Work/code/KadanoffBaym.jl")
#include("/home/harsh/Work/code/KadanoffBaym.jl")
#import KandaoffBaym

using Revise #Remove

using KadanoffBaym
using Catalyst
using LinearAlgebra
using BlockArrays
using ChainRulesCore
using Distributions
using PyPlot
using IterTools
using Einsum

include("Struct.jl")
#includet("Struct.jl")
export ReactionStructure

include("Var.jl")
#includet("Var.jl")
export Response, ReactionVariables

include("Dynamics.jl")
#includet("Dynamics.jl")
export solve_dynamics!

end #Module End