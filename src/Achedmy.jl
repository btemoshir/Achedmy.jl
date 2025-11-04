module Achedmy

"""
Achedmy -- Adaptive CHEmical Dynamics using MemorY

This module implements the memory corrections (gSBR - generalized self-consistent bubble resummation), SBR and Mode Coupling approximation for the dynamics of Chemical Reaction Networks (CRNs), calculating the one point and two point quantities accurately.

The CRNs are defined using catalyst.jl and the module uses KB.jl to solve the resulting two time equations using adaptive time step given some tolerance values.

Author: Moshir Harsh
email : btemoshir@gmail.com
"""

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
export ReactionStructure

include("Var.jl")
export Response, ReactionVariables

include("Dynamics.jl")
export solve_dynamics!

end #Module End