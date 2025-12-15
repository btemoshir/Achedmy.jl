"""
    Achedmy

Adaptive CHEmical Dynamics using MemorY

A Julia package for computing memory-corrected dynamics of chemical reaction networks (CRNs) 
with discrete molecular number fluctuations using dynamical free energy and Plefka expansions.

# Approximation Methods

- **gSBR**: Generalized self-consistent bubble resummation (recommended)
- **SBR**: Self-consistent bubble resummation (ignores the joint fluctuations induced by different reactions)
- **MCA**: Mode coupling approximation (O(α^2))
- **MAK**: Mass action kinetics (mean-field)

# Main Types

- [`ReactionStructure`](@ref): Stores reaction network stoichiometry and rates
- [`ReactionVariables`](@ref): Stores dynamical variables (means, correlations, responses, self-energies)

# Main Functions

- [`solve_dynamics!`](@ref): Integrate the memory-corrected equations
- [`ReactionStructure`](@ref): Constructor from Catalyst networks
- [`ReactionVariables`](@ref): Initialize variables container

# Quick Example

```julia
using Achedmy, Catalyst

# Define reaction network
rn = @reaction_network begin
    @species S(t)=1.0 P(t)=0.1
    @parameters k=1.0 d=0.1
    k, S --> S + P
    d, P --> 0
end

# Setup and solve
structure = ReactionStructure(rn)
variables = ReactionVariables(structure, "cross")
sol = solve_dynamics!(structure, variables, 
                     selfEnergy="gSBR", tmax=10.0)

# Extract results
mean_P = variables.μ[2, :]  # Mean protein count
variance_P = diag(variables.N[2,2,:,:])  # Variance over time
```

# Citation

If you use Achedmy.jl in your research, please cite:

Harsh, M. (2025). Memory-corrected dynamics of chemical reaction networks 
via Plefka expansion. In preparation.

# See Also

- [GitHub Repository](https://github.com/btemoshir/achedmy)
- [Catalyst.jl](https://docs.sciml.ai/Catalyst/stable/)
- [KadanoffBaym.jl](https://nonequilibriumdynamics.github.io/KadanoffBaym.jl/)
"""
module Achedmy

using Catalyst
using LinearAlgebra
using BlockArrays
using ChainRulesCore
using Distributions
using PyPlot
using IterTools
using Einsum
using KadanoffBaym
include("KadanoffBaym-1.2.1/KadanoffBaym.jl")
using .KadanoffBaym

include("Struct.jl")
export ReactionStructure

include("Var.jl")
export Response, ReactionVariables

include("Dynamics.jl")
export solve_dynamics!

end #Module End