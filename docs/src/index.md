# Achedmy.jl Documentation

**Achedmy.jl** is a Julia package for simulating chemical reaction network dynamics using memory-corrected equations of motion derived from the Plefka expansion. The package provides fast, accurate approximations to the exact stochastic dynamics without requiring computationally expensive Gillespie simulations.

## Overview

Achedmy implements several approximation schemes for computing two-time response and correlation functions:

- **MAK (Mean-field Approximation)**: Fastest, suitable for mean-field dominated systems
- **MCA (Mode Coupling Approximation)**: O(α²) perturbative expansion
- **SBR (Single-species Bubble Resummation)**: Self-consistent bubbles for each species
- **gSBR (Generalized SBR)**: Full cross-species correlations (most accurate)

## Key Features

- 🚀 **Fast**: 10-1000× faster than Gillespie for two-time quantities
- 📊 **Accurate**: Captures memory effects and fluctuations beyond mean-field
- 🔧 **Flexible**: Works with any Catalyst.jl reaction network
- 📈 **Adaptive**: Automatic time-grid refinement via KadanoffBaym.jl
- 🧮 **Complete**: Computes responses, correlations, and equal-time variances

## Quick Example

```julia
using Catalyst, Achedmy

# Define enzyme kinetics network
rn = @reaction_network begin
    k1, S + E --> SE
    k2, SE --> S + E  
    k3, SE --> P + E
end

# Set up system
structure = ReactionStructure(rn, [:S=>100, :E=>10, :SE=>0, :P=>0])
variables = ReactionVariables(structure, "single")

# Solve with gSBR approximation
sol = solve_dynamics!(structure, variables, 
                     selfEnergy="gSBR", 
                     tmax=5.0, 
                     abstol=1e-6)

# Access results
means = variables.μ  # Mean densities
correlations = variables.N  # Two-time correlations
responses = variables.R  # Two-time responses
```

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/btemoshir/Achedmy.jl")
```

## Documentation Structure

```@contents
Pages = [
    "tutorial.md",
    "theory.md",
    "examples.md",
    "api.md"
]
Depth = 2
```

## Citation

If you use Achedmy.jl in your research, please cite:

```bibtex
@article{achedmy2024,
  title={Memory-corrected dynamics of chemical reaction networks},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

## Index

```@index
```
