# Achedmy.jl Documentation

**Achedmy.jl** is a Julia package for simulating chemical reaction network dynamics using memory-corrected equations of motion derived from the Plefka expansion. The package provides fast, accurate approximations to the exact stochastic dynamics without requiring computationally expensive Gillespie simulations.

The theory and methods are described in M. Harsh and P. Sollich, [*Dynamical Variational Theory of Stochastic Reaction Networks*](https://arxiv.org/abs/2611.25491), arXiv:2611.25491 (2026).

## Overview

Achedmy implements several approximation schemes for computing two-time response and correlation functions:

- **MAK (Mass-Action Kinetics)**: Mean-field, no memory; fastest
- **MCA (Mode Coupling Approximation)**: O(α²) perturbative expansion
- **SBR (Self-consistent Bubble Resummation)**: Single-species self-energies, ignores cross-reaction fluctuations
- **gSBR (Generalized SBR)**: Full cross-species and cross-reaction corrections (most accurate)

## Key Features

- 🚀 **Fast**: Much faster than Gillespie for accurate one and two-time quantities in the regime of small copy numbers or large intrinsic noise
- 📊 **Accurate**: Captures memory effects and fluctuations beyond mean-field
- 🔧 **Flexible**: Works with any Catalyst.jl reaction network with polynomial propensities and at most binary reactions
- 📈 **Adaptive**: Automatic time-grid refinement via KadanoffBaym.jl
- 🧮 **Complete**: Computes responses, correlations, and equal-time variances

## Quick Example

```julia
using Catalyst, Achedmy

# Define enzyme kinetics network. Achedmy reads the rates and the initial
# mean copy numbers from the network's default values.
rn = @reaction_network begin
    @species S(t)=1.0 E(t)=0.9 SE(t)=0.1 P(t)=0.0
    @parameters k1=1.0 k2=0.1 k3=1.0
    k1, S + E --> SE
    k2, SE --> S + E
    k3, SE --> P + E
end

# Set up system
structure = ReactionStructure(rn)
variables = ReactionVariables(structure, "cross")

# Solve with gSBR approximation
sol = solve_dynamics!(structure, variables,
                     selfEnergy="gSBR",
                     tmax=5.0,
                     atol=1e-3, rtol=1e-2)

# Access results
means = variables.μ         # μ[i, t]: mean copy numbers
correlations = variables.N  # N[i, j, t, t′]: two-time number-number correlations
responses = variables.R     # R[i, j, t, t′]: two-time responses
```

## Installation

```julia
using Pkg
Pkg.add("Achedmy")
```

For the latest development version, install from GitHub instead:

```julia
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

M. Harsh and P. Sollich, [*Dynamical Variational Theory of Stochastic Reaction Networks*](https://arxiv.org/abs/2611.25491), arXiv:2611.25491 (2026)

```bibtex
@misc{harsh2026dynamical,
  title={Dynamical Variational Theory of Stochastic Reaction Networks},
  author={Harsh, Moshir and Sollich, Peter},
  year={2026},
  eprint={2611.25491},
  archivePrefix={arXiv},
  primaryClass={cond-mat.stat-mech},
  url={https://arxiv.org/abs/2611.25491}
}
```

## Index

```@index
```
