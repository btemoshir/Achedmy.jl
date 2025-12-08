<!-- # Achedmy.jl -- Adaptive CHEmical Dynamics using MemorY -->

<!-- This package implements the **memory corrections** to the mean field dynamics of chemcial reaction networks (CRNs) with discrete number intrinsic noise which are significant in the regime of large flucatiations or small molecules.

The observables like the mean molecular numbers $\mu(t)$ for the entire course of the dynamics and all two-time quantities like the response function $R(t,t')$, correlation function $C(t,t')$ and number-number correlation functions $N(t,t')$ are calculated.

The following approximations to the dynamics are implemented in this package:
1. **gSBR** - generalized self-consistent bubble resummation approximation.
2. **SBR** - self-consistent bubble resummation approximation.
3. **MCA** - mode coupling approximation.
4. **MAK** - mass action kinetics (mean field dynamics without memory corrections).

The CRNs are defined using [`Catalyst.jl`](https://docs.sciml.ai/Catalyst/stable/) and at the backend the package uses [`KB.jl`](https://nonequilibriumdynamics.github.io/KadanoffBaym.jl/stable/) to solve the resulting two time equations using adaptive time steps.

Author: Moshir Harsh

Email : btemoshir@gmail.com

Dependencies:
```
Catalyst.jl
KadanoffBaym.jl
LinearAlgebra.jl
....
```



TODO: Implement proper handling of initial correlation $C_{ij}(0,0)$ values! -->


# Achedmy.jl -- Adaptive CHEmical Dynamics using MemorY

<!-- [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
[![CI](https://github.com/btemoshir/achedmy/workflows/CI/badge.svg)](https://github.com/btemoshir/achedmy/actions)  
[![codecov](https://codecov.io/gh/btemoshir/achedmy/branch/main/graph/badge.svg)](https://codecov.io/gh/btemoshir/achedmy) -->

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
[![Julia](https://img.shields.io/badge/Julia-1.6+-9558B2.svg)](https://julialang.org/)  
[![Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://btemoshir.github.io/Achedmy.jl/stable/)  
[![CI](https://github.com/btemoshir/Achedmy.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/btemoshir/Achedmy.jl/actions/workflows/CI.yml)  
[![codecov](https://codecov.io/gh/btemoshir/Achedmy.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/btemoshir/Achedmy.jl)

A Julia package implementing **memory-corrected dynamics** for chemical reaction networks (CRNs) with discrete molecular number fluctuations. Achedmy captures the effects of intrinsic noise that become significant in the regime of small molecule numbers or large fluctuations, going beyond standard mean-field approximations.

**Author:** Moshir Harsh  
**Email:** btemoshir@gmail.com  
**Related Paper:** In preparation 
<!-- Plefka expansion for chemical reaction networks (PRX, in preparation) -->

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Usage Guide](#usage-guide)
6. [Approximation Methods](#approximation-methods)
7. [Examples](#examples)
8. [Package Structure](#package-structure)
9. [Dependencies](#dependencies)
10. [Advanced Usage](#advanced-usage)
11. [Testing](#testing)
12. [Citation](#citation)
13. [License](#license)

---

## Overview

Chemical reaction networks in biological systems often involve small numbers of molecules, leading to significant stochastic fluctuations. Traditional mean-field approaches (Mass Action Kinetics) fail to capture these effects accurately. Achedmy implements a hierarchy of approximations based on **dynamical variational free energy approximation** with **Plefka-type expansion** that systematically incorporates memory corrections to the dynamics.

The CRNs are defined using [`Catalyst.jl`](https://docs.sciml.ai/Catalyst/stable/) and at the backend the package uses [`KB.jl`](https://nonequilibriumdynamics.github.io/KadanoffBaym.jl/stable/) to solve the resulting two time equations using adaptive time steps.

### What Achedmy Computes

- **Mean molecular numbers** $\langle n_i(t) \rangle$ or $\mu_i(t)$ over time
- **Response functions** $R_{ij}(t,t')$ - how perturbations propagate
- **Correlation functions** $C_{ij}(t,t')$ - connected correlations
- **Number-number correlations** $N_{ij}(t,t') = \langle \delta n_i(t) \delta n_j(t') \rangle$
- **Variances and covariances** at equal and unequal times
- **Associated self-energies** $\Sigma_{ij}(t,t')$ encoding memory effects

All quantities are computed with [adaptive two-time solvers](https://nonequilibriumdynamics.github.io/KadanoffBaym.jl/stable/) for efficiency and accuracy. This ensures that the we can simulate multiple orders of magnitude in time scales without excessive computational cost.

---

## Features

- ✅ **Multiple approximation schemes** (gSBR, SBR, MCA, MAK)
- ✅ **Full two-time dynamics** including memory effects
- ✅ **Adaptive time-stepping** using Kadanoff-Baym integrators
- ✅ **Cross-response calculations** for multi-species correlations
- ✅ **Compatible with Catalyst.jl** for easy reaction network definition
- ✅ **Handles both single and cross-response formulations**
- ✅ **Efficient caching** of intermediate calculations

---

## Installation

### Prerequisites

- Julia 1.6 or later
- Git (for cloning the repository)

### Install from Source

```bash
git clone https://github.com/btemoshir/achedmy.git
cd achedmy
```

In Julia REPL:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

### Add as Local Package

```julia
using Pkg
Pkg.develop(path="/path/to/achedmy")
```

###  Use Julia's package manager from terminal
```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Verifying Installation

```julia
using Pkg
Pkg.activate(".")
Pkg.test("Achedmy")  # Run the test suite
```

---

## Quick Start

Here's a minimal example computing the dynamics of a gene regulation system:

```julia
using Achedmy
using Catalyst

# Define the reaction network using Catalyst
gene_system = @reaction_network begin
    @species G(t)=0 P(t)=10
    @parameters k_on=0.1 k_off=1.0 k_p=10.0 k_d=1.0
    (k_on, k_off), 0 <--> G
    k_p, G --> G + P
    k_d, P --> 0
end

# Create structure and variables
structure = Achedmy.ReactionStructure(gene_system)
variables = Achedmy.ReactionVariables(structure, "cross")

# Solve dynamics using gSBR approximation
sol = Achedmy.solve_dynamics!(
    structure, 
    variables,
    selfEnergy = "gSBR",
    tmax = 10.0,
    tstart = 0.0,
    atol = 1e-3,
    rtol = 1e-2
)

# Access results
mean_proteins = variables.μ[2, :]  # Mean protein number over time
variance_proteins = diag(variables.N[2, 2, :, :])  # Variance over time
```

---

## Usage Guide

### Step 1: Define Your Reaction Network

Use [`Catalyst.jl`](https://docs.sciml.ai/Catalyst/stable/) to define your CRN:

```julia
using Catalyst

# Example: Enzyme kinetics (Michaelis-Menten)
enzyme_system = @reaction_network begin
    @species S(t)=1.0 E(t)=0.9 C(t)=0.1 X(t)=0.1
    @parameters k_f=1.0 k_b=0.1 k_d=1.0 k_2X=1.0 k_2S=1.0 k_1S=1.0
    (k_f, k_b), S + E <--> C
    k_d, C --> E + X
    k_2X, X --> 0
    (k_2S, k_1S), S <--> 0
end
```

### Step 2: Create Structure and Variables

```julia
using Achedmy

# Create reaction structure (stoichiometry, rates, etc.)
structure = Achedmy.ReactionStructure(enzyme_system)

# Create variables container
# Use "cross" for full cross-correlations, "single" for single-species only
variables = Achedmy.ReactionVariables(structure, "cross")
```

### Step 3: Solve the Dynamics

```julia
sol = Achedmy.solve_dynamics!(
    structure,
    variables,
    selfEnergy = "gSBR",    # Choose: "gSBR", "SBR", "MCA", "MAK"
    tmax = 10.0,            # Final time
    tstart = 0.0,           # Initial time
    atol = 1e-3,            # Absolute tolerance
    rtol = 1e-2             # Relative tolerance
)
```

### Step 4: Extract Results

```julia
# Time grid
time = sol.t

# Mean trajectories for species i
mean_trajectory = variables.μ[i, :]

# Variance for species i at time t
variance_i = diag(variables.N[i, i, :, :])

# Cross-correlation between species i and j at time (t, t')
cross_corr = variables.N[i, j, t_idx, tp_idx]

# Response function R_ij(t, t')
response = variables.R[i, j, t_idx, tp_idx]

# Correlation function C_ij(t, t')
correlation = variables.C[i, j, t_idx, tp_idx]
```

---

## Approximation Methods

Achedmy implements four approximation schemes of increasing accuracy:

### 1. MAK (Mass Action Kinetics)
- **Description:** Standard mean-field theory, no fluctuations
- **Use case:** Quick baseline comparison
- **Accuracy:** Poor for small copy numbers
- **Cost:** Lowest

### 2. MCA (Mode Coupling Approximation)
- **Description:** Includes correlations via mode coupling
- **Use case:** Intermediate systems
- **Accuracy:** Better than MAK, but limited
- **Cost:** Moderate

### 3. SBR (Self-consistent Bubble Resummation)
- **Description:** Self-consistent treatment of single-species bubbles
- **Use case:** Systems where cross-correlations are weak
- **Accuracy:** Good for weakly coupled species
- **Cost:** Moderate-High

### 4. gSBR (Generalized SBR) ⭐ **Recommended**
- **Description:** Full self-consistent treatment with cross-correlations
- **Use case:** General chemical reaction networks
- **Accuracy:** Best available, validated against master equation
- **Cost:** Highest (but worth it!)

**Rule of thumb:** Always start with gSBR unless computational cost is prohibitive.

---

## Examples

The `examples/` directory contains detailed Jupyter notebooks for three systems:

### 1. Gene Regulation (`gene_regulation.ipynb`)
- Telegraphic model of gene switching
- Compares gSBR, MAK, LNA, Master equation
- Demonstrates corrections to mean-field theory
- Shows importance of memory in bursty dynamics

### 2. Enzyme Kinetics (`enzyme_kinetics.ipynb`)
- Michaelis-Menten kinetics with 4 species
- Full comparison of all methods (gSBR, MAK, MCA, LNA, Master, Gillespie)
- Cross-correlations and cross-responses
- Significant deviations from mean-field predictions

### 3. SIR Infection Dynamics (`SIR_infection_dynamics.ipynb`)
- Epidemic spreading in finite populations
- Time-dependent infection and recovery
- Critical role of fluctuations near transitions
- Population-size effects drastically alter dynamics; mean field predicts infection spreads to entire population, while gSBR correctly predicts extinction in finite populations.

### Running Examples

```bash
cd examples/
jupyter notebook enzyme_kinetics.ipynb
```

**Note:** Examples require additional Python libraries for comparison with other methods:
- `cheMASTER` (Master equation solver)
- `emre` (EMRE/LNA solver)
<!-- - `tqdm` (progress bars - may need disabling, see [Troubleshooting](#troubleshooting)) -->

---

## Package Structure


```
achedmy/
├── src/
│   └── Achedmy/
│       ├── Achedmy.jl          # Main module
│       ├── Cmn.jl              # Coefficient calculations (c_mn)
│       ├── SelfEnergy.jl       # Self-energy Σ computations
│       ├── Struct.jl           # ReactionStructure type definitions
│       ├── BlockOp.jl          # Block operator definitions
│       ├── Dynamics.jl         # Runs the dynamics and integrates the self-energies
│       └── Var.jl              # ReactionVariables type definitions
├── test/
│   ├── runtests.jl            # Main test suite entry point
│   ├── test_structure.jl      # Tests for ReactionStructure
│   ├── test_variables.jl      # Tests for ReactionVariables
│   └── test_dynamics.jl       # Tests for solve_dynamics!
├── examples/
│   ├── enzyme_kinetics.ipynb
│   ├── gene_regulation.ipynb
│   └── SIR_infection_dynamics.ipynb
├── extras/
│   └── other_dynamics/
│       ├── cheMASTER/          # Master equation solver (Python)
│       └── emre/               # EMRE/LNA solver (Python)
├── .github/
│   └── workflows/
│       ├── CI.yml              # Continuous Integration workflow
│       ├── CompatHelper.yml    # Dependency compatibility checker
│       └── TagBot.yml          # Automatic version tagging
├── plots/                      # Generated figures
├── LICENSE                     # MIT License
├── Project.toml                # Project dependencies
└── README.md                   # This file
```

### Key Files

- **`Achedmy.jl`**: Main entry point, exports primary functions
- **`Struct.jl`**: Parses Catalyst reactions into stoichiometry matrices, rates, etc.
- **`Var.jl`**: Storage for means, correlations, responses, self-energies and all other dynamic variables
- **`SelfEnergy.jl`**: Core algorithm - computes memory kernels Σ
- **`Dynamics.jl`**: Integrates the two-time equations using Kadanoff-Baym solvers
- **`BlockOp.jl`**: Defines block operators for efficient matrix operations
- **`Cmn.jl`**: Helper functions for coefficient calculations

---

## Dependencies

### Julia Packages

```julia
Catalyst          # Reaction network DSL
KadanoffBaym      # Adaptive time-stepping for memory equations
LinearAlgebra     # Matrix operations
DifferentialEquations  # ODE solvers
Serialization     # Save/load results
```

### Optional (for examples)

```julia
PyPlot           # Plotting
PyCall           # Python interop
LaTeXStrings     # LaTeX labels
MomentClosure    # Normal closure comparison
```

### Installation

```julia
using Pkg
Pkg.add(["Catalyst", "LinearAlgebra", "DifferentialEquations", 
         "Serialization", "PyPlot", "LaTeXStrings"])
Pkg.add(url="https://github.com/NonequilibriumDynamics/KadanoffBaym.jl")
```

---

## Advanced Usage

### Saving and Loading Results

```julia
using Serialization

# Save
open("results.jls", "w") do f
    serialize(f, (sol=sol, vars=variables))
end

# Load
sol, variables = open("results.jls", "r") do f
    deserialize(f)
end
```

### Parameter Sweeps

```julia
alpha_range = [0.01, 0.1, 1.0, 10.0]
SOL, VAR = [], []

for α in alpha_range
    enzyme_system.defaults[k_f] = α
    
    structure = Achedmy.ReactionStructure(enzyme_system)
    variables = Achedmy.ReactionVariables(structure, "cross")
    sol = Achedmy.solve_dynamics!(structure, variables, selfEnergy="gSBR")
    
    push!(SOL, sol)
    push!(VAR, variables)
end
```

### Custom Initial Conditions

```julia
# Modify initial conditions in Catalyst definition
@reaction_network begin
    @species S(t)=100.0 E(t)=50.0  # Custom initial values
    # ...
end
```

---

## Testing

### Running Tests

To run the full test suite:

```julia
using Pkg
Pkg.activate(".")
Pkg.test("Achedmy")
```

To run specific test files:

```julia
using Pkg
Pkg.activate(".")
include("test/test_structure.jl")
```

### Test Coverage

The test suite covers:
- ✅ Module loading and exports
- ✅ Reaction network structure creation
- ✅ Variable initialization (cross and single response)
- ✅ Dynamics integration for all methods (MAK, MCA, SBR, gSBR)
- ✅ Physical constraints (positivity, causality)
- ✅ Self-energy calculations
- ✅ Example systems (enzyme kinetics, gene regulation, SIR)

### Continuous Integration

The package uses GitHub Actions for automated testing:
- **CI.yml**: Tests on Julia 1.6, 1.9, and latest across Linux, macOS, Windows
- **CompatHelper.yml**: Automatically updates dependency compatibility
- **TagBot.yml**: Automatic version tagging

View build status: [![CI](https://github.com/btemoshir/achedmy/workflows/CI/badge.svg)](https://github.com/btemoshir/achedmy/actions)

### Writing New Tests

To add tests for new features:

1. Create a new test file in `test/` (e.g., `test_newfeature.jl`)
2. Add `include("test_newfeature.jl")` to `test/runtests.jl`
3. Use `@testset` blocks to organize tests
4. Run locally before pushing

Example:
```julia
@testset "New Feature Tests" begin
    @test 1 + 1 == 2
    @test_throws ErrorException error("expected error")
end
```

## Troubleshooting

### Memory Issues with Large Systems

For systems with many species (>10) or long times:
- Use `"single"` response type instead of `"cross"`
- Increase `atol` and `rtol` tolerances
- Reduce time range or increase `dt_min`

### Numerical Instabilities

If you see negative variances or diverging solutions:
- Decrease tolerances (`atol`, `rtol`)
- Check initial conditions are physical (non-negative)
- Try MAK first and then move to SBR, gSBR with single species response and cross species response in order. 
- In case of instabilities, the cross species response gSBR can be more stable than single species response gSBR or any SBR.
- Decrease `dt_max` to force smaller time steps. 

---

## Performance Tips

1. **Use cross vs. single wisely:** Cross-correlations are $O(N^2)$ in memory
2. **Adjust tolerances:** Looser tolerances = faster, but less accurate
3. **Pre-compile:** First run includes compilation overhead
4. **Parallelize parameter sweeps:** Use `@threads` or `pmap`

```julia
using Base.Threads

@threads for α in alpha_range
    # Run simulation
end
```

---

## Citation

If you use Achedmy.jl or gSBR/SBR methods in your research, please cite:

<!-- ```bibtex
@article{harsh2025plefka,
  title={Memory-corrected dynamics of chemical reaction networks via Plefka expansion},
  author={Harsh, Moshir and [Co-authors]},
  journal={Physical Review X},
  year={2025},
  note={In preparation}
} -->
```

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
Pkg.test()  # Run test suite (if available)
```

---

## Roadmap

- [ ] Implement proper handling of initial correlations $C_{ij}(0,0)$
- [ ] Add GPU acceleration for large systems
- [ ] Extend to time-dependent parameters
- [ ] Add more benchmark examples
- [ ] Create Python wrapper for broader accessibility
- [ ] Optimize memory allocation in two-time loops

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

**Moshir Harsh**  
Email: btemoshir@gmail.com  
GitHub: [@yourusername](https://github.com/yourusername) 

- Work done at Institute for Theoretical Physics, University of Göttingen
- Current affiliation: Harvard Medical School, Harvard University

For bug reports and feature requests, please use the [GitHub Issues](https://github.com/yourusername/achedmy/issues) page.

---

## Acknowledgments

- Built on [`Catalyst.jl`](https://docs.sciml.ai/Catalyst/stable/) by the SciML ecosystem
- Uses [`KadanoffBaym.jl`](https://nonequilibriumdynamics.github.io/KadanoffBaym.jl/stable/) for memory equation integration

---

**Happy simulating! 🧪🔬**
