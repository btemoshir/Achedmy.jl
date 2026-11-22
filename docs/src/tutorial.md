# Getting Started Tutorial

This tutorial will walk you through using Achedmy.jl to simulate chemical reaction networks, from defining the system to analyzing results.

## Prerequisites

Make sure you have installed:

```julia
using Pkg
Pkg.add(url="https://github.com/btemoshir/Achedmy.jl")
Pkg.add("Catalyst")
Pkg.add("Plots")  # For visualization
```

## Step 1: Define Your Reaction Network

Achedmy uses Catalyst.jl's Domain-Specific Language (DSL) to define reactions.
Achedmy reads every rate and every initial mean copy number from the network's
**default values**, so give each parameter and species one with `@parameters`
and `@species`:

```julia
using Catalyst

# Simple gene expression: ∅ ⇄ mRNA → Protein
rn = @reaction_network begin
    @species M(t)=0.0 P(t)=0.0             # initial mean copy numbers
    @parameters k_on=1.0 k_off=1.0 k_p=2.0 γ_p=1.0
    k_on,  ∅ --> M      # Transcription
    k_off, M --> ∅      # mRNA degradation
    k_p,   M --> M + P  # Translation
    γ_p,   P --> ∅      # Protein degradation
end
```

Each rate must be a single parameter: plain numbers (`1.0, ∅ --> M`) and rate
expressions (e.g. Hill functions) are not supported.

### Supported Reaction Types

- **Zeroth-order**: `∅ → A` (production)
- **First-order**: `A → B`, `A → ∅` (conversion, degradation)
- **Second-order**: `A + B → C`, `2A → B` (binding, dimerization)
- **Reversible**: Any combination with `⇄`

## Step 2: Create the System Structure

The `ReactionStructure` parses your network, its rates and its initial conditions:

```julia
using Achedmy

# Create structure (handles stoichiometry, rates, etc.)
structure = ReactionStructure(rn)

# Inspect the structure
println("Number of species: ", structure.num_species)
println("Number of interaction reactions: ", structure.num_interactions)
println("Initial mean copy numbers: ", structure.initial_values)
```

To change a rate or initial value without redefining the network, edit the
default in the `@reaction_network` definition.

### What's Inside ReactionStructure?

- `stochiometry_prod`: Product stoichiometry matrix (S)
- `stochiometry_react`: Reactant stoichiometry matrix (R)
- `rate_creation`, `rate_destruction`: Rates of the creation (∅ → A) and degradation (A → ∅) reactions
- `rate_interaction`: Rates of all other reactions (k)
- `num_species`: Number of species (N)
- `num_interactions`: Number of reactions other than simple creation/degradation
- `initial_values`: Initial mean copy numbers

## Step 3: Initialize Variables

`ReactionVariables` stores the dynamical quantities (responses, correlations):

```julia
# Choose response function type:
# "single" = diagonal only (fast, memory efficient)
# "cross"  = full cross-species (slower, more accurate)

variables = ReactionVariables(structure, "single")
```

### Variable Storage

The `variables` object contains:

- `μ`: Mean copy numbers μᵢ(t), indexed `μ[i, t]`
- `R`: Response functions, `R[i, t, t′]` ("single") or `R[i, j, t, t′]` ("cross")
- `N`: Number-number correlation functions, `N[i, t, t′]` ("single") or `N[i, j, t, t′]` ("cross")
- `Σ_R`, `Σ_μ`, `Σ_B`: Self-energies (memory kernels)

## Step 4: Solve the Dynamics

Now run the simulation with your chosen approximation method:

```julia
# Solve with gSBR (generalized SBR - most accurate)
sol = solve_dynamics!(
    structure, 
    variables,
    selfEnergy = "gSBR",   # Approximation: "MAK", "MCA", "SBR", "gSBR"
    tmax = 10.0,            # Maximum simulation time
    atol = 1e-4,            # Absolute tolerance of the adaptive time stepper
    rtol = 1e-3,            # Relative tolerance
)
```

The time grid is adaptive. `dtini` sets the initial step and `dtmax` caps the
step size; see [`solve_dynamics!`](@ref) for all options.

### Choosing an Approximation Method

| Method | Speed | Accuracy | When to Use |
|--------|-------|----------|-------------|
| **MAK** | ⚡⚡⚡⚡ | ⭐ | Quick estimates, mean-field systems |
| **MCA** | ⚡⚡⚡ | ⭐⭐ | Weak fluctuations, perturbative regime |
| **SBR** | ⚡⚡ | ⭐⭐⭐ | Reaction independent fluctuations, moderate coupling |
| **gSBR** | ⚡ | ⭐⭐⭐⭐ | Strong coupling, cross-species cross-reaction correlations |

## Step 5: Access Results

After solving, extract the results from `variables`:

```julia
# Time grid (adaptive, from KadanoffBaym.jl)
times = sol.t

using LinearAlgebra  # for diag

# Mean copy numbers over time
μ_M = variables.μ[1, :]  # mRNA
μ_P = variables.μ[2, :]  # Protein

# Equal-time variances σ²(t): the diagonal of the two-time correlation
# ("single" response type: N[i, t, t′])
var_M = diag(variables.N[1, :, :])
var_P = diag(variables.N[2, :, :])

# Two-time autocorrelation of the protein ⟨ΔP(t) ΔP(t′)⟩
N_PP = variables.N[2, :, :]
```

With the `"cross"` response type the arrays carry two species indices, e.g.
`variables.N[1, 2, :, :]` is the cross-correlation ⟨ΔM(t) ΔP(t′)⟩ and
`diag(variables.N[2, 2, :, :])` the protein variance.

### Two-Time Structure

The correlation functions `N[i,j,t,t′]` and responses `R[i,j,t,t′]` (or `N[i,t,t′]`
and `R[i,t,t′]` for `"single"`) have **two time indices** on the adaptive grid `sol.t`:

- `t`: Later time (rows)
- `t′`: Earlier time (columns); entries with `t′ > t` are zero
- Diagonal `N[i,i,t,t]`: Equal-time variance σᵢ²(t)
- Off-diagonal `N[i,i,t,t′]`: Autocorrelation (t ≠ t′)

## Step 6: Visualization

Plot your results using Plots.jl:

```julia
using Plots

# Plot mean densities
plot(times, μ_M, label="⟨M⟩", xlabel="Time", ylabel="Density")
plot!(times, μ_P, label="⟨P⟩")

# Plot variances
plot(times, var_M, label="Var(M)", xlabel="Time", ylabel="Variance")
plot!(times, var_P, label="Var(P)")

# Heatmap of the protein autocorrelation function
heatmap(times, times, N_PP,
        xlabel="t′", ylabel="t", title="⟨ΔP(t) ΔP(t′)⟩")
```

## Common Issues and Solutions

### Issue: Simulation is too slow

**Solutions:**
1. Use faster approximation (`"SBR"` or `"MAK"` instead of `"gSBR"`)
2. Increase tolerances (e.g. `atol=1e-2, rtol=1e-1`)
3. Use `"single"` response type instead of `"cross"`
4. Reduce `tmax`: the cost grows with the cube of the number of time points

### Issue: Results are inaccurate

**Solutions:**
1. Use more accurate approximation (`"gSBR"` instead of `"MCA"`)
2. Decrease tolerances (e.g. `atol=1e-5, rtol=1e-4`)
3. Use `"cross"` response type for strongly coupled species
4. Check if initial conditions are in valid regime

### Issue: Memory errors

**Solutions:**
1. Use `"single"` instead of `"cross"` response type
2. Reduce `tmax` (fewer time points)
3. Increase `atol`/`rtol` (coarser adaptive grid)
4. Use MAK or MCA (smaller memory footprint)

## Next Steps

- Read the [Theory](theory.md) page to understand the mathematical framework
- Explore [Examples](examples.md) for complete worked examples
- Check the [API Reference](api.md) for detailed function documentation

## Complete Example

Putting it all together:

```julia
using Catalyst, Achedmy, Plots

# Define system (rates and initial copy numbers as defaults)
rn = @reaction_network begin
    @species M(t)=0.0 P(t)=0.0
    @parameters k_on=1.0 k_off=1.0 k_p=2.0 γ_p=1.0
    k_on,  ∅ --> M
    k_off, M --> ∅
    k_p,   M --> M + P
    γ_p,   P --> ∅
end

# Set up and solve
structure = ReactionStructure(rn)
variables = ReactionVariables(structure, "single")
sol = solve_dynamics!(structure, variables, 
                     selfEnergy="gSBR", tmax=10.0)

# Plot results
times = sol.t
plot(times, variables.μ[1,:], label="⟨M⟩")
plot!(times, variables.μ[2,:], label="⟨P⟩")
```

That's it! You've successfully simulated a chemical reaction network with Achedmy.jl.
