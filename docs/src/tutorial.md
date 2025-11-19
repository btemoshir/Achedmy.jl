# Getting Started Tutorial

This tutorial will walk you through using Achedmy.jl to simulate chemical reaction networks, from defining the system to analyzing results.

## Prerequisites

Make sure you have installed:

```julia
using Pkg
Pkg.add("Catalyst")
Pkg.add("KadanoffBaym")
Pkg.add("Plots")  # For visualization
```

## Step 1: Define Your Reaction Network

Achedmy uses Catalyst.jl's Domain-Specific Language (DSL) to define reactions:

```julia
using Catalyst

# Simple gene expression: ∅ ⇄ mRNA → Protein
rn = @reaction_network begin
    k_on,  ∅ --> M    # Transcription
    k_off, M --> ∅    # mRNA degradation
    k_p,   M --> M + P  # Translation
    γ_p,   P --> ∅    # Protein degradation
end

# Set parameter values
k_on = 1.0
k_off = 0.1
k_p = 10.0
γ_p = 1.0
```

### Supported Reaction Types

- **Zeroth-order**: `∅ → A` (production)
- **First-order**: `A → B`, `A → ∅` (conversion, degradation)
- **Second-order**: `A + B → C`, `2A → B` (binding, dimerization)
- **Reversible**: Any combination with `⇄`

## Step 2: Create the System Structure

The `ReactionStructure` parses your network and sets initial conditions:

```julia
using Achedmy

# Define initial species counts
initial_conditions = [:M => 0, :P => 0]

# Create structure (handles stoichiometry, rates, etc.)
structure = ReactionStructure(rn, initial_conditions)

# Inspect the structure
println("Number of species: ", structure.num_species)
println("Number of reactions: ", structure.num_interactions)
println("Initial densities: ", structure.μ0)
```

### What's Inside ReactionStructure?

- `stochiometry_prod`: Product stoichiometry matrix (S)
- `stochiometry_react`: Reactant stoichiometry matrix (R)  
- `rate_interaction`: Reaction rates (k)
- `num_species`: Number of species (N)
- `num_interactions`: Number of reactions
- `μ0`: Initial mean densities

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

- `μ`: Mean densities μᵢ(t)
- `R`: Response functions Rᵢⱼ(t,t′)
- `N`: Correlation functions Nᵢⱼ(t,t′)
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
    abstol = 1e-6,          # Absolute tolerance
    reltol = 1e-6,          # Relative tolerance
    dt = 0.1                # Initial time step
)
```

### Choosing an Approximation Method

| Method | Speed | Accuracy | When to Use |
|--------|-------|----------|-------------|
| **MAK** | ⚡⚡⚡⚡ | ⭐ | Quick estimates, mean-field systems |
| **MCA** | ⚡⚡⚡ | ⭐⭐ | Weak fluctuations, perturbative regime |
| **SBR** | ⚡⚡ | ⭐⭐⭐ | Single-species dominant, moderate coupling |
| **gSBR** | ⚡ | ⭐⭐⭐⭐ | Strong coupling, cross-species correlations |

## Step 5: Access Results

After solving, extract the results from `variables`:

```julia
# Time grid (adaptive, from KadanoffBaym.jl)
times = sol.t

# Mean densities over time
μ_M = variables.μ[1, :]  # mRNA
μ_P = variables.μ[2, :]  # Protein

# Equal-time variances
var_M = variables.N[1, 1, :, :]  # Diagonal: σ²_M(t)
var_P = variables.N[2, 2, :, :]

# Cross-correlation (if using "cross" response type)
corr_MP = variables.N[1, 2, :, :]  # ⟨ΔM(t) ΔP(t′)⟩
```

### Two-Time Structure

The correlation functions `N[i,j,t,t′]` and responses `R[i,j,t,t′]` have **two time indices**:

- `t`: Later time (rows)
- `t′`: Earlier time (columns)
- Diagonal `N[i,i,t,t]`: Equal-time variance σᵢ²(t)
- Off-time `N[i,i,t,t′]`: Autocorrelation (t ≠ t′)

## Step 6: Visualization

Plot your results using Plots.jl:

```julia
using Plots

# Plot mean densities
plot(times, μ_M, label="⟨M⟩", xlabel="Time", ylabel="Density")
plot!(times, μ_P, label="⟨P⟩")

# Plot variances
plot(times, [var_M[i,i] for i in 1:length(times)], 
     label="Var(M)", xlabel="Time", ylabel="Variance")
plot!(times, [var_P[i,i] for i in 1:length(times)], 
      label="Var(P)")

# Heatmap of correlation function
heatmap(times, times, corr_MP, 
        xlabel="t′", ylabel="t", title="⟨ΔM(t) ΔP(t′)⟩")
```

## Common Issues and Solutions

### Issue: Simulation is too slow

**Solutions:**
1. Use faster approximation (`"SBR"` or `"MAK"` instead of `"gSBR"`)
2. Increase tolerances (`abstol=1e-4, reltol=1e-4`)
3. Use `"single"` response type instead of `"cross"`
4. Reduce `tmax` or increase `dt`

### Issue: Results are inaccurate

**Solutions:**
1. Use more accurate approximation (`"gSBR"` instead of `"MCA"`)
2. Decrease tolerances (`abstol=1e-8, reltol=1e-8`)
3. Use `"cross"` response type for strongly coupled species
4. Check if initial conditions are in valid regime

### Issue: Memory errors

**Solutions:**
1. Use `"single"` instead of `"cross"` response type
2. Reduce `tmax` (fewer time points)
3. Increase `dt` (coarser grid)
4. Use MAK or MCA (smaller memory footprint)

## Next Steps

- Read the [Theory](theory.md) page to understand the mathematical framework
- Explore [Examples](examples.md) for complete worked examples
- Check the [API Reference](api.md) for detailed function documentation

## Complete Example

Putting it all together:

```julia
using Catalyst, Achedmy, Plots

# Define system
rn = @reaction_network begin
    1.0, ∅ --> M
    0.1, M --> ∅
    10.0, M --> M + P
    1.0, P --> ∅
end

# Set up and solve
structure = ReactionStructure(rn, [:M => 0, :P => 0])
variables = ReactionVariables(structure, "single")
sol = solve_dynamics!(structure, variables, 
                     selfEnergy="gSBR", tmax=10.0)

# Plot results
times = sol.t
plot(times, variables.μ[1,:], label="⟨M⟩")
plot!(times, variables.μ[2,:], label="⟨P⟩")
```

That's it! You've successfully simulated a chemical reaction network with Achedmy.jl. 🎉
