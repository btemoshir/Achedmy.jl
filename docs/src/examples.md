# Examples

Complete worked examples demonstrating Achedmy.jl for different biological systems.

## Example 1: Enzyme Kinetics (Michaelis-Menten)

Classic enzyme-substrate dynamics with product formation.

### System Definition

```julia
using Catalyst, Achedmy, Plots

# Michaelis-Menten mechanism
rn_enzyme = @reaction_network begin
    kf, S + E --> SE    # Substrate + Enzyme → Complex
    kr, SE --> S + E    # Complex → Substrate + Enzyme (unbinding)
    kcat, SE --> P + E  # Complex → Product + Enzyme (catalysis)
end

# Parameters (typical enzyme kinetics)
kf = 0.01      # Binding rate
kr = 1.0       # Unbinding rate  
kcat = 0.5     # Catalytic rate

# Initial conditions
S0 = 100       # Substrate molecules
E0 = 10        # Enzyme molecules
SE0 = 0        # Complex (initially zero)
P0 = 0         # Product (initially zero)
```

### Setup and Solve

```julia
# Create system structure
structure = ReactionStructure(rn_enzyme, 
    [:S => S0, :E => E0, :SE => SE0, :P => P0])

# Initialize variables with cross-species responses
# (Important for S-E-SE coupling!)
variables = ReactionVariables(structure, "cross")

# Solve with gSBR
sol = solve_dynamics!(structure, variables,
    selfEnergy = "gSBR",
    tmax = 10.0,
    abstol = 1e-6,
    reltol = 1e-6
)
```

### Analyze Results

```julia
times = sol.t

# Mean densities
μ_S = variables.μ[1, :]   # Substrate
μ_E = variables.μ[2, :]   # Enzyme  
μ_SE = variables.μ[3, :]  # Complex
μ_P = variables.μ[4, :]   # Product

# Plot mean dynamics
plot(times, μ_S, label="⟨S⟩", linewidth=2, xlabel="Time", ylabel="Count")
plot!(times, μ_E, label="⟨E⟩", linewidth=2)
plot!(times, μ_SE, label="⟨SE⟩", linewidth=2)
plot!(times, μ_P, label="⟨P⟩", linewidth=2)
title!("Enzyme Kinetics: Mean Dynamics")

# Equal-time variances
var_S = [variables.N[1,1,i,i] for i in 1:length(times)]
var_SE = [variables.N[3,3,i,i] for i in 1:length(times)]

plot(times, var_S, label="Var(S)", linewidth=2)
plot!(times, var_SE, label="Var(SE)", linewidth=2)
xlabel!("Time")
ylabel!("Variance")
title!("Fluctuations in Substrate and Complex")

# Cross-correlation between S and E
corr_SE = variables.N[1, 2, :, :]  # S-E correlation
heatmap(times, times, corr_SE, 
    xlabel="t′", ylabel="t", 
    title="⟨ΔS(t) ΔE(t′)⟩",
    color=:viridis)
```

### Physical Insights

1. **Mean behavior**: Substrate decreases, product increases, complex reaches steady-state
2. **Fluctuations**: Variance in SE larger than Poisson (σ² > μ) due to binding bursts
3. **Correlations**: S-E anticorrelated (enzyme binding removes substrate)

### Comparison of Methods

```julia
# Compare different approximations
methods = ["MAK", "MCA", "SBR", "gSBR"]
μ_P_methods = []

for method in methods
    vars = ReactionVariables(structure, "single")
    solve_dynamics!(structure, vars, selfEnergy=method, tmax=10.0)
    push!(μ_P_methods, vars.μ[4, :])
end

# Plot comparison
plot(times, μ_P_methods[1], label="MAK", linewidth=2)
plot!(times, μ_P_methods[2], label="MCA", linewidth=2)
plot!(times, μ_P_methods[3], label="SBR", linewidth=2)  
plot!(times, μ_P_methods[4], label="gSBR", linewidth=2, linestyle=:dash)
title!("Product Formation: Method Comparison")
```

---

## Example 2: Gene Regulation (Toggle Switch)

Bistable genetic circuit with mutual repression.

### System Definition

```julia
# Mutual repression network
rn_gene = @reaction_network begin
    k1 / (1 + (P2/K)^n), ∅ --> P1    # P1 production (repressed by P2)
    k2 / (1 + (P1/K)^n), ∅ --> P2    # P2 production (repressed by P1)
    γ1, P1 --> ∅                      # P1 degradation
    γ2, P2 --> ∅                      # P2 degradation
end

# Parameters (bistable regime)
k1 = 10.0     # Max production rate P1
k2 = 10.0     # Max production rate P2  
K = 50.0      # Repression threshold
n = 2.0       # Hill coefficient
γ1 = 1.0      # Degradation rate P1
γ2 = 1.0      # Degradation rate P2
```

### Exploring Bistability

```julia
# Try two different initial conditions
IC_high_P1 = [:P1 => 80, :P2 => 10]
IC_high_P2 = [:P1 => 10, :P2 => 80]

# Simulate both
structure1 = ReactionStructure(rn_gene, IC_high_P1)
variables1 = ReactionVariables(structure1, "single")
sol1 = solve_dynamics!(structure1, variables1, 
    selfEnergy="SBR", tmax=20.0)

structure2 = ReactionStructure(rn_gene, IC_high_P2)
variables2 = ReactionVariables(structure2, "single")
sol2 = solve_dynamics!(structure2, variables2,
    selfEnergy="SBR", tmax=20.0)

# Plot phase portrait
plot(variables1.μ[1,:], variables1.μ[2,:], 
    label="IC: High P1", linewidth=2, 
    xlabel="⟨P1⟩", ylabel="⟨P2⟩")
plot!(variables2.μ[1,:], variables2.μ[2,:], 
    label="IC: High P2", linewidth=2)
scatter!([IC_high_P1[1][2]], [IC_high_P1[2][2]], 
    label="Start 1", markersize=8)
scatter!([IC_high_P2[1][2]], [IC_high_P2[2][2]], 
    label="Start 2", markersize=8)
title!("Toggle Switch: Bistable Dynamics")
```

### Switching Dynamics

Study response to perturbations:

```julia
# Start in State 1 (high P1)
structure = ReactionStructure(rn_gene, IC_high_P1)
variables = ReactionVariables(structure, "cross")

# Solve to equilibrium
sol = solve_dynamics!(structure, variables, 
    selfEnergy="gSBR", tmax=30.0)

# Response function shows how system reacts to perturbations
R_P1_to_P2 = variables.R[1, 2, :, :]  # How P1(t) responds to δP2(t')

heatmap(sol.t, sol.t, R_P1_to_P2,
    xlabel="Perturbation time t′",
    ylabel="Response time t",
    title="R_{P1,P2}(t,t′): Cross-Response")
```

---

## Example 3: SIR Infection Dynamics

Susceptible-Infected-Recovered epidemic model.

### System Definition

```julia
# SIR model with birth/death
rn_sir = @reaction_network begin
    β, S + I --> 2I      # Infection
    γ, I --> R           # Recovery
    μ, S --> ∅           # Natural death (S)
    μ, I --> ∅           # Natural death (I)
    μ, R --> ∅           # Natural death (R)
    μ*N_total, ∅ --> S   # Birth (maintain population)
end

# Parameters (endemic regime)
β = 0.001     # Infection rate
γ = 0.1       # Recovery rate
μ = 0.01      # Birth/death rate
N_total = 1000  # Total population

# Initial conditions (small outbreak)
S0 = 990
I0 = 10
R0 = 0
```

### Epidemic Outbreak

```julia
structure = ReactionStructure(rn_sir, [:S => S0, :I => I0, :R => R0])
variables = ReactionVariables(structure, "single")

# Solve outbreak dynamics
sol = solve_dynamics!(structure, variables,
    selfEnergy = "SBR",
    tmax = 200.0,
    dt = 0.5
)

times = sol.t
μ_S = variables.μ[1, :]
μ_I = variables.μ[2, :]
μ_R = variables.μ[3, :]

# Plot SIR curves
plot(times, μ_S, label="Susceptible", linewidth=2, 
    xlabel="Time (days)", ylabel="Population")
plot!(times, μ_I, label="Infected", linewidth=2)
plot!(times, μ_R, label="Recovered", linewidth=2)
title!("SIR Epidemic Dynamics")

# Find peak infection time
peak_idx = argmax(μ_I)
peak_time = times[peak_idx]
peak_infected = μ_I[peak_idx]

println("Peak infection: $(peak_infected) at time $(peak_time)")
```

### Variance and Stochastic Effects

```julia
# Infection variance shows stochastic fluctuations
var_I = [variables.N[2,2,i,i] for i in 1:length(times)]
std_I = sqrt.(var_I)

plot(times, μ_I, ribbon=std_I, fillalpha=0.3,
    label="⟨I⟩ ± σ", linewidth=2,
    xlabel="Time (days)", ylabel="Infected Count")
title!("Infection Dynamics with Uncertainty")

# Coefficient of variation
CV_I = std_I ./ μ_I
plot(times, CV_I, label="CV(I)", linewidth=2,
    xlabel="Time (days)", ylabel="Coefficient of Variation")
title!("Relative Fluctuations in Infected Population")
```

### R₀ Estimation from Dynamics

```julia
# Basic reproduction number from early growth
# dI/dt ≈ (β⟨S⟩/N - γ)⟨I⟩ at t→0

early_idx = 1:10  # First 10 time points
growth_rates = diff(log.(μ_I[early_idx])) ./ diff(times[early_idx])
avg_growth = mean(growth_rates)

R0_estimated = 1 + avg_growth / γ
R0_theoretical = β * S0 / (γ * N_total)

println("R₀ (theoretical): $(R0_theoretical)")
println("R₀ (from dynamics): $(R0_estimated)")
```

---

## Example 4: Comparing with Gillespie SSA

Validate Achedmy against exact stochastic simulations.

```julia
using JumpProcesses  # For Gillespie SSA

# Simple birth-death process
rn = @reaction_network begin
    k_b, ∅ --> X
    k_d, X --> ∅
end

k_b = 10.0
k_d = 1.0
X0 = 0

# Gillespie simulation (many trajectories)
function gillespie_mean_variance(rn, X0, tmax, n_trajectories=1000)
    # ... (Gillespie implementation)
    # Returns: times, mean_X, var_X
end

times_gillespie, μ_gillespie, var_gillespie = gillespie_mean_variance(rn, X0, 10.0)

# Achedmy simulation
structure = ReactionStructure(rn, [:X => X0])
variables = ReactionVariables(structure, "single")
sol = solve_dynamics!(structure, variables, 
    selfEnergy="MAK", tmax=10.0)  # MAK exact for linear systems

# Compare
plot(times_gillespie, μ_gillespie, label="Gillespie (1000 runs)", 
    marker=:circle, linewidth=2)
plot!(sol.t, variables.μ[1,:], label="Achedmy (MAK)", 
    linewidth=2, linestyle=:dash)
xlabel!("Time")
ylabel!("⟨X⟩")
title!("Birth-Death: Achedmy vs Gillespie")

# Variance comparison
var_achedmy = [variables.N[1,1,i,i] for i in 1:length(sol.t)]
plot(times_gillespie, var_gillespie, label="Gillespie", 
    marker=:circle, linewidth=2)
plot!(sol.t, var_achedmy, label="Achedmy", 
    linewidth=2, linestyle=:dash)
xlabel!("Time")
ylabel!("Var(X)")
title!("Variance: Perfect Agreement")
```

---

## Performance Benchmarking

```julia
using BenchmarkTools

# Benchmark different methods on enzyme kinetics
structure = ReactionStructure(rn_enzyme, [:S=>100, :E=>10, :SE=>0, :P=>0])

for method in ["MAK", "MCA", "SBR", "gSBR"]
    variables = ReactionVariables(structure, 
        method in ["gSBR"] ? "cross" : "single")
    
    t_bench = @belapsed solve_dynamics!(\$structure, \$variables, 
        selfEnergy=\$method, tmax=10.0)
    
    println("\$(method): \$(round(t_bench, digits=3))s")
end

# Typical output:
# MAK: 0.021s
# MCA: 0.145s  
# SBR: 0.183s
# gSBR: 1.247s
```

---

## Tips and Tricks

### Speeding Up Large Systems

```julia
# For large networks, use:
# 1. Coarser time grid
sol = solve_dynamics!(structure, variables, dt=0.5)  # Instead of 0.1

# 2. Relaxed tolerances  
sol = solve_dynamics!(structure, variables, abstol=1e-4, reltol=1e-4)

# 3. Faster methods
sol = solve_dynamics!(structure, variables, selfEnergy="SBR")  # Instead of gSBR
```

### Saving and Loading Results

```julia
using JLD2

# Save results
@save "enzyme_results.jld2" structure variables sol

# Load later
@load "enzyme_results.jld2" structure variables sol

# Continue simulation from saved state
sol_extended = solve_dynamics!(structure, variables,
    selfEnergy="gSBR", tmax=20.0)  # Extends from previous tmax
```

### Extracting Specific Quantities

```julia
# Autocorrelation function for species i
function autocorr(variables, species_idx)
    N_ii = variables.N[species_idx, species_idx, :, :]
    μ_i = variables.μ[species_idx, :]
    
    # Normalized: C(t,t') = ⟨ΔX(t)ΔX(t')⟩ / σ(t)σ(t')
    σ_t = sqrt.(diag(N_ii))
    C = N_ii ./ (σ_t * σ_t')
    
    return C
end

# Lag correlation: C(τ) where τ = t - t'
function lag_correlation(variables, species_idx)
    C = autocorr(variables, species_idx)
    n = size(C, 1)
    
    # Extract τ=0, τ=1, τ=2, ... diagonals
    lags = 0:(n-1)
    C_lag = [mean(diag(C, k)) for k in lags]
    
    return lags, C_lag
end

# Usage
lags, C_S = lag_correlation(variables, 1)  # Substrate autocorrelation
plot(lags, C_S, marker=:circle, label="C_S(τ)", 
    xlabel="Lag τ", ylabel="Correlation")
```

## Next Steps

- Explore your own reaction networks
- Tune parameters for your specific system
- Compare methods to find the best speed/accuracy trade-off
- Check out the [Theory](theory.md) page for mathematical details
- See the [API Reference](api.md) for all available functions
