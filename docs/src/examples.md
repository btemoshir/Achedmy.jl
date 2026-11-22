# Examples

Complete worked examples demonstrating Achedmy.jl for different biological systems.
The networks follow the example notebooks in
[`examples/`](https://github.com/btemoshir/Achedmy.jl/tree/main/examples).

Achedmy reads every rate and initial mean copy number from the network's default
values, so each example sets them with `@parameters` and `@species`.

## Example 1: Enzyme Kinetics (Michaelis-Menten)

Classic enzyme-substrate dynamics with product formation.

### System Definition

```julia
using Catalyst, Achedmy, Plots
using LinearAlgebra  # for diag

# Michaelis-Menten mechanism, with initial mean copy numbers and rates as defaults
rn_enzyme = @reaction_network begin
    @species S(t)=1.0 E(t)=0.9 SE(t)=0.1 P(t)=0.0
    @parameters kf=1.0 kr=0.1 kcat=1.0
    kf, S + E --> SE    # Substrate + Enzyme → Complex
    kr, SE --> S + E    # Complex → Substrate + Enzyme (unbinding)
    kcat, SE --> P + E  # Complex → Product + Enzyme (catalysis)
end
```

### Setup and Solve

```julia
# Create system structure
structure_enzyme = ReactionStructure(rn_enzyme)

# Initialize variables with cross-species responses
# (Important for S-E-SE coupling!)
variables_enzyme = ReactionVariables(structure_enzyme, "cross")

# Solve with gSBR
sol_enzyme = solve_dynamics!(structure_enzyme, variables_enzyme,
    selfEnergy = "gSBR",
    tmax = 10.0,
    atol = 1e-3,
    rtol = 1e-2
)
```

### Analyze Results

```julia
times = sol_enzyme.t

# Mean copy numbers
μ_S = variables_enzyme.μ[1, :]   # Substrate
μ_E = variables_enzyme.μ[2, :]   # Enzyme
μ_SE = variables_enzyme.μ[3, :]  # Complex
μ_P = variables_enzyme.μ[4, :]   # Product

# Plot mean dynamics
plot(times, μ_S, label="⟨S⟩", linewidth=2, xlabel="Time", ylabel="Count")
plot!(times, μ_E, label="⟨E⟩", linewidth=2)
plot!(times, μ_SE, label="⟨SE⟩", linewidth=2)
plot!(times, μ_P, label="⟨P⟩", linewidth=2)
title!("Enzyme Kinetics: Mean Dynamics")

# Equal-time variances ("cross" response type: N[i, j, t, t′])
var_S = diag(variables_enzyme.N[1, 1, :, :])
var_SE = diag(variables_enzyme.N[3, 3, :, :])

plot(times, var_S, label="Var(S)", linewidth=2)
plot!(times, var_SE, label="Var(SE)", linewidth=2)
xlabel!("Time")
ylabel!("Variance")
title!("Fluctuations in Substrate and Complex")

# Two-time cross-correlation between S and E
corr_SE = variables_enzyme.N[1, 2, :, :]  # ⟨ΔS(t) ΔE(t′)⟩
heatmap(times, times, corr_SE,
    xlabel="t′", ylabel="t",
    title="⟨ΔS(t) ΔE(t′)⟩",
    color=:viridis)
```

### Physical Insights

1. **Mean behavior**: Substrate is converted to product; the complex rises and then decays as the substrate runs out
2. **Fluctuations**: With copy numbers of order one, the variances are of the same order as the means, so fluctuation corrections to mass-action kinetics matter
3. **Correlations**: S and E are anticorrelated, since binding removes one of each

### Comparison with Mass-Action Kinetics

```julia
# Compare MAK with gSBR using single-species and cross-species responses.
# Each solve has its own adaptive time grid, so keep the times with the trajectory.
results = Dict()
for (method, response) in [("MAK", "single"), ("gSBR", "single")]
    vars = ReactionVariables(structure_enzyme, response)
    sol = solve_dynamics!(structure_enzyme, vars, selfEnergy=method, tmax=10.0)
    results["$method ($response)"] = (sol.t, vars.μ[4, :])
end

plot(results["MAK (single)"]..., label="MAK", linewidth=2)
plot!(results["gSBR (single)"]..., label="gSBR (single)", linewidth=2)
plot!(times, μ_P, label="gSBR (cross)", linewidth=2, linestyle=:dash)
title!("Product Formation: Method Comparison")
```

MCA and SBR are not stable for this network: with `"single"` responses and
`tmax=10.0` the solver fails with `InexactError` (a NaN in the dynamics), and MCA
with `"cross"` responses gives negative means. MCA is a perturbative expansion
and SBR ignores the joint fluctuations of different reactions, so both are
reliable only for weakly coupled systems. Example 2 compares all four methods
on a network where they are stable.

---

## Example 2: Gene Regulation (Negative Autoregulation)

A gene that produces a protein which binds back to the gene and switches it off.
Rates must be mass-action, so the repression is modelled explicitly through binding
rather than with a Hill function.

### System Definition

```julia
# G: active gene, G_star: gene with bound protein (inactive), P: protein
rn_gene = @reaction_network begin
    @species G(t)=0.01 G_star(t)=1.0 P(t)=0.1
    @parameters k_t=10.0 k_b=1.0 k_u=0.1 k_2P=2.0
    k_t, G --> G + P             # Transcription/translation from the active gene
    (k_b, k_u), G + P <--> G_star  # Protein binding switches the gene off
    k_2P, P --> ∅                # Protein degradation
end
```

### Comparison of Methods

```julia
structure_gene = ReactionStructure(rn_gene)

# All four approximations with single-species responses. Each solve has its
# own adaptive time grid, so keep the times with the trajectory.
results_gene = Dict()
for method in ["MAK", "MCA", "SBR", "gSBR"]
    vars = ReactionVariables(structure_gene, "single")
    sol = solve_dynamics!(structure_gene, vars, selfEnergy=method, tmax=3.5)
    results_gene[method] = (sol.t, vars.μ[3, :])
end

# gSBR with cross-species responses, kept for the response functions below
variables_gene = ReactionVariables(structure_gene, "cross")
sol_gene = solve_dynamics!(structure_gene, variables_gene,
    selfEnergy="gSBR", tmax=3.5)

plot(results_gene["MAK"]..., label="MAK", linewidth=2,
    xlabel="Time", ylabel="⟨P⟩")
plot!(results_gene["MCA"]..., label="MCA", linewidth=2)
plot!(results_gene["SBR"]..., label="SBR", linewidth=2)
plot!(results_gene["gSBR"]..., label="gSBR (single)", linewidth=2)
plot!(sol_gene.t, variables_gene.μ[3, :], label="gSBR (cross)", linewidth=2, linestyle=:dash)
title!("Protein Number: Effect of Fluctuations")
```

Mass-action kinetics predicts a protein number about three times larger at
`t = 3.5` than the fluctuation-corrected methods, which agree closely with each
other here.

### Response to Perturbations

```julia
# Response function shows how the system reacts to perturbations
R_G_to_P = variables_gene.R[1, 3, :, :]  # How G(t) responds to δP(t′)

heatmap(sol_gene.t, sol_gene.t, R_G_to_P,
    xlabel="Perturbation time t′",
    ylabel="Response time t",
    title="R_{G,P}(t,t′): Cross-Response")
```

---

## Example 3: SIR Infection Dynamics

Susceptible-Infected epidemic model with recovery.

### System Definition

```julia
using Statistics  # for mean

rn_sir = @reaction_network begin
    @species S(t)=5.0 I(t)=1.0
    @parameters k_i=0.75 k_2I=1.0
    k_i, S + I --> 2I   # Infection
    k_2I, I --> ∅       # Recovery (removal)
end
```

### Epidemic Outbreak

```julia
structure_sir = ReactionStructure(rn_sir)
variables_sir = ReactionVariables(structure_sir, "single")

# Solve outbreak dynamics
sol_sir = solve_dynamics!(structure_sir, variables_sir,
    selfEnergy = "SBR",
    tmax = 10.0,
    dtmax = 0.05
)

times = sol_sir.t
μ_S = variables_sir.μ[1, :]
μ_I = variables_sir.μ[2, :]

# Plot SIR curves
plot(times, μ_S, label="Susceptible", linewidth=2,
    xlabel="Time", ylabel="Population")
plot!(times, μ_I, label="Infected", linewidth=2)
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
# ("single" response type: N[i, t, t′])
var_I = diag(variables_sir.N[2, :, :])
std_I = sqrt.(var_I)

plot(times, μ_I, ribbon=std_I, fillalpha=0.3,
    label="⟨I⟩ ± σ", linewidth=2,
    xlabel="Time", ylabel="Infected Count")
title!("Infection Dynamics with Uncertainty")

# Coefficient of variation
CV_I = std_I ./ μ_I
plot(times, CV_I, label="CV(I)", linewidth=2,
    xlabel="Time", ylabel="Coefficient of Variation")
title!("Relative Fluctuations in Infected Population")
```

### R₀ Estimation from Dynamics

```julia
# Basic reproduction number from early growth:
# d⟨I⟩/dt ≈ (k_i S0 - k_2I)⟨I⟩ at t→0, so R₀ = k_i S0 / k_2I
k_i, k_2I, S0 = 0.75, 1.0, 5.0

early_idx = 1:10  # First 10 time points
growth_rates = diff(log.(μ_I[early_idx])) ./ diff(times[early_idx])
avg_growth = mean(growth_rates)

R0_estimated = 1 + avg_growth / k_2I
R0_theoretical = k_i * S0 / k_2I

println("R₀ (theoretical, mass action): $(R0_theoretical)")
println("R₀ (from early growth of ⟨I⟩): $(R0_estimated)")
```

---

## Example 4: Comparing with Gillespie SSA

Validate Achedmy against exact stochastic simulations.

```julia
using Statistics  # for mean, var

# Simple birth-death process, starting from zero molecules
rn_bd = @reaction_network begin
    @species X(t)=0.0
    @parameters k_b=10.0 k_d=1.0
    k_b, ∅ --> X
    k_d, X --> ∅
end

# Minimal Gillespie SSA for the birth-death process, sampled on a fixed grid
function gillespie_mean_variance(k_b, k_d, X0, tmax; n_traj=1000, n_grid=51)
    tgrid = range(0, tmax, length=n_grid)
    X = zeros(n_traj, n_grid)
    for r in 1:n_traj
        t, x, j = 0.0, X0, 1
        while j <= n_grid
            a_b, a_d = k_b, k_d * x           # propensities
            τ = -log(rand()) / (a_b + a_d)    # time to the next reaction
            while j <= n_grid && tgrid[j] < t + τ
                X[r, j] = x; j += 1
            end
            t += τ
            x += rand() * (a_b + a_d) < a_b ? 1 : -1
        end
    end
    return tgrid, vec(mean(X, dims=1)), vec(var(X, dims=1))
end

times_gillespie, μ_gillespie, var_gillespie = gillespie_mean_variance(10.0, 1.0, 0, 10.0)

# Achedmy simulation
structure_bd = ReactionStructure(rn_bd)
variables_bd = ReactionVariables(structure_bd, "single")
sol_bd = solve_dynamics!(structure_bd, variables_bd,
    selfEnergy="MAK", tmax=10.0)  # MAK exact for linear systems

# Compare
plot(times_gillespie, μ_gillespie, label="Gillespie (1000 runs)",
    marker=:circle, linewidth=2)
plot!(sol_bd.t, variables_bd.μ[1,:], label="Achedmy (MAK)",
    linewidth=2, linestyle=:dash)
xlabel!("Time")
ylabel!("⟨X⟩")
title!("Birth-Death: Achedmy vs Gillespie")

# Variance comparison
var_achedmy = diag(variables_bd.N[1, :, :])
plot(times_gillespie, var_gillespie, label="Gillespie",
    marker=:circle, linewidth=2)
plot!(sol_bd.t, var_achedmy, label="Achedmy",
    linewidth=2, linestyle=:dash)
xlabel!("Time")
ylabel!("Var(X)")
title!("Variance: Agreement up to Sampling Noise")
```

---

## Performance Benchmarking

```julia
# Time the different methods on the gene regulation network. Each run needs
# fresh variables, and the first call of each method includes compilation.
for method in ["MAK", "MCA", "SBR", "gSBR"]
    response = method == "gSBR" ? "cross" : "single"
    solve_dynamics!(structure_gene, ReactionVariables(structure_gene, response),
        selfEnergy=method, tmax=3.5)  # warm-up (compilation)

    vars = ReactionVariables(structure_gene, response)
    t_run = @elapsed solve_dynamics!(structure_gene, vars,
        selfEnergy=method, tmax=3.5)

    println("$(method): $(round(t_run, digits=3))s")
end
```

---

## Tips and Tricks

### Speeding Up Large Systems

```julia
# For large networks, use:
# 1. Relaxed tolerances (fewer adaptive time points)
sol = solve_dynamics!(structure_gene, ReactionVariables(structure_gene, "single"),
    atol=1e-2, rtol=1e-1)

# 2. Single-species responses instead of "cross"
# 3. Faster methods, where they are stable for your network (see Example 1)
sol = solve_dynamics!(structure_gene, ReactionVariables(structure_gene, "single"),
    selfEnergy="SBR")  # Instead of gSBR
```

### Saving and Loading Results

```julia
using Serialization

# Save results
serialize("enzyme_results.jls", (structure_enzyme, variables_enzyme, sol_enzyme))

# Load later (requires the same package versions)
structure_enzyme, variables_enzyme, sol_enzyme = deserialize("enzyme_results.jls")
```

### Extracting Specific Quantities

```julia
using LinearAlgebra, Statistics

# Normalized autocorrelation for species i (needs the "cross" response type)
function autocorr(variables, species_idx)
    N_ii = variables.N[species_idx, species_idx, :, :]

    # Normalized: C(t,t') = ⟨ΔX(t)ΔX(t')⟩ / σ(t)σ(t')
    σ_t = sqrt.(diag(N_ii))
    C = N_ii ./ (σ_t * σ_t')

    return C
end

# Lag correlation C(k), averaged over pairs of time points k steps apart.
# N is stored for t ≥ t′ only, so take the k-th diagonal *below* the main one.
# The grid is adaptive, so k counts time steps, not a fixed time lag.
function lag_correlation(variables, species_idx)
    C = autocorr(variables, species_idx)
    n = size(C, 1)

    lags = 0:(n-1)
    C_lag = [mean(diag(C, -k)) for k in lags]

    return lags, C_lag
end

# Usage
lags, C_S = lag_correlation(variables_enzyme, 1)  # Substrate autocorrelation
plot(lags, C_S, marker=:circle, label="C_S(k)",
    xlabel="Lag k (time steps)", ylabel="Correlation")
```

## Next Steps

- Explore your own reaction networks
- Tune parameters for your specific system
- Compare methods to find the best speed/accuracy trade-off
- Check out the [Theory](theory.md) page for mathematical details
- See the [API Reference](api.md) for all available functions
